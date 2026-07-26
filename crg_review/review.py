"""LLM Review Layer for Code Review Graph - Monolith Implementation"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import httpx
import jsonschema

# Upstream imports
from code_review_graph.tools.review import get_review_context


# =============================================================================
# CONSTANTS
# =============================================================================

SYSTEM_PROMPT = """You are a senior code reviewer with deep expertise in software engineering.
Review the provided code changes thoroughly and constructively.

SEVERITY GUIDELINES:
- CRITICAL: Bugs causing crashes, data loss, security vulnerabilities, incorrect behavior that could be exploited
- WARNING: Performance issues, missing error handling, edge cases that could fail under specific conditions
- INFO: Readability improvements, naming suggestions, structural recommendations, opportunities for simplification

WHAT TO IGNORE:
- Trivial style issues (formatting, whitespace — let linters handle those)
- Subjective preferences unless they violate established conventions
- Changes that are intentional and documented

OUTPUT FORMAT:
Return ONLY valid JSON with this structure:
{
  "summary": "2-3 sentence overall assessment",
  "issues": [
    {"file": "<relative_path>", "line": <int>, "end_line": <int|null>,
     "severity": "critical|warning|info",
     "category": "bug|security|performance|error_handling|readability|architecture",
     "message": "<specific description>", "suggestion": "<concrete fix with code example>"}
  ],
  "positive": ["<what was done well>"]
}
If the code is well-written, say so — don't fabricate issues.
"""

# Auto-injected system prompt addition for the main coding LLM
PAIR_REVIEWER_SYSTEM_ADDITION = """
## 🧠 You have a background pair reviewer running.

### When to check feedback:
- Before declaring any non-trivial task "done"
- After writing new functions, classes, or complex logic
- When you suspect edge cases or security issues

### How to check:
Call `get_reviewer_feedback()` — returns latest structured findings (line, severity, category, message, suggestion).

### On issues found:
- Fix directly if suggestion is clear
- Ask clarifying question if ambiguous
- Don't ignore CRITICAL/WARNING — real bugs
"""

REVIEW_SCHEMA = {
    "type": "object",
    "required": ["summary", "issues", "positive"],
    "properties": {
        "summary": {"type": "string"},
        "positive": {"type": "array", "items": {"type": "string"}},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["file", "line", "severity", "category", "message", "suggestion"],
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": ["integer", "null"], "minimum": 1},
                    "severity": {"type": "string", "enum": ["critical", "warning", "info"]},
                    "category": {"type": "string", "enum": ["bug", "security", "performance", "error_handling", "readability", "architecture"]},
                    "message": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
            },
        },
    },
}

LANG_MAP = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".rs": "rust",
    ".go": "go", ".java": "java", ".cpp": "cpp", ".c": "c",
    ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".kt": "kotlin", ".scala": "scala", ".sql": "sql",
    ".mjs": "javascript", ".cjs": "javascript",
    ".vue": "vue", ".svelte": "svelte", ".astro": "astro",
}

# Shared state for background watcher
STATE = {
    "latest_review": {"summary": "No review yet", "issues": [], "positive": []},
    "watcher_task": None,
    "watcher_running": False,
}


# =============================================================================
# BACKGROUND WATCHER (asyncio polling - no watchdog dependency)
# =============================================================================

async def _watcher_loop(repo_root: str = ".", debounce: float = 2.0) -> None:
    """Background task: polls git for changed files, triggers review on save."""
    last_hash = ""
    while STATE["watcher_running"]:
        try:
            r = subprocess.run(
                ["git", "-C", repo_root, "diff", "--name-only", "HEAD"],
                capture_output=True, text=True,
            )
            changes = r.stdout.strip().splitlines() if r.stdout.strip() else []
            current_hash = "\n".join(changes)
            if changes and current_hash != last_hash:
                await asyncio.sleep(debounce)
                # Re-check after debounce — only review if changes still present
                r2 = subprocess.run(
                    ["git", "-C", repo_root, "diff", "--name-only", "HEAD"],
                    capture_output=True, text=True,
                )
                if r2.stdout.strip():
                    print(f"[crg-review] {len(changes)} file(s) changed", file=sys.stderr)
                    await review_changes(repo_root=repo_root)
                last_hash = current_hash
        except Exception as e:
            print(f"[crg-review] Watcher error: {e}", file=sys.stderr)

        await asyncio.sleep(1.0)  # Poll interval


def start_watcher(repo_root: str = ".", debounce: float = 2.0) -> None:
    """Start the background file watcher."""
    if STATE["watcher_running"]:
        return
    STATE["watcher_running"] = True
    STATE["watcher_task"] = asyncio.create_task(_watcher_loop(repo_root, debounce))


def stop_watcher() -> None:
    """Stop the background file watcher."""
    if STATE["watcher_task"]:
        STATE["watcher_task"].cancel()
        STATE["watcher_task"] = None
    STATE["watcher_running"] = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_language(file_path: str) -> str:
    return LANG_MAP.get(Path(file_path).suffix.lower(), "text")


async def call_llm(
    prompt: str,
    endpoint: str,
    model: str,
    api_key: str,
    timeout: int = 120,
) -> dict:
    """POST /chat/completions to OpenAI-compatible endpoint. Returns raw response dict."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{endpoint.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 4096,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        return resp.json()


def validate_llm_response(content: dict) -> dict:
    jsonschema.validate(content, REVIEW_SCHEMA)
    return content


async def _call_llm_parsed(
    prompt: str,
    endpoint: str,
    model: str,
    api_key: str,
    base_prompt: Optional[str] = None,
    timeout: int = 120,
) -> Optional[dict]:
    """Call LLM, parse JSON, validate schema. One retry on parse failure."""
    content_str = ""
    for attempt in range(2):
        try:
            response = await call_llm(prompt, endpoint, model, api_key, timeout)
            content_str = response["choices"][0]["message"]["content"]
            content = validate_llm_response(json.loads(content_str))
            return {"response": response, "content": content}
        except (json.JSONDecodeError, jsonschema.ValidationError):
            if attempt == 0:
                prompt = (base_prompt or prompt) + "\n\nIMPORTANT: Return ONLY valid JSON. No markdown, no commentary, no code fences."
            else:
                return {"error": "Parse failed after retry", "raw_output": content_str[:500]}
        except httpx.HTTPError as e:
            return {"error": f"LLM endpoint error: {e}"}
        except Exception as e:
            return {"error": str(e)}
    return None


def _truncate_at_hunk_boundary(diff: str, max_chars: int) -> str:
    """Truncate diff at last `@@` hunk header boundary, not mid-hunk."""
    if len(diff) <= max_chars:
        return diff
    cut = diff.rfind("\n@@", 0, max_chars)
    if cut == -1:
        return diff[:max_chars] + "\n... [diff truncated, no hunk boundary found]"
    return diff[:cut] + "\n... [diff truncated at hunk boundary]"


def build_diff(
    repo_root: str,
    base: str,
    changed_files: list[str],
    max_total: int = 8000,
    max_per_file: int = 4000,
) -> str:
    """git diff per file, truncated at hunk boundaries."""
    diffs = []
    for f in changed_files:
        result = subprocess.run(
            ["git", "-C", repo_root, "diff", base, "--", f],
            capture_output=True, text=True,
        )
        if result.stdout:
            diffs.append(_truncate_at_hunk_boundary(result.stdout, max_per_file))
    combined = "\n".join(diffs)
    return _truncate_at_hunk_boundary(combined, max_total)


async def review_changes(
    repo_root: str = ".",
    base: str = "HEAD~1",
    endpoint_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """Review local changes using local LLM with graph context."""
    endpoint_url = endpoint_url or os.getenv("CRG_REVIEW_ENDPOINT_URL", "http://localhost:8000/v1")
    model = model or os.getenv("CRG_REVIEW_MODEL", "mtplx-qwen36-35b-a3b-optimized-balance")
    api_key = api_key or os.getenv("CRG_REVIEW_API_KEY", "")

    # Get graph-augmented context from upstream
    graph_result = get_review_context(repo_root=repo_root, base=base, detail_level="standard")
    if graph_result.get("status") != "ok":
        return {"summary": "No changes detected or graph error", "issues": [], "positive": [], "tokens_used": 0}

    ctx = graph_result.get("context", {})
    changed_files = ctx.get("changed_files", [])
    if not changed_files:
        return {"summary": "No changed files", "issues": [], "positive": [], "tokens_used": 0}

    # Enrich with signatures of impacted files (from graph nodes, no extra calls)
    max_impacted = int(os.getenv("CRG_REVIEW_MAX_IMPACTED_FILES", "10"))
    impacted_files = ctx.get("impacted_files", [])[:max_impacted]
    impacted_nodes = ctx.get("graph", {}).get("impacted_nodes", [])
    edges = ctx.get("graph", {}).get("edges", [])

    # Group signatures by file from impacted_nodes (free from get_review_context)
    signatures: dict[str, list[str]] = {}
    for node in impacted_nodes:
        fp = node.get("file_path", "")
        if fp and fp in impacted_files:
            signatures.setdefault(fp, []).append(
                f"{node.get('kind', '')} {node.get('qualified_name', node.get('name', ''))} (L{node.get('line_start', '?')}-{node.get('line_end', '?')})"
            )

    # MVP: One LLM call per changed file (no blast-radius clustering yet)
    all_issues, all_positive, total_tokens = [], [], 0

    for file_path in changed_files:
        diff = build_diff(repo_root, base, [file_path])
        if not diff.strip():
            continue
        language = detect_language(file_path)

        # Build prompt with graph context + signatures for this file
        impacted_count = 1 if file_path in impacted_files else 0
        caller_count = len([e for e in edges if e.get("source") == file_path or e.get("target") == file_path])

        # Add signature context from impacted nodes (already gathered above)
        sig_context = ""
        if signatures:
            sig_context = "\n**Impacted File Signatures (from blast radius):**\n"
            for sig_file, sigs in signatures.items():
                sig_context += f"\n--- {sig_file} ---\n"
                for s in sigs[:5]:
                    sig_context += f"  {s}\n"

        prompt = f"""## Code Review Request

**Language**: {language}

**Changed File**:
{file_path}

**Blast Radius (from code graph)**:
- {impacted_count} impacted files in this call
- {caller_count} call/dependency edges analyzed
{sig_context}

**Diff**:
```diff
{diff}
```

Review thoroughly. Focus on:
1. Correctness & edge cases
2. Security vulnerabilities
3. Performance implications
4. Error handling completeness
5. API contract changes affecting callers
6. Test coverage gaps in blast radius

Return JSON only per the output format.
"""
        parsed = await _call_llm_parsed(prompt, endpoint_url, model, api_key, base_prompt=prompt)
        if parsed is None or "error" in parsed:
            return {"summary": parsed.get("error", "LLM call failed") if parsed else "LLM call failed", "issues": [], "positive": [], "tokens_used": 0, **(parsed or {})}
        llm_response = parsed["response"]
        content = parsed["content"]
        tokens = llm_response.get("usage", {}).get("total_tokens", 0)
        all_issues.extend(content.get("issues", []))
        all_positive.extend(content.get("positive", []))
        total_tokens += tokens

    result = {
        "summary": f"Reviewed {len(changed_files)} files, found {len(all_issues)} issues",
        "issues": all_issues,
        "positive": list(dict.fromkeys(all_positive)),
        "tokens_used": total_tokens,
    }
    STATE["latest_review"] = result
    return result


async def review_file(
    file_path: str,
    repo_root: str = ".",
    endpoint_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """Review a single file using local LLM with graph context."""
    endpoint_url = endpoint_url or os.getenv("CRG_REVIEW_ENDPOINT_URL", "http://localhost:8000/v1")
    model = model or os.getenv("CRG_REVIEW_MODEL", "mtplx-qwen36-35b-a3b-optimized-balance")
    api_key = api_key or os.getenv("CRG_REVIEW_API_KEY", "")

    # Get graph context for this file
    graph_result = get_review_context(repo_root=repo_root, base="HEAD", detail_level="standard", changed_files=[file_path])
    if graph_result.get("status") != "ok":
        return {"summary": "No graph context available", "issues": [], "positive": [], "tokens_used": 0}

    ctx = graph_result.get("context", {})
    diff = build_diff(repo_root, "HEAD", [file_path])
    if not diff.strip():
        return {"summary": f"No changes in {file_path}", "issues": [], "positive": [], "tokens_used": 0}

    language = detect_language(file_path)

    prompt = f"""## Code Review Request

**Language**: {language}

**Changed File**:
{file_path}

**Diff**:
```diff
{diff}
```

Review thoroughly. Return JSON only per the output format.
"""
    parsed = await _call_llm_parsed(prompt, endpoint_url, model, api_key, base_prompt=prompt)
    if parsed is None or "error" in parsed:
        return {"summary": parsed.get("error", "LLM call failed") if parsed else "LLM call failed", "issues": [], "positive": [], "tokens_used": 0, **(parsed or {})}
    llm_response = parsed["response"]
    content = parsed["content"]
    tokens = llm_response.get("usage", {}).get("total_tokens", 0)

    result = {
        "summary": f"Reviewed {file_path}, found {len(content.get('issues', []))} issues",
        "issues": content.get("issues", []),
        "positive": content.get("positive", []),
        "tokens_used": tokens,
    }
    STATE["latest_review"] = result
    return result


async def review_pr(
    repo_root: str = ".",
    pr_number: int = 0,
    base_branch: str = "main",
    head_branch: str = "",
    endpoint_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """Review a GitHub PR using local LLM. Requires `gh` CLI authenticated."""
    endpoint_url = endpoint_url or os.getenv("CRG_REVIEW_ENDPOINT_URL", "http://localhost:8000/v1")
    model = model or os.getenv("CRG_REVIEW_MODEL", "mtplx-qwen36-35b-a3b-optimized-balance")
    api_key = api_key or os.getenv("CRG_REVIEW_API_KEY", "")

    # Fetch PR diff via gh CLI
    cmd = ["gh", "pr", "diff", str(pr_number), "--repo", "."]
    if head_branch:
        cmd.extend(["--head", head_branch])
    if base_branch:
        cmd.extend(["--base", base_branch])

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    if result.returncode != 0:
        return {"summary": f"Failed to fetch PR diff: {result.stderr}", "issues": [], "positive": [], "tokens_used": 0}

    diff = result.stdout
    if not diff.strip():
        return {"summary": "No changes in PR", "issues": [], "positive": [], "tokens_used": 0}

    # Determine changed files from diff
    changed_files = []
    for line in diff.splitlines():
        if line.startswith("diff --git a/"):
            parts = line.split()
            if len(parts) >= 4:
                changed_files.append(parts[3][2:])  # strip "b/"

    # Get graph context
    graph_result = get_review_context(repo_root=repo_root, base=base_branch, detail_level="standard", changed_files=changed_files)
    if graph_result.get("status") != "ok":
        return {"summary": "No graph context available", "issues": [], "positive": [], "tokens_used": 0}

    ctx = graph_result.get("context", {})
    impacted_files = ctx.get("impacted_files", [])
    edges = ctx.get("graph", {}).get("edges", [])

    language = "multiple"
    prompt = f"""## Pull Request Review

**PR**: #{pr_number}
**Base**: {base_branch}
**Head**: {head_branch or 'auto'}

**Changed Files**:
{', '.join(changed_files)}

**Blast Radius (from code graph)**:
- {len(impacted_files)} impacted files: {', '.join(impacted_files[:10])}
- {len(edges)} call/dependency edges analyzed

**Diff**:
```diff
{_truncate_at_hunk_boundary(diff, 8000)}
```

Review thoroughly. Focus on:
1. Correctness & edge cases
2. Security vulnerabilities
3. Performance implications
4. Error handling completeness
5. API contract changes affecting callers
6. Test coverage gaps in blast radius

Return JSON only per the output format.
"""
    parsed = await _call_llm_parsed(prompt, endpoint_url, model, api_key, base_prompt=prompt)
    if parsed is None or "error" in parsed:
        return {"summary": parsed.get("error", "LLM call failed") if parsed else "LLM call failed", "issues": [], "positive": [], "tokens_used": 0, **(parsed or {})}
    llm_response = parsed["response"]
    content = parsed["content"]
    tokens = llm_response.get("usage", {}).get("total_tokens", 0)

    result = {
        "summary": f"Reviewed PR #{pr_number}, found {len(content.get('issues', []))} issues",
        "issues": content.get("issues", []),
        "positive": content.get("positive", []),
        "tokens_used": tokens,
    }
    STATE["latest_review"] = result
    return result


# =============================================================================
# MCP TOOLS REGISTRATION
# =============================================================================

def register_review_tools(mcp: Any) -> None:
    """Register all review tools with the FastMCP server."""

    @mcp.tool(name="review_changes")
    async def _review_changes(
        repo_root: str = ".",
        base: str = "HEAD~1",
        endpoint_url: str = "http://localhost:8000/v1",
        model: str = "mtplx-qwen36-35b-a3b-optimized-balance",
        api_key: str = "",
    ) -> dict:
        """Review local changes using local LLM with graph context."""
        return await review_changes(repo_root, base, endpoint_url, model, api_key)

    @mcp.tool(name="review_file")
    async def _review_file(
        file_path: str,
        repo_root: str = ".",
        endpoint_url: str = "http://localhost:8000/v1",
        model: str = "mtplx-qwen36-35b-a3b-optimized-balance",
        api_key: str = "",
    ) -> dict:
        """Review a single file using local LLM with graph context."""
        return await review_file(file_path, repo_root, endpoint_url, model, api_key)

    @mcp.tool(name="review_pr")
    async def _review_pr(
        repo_root: str = ".",
        pr_number: int = 0,
        base_branch: str = "main",
        head_branch: str = "",
        endpoint_url: str = "http://localhost:8000/v1",
        model: str = "mtplx-qwen36-35b-a3b-optimized-balance",
        api_key: str = "",
    ) -> dict:
        """Review a GitHub PR using local LLM."""
        return await review_pr(repo_root, pr_number, base_branch, head_branch, endpoint_url, model, api_key)

    @mcp.tool(name="get_reviewer_feedback")
    async def _get_reviewer_feedback() -> dict:
        """Get latest findings from background pair reviewer."""
        return STATE["latest_review"]

    @mcp.resource("review://latest")
    def _latest_review_resource() -> str:
        """MCP Resource exposing latest review findings."""
        return json.dumps(STATE["latest_review"], indent=2)