"""Tests for crg_review - Local LLM Code Review Layer"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import the module under test
from crg_review.review import (
    PAIR_REVIEWER_SYSTEM_ADDITION,
    REVIEW_SCHEMA,
    STATE,
    _call_llm_parsed,
    _truncate_at_hunk_boundary,
    build_diff,
    call_llm,
    detect_language,
    register_review_tools,
    start_watcher,
    stop_watcher,
    validate_llm_response,
)


class TestReviewSchema:
    """Test the review schema validation."""

    def test_valid_review_passes(self):
        """A valid review structure should pass validation."""
        valid = {
            "summary": "Code looks good",
            "issues": [
                {
                    "file": "test.py",
                    "line": 10,
                    "end_line": 12,
                    "severity": "warning",
                    "category": "performance",
                    "message": "Inefficient loop",
                    "suggestion": "Use list comprehension",
                }
            ],
            "positive": ["Clean structure"],
        }
        validate_llm_response(valid)  # Should not raise

    def test_missing_required_field_fails(self):
        """Missing required fields should fail validation."""
        invalid = {
            "summary": "Code looks good",
            "issues": [],
            # missing "positive"
        }
        with pytest.raises(Exception):
            validate_llm_response(invalid)

    def test_invalid_severity_fails(self):
        """Invalid severity enum should fail."""
        invalid = {
            "summary": "Code looks good",
            "issues": [
                {
                    "file": "test.py",
                    "line": 10,
                    "end_line": 12,
                    "severity": "invalid_severity",
                    "category": "performance",
                    "message": "Inefficient loop",
                    "suggestion": "Use list comprehension",
                }
            ],
            "positive": ["Clean structure"],
        }
        with pytest.raises(Exception):
            validate_llm_response(invalid)

    def test_end_line_can_be_null(self):
        """end_line should accept null."""
        valid = {
            "summary": "Code looks good",
            "issues": [
                {
                    "file": "test.py",
                    "line": 10,
                    "end_line": None,
                    "severity": "warning",
                    "category": "performance",
                    "message": "Inefficient loop",
                    "suggestion": "Use list comprehension",
                }
            ],
            "positive": ["Clean structure"],
        }
        validate_llm_response(valid)  # Should not raise


class TestTruncateAtHunkBoundary:
    """Test diff truncation at hunk boundaries."""

    def test_short_diff_unchanged(self):
        """Diff under max_chars should be returned unchanged."""
        diff = "@@ -1,5 +1,5 @@\n line1\n-line2\n+line2_new\n line3\n"
        result = _truncate_at_hunk_boundary(diff, 1000)
        assert result == diff

    def test_long_diff_truncated_at_hunk(self):
        """Long diff should be truncated at last @@ boundary."""
        hunk1 = "@@ -1,5 +1,5 @@\n line1\n-line2\n+line2_new\n line3\n"
        hunk2 = "@@ -10,5 +10,5 @@\n line10\n-line11\n+line11_new\n line12\n"
        diff = hunk1 + hunk2
        result = _truncate_at_hunk_boundary(diff, len(hunk1) + 10)
        assert result.startswith(hunk1)
        assert "[diff truncated at hunk boundary]" in result
        assert hunk2 not in result

    def test_no_hunk_boundary_found(self):
        """If no @@ found, truncate at max_chars with message."""
        diff = "a" * 2000
        result = _truncate_at_hunk_boundary(diff, 1000)
        assert len(result) <= 1000 + 100  # message length
        assert "[diff truncated, no hunk boundary found]" in result


class TestBuildDiff:
    """Test git diff building."""

    def test_build_diff_empty(self, tmp_path):
        """No changes should return empty string."""
        repo = tmp_path
        (repo / "test.py").write_text("print('hello')\n")
        # No git repo initialized, but function handles it gracefully
        # Just test it doesn't crash
        result = build_diff(str(repo), "HEAD", ["test.py"])
        assert isinstance(result, str)


class TestDetectLanguage:
    """Test language detection from file extension."""

    @pytest.mark.parametrize(
        "ext,expected",
        [
            (".py", "python"),
            (".ts", "typescript"),
            (".tsx", "typescript"),
            (".js", "javascript"),
            (".jsx", "javascript"),
            (".rs", "rust"),
            (".go", "go"),
            (".java", "java"),
            (".cpp", "cpp"),
            (".c", "c"),
            (".cs", "csharp"),
            (".rb", "ruby"),
            (".php", "php"),
            (".swift", "swift"),
            (".kt", "kotlin"),
            (".scala", "scala"),
            (".sql", "sql"),
            (".vue", "vue"),
            (".svelte", "svelte"),
            (".astro", "astro"),
            (".unknown", "text"),
        ],
    )
    def test_extensions(self, ext, expected):
        assert detect_language(f"file{ext}") == expected


class TestCallLLM:
    """Test LLM calling functions with mocks."""

    @pytest.mark.asyncio
    async def test_call_llm_success(self):
        """call_llm should return parsed JSON on success."""
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"summary": "ok", "issues": [], "positive": []}'}}]
        }
        mock_resp.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)
            result = await call_llm("test prompt", "http://localhost:8000/v1", "model", "key")
            assert "choices" in result

    @pytest.mark.asyncio
    async def test_call_llm_parsed_success(self):
        """_call_llm_parsed should return parsed content on success."""
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"summary": "ok", "issues": [], "positive": ["good"]}'}}],
            "usage": {"total_tokens": 100},
        }
        mock_resp.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)
            result = await _call_llm_parsed("test", "http://localhost", "model", "key")
            assert result is not None
            assert result["content"]["summary"] == "ok"
            assert result["content"]["positive"] == ["good"]

    @pytest.mark.asyncio
    async def test_call_llm_parsed_retry_on_invalid_json(self):
        """_call_llm_parsed should retry once on JSON decode error."""
        mock_resp1 = Mock()
        mock_resp1.json.return_value = {"choices": [{"message": {"content": "not valid json"}}]}
        mock_resp1.raise_for_status = Mock()

        mock_resp2 = Mock()
        mock_resp2.json.return_value = {
            "choices": [{"message": {"content": '{"summary": "ok", "issues": [], "positive": []}'}}],
            "usage": {"total_tokens": 50},
        }
        mock_resp2.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=[mock_resp1, mock_resp2])
            result = await _call_llm_parsed("test", "http://localhost", "model", "key")
            assert result is not None
            assert result["content"]["summary"] == "ok"


class TestStateManagement:
    """Test shared state for watcher."""

    def test_initial_state(self):
        """STATE should have expected initial values."""
        assert STATE["latest_review"] == {"summary": "No review yet", "issues": [], "positive": []}
        assert STATE["watcher_task"] is None
        assert STATE["watcher_running"] is False


class TestWatcher:
    """Test background watcher functions."""

    @pytest.mark.asyncio
    async def test_start_watcher_creates_task(self):
        """start_watcher should set running flag and create task."""
        stop_watcher()  # Ensure clean state
        start_watcher(".")
        assert STATE["watcher_running"] is True
        assert STATE["watcher_task"] is not None
        stop_watcher()

    @pytest.mark.asyncio
    async def test_start_watcher_idempotent(self):
        """Calling start_watcher twice should not create two tasks."""
        stop_watcher()
        start_watcher(".")
        task1 = STATE["watcher_task"]
        start_watcher(".")
        task2 = STATE["watcher_task"]
        assert task1 is task2
        stop_watcher()

    @pytest.mark.asyncio
    async def test_stop_watcher_cancels_task(self):
        """stop_watcher should cancel task and clear state."""
        stop_watcher()
        start_watcher(".")
        assert STATE["watcher_running"] is True
        stop_watcher()
        assert STATE["watcher_running"] is False
        assert STATE["watcher_task"] is None


class TestMCPRegistration:
    """Test MCP tool registration."""

    def test_register_review_tools_adds_all_tools(self):
        """register_review_tools should register all 4 review tools + resource."""
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        register_review_tools(mcp)

        # Check tools registered
        import asyncio

        async def check():
            tools = await mcp.list_tools()
            tool_names = {t.name for t in tools}
            assert "review_changes" in tool_names
            assert "review_file" in tool_names
            assert "review_pr" in tool_names
            assert "get_reviewer_feedback" in tool_names

            resources = await mcp.list_resources()
            resource_uris = {str(r.uri) for r in resources}
            assert "review://latest" in resource_uris

        asyncio.run(check())


class TestPromptInjection:
    """Test the PAIR_REVIEWER_SYSTEM_ADDITION constant."""

    def test_constant_exists_and_non_empty(self):
        assert PAIR_REVIEWER_SYSTEM_ADDITION
        assert len(PAIR_REVIEWER_SYSTEM_ADDITION) > 100

    def test_contains_key_instructions(self):
        """Should contain key instructional phrases."""
        assert "get_reviewer_feedback()" in PAIR_REVIEWER_SYSTEM_ADDITION
        assert "CRITICAL" in PAIR_REVIEWER_SYSTEM_ADDITION
        assert "WARNING" in PAIR_REVIEWER_SYSTEM_ADDITION
        assert "pair reviewer" in PAIR_REVIEWER_SYSTEM_ADDITION.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])