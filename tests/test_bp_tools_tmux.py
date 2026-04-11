#!/usr/bin/env python3
"""
Unit tests for TmuxReadTool incremental reading logic.

These tests mock tmux so they run without a real tmux server.
"""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smolagents.bp_tools_tmux import (
    TmuxReadTool,
    _last_read,
    _make_fingerprint,
)


@pytest.fixture(autouse=True)
def clear_state():
    """Reset incremental read state between tests."""
    _last_read.clear()
    yield
    _last_read.clear()


def _mock_capture(lines: list[str]):
    """Return a mock _run_tmux result simulating capture-pane output."""
    result = MagicMock()
    result.returncode = 0
    result.stdout = "\n".join(lines) + "\n"
    return result


class TestMakeFingerprint:
    def test_three_or_more_lines(self):
        assert _make_fingerprint(["a", "b", "c"]) == ("a", "b", "c")
        assert _make_fingerprint(["a", "b", "c", "d"]) == ("b", "c", "d")

    def test_two_lines(self):
        assert _make_fingerprint(["a", "b"]) == ("", "a", "b")

    def test_one_line(self):
        assert _make_fingerprint(["a"]) == ("", "a", "a")


class TestTmuxReadIncremental:
    """Test the incremental reading logic of TmuxReadTool."""

    def setup_method(self):
        self.tool = TmuxReadTool()

    @patch("smolagents.bp_tools_tmux._run_tmux")
    def test_first_read_returns_full_tail(self, mock_tmux):
        lines = ["line1", "line2", "line3", "line4", "line5"]
        mock_tmux.return_value = _mock_capture(lines)

        result = self.tool.forward("sess", lines=3)
        assert result == "line3\nline4\nline5"

    @patch("smolagents.bp_tools_tmux._run_tmux")
    def test_incremental_returns_only_new_lines(self, mock_tmux):
        # First read: 5 lines
        lines1 = ["line1", "line2", "line3", "line4", "prompt$"]
        mock_tmux.return_value = _mock_capture(lines1)
        self.tool.forward("sess", lines=20)

        # Second read: 3 new lines appended
        lines2 = lines1 + ["output_a", "output_b", "new_prompt$"]
        mock_tmux.return_value = _mock_capture(lines2)
        result = self.tool.forward("sess", lines=20)

        # Should see everything after the anchor (line3), which is:
        # line4 (old last line), output_a, output_b, new_prompt$
        assert "output_a" in result
        assert "output_b" in result
        assert "new_prompt$" in result

    @patch("smolagents.bp_tools_tmux._run_tmux")
    def test_no_new_output(self, mock_tmux):
        lines = ["line1", "line2", "line3", "prompt$"]
        mock_tmux.return_value = _mock_capture(lines)
        self.tool.forward("sess", lines=20)

        # Same content again
        mock_tmux.return_value = _mock_capture(lines)
        result = self.tool.forward("sess", lines=20)
        assert result == "(no new output)"

    @patch("smolagents.bp_tools_tmux._run_tmux")
    def test_non_incremental_always_returns_full_tail(self, mock_tmux):
        lines = ["line1", "line2", "line3", "prompt$"]
        mock_tmux.return_value = _mock_capture(lines)
        self.tool.forward("sess", lines=20)

        # Same content, but incremental=False
        mock_tmux.return_value = _mock_capture(lines)
        result = self.tool.forward("sess", lines=20, incremental=False)
        assert "line1" in result
        assert "prompt$" in result

    @patch("smolagents.bp_tools_tmux._run_tmux")
    def test_trailing_blanks_do_not_break_anchor(self, mock_tmux):
        """Issue #2: varying trailing blanks should not cause fallback."""
        lines1 = ["line1", "line2", "line3", "prompt$"]
        mock_tmux.return_value = _mock_capture(lines1)
        self.tool.forward("sess", lines=20)

        # Same real content but with trailing blanks appended (tmux does this)
        lines2 = ["line1", "line2", "line3", "prompt$", "", "", ""]
        mock_tmux.return_value = _mock_capture(lines2)
        result = self.tool.forward("sess", lines=20)
        assert result == "(no new output)"

    @patch("smolagents.bp_tools_tmux._run_tmux")
    def test_single_line_content_change_detected(self, mock_tmux):
        """Issue #5: progress counter updating in-place should be visible."""
        lines1 = ["progress: 50%"]
        mock_tmux.return_value = _mock_capture(lines1)
        self.tool.forward("sess", lines=20)

        # Content changed (in-place update)
        lines2 = ["progress: 100%"]
        mock_tmux.return_value = _mock_capture(lines2)
        result = self.tool.forward("sess", lines=20)
        # Anchor won't match → falls back to full tail → we see the update
        assert "100%" in result
        assert result != "(no new output)"

    @patch("smolagents.bp_tools_tmux._run_tmux")
    def test_single_line_unchanged(self, mock_tmux):
        lines = ["waiting..."]
        mock_tmux.return_value = _mock_capture(lines)
        self.tool.forward("sess", lines=20)

        mock_tmux.return_value = _mock_capture(lines)
        result = self.tool.forward("sess", lines=20)
        assert result == "(no new output)"

    @patch("smolagents.bp_tools_tmux._run_tmux")
    def test_duplicate_lines_use_two_line_fingerprint(self, mock_tmux):
        """Two-line fingerprint should disambiguate duplicate anchor lines."""
        # The anchor line "---" appears multiple times
        lines1 = ["header", "---", "body1", "---", "prompt$"]
        mock_tmux.return_value = _mock_capture(lines1)
        self.tool.forward("sess", lines=20)

        # New content after the last "---"
        lines2 = ["header", "---", "body1", "---", "prompt$", "new_output", "prompt2$"]
        mock_tmux.return_value = _mock_capture(lines2)
        result = self.tool.forward("sess", lines=20)
        # Fingerprint is ("body1", "---"), matching the LAST occurrence.
        # New lines start after that "---" → prompt$, new_output, prompt2$
        assert "new_output" in result

    @patch("smolagents.bp_tools_tmux._run_tmux")
    def test_empty_screen(self, mock_tmux):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "\n"
        mock_tmux.return_value = result
        assert self.tool.forward("sess") == "(empty screen)"

    @patch("smolagents.bp_tools_tmux._run_tmux")
    def test_error_propagated(self, mock_tmux):
        result = MagicMock()
        result.returncode = 1
        result.stderr = "session not found"
        mock_tmux.return_value = result
        assert "ERROR" in self.tool.forward("sess")

    @patch("smolagents.bp_tools_tmux._run_tmux")
    def test_lines_cap_respected(self, mock_tmux):
        lines = [f"line{i}" for i in range(50)]
        mock_tmux.return_value = _mock_capture(lines)
        result = self.tool.forward("sess", lines=5)
        assert len(result.splitlines()) == 5

    @patch("smolagents.bp_tools_tmux._run_tmux")
    def test_buffer_trim_causes_graceful_fallback(self, mock_tmux):
        """Issue #1: if tmux trims old lines, anchor search falls back to full tail."""
        lines1 = [f"line{i}" for i in range(20)] + ["prompt$"]
        mock_tmux.return_value = _mock_capture(lines1)
        self.tool.forward("sess", lines=20)

        # Buffer was trimmed — old anchor lines are gone
        lines2 = ["totally_new1", "totally_new2", "totally_new3", "prompt2$"]
        mock_tmux.return_value = _mock_capture(lines2)
        result = self.tool.forward("sess", lines=20)
        # Falls back to full tail — we see new content
        assert "totally_new1" in result

    @patch("smolagents.bp_tools_tmux._run_tmux")
    def test_three_consecutive_reads(self, mock_tmux):
        """Fingerprint updates correctly across multiple incremental reads."""
        lines1 = ["a", "b", "c", "prompt1$"]
        mock_tmux.return_value = _mock_capture(lines1)
        r1 = self.tool.forward("sess", lines=20)
        assert "a" in r1

        lines2 = ["a", "b", "c", "prompt1$", "d", "e", "prompt2$"]
        mock_tmux.return_value = _mock_capture(lines2)
        r2 = self.tool.forward("sess", lines=20)
        assert "d" in r2
        assert "a" not in r2

        lines3 = ["a", "b", "c", "prompt1$", "d", "e", "prompt2$", "f", "prompt3$"]
        mock_tmux.return_value = _mock_capture(lines3)
        r3 = self.tool.forward("sess", lines=20)
        assert "f" in r3
        assert "d" not in r3
