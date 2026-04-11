#!/usr/bin/env python3
"""
Unit tests for bp_utils_readable_compress.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smolagents.bp_utils_readable_compress import (
    readable_compress,
    _strip_ansi,
    _normalize_whitespace,
    _collapse_repeated_lines,
    _collapse_progress_lines,
    _strip_timestamps,
)


# ── ANSI stripping ──────────────────────────────────────────────────────

class TestStripAnsi:
    def test_removes_colour_codes(self):
        assert _strip_ansi("\x1b[31mERROR\x1b[0m") == "ERROR"

    def test_removes_cursor_move(self):
        assert _strip_ansi("\x1b[2Ahello") == "hello"

    def test_plain_text_unchanged(self):
        assert _strip_ansi("hello world") == "hello world"

    def test_osc_sequence(self):
        assert _strip_ansi("\x1b]0;title\x07rest") == "rest"


# ── Whitespace normalization ────────────────────────────────────────────

class TestNormalizeWhitespace:
    def test_strips_trailing_spaces(self):
        assert _normalize_whitespace(["hello   ", "world  "]) == ["hello", "world"]

    def test_collapses_blank_runs(self):
        lines = ["a", "", "", "", "b"]
        assert _normalize_whitespace(lines) == ["a", "", "b"]

    def test_single_blank_preserved(self):
        lines = ["a", "", "b"]
        assert _normalize_whitespace(lines) == ["a", "", "b"]


# ── Duplicate collapsing ────────────────────────────────────────────────

class TestCollapseRepeatedLines:
    def test_no_repeats(self):
        lines = ["a", "b", "c"]
        assert _collapse_repeated_lines(lines) == ["a", "b", "c"]

    def test_two_repeats_kept(self):
        lines = ["a", "a", "b"]
        assert _collapse_repeated_lines(lines) == ["a", "a", "b"]

    def test_many_repeats_collapsed(self):
        lines = ["x"] * 50
        result = _collapse_repeated_lines(lines)
        assert len(result) == 2
        assert result[0] == "x"
        assert "49 more" in result[1]

    def test_empty_input(self):
        assert _collapse_repeated_lines([]) == []

    def test_mixed_runs(self):
        lines = ["a", "a", "a", "b", "b", "c"]
        result = _collapse_repeated_lines(lines)
        assert result == ["a", "  ... (repeated 2 more times)", "b", "b", "c"]


# ── Progress line removal ──────────────────────────────────────────────

class TestCollapseProgressLines:
    def test_percentage_with_bar(self):
        lines = [
            "Downloading [##        ] 20%",
            "Downloading [#####     ] 50%",
            "Downloading [##########] 100%",
            "Done.",
        ]
        result = _collapse_progress_lines(lines)
        assert result == ["Downloading [##########] 100%", "Done."]

    def test_spinner_lines(self):
        lines = ["|", "/", "-", "\\", "finished"]
        result = _collapse_progress_lines(lines)
        assert result == ["\\", "finished"]

    def test_normal_lines_unchanged(self):
        lines = ["compiling foo.c", "compiling bar.c"]
        assert _collapse_progress_lines(lines) == lines

    def test_percentage_without_bar_not_matched(self):
        lines = ["Test passed: 100% coverage"]
        assert _collapse_progress_lines(lines) == lines

    def test_empty(self):
        assert _collapse_progress_lines([]) == []


# ── Timestamp stripping ─────────────────────────────────────────────────

class TestStripTimestamps:
    def test_iso_timestamp(self):
        lines = ["2026-04-10T12:00:01Z hello", "2026-04-10T12:00:02Z world"]
        result = _strip_timestamps(lines)
        assert result[0].startswith("[timestamps stripped")
        assert "hello" in result[1]
        assert "world" in result[2]

    def test_bracketed_time(self):
        lines = ["[12:00:01] info message"]
        result = _strip_timestamps(lines)
        assert len(result) == 2
        assert "info message" in result[1]

    def test_syslog_timestamp(self):
        lines = ["Apr 10 12:00:01 server msg"]
        result = _strip_timestamps(lines)
        assert "server msg" in result[1]

    def test_no_timestamps(self):
        lines = ["hello", "world"]
        result = _strip_timestamps(lines)
        assert result == ["hello", "world"]


# ── Full pipeline ───────────────────────────────────────────────────────

class TestReadableCompress:
    def test_full_pipeline(self):
        text = (
            "\x1b[32m2026-04-10T12:00:01Z Starting build\x1b[0m\n"
            "2026-04-10T12:00:02Z Compiling\n"
            "Downloading [##        ] 20%\n"
            "Downloading [#####     ] 50%\n"
            "Downloading [##########] 100%\n"
            "line\n"
            "line\n"
            "line\n"
            "line\n"
            "line\n"
            "\n"
            "\n"
            "\n"
            "done"
        )
        result = readable_compress(text)
        # ANSI stripped
        assert "\x1b" not in result
        # Timestamps stripped
        assert "[timestamps stripped" in result
        # Progress collapsed — only 100% kept
        assert "20%" not in result
        assert "100%" in result
        # Duplicate lines collapsed
        assert "repeated" in result
        # Blank line run collapsed
        assert "\n\n\n" not in result
        # Content preserved
        assert "done" in result

    def test_plain_text_passthrough(self):
        text = "hello\nworld"
        assert readable_compress(text) == "hello\nworld"
