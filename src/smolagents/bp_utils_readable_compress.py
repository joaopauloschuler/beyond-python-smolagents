# BPSA - Beyond Python SmolAgents
# https://github.com/joaopauloschuler/beyond-python-smolagents
#
# Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and others.
# Refer to the git commit history for individual authorship.
# Licensed under the Apache License, Version 2.0

"""
Readable text compression for agent output.

Compresses text while keeping it human-readable. Designed for tmux
scrollback and similar command output where verbosity wastes context
tokens without adding information.

Passes (in order):
1. ANSI escape stripping
2. Whitespace normalization (trailing spaces, blank-line runs)
3. Exact consecutive duplicate line collapsing
4. Progress line removal (keep only the final state)
5. Separator line collapsing (e.g. ====, ----, ####)
6. Timestamp prefix stripping
"""

import re

# ---------------------------------------------------------------------------
# 1. ANSI escape stripping
# ---------------------------------------------------------------------------

# Matches CSI sequences (e.g. colours, cursor moves) and OSC sequences.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b\[[\d;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences."""
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# 2. Whitespace normalization
# ---------------------------------------------------------------------------


def _normalize_whitespace(lines: list[str]) -> list[str]:
    """Strip trailing whitespace per line and collapse runs of blank lines."""
    result: list[str] = []
    prev_blank = False
    for line in lines:
        stripped = line.rstrip()
        is_blank = stripped == ""
        if is_blank and prev_blank:
            continue
        result.append(stripped)
        prev_blank = is_blank
    return result


# ---------------------------------------------------------------------------
# 3. Exact consecutive duplicate collapsing
# ---------------------------------------------------------------------------


def _collapse_repeated_lines(lines: list[str]) -> list[str]:
    """Collapse runs of identical consecutive lines."""
    if not lines:
        return lines
    result: list[str] = []
    run_line = lines[0]
    run_count = 1
    for line in lines[1:]:
        if line == run_line:
            run_count += 1
        else:
            _flush_run(result, run_line, run_count)
            run_line = line
            run_count = 1
    _flush_run(result, run_line, run_count)
    return result


def _flush_run(result: list[str], line: str, count: int) -> None:
    """Append a run to *result*, summarising if count > 2."""
    if count <= 2:
        result.extend([line] * count)
    else:
        result.append(line)
        result.append(f"  ... (repeated {count - 1} more times)")


# ---------------------------------------------------------------------------
# 4. Progress line removal
# ---------------------------------------------------------------------------

# Matches lines that look like progress indicators:
#   - contain a percentage  (e.g. " 45%", "100%")
#   - or a progress bar     (e.g. [####    ], [=====>  ])
#   - or common spinners    (|, /, -, \)
_PERCENT_RE = re.compile(r"\d{1,3}%")
_PROGRESS_BAR_RE = re.compile(r"\[[\s#=\->]+\]")
_SPINNER_RE = re.compile(r"^[\s]*[|/\\\-][\s]*$")


def _collapse_progress_lines(lines: list[str]) -> list[str]:
    """Remove intermediate progress lines, keeping only the last in a run."""
    if not lines:
        return lines
    result: list[str] = []
    progress_run: list[str] = []
    for line in lines:
        if _is_progress_line(line):
            progress_run.append(line)
        else:
            if progress_run:
                result.append(progress_run[-1])
                progress_run = []
            result.append(line)
    if progress_run:
        result.append(progress_run[-1])
    return result


def _is_progress_line(line: str) -> bool:
    """Heuristic: does *line* look like a progress indicator?"""
    if _SPINNER_RE.match(line):
        return True
    if _PERCENT_RE.search(line) and _PROGRESS_BAR_RE.search(line):
        return True
    return False


# ---------------------------------------------------------------------------
# 5. Separator line collapsing
# ---------------------------------------------------------------------------

# Runs of 4+ identical separator characters anywhere in text.
_SEPARATOR_RE = re.compile(r"([=\-#_*~.+])\1{3,}")
_SEPARATOR_MARKER = "---"


def _collapse_separators(lines: list[str]) -> list[str]:
    """Replace runs of repeated separator characters with a short marker."""
    return [_SEPARATOR_RE.sub(_SEPARATOR_MARKER, line) for line in lines]


# ---------------------------------------------------------------------------
# 6. Timestamp prefix stripping
# ---------------------------------------------------------------------------

# Common log timestamp patterns at the start of a line:
#   2026-04-10T12:00:01.123Z  |  2026-04-10 12:00:01  |  [12:00:01]  |
#   Apr 10 12:00:01
_TIMESTAMP_RE = re.compile(
    r"^("
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[\.\d]*[Z]?\s*"
    r"|\[\d{2}:\d{2}:\d{2}\]\s*"
    r"|[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s*"
    r")"
)


def _strip_timestamps(lines: list[str]) -> list[str]:
    """Strip common timestamp prefixes and prepend a note if any were found."""
    stripped: list[str] = []
    count = 0
    for line in lines:
        new_line, n = _TIMESTAMP_RE.subn("", line, count=1)
        count += n
        stripped.append(new_line)
    if count > 0:
        stripped.insert(0, f"[timestamps stripped from {count} lines]")
    return stripped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def readable_compress(text: str) -> str:
    """Apply all readable compression passes and return compressed text."""
    text = _strip_ansi(text)
    lines = text.splitlines()
    lines = _normalize_whitespace(lines)
    lines = _collapse_repeated_lines(lines)
    lines = _collapse_progress_lines(lines)
    lines = _collapse_separators(lines)
    lines = _strip_timestamps(lines)
    return "\n".join(lines)
