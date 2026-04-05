# BPSA - Beyond Python SmolAgents
# https://github.com/joaopauloschuler/beyond-python-smolagents
#
# Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and others.
# Refer to the git commit history for individual authorship.
# Licensed under the Apache License, Version 2.0

"""
Tmux multi-screen tools for BPSA agents.

Allows agents to create and operate multiple independent shell sessions
concurrently via tmux. Each session is named and can be independently
controlled (send keystrokes, read output, wait for patterns).

Dependencies:
- tmux (required, must be installed on the system)

Enable via ``--tmux`` CLI flag or ``BPSA_TMUX=1`` environment variable.
"""

import atexit
import shlex
import subprocess
import time

from .tools import Tool


TMUX_PREFIX = "bpsa_"
_MAX_READ_LINES = 2000


def _full_name(session_name: str) -> str:
    """Return the prefixed tmux session name."""
    return TMUX_PREFIX + session_name


def _run_tmux(*args: str) -> subprocess.CompletedProcess:
    """Run a tmux command and return the CompletedProcess."""
    cmd = ["tmux"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=15)


def _cleanup_all_sessions():
    """Kill all bpsa_ prefixed tmux sessions (registered via atexit)."""
    try:
        result = _run_tmux("list-sessions", "-F", "#{session_name}")
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                if line.startswith(TMUX_PREFIX):
                    _run_tmux("kill-session", "-t", line)
    except Exception:
        pass


# Register cleanup so sessions don't leak when the process exits.
atexit.register(_cleanup_all_sessions)


# ======================================================================
# tmux_create — create a new named session
# ======================================================================


class TmuxCreateTool(Tool):
    """Create a new named tmux screen session."""

    name = "tmux_create"
    description = (
        "Creates a new independent named tmux screen session. "
        "Use this to run long-running processes (servers, builds, watches) "
        "in the background while continuing other work. "
        "Each session is a full shell you can send commands to and read output from."
    )
    inputs = {
        "session_name": {
            "type": "string",
            "description": "Unique name for this screen session (e.g. 'server', 'build', 'tests').",
        },
        "command": {
            "type": "string",
            "description": "Optional initial command to run in the session. Defaults to 'bash'.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, session_name: str, command: str | None = None) -> str:
        full = _full_name(session_name)
        cmd = command or "bash"
        result = _run_tmux(
            "new-session", "-d", "-s", full, "-x", "200", "-y", "50", cmd,
        )
        if result.returncode != 0:
            return f"ERROR: {result.stderr.strip()}"
        return f"Session '{session_name}' created."


# ======================================================================
# tmux_send — send keystrokes to a session
# ======================================================================


class TmuxSendTool(Tool):
    """Send keystrokes to a named tmux screen session."""

    name = "tmux_send"
    description = (
        "Sends text (keystrokes) to a named tmux screen session. "
        "By default presses Enter after the text. "
        "Use press_enter=False to type without submitting (useful for interactive prompts)."
    )
    inputs = {
        "session_name": {
            "type": "string",
            "description": "Name of the target screen session.",
        },
        "text": {
            "type": "string",
            "description": "Text to type into the session.",
        },
        "press_enter": {
            "type": "boolean",
            "description": "Whether to press Enter after the text. Defaults to True.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, session_name: str, text: str, press_enter: bool | None = None) -> str:
        if press_enter is None:
            press_enter = True
        full = _full_name(session_name)
        # Use tmux send-keys with literal flag to avoid key-name interpretation
        args = ["send-keys", "-t", full, "-l", text]
        result = _run_tmux(*args)
        if result.returncode != 0:
            return f"ERROR: {result.stderr.strip()}"
        if press_enter:
            result = _run_tmux("send-keys", "-t", full, "Enter")
            if result.returncode != 0:
                return f"ERROR sending Enter: {result.stderr.strip()}"
        return "Keys sent."


# ======================================================================
# tmux_read — read the visible content of a session
# ======================================================================


class TmuxReadTool(Tool):
    """Read the current visible content of a named tmux screen session."""

    name = "tmux_read"
    description = (
        "Reads and returns the current text content (scrollback buffer) of a named "
        "tmux screen session. Use this to check command output, monitor progress, "
        "or see the current state of an interactive program."
    )
    inputs = {
        "session_name": {
            "type": "string",
            "description": "Name of the target screen session.",
        },
        "lines": {
            "type": "integer",
            "description": "Number of scrollback lines to capture. Defaults to 100.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, session_name: str, lines: int | None = None) -> str:
        if lines is None:
            lines = 100
        lines = min(lines, _MAX_READ_LINES)
        full = _full_name(session_name)
        result = _run_tmux("capture-pane", "-t", full, "-p", "-S", f"-{lines}")
        if result.returncode != 0:
            return f"ERROR: {result.stderr.strip()}"
        # Strip trailing blank lines for cleaner output
        output = result.stdout.rstrip("\n")
        return output if output else "(empty screen)"


# ======================================================================
# tmux_list — list active sessions
# ======================================================================


class TmuxListTool(Tool):
    """List all active tmux screen sessions."""

    name = "tmux_list"
    description = (
        "Lists all active tmux screen sessions created by this agent. "
        "Returns session names and their status."
    )
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        result = _run_tmux("list-sessions", "-F", "#{session_name} (#{session_windows} windows, created #{session_created_string})")
        if result.returncode != 0:
            if "no server running" in result.stderr or "no sessions" in result.stderr:
                return "No active sessions."
            return f"ERROR: {result.stderr.strip()}"
        lines = []
        for line in result.stdout.strip().splitlines():
            if line.startswith(TMUX_PREFIX):
                # Strip the prefix for display
                lines.append(line[len(TMUX_PREFIX):])
        if not lines:
            return "No active sessions."
        return "\n".join(lines)


# ======================================================================
# tmux_destroy — kill a session
# ======================================================================


class TmuxDestroyTool(Tool):
    """Destroy (kill) a named tmux screen session."""

    name = "tmux_destroy"
    description = (
        "Destroys a named tmux screen session and all processes running in it. "
        "Use this to clean up sessions you no longer need."
    )
    inputs = {
        "session_name": {
            "type": "string",
            "description": "Name of the session to destroy.",
        },
    }
    output_type = "string"

    def forward(self, session_name: str) -> str:
        full = _full_name(session_name)
        result = _run_tmux("kill-session", "-t", full)
        if result.returncode != 0:
            return f"ERROR: {result.stderr.strip()}"
        return f"Session '{session_name}' destroyed."


# ======================================================================
# tmux_wait — wait for a pattern to appear in a session
# ======================================================================


class TmuxWaitTool(Tool):
    """Wait for a text pattern to appear in a tmux screen session."""

    name = "tmux_wait"
    description = (
        "Polls a named tmux screen session until a given text pattern appears "
        "in its output, or until the timeout is reached. "
        "Useful for waiting until a server is ready, a build completes, etc."
    )
    inputs = {
        "session_name": {
            "type": "string",
            "description": "Name of the target screen session.",
        },
        "pattern": {
            "type": "string",
            "description": "Text pattern to wait for (plain substring match).",
        },
        "timeout": {
            "type": "integer",
            "description": "Maximum seconds to wait. Defaults to 60.",
            "nullable": True,
        },
        "interval": {
            "type": "number",
            "description": "Seconds between polls. Defaults to 1.0.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        session_name: str,
        pattern: str,
        timeout: int | None = None,
        interval: float | None = None,
    ) -> str:
        if timeout is None:
            timeout = 60
        if interval is None:
            interval = 1.0
        full = _full_name(session_name)
        deadline = time.monotonic() + timeout
        last_output = ""
        while time.monotonic() < deadline:
            result = _run_tmux("capture-pane", "-t", full, "-p", "-S", "-200")
            if result.returncode != 0:
                return f"ERROR: {result.stderr.strip()}"
            last_output = result.stdout
            if pattern in last_output:
                return f"Pattern '{pattern}' found.\n\n{last_output.rstrip()}"
            time.sleep(interval)
        return f"TIMEOUT: pattern '{pattern}' not found after {timeout}s.\n\nLast output:\n{last_output.rstrip()}"


# ======================================================================
# Factory function
# ======================================================================


def create_tmux_tools() -> list[Tool]:
    """Create and return the tmux multi-screen tools."""
    return [
        TmuxCreateTool(),
        TmuxSendTool(),
        TmuxReadTool(),
        TmuxListTool(),
        TmuxDestroyTool(),
        TmuxWaitTool(),
    ]
