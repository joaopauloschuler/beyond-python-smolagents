"""Tests for the approval prompt when stdin is not a terminal, e.g. `echo task | bpsa` or `bpsa run` in cron or CI.

The prompt reads the key from stdin when stdin is a terminal. Otherwise it reads it from the terminal itself
(bp_cli._TTY_PATH, a pseudo-terminal in these tests), but only when stdout is a terminal too, so that the question
can be seen. When nobody can answer, `bpsa run` stops at startup and the prompt answers no, both with advice,
instead of crashing with termios.error.
"""

import io
import os
import sys
import threading
import time
import types

import pytest
from rich.console import Console

from smolagents import bp_cli


termios = pytest.importorskip("termios")  # Unix only: on Windows the prompt reads the console with msvcrt

ADVICE = ("--auto-approve on", "BPSA_AUTO_APPROVE=on")


class FakeStream(io.StringIO):
    """A text stream that says whether it is a terminal and, optionally, which file descriptor it uses."""

    def __init__(self, tty=False, fd=None):
        super().__init__()
        self.tty, self.fd = tty, fd

    def isatty(self):
        return self.tty

    def fileno(self):
        if self.fd is None:
            raise io.UnsupportedOperation("fileno")
        return self.fd


class Stop(Exception):
    """Raised by the build_model stub: run_one_shot() got past its startup checks."""


@pytest.fixture
def terminal():
    """A pseudo-terminal: the fd to type on, the fd of the terminal, and its path."""
    master, slave = os.openpty()
    yield master, slave, os.ttyname(slave)
    os.close(master)
    os.close(slave)


@pytest.fixture
def output(monkeypatch):
    """Record what bp_cli prints."""
    recorder = Console(record=True, width=200, file=io.StringIO())
    monkeypatch.setattr(bp_cli, "console", recorder)
    return recorder


@pytest.fixture
def one_shot(monkeypatch, tmp_path, output):
    """Let run_one_shot() run until it would build the model: no ./.env, no required settings."""
    monkeypatch.chdir(tmp_path)
    for name in ("_auto_approve", "_auto_approve_source", "_verbose"):
        monkeypatch.setattr(bp_cli, name, getattr(bp_cli, name))  # restored after the test
    monkeypatch.setattr(bp_cli, "check_required_env", lambda: None)

    def build_model():
        raise Stop

    monkeypatch.setattr(bp_cli, "build_model", build_model)


def _streams(monkeypatch, stdin, stdout):
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)


def _text(console):
    # export_text() clears the recording unless clear=False, and a test may read it more than once
    return " ".join(console.export_text(clear=False).split())


@pytest.mark.parametrize(
    "stdin, stdout, tty_path, expected",
    [
        ("terminal", "pipe", "missing", "stdin"),
        ("pipe", "terminal", "terminal", "tty"),
        ("pipe", "pipe", "terminal", None),
        ("pipe", "terminal", "missing", None),
        ("pipe", "terminal", "file", None),
        ("none", "closed", "terminal", None),
    ],
    ids=["stdin-is-a-terminal", "piped-task", "output-captured", "no-terminal", "file-not-a-terminal", "no-streams"],
)
def test_approval_key_source(monkeypatch, tmp_path, terminal, stdin, stdout, tty_path, expected):
    closed = io.StringIO()
    closed.close()
    streams = {"terminal": FakeStream(tty=True), "pipe": FakeStream(), "none": None, "closed": closed}
    (tmp_path / "file").write_text("")
    paths = {"terminal": terminal[2], "missing": str(tmp_path / "missing"), "file": str(tmp_path / "file")}
    monkeypatch.setattr(bp_cli, "_TTY_PATH", paths[tty_path])
    _streams(monkeypatch, streams[stdin], streams[stdout])
    assert bp_cli.approval_key_source() == expected


def test_windows_keeps_reading_the_console(monkeypatch):
    monkeypatch.setitem(sys.modules, "msvcrt", types.ModuleType("msvcrt"))
    _streams(monkeypatch, FakeStream(), FakeStream())
    assert bp_cli.approval_key_source() == "console"


def test_getch_reads_the_terminal_when_stdin_is_piped(monkeypatch, terminal):
    master, slave, path = terminal
    monkeypatch.setattr(bp_cli, "_TTY_PATH", path)
    _streams(monkeypatch, FakeStream(), FakeStream(tty=True))
    mode_before = termios.tcgetattr(slave)
    result = {}

    def read_key():
        try:
            result["key"] = bp_cli._getch()
        except Exception as err:
            result["error"] = repr(err)

    reader = threading.Thread(target=read_key, daemon=True)
    reader.start()
    deadline = time.monotonic() + 5
    while reader.is_alive() and time.monotonic() < deadline:
        os.write(master, b"Y")  # again and again: switching to raw mode discards keys typed before it
        reader.join(0.05)
    assert result == {"key": "y"}
    assert termios.tcgetattr(slave) == mode_before


def test_getch_raises_eof_when_no_terminal_can_answer(monkeypatch):
    _streams(monkeypatch, FakeStream(), FakeStream())
    with pytest.raises(EOFError):
        bp_cli._getch()


def test_getch_turns_a_terminal_error_into_eof(monkeypatch):
    with open(os.devnull) as not_a_terminal:  # stdin claims to be a terminal, but termios fails on it
        _streams(monkeypatch, FakeStream(tty=True, fd=not_a_terminal.fileno()), FakeStream(tty=True))
        with pytest.raises(EOFError):
            bp_cli._getch()


def test_prompt_answers_no_and_explains_when_no_key_can_be_read(monkeypatch, output):
    monkeypatch.setattr(bp_cli, "_auto_approve", False)
    _streams(monkeypatch, FakeStream(), FakeStream())
    assert bp_cli.interactive_approval_callback("runcode", "print('hello')") is False
    text = _text(output)
    assert "Approval required: RUNCODE" in text
    assert "Not approved" in text
    assert all(advice in text for advice in ADVICE), text


def test_run_one_shot_stops_at_startup_when_nobody_can_approve(monkeypatch, one_shot, output):
    _streams(monkeypatch, FakeStream(), FakeStream())
    with pytest.raises(SystemExit) as stopped:
        bp_cli.run_one_shot("task", auto_approve=False)
    assert stopped.value.code == 1
    text = _text(output)
    assert "Cannot ask for approval" in text
    assert all(advice in text for advice in ADVICE), text


@pytest.mark.parametrize(
    "auto_approve, stdin_is_a_terminal", [(True, False), (False, True)], ids=["auto-approve-on", "stdin-is-a-terminal"]
)
def test_run_one_shot_goes_on_when_approval_is_possible(monkeypatch, one_shot, auto_approve, stdin_is_a_terminal):
    _streams(monkeypatch, FakeStream(tty=stdin_is_a_terminal), FakeStream())
    with pytest.raises(Stop):
        bp_cli.run_one_shot("task", auto_approve=auto_approve)
