"""Tests for /show-config: cmd_show_config prints the effective settings with their source and never the full key."""

import io
from types import SimpleNamespace

import pytest
from rich.console import Console

from smolagents.bp_compression import CompressionConfig

FAKE_KEY = "sk-or-v1-0123456789abcdefFAKEKEYabcdef9876543210"


def _fake_agent(model_id="main-model", compression_config=None):
    model = SimpleNamespace(model_id=model_id, api_base="https://openrouter.ai/api/v1")
    return SimpleNamespace(
        model=model,
        executor_type="exec",
        max_steps=200,
        planning_interval=None,
        compression_config=compression_config,
        memory=SimpleNamespace(steps=[], knowledge=""),
    )


def _capture(monkeypatch):
    from smolagents import bp_cli

    console = Console(file=io.StringIO(), record=True, width=300, force_terminal=False)
    monkeypatch.setattr(bp_cli, "console", console)
    return console


def _clear_bpsa_env(monkeypatch):
    import os

    for name in list(os.environ):
        if name.startswith("BPSA_"):
            monkeypatch.delenv(name, raising=False)


def _unset_env(monkeypatch, *names):
    """Unset names and have monkeypatch restore them afterwards, even when load_dotenv sets them during the test."""
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _row(text: str, label: str) -> str:
    lines = [line for line in text.splitlines() if line.strip().startswith(label)]
    assert lines, f"row {label!r} missing in:\n{text}"
    return lines[0]


def test_key_is_masked_and_full_key_absent(monkeypatch):
    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    monkeypatch.setenv("BPSA_KEY_VALUE", FAKE_KEY)
    console = _capture(monkeypatch)
    bp_cli.cmd_show_config(_fake_agent())
    text = console.export_text()
    assert "sk-o...3210" in _row(text, "API key")
    assert FAKE_KEY not in text
    assert FAKE_KEY[4:-4] not in text


def test_mask_secret_edge_cases():
    from smolagents import bp_cli

    assert bp_cli.mask_secret(None) == "(not set)"
    assert bp_cli.mask_secret("") == "(not set)"
    assert bp_cli.mask_secret("short-key") == "****"
    assert bp_cli.mask_secret("abcd12345678wxyz") == "abcd...wxyz"


def test_key_not_set(monkeypatch):
    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    console = _capture(monkeypatch)
    bp_cli.cmd_show_config(_fake_agent())
    row = _row(console.export_text(), "API key")
    assert "(not set)" in row
    assert "default" in row


def test_session_id_row_when_enabled(monkeypatch):
    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    monkeypatch.setenv("BPSA_HAS_SESSION_ID", "1")
    monkeypatch.setattr(bp_cli, "_session_id", "bpsa-feedfacefeedface")
    console = _capture(monkeypatch)
    bp_cli.cmd_show_config(_fake_agent())
    row = _row(console.export_text(), "OpenRouter session id")
    assert "bpsa-feedfacefeedface" in row
    assert "env" in row


def test_session_id_row_when_disabled(monkeypatch):
    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    monkeypatch.setattr(bp_cli, "_session_id", None)
    console = _capture(monkeypatch)
    bp_cli.cmd_show_config(_fake_agent())
    row = _row(console.export_text(), "OpenRouter session id")
    assert "(disabled)" in row
    assert "default" in row
    assert bp_cli._session_id is None


def test_env_source_env_dotenv_and_default(monkeypatch):
    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    monkeypatch.setenv("BPSA_PROVIDER_ORDER", "Alpha,Beta")
    monkeypatch.setenv("BPSA_MAX_TOKENS", "1234")
    monkeypatch.setattr(bp_cli, "_dotenv_keys", {"BPSA_MAX_TOKENS"})
    assert bp_cli.env_source("BPSA_PROVIDER_ORDER") == "env"
    assert bp_cli.env_source("BPSA_MAX_TOKENS") == ".env"
    assert bp_cli.env_source("BPSA_SYSTEM_PROMPT_FIRST") == "default"
    console = _capture(monkeypatch)
    bp_cli.cmd_show_config(_fake_agent())
    text = console.export_text()
    assert "Alpha,Beta" in _row(text, "Provider order") and "env" in _row(text, "Provider order")
    assert "1234" in _row(text, "Max tokens") and ".env" in _row(text, "Max tokens")
    assert "first" in _row(text, "System prompt position") and "default" in _row(text, "System prompt position")


def test_try_load_dotenv_records_only_keys_it_added(monkeypatch, tmp_path):
    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    monkeypatch.setenv("BPSA_MAX_TOKENS", "from-process")
    (tmp_path / ".env").write_text("BPSA_MAX_TOKENS=from-file\nBPSA_PROVIDER_ORDER=Alpha\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bp_cli, "_dotenv_keys", set())
    _capture(monkeypatch)
    bp_cli.try_load_dotenv()
    assert bp_cli._dotenv_keys == {"BPSA_PROVIDER_ORDER"}
    assert bp_cli.env_source("BPSA_PROVIDER_ORDER") == ".env"
    assert bp_cli.env_source("BPSA_MAX_TOKENS") == "env"
    monkeypatch.delenv("BPSA_PROVIDER_ORDER")  # load_dotenv wrote into os.environ; keep other tests clean


def test_compression_rows_reflect_agent_config(monkeypatch):
    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    config = CompressionConfig(
        enabled=True, keep_recent_steps=7, max_uncompressed_steps=9, keep_compressed_steps=3,
        max_compressed_steps=5, estimated_token_threshold=12000,
        compression_model=SimpleNamespace(model_id="cheap-compressor"),
    )
    console = _capture(monkeypatch)
    bp_cli.cmd_show_config(_fake_agent(compression_config=config))
    text = console.export_text()
    assert "True" in _row(text, "Compression enabled")
    assert "7" in _row(text, "Compression keep_recent_steps")
    assert "/compression-keep-recent-steps" in _row(text, "Compression keep_recent_steps")
    assert "9" in _row(text, "Compression max_uncompressed_steps")
    assert "3" in _row(text, "Compression keep_compressed_steps")
    assert "5" in _row(text, "Compression max_compressed_steps")
    assert "12000" in _row(text, "Compression estimated_token_threshold")
    assert "cheap-compressor" in _row(text, "Compression model")


def test_compression_model_same_as_main_and_no_config(monkeypatch):
    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    console = _capture(monkeypatch)
    bp_cli.cmd_show_config(_fake_agent(compression_config=CompressionConfig()))
    assert "same as main" in _row(console.export_text(), "Compression model")
    console = _capture(monkeypatch)
    bp_cli.cmd_show_config(_fake_agent())
    assert "no config on this agent" in _row(console.export_text(), "Compression")


def test_model_and_tool_rows(monkeypatch):
    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    monkeypatch.setenv("BPSA_MODEL_ID", "startup-model")
    monkeypatch.setenv("BPSA_TMUX", "1")
    monkeypatch.setattr(bp_cli, "_auto_approve", True)
    console = _capture(monkeypatch)
    agent = _fake_agent(model_id="switched-model")
    agent.max_steps = 33
    agent.planning_interval = 4
    bp_cli.cmd_show_config(agent, browser_enabled=True, tmux_enabled=True, mcp_servers=["a", "b"])
    text = console.export_text()
    assert "SimpleNamespace" in _row(text, "Model class")
    assert "switched-model" in _row(text, "Model id") and "/model" in _row(text, "Model id")
    assert "https://openrouter.ai/api/v1" in _row(text, "Endpoint")
    assert "exec" in _row(text, "Executor")
    assert "33" in _row(text, "Max steps") and "/set-max-steps" in _row(text, "Max steps")
    assert "4" in _row(text, "Planning interval") and "/plan" in _row(text, "Planning interval")
    assert "on" in _row(text, "Auto-approve")
    assert "on" in _row(text, "Browser tools") and "--browser" in _row(text, "Browser tools")
    assert "off" in _row(text, "GUI tools") and "default" in _row(text, "GUI tools")
    assert "on" in _row(text, "Tmux tools") and "env" in _row(text, "Tmux tools")
    assert "2" in _row(text, "MCP servers") and "--mcp" in _row(text, "MCP servers")
    assert "~/.bpsa.yaml" in text


def test_completer_and_help_list_show_config():
    from smolagents import bp_cli

    assert "/show-config" in bp_cli.SLASH_COMMANDS
    assert bp_cli.SLASH_COMMANDS == sorted(bp_cli.SLASH_COMMANDS)
    console = Console(file=io.StringIO(), record=True, width=300, force_terminal=False)
    original = bp_cli.console
    bp_cli.console = console
    try:
        bp_cli.print_help()
    finally:
        bp_cli.console = original
    assert "/show-config" in console.export_text()


def test_resolve_auto_approve_precedence(monkeypatch):
    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    monkeypatch.setattr(bp_cli, "_dotenv_keys", set())
    assert bp_cli.resolve_auto_approve(None) == (False, "default")
    monkeypatch.setenv("BPSA_AUTO_APPROVE", " ON ")
    assert bp_cli.resolve_auto_approve(None) == (True, "env")
    assert bp_cli.resolve_auto_approve(False) == (False, "--auto-approve")
    monkeypatch.setenv("BPSA_AUTO_APPROVE", "off")
    assert bp_cli.resolve_auto_approve(None) == (False, "env")
    assert bp_cli.resolve_auto_approve(True) == (True, "--auto-approve")


def test_resolve_auto_approve_warns_and_ignores_invalid_value(monkeypatch):
    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    monkeypatch.setenv("BPSA_AUTO_APPROVE", "yes")
    console = _capture(monkeypatch)
    assert bp_cli.resolve_auto_approve(None) == (False, "default")
    text = " ".join(console.export_text().split())
    assert "Ignoring BPSA_AUTO_APPROVE='yes': expected on or off." in text


@pytest.mark.parametrize("entry", ["run_one_shot", "run_repl"])
def test_entry_points_take_verbose_and_auto_approve_from_dotenv(monkeypatch, tmp_path, entry):
    from smolagents import bp_cli

    class Stop(Exception):
        pass

    def stop():
        raise Stop

    _clear_bpsa_env(monkeypatch)
    _unset_env(monkeypatch, "BPSA_AUTO_APPROVE", "BPSA_VERBOSE")
    (tmp_path / ".env").write_text("BPSA_AUTO_APPROVE=on\nBPSA_VERBOSE=1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bp_cli, "_dotenv_keys", set())
    for name in ("_auto_approve", "_auto_approve_source", "_verbose"):
        monkeypatch.setattr(bp_cli, name, getattr(bp_cli, name))  # restored after the entry point sets them
    monkeypatch.setattr(bp_cli, "check_required_env", stop)  # stop right after auto-approve is resolved
    _capture(monkeypatch)
    kwargs = {"task": "t"} if entry == "run_one_shot" else {}
    with pytest.raises(Stop):
        getattr(bp_cli, entry)(auto_approve=None, **kwargs)
    assert (bp_cli._verbose, bp_cli._auto_approve, bp_cli._auto_approve_source) == (True, True, ".env")


def test_main_passes_auto_approve_only_when_flag_given(monkeypatch, tmp_path):
    import sys

    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    monkeypatch.setenv("BPSA_AUTO_APPROVE", "on")  # resolved later, after .env is loaded
    monkeypatch.chdir(tmp_path)  # main() loads ./.env; an empty folder keeps a local .env out of the test
    _capture(monkeypatch)
    seen = []
    monkeypatch.setattr(bp_cli, "run_one_shot", lambda task, **kwargs: seen.append(kwargs["auto_approve"]))
    for flag in ([], ["--auto-approve", "on"], ["--auto-approve", "off"]):
        monkeypatch.setattr(sys, "argv", ["bpsa", *flag, "run", "task"])
        bp_cli.main()
    assert seen == [None, True, False]


def test_show_config_auto_approve_row_shows_its_source(monkeypatch):
    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    monkeypatch.setattr(bp_cli, "_auto_approve", True)
    monkeypatch.setattr(bp_cli, "_auto_approve_source", ".env")
    console = _capture(monkeypatch)
    bp_cli.cmd_show_config(_fake_agent())
    cells = _row(console.export_text(), "Auto-approve").replace("│", " ").split()
    assert cells == ["Auto-approve", "on", ".env"]


def test_main_reads_feature_flags_from_dotenv(monkeypatch, tmp_path):
    import sys

    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    _unset_env(monkeypatch, "BPSA_BROWSER", "BPSA_GUI", "BPSA_IMAGE", "BPSA_TMUX", "BPSA_MCP")
    (tmp_path / ".env").write_text("BPSA_BROWSER=1\nBPSA_GUI=1\nBPSA_IMAGE=1\nBPSA_TMUX=1\nBPSA_MCP=my-mcp-server\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bp_cli, "_dotenv_keys", set())
    _capture(monkeypatch)
    monkeypatch.setattr(bp_cli, "_parse_mcp_servers", lambda servers: servers)  # skip building MCP server parameters
    seen = {}
    monkeypatch.setattr(bp_cli, "run_one_shot", lambda task, **kwargs: seen.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["bpsa", "run", "task"])
    bp_cli.main()
    flags = [seen[name] for name in ("browser_enabled", "gui_enabled", "image_enabled", "tmux_enabled")]
    assert (flags, seen["mcp_servers"]) == ([True] * 4, ["my-mcp-server"])


def test_try_load_dotenv_handles_each_path_once(monkeypatch, tmp_path):
    import os

    from smolagents import bp_cli

    _clear_bpsa_env(monkeypatch)
    _unset_env(monkeypatch, "BPSA_MAX_TOKENS")
    (tmp_path / "with_env").mkdir()
    (tmp_path / "with_env" / ".env").write_text("BPSA_MAX_TOKENS=123\n")
    (tmp_path / "without_env").mkdir()
    monkeypatch.setattr(bp_cli, "_dotenv_keys", set())
    console = _capture(monkeypatch)
    for folder in ("with_env", "with_env", "without_env", "without_env"):
        monkeypatch.chdir(tmp_path / folder)
        bp_cli.try_load_dotenv()  # main() calls it, then run_one_shot() or run_repl() calls it again
    text = console.export_text()
    assert (text.count("Loaded .env from:"), text.count("No .env file found")) == (1, 1)
    assert (os.environ["BPSA_MAX_TOKENS"], bp_cli._dotenv_keys) == ("123", {"BPSA_MAX_TOKENS"})
