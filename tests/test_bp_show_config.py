"""Tests for /show-config: cmd_show_config prints the effective settings with their source and never the full key."""

import io
from types import SimpleNamespace

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
