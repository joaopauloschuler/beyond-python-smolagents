"""Tests for the model context length: BPSA_CONTEXT_LENGTH wins over fetch_context_length, the OpenRouter /models
payload is parsed without network, and the turn summary, banner, compression default and /show-config use it."""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from smolagents.bp_compression import CompressionConfig
from smolagents.monitoring import TokenUsage

FAKE_KEY = "sk-or-v1-0123456789abcdefFAKEKEYabcdef9876543210"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _model(model_id="deepseek/deepseek-chat-v3-0324", api_base=OPENROUTER_BASE):
    return SimpleNamespace(model_id=model_id, api_base=api_base, api_key=FAKE_KEY)


def _fake_models_payload():
    return {"data": [
        {"id": "other/model", "context_length": 8192},
        {"id": "deepseek/deepseek-chat-v3-0324", "context_length": 163840,
         "top_provider": {"context_length": 163840}},
        {"id": "no-field/model"},
        {"id": "non-int/model", "context_length": "128k"},
        {"id": "provider-only/model", "top_provider": {"context_length": 32000}},
    ]}


def _patch_requests(monkeypatch, payload=None, error=None):
    """Replace requests.get with a MagicMock returning a response whose json() gives payload, or raising error."""
    import requests

    response = MagicMock()
    response.json.return_value = payload if payload is not None else _fake_models_payload()
    fake_get = MagicMock(return_value=response, side_effect=error)
    monkeypatch.setattr(requests, "get", fake_get)
    return fake_get


def _capture(monkeypatch):
    from smolagents import bp_cli

    console = Console(file=io.StringIO(), record=True, width=300, force_terminal=False)
    monkeypatch.setattr(bp_cli, "console", console)
    return console


def _fake_agent(context_length=None, last_input_tokens=0, threshold=0):
    step = SimpleNamespace(token_usage=TokenUsage(input_tokens=last_input_tokens, output_tokens=5))
    agent = SimpleNamespace(
        model=_model(),
        executor_type="exec",
        max_steps=200,
        planning_interval=None,
        compression_config=CompressionConfig(estimated_token_threshold=threshold),
        compressor=None,
        memory=SimpleNamespace(steps=[step] if last_input_tokens else [], knowledge=""),
        get_context_char_size=lambda: 12345,
    )
    if context_length is not None:
        agent.context_length = context_length
        agent.context_length_source = "env"
    return agent


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("BPSA_CONTEXT_LENGTH", raising=False)
    monkeypatch.delenv("BPSA_COMPRESSION_TOKEN_THRESHOLD", raising=False)


def test_env_var_wins_over_fetch(monkeypatch):
    from smolagents import bp_cli

    fake_get = _patch_requests(monkeypatch)
    monkeypatch.setenv("BPSA_CONTEXT_LENGTH", "128000")
    assert bp_cli.resolve_context_length(_model()) == 128000
    fake_get.assert_not_called()


@pytest.mark.parametrize("value", ["0", "-5", "abc", ""])
def test_unusable_env_var_falls_back_to_fetch(monkeypatch, value):
    from smolagents import bp_cli

    _patch_requests(monkeypatch)
    monkeypatch.setenv("BPSA_CONTEXT_LENGTH", value)
    assert bp_cli.resolve_context_length(_model()) == 163840


def test_fetch_exact_id_match_sends_key_once_with_timeout(monkeypatch):
    from smolagents import bp_cli

    fake_get = _patch_requests(monkeypatch)
    assert bp_cli.fetch_context_length(_model()) == 163840
    fake_get.assert_called_once()
    assert fake_get.call_args.args[0] == f"{OPENROUTER_BASE}/models"
    assert fake_get.call_args.kwargs["headers"] == {"Authorization": f"Bearer {FAKE_KEY}"}
    assert fake_get.call_args.kwargs["timeout"] == bp_cli.CONTEXT_LENGTH_FETCH_TIMEOUT_SECONDS


def test_fetch_reads_openai_model_client_kwargs(monkeypatch):
    from smolagents import bp_cli

    fake_get = _patch_requests(monkeypatch)
    model = SimpleNamespace(model_id="deepseek/deepseek-chat-v3-0324",
                            client_kwargs={"base_url": OPENROUTER_BASE, "api_key": FAKE_KEY})
    assert bp_cli.model_endpoint(model) == (OPENROUTER_BASE, FAKE_KEY)
    assert bp_cli.fetch_context_length(model) == 163840
    assert fake_get.call_args.kwargs["headers"] == {"Authorization": f"Bearer {FAKE_KEY}"}
    assert bp_cli.model_endpoint(SimpleNamespace(model_id="x")) == (None, None)


def test_fetch_strips_leading_tilde_when_no_exact_match(monkeypatch):
    from smolagents import bp_cli

    _patch_requests(monkeypatch)
    assert bp_cli.fetch_context_length(_model("~deepseek/deepseek-chat-v3-0324")) == 163840


def test_fetch_prefers_exact_tilde_id_when_listed(monkeypatch):
    from smolagents import bp_cli

    payload = {"data": [{"id": "x/m", "context_length": 1}, {"id": "~x/m", "context_length": 2}]}
    _patch_requests(monkeypatch, payload=payload)
    assert bp_cli.fetch_context_length(_model("~x/m")) == 2


def test_fetch_uses_top_provider_when_context_length_missing(monkeypatch):
    from smolagents import bp_cli

    _patch_requests(monkeypatch)
    assert bp_cli.fetch_context_length(_model("provider-only/model")) == 32000


@pytest.mark.parametrize("model_id", ["missing/model", "no-field/model", "non-int/model"])
def test_fetch_returns_none_for_missing_model_or_field(monkeypatch, model_id):
    from smolagents import bp_cli

    _patch_requests(monkeypatch)
    assert bp_cli.fetch_context_length(_model(model_id)) is None


@pytest.mark.parametrize("error", [RuntimeError("boom"), TimeoutError("timed out")])
def test_fetch_returns_none_on_exception_or_timeout(monkeypatch, error):
    from smolagents import bp_cli

    _patch_requests(monkeypatch, error=error)
    assert bp_cli.fetch_context_length(_model()) is None


def test_fetch_returns_none_on_bad_payload(monkeypatch):
    from smolagents import bp_cli

    _patch_requests(monkeypatch, payload={"data": "not a list"})
    assert bp_cli.fetch_context_length(_model()) is None


@pytest.mark.parametrize("api_base", ["https://api.openai.com/v1", "https://api.poe.com/v1", None])
def test_non_openrouter_base_skips_http(monkeypatch, api_base):
    from smolagents import bp_cli

    fake_get = _patch_requests(monkeypatch)
    assert bp_cli.fetch_context_length(_model(api_base=api_base)) is None
    assert bp_cli.resolve_context_length(_model(api_base=api_base)) is None
    fake_get.assert_not_called()


def test_turn_summary_shows_percent_when_known(monkeypatch):
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    bp_cli.print_turn_summary(1, 2.0, 61000, 10, _fake_agent(context_length=100000, last_input_tokens=61000))
    text = console.export_text()
    assert "Context: 61%" in text
    assert "chars" not in text.split("Knowledge")[0]


@pytest.mark.parametrize("percent, colour", [(50, "green"), (70, "yellow"), (90, "red")])
def test_context_percent_colours(percent, colour):
    from smolagents import bp_cli

    assert bp_cli.format_context_percent(percent) == f"[{colour}]{percent}%[/]"


def test_turn_summary_keeps_chars_when_unknown(monkeypatch):
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    bp_cli.print_turn_summary(1, 2.0, 61000, 10, _fake_agent(context_length=None, last_input_tokens=61000))
    text = console.export_text()
    assert "Context: 12,345 chars" in text
    assert "%" not in text.split("Context")[1].split("|")[0]


def test_turn_summary_keeps_chars_when_no_step_reports_tokens(monkeypatch):
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    bp_cli.print_turn_summary(1, 2.0, 0, 0, _fake_agent(context_length=100000, last_input_tokens=0))
    assert "Context: 12,345 chars" in console.export_text()


def test_last_input_tokens_comes_from_the_latest_step_with_usage():
    from smolagents import bp_cli

    steps = [
        SimpleNamespace(token_usage=TokenUsage(input_tokens=100, output_tokens=1)),
        SimpleNamespace(token_usage=TokenUsage(input_tokens=250, output_tokens=1)),
        SimpleNamespace(),
        SimpleNamespace(token_usage=None),
    ]
    agent = SimpleNamespace(memory=SimpleNamespace(steps=steps))
    assert bp_cli.get_agent_last_input_tokens(agent) == 250
    assert bp_cli.get_agent_last_input_tokens(SimpleNamespace()) == 0


def test_banner_shows_context_row_only_when_known(monkeypatch):
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    bp_cli.print_banner("m", "OpenAIServerModel", 3, context_length=128000)
    assert "Context: 128,000 tokens" in console.export_text()
    console = _capture(monkeypatch)
    bp_cli.print_banner("m", "OpenAIServerModel", 3)
    assert "Context:" not in console.export_text()


def test_apply_defaults_compression_threshold_to_75_percent_when_env_unset(monkeypatch):
    from smolagents import bp_cli
    from smolagents.bp_thinkers import DEFAULT_THINKER_COMPRESSION

    agent = _fake_agent()
    shared = agent.compression_config
    agent.compressor = SimpleNamespace(config=shared)
    bp_cli.apply_context_length(agent, 100000)
    assert agent.context_length == 100000
    assert agent.context_length_source == "openrouter"
    assert agent.compression_config.estimated_token_threshold == 75000
    assert agent.compressor.config is agent.compression_config
    assert shared.estimated_token_threshold == 0  # the original config object is left alone
    assert DEFAULT_THINKER_COMPRESSION.estimated_token_threshold == 0


@pytest.mark.parametrize("env_value, expected", [("12345", 12345), ("0", 0)])
def test_apply_keeps_explicit_env_threshold(monkeypatch, env_value, expected):
    from smolagents import bp_cli

    monkeypatch.setenv("BPSA_COMPRESSION_TOKEN_THRESHOLD", env_value)
    agent = _fake_agent(threshold=expected)
    bp_cli.apply_context_length(agent, 100000)
    assert agent.compression_config.estimated_token_threshold == expected


def test_apply_unknown_context_restores_env_default_and_reports_source(monkeypatch):
    from smolagents import bp_cli

    agent = _fake_agent(threshold=75000)
    bp_cli.apply_context_length(agent, None)
    assert agent.context_length is None
    assert agent.context_length_source == "unknown"
    assert agent.compression_config.estimated_token_threshold == 0
    monkeypatch.setenv("BPSA_CONTEXT_LENGTH", "4096")
    bp_cli.apply_context_length(agent, 4096)
    assert agent.context_length_source == "env"
    assert agent.compression_config.estimated_token_threshold == 3072


def test_show_config_rows(monkeypatch):
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    agent = _fake_agent()
    bp_cli.apply_context_length(agent, 200000)
    bp_cli.cmd_show_config(agent)
    text = console.export_text()
    row = [line for line in text.splitlines() if line.strip().startswith("Context length")][0]
    assert "200,000 tokens" in row and "openrouter" in row
    threshold_row = [line for line in text.splitlines() if "estimated_token_threshold" in line][0]
    assert "150000" in threshold_row and "75% of context length" in threshold_row
    console = _capture(monkeypatch)
    bp_cli.cmd_show_config(_fake_agent())
    row = [line for line in console.export_text().splitlines() if line.strip().startswith("Context length")][0]
    assert "unknown" in row
