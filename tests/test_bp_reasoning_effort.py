"""Tests for BPSA_REASONING_EFFORT: build_model forwards it as reasoning_effort to OpenAI-compatible models only."""

from unittest.mock import MagicMock


def _openrouter_env(monkeypatch, effort=None):
    monkeypatch.setenv("BPSA_SERVER_MODEL", "OpenAIServerModel")
    monkeypatch.setenv("BPSA_MODEL_ID", "openai/gpt-5.6-luna")
    monkeypatch.setenv("BPSA_KEY_VALUE", "x")
    monkeypatch.setenv("BPSA_API_ENDPOINT", "https://openrouter.ai/api/v1")
    monkeypatch.delenv("BPSA_PROVIDER_ORDER", raising=False)
    monkeypatch.delenv("BPSA_HAS_SESSION_ID", raising=False)
    if effort is None:
        monkeypatch.delenv("BPSA_REASONING_EFFORT", raising=False)
    else:
        monkeypatch.setenv("BPSA_REASONING_EFFORT", effort)


def _patch_model_class(monkeypatch):
    import smolagents
    from smolagents import bp_cli

    monkeypatch.setattr(bp_cli, "_session_id", None)
    model_class = MagicMock(side_effect=lambda model_id, **kwargs: MagicMock(model_id=model_id))
    monkeypatch.setattr(smolagents, "OpenAIServerModel", model_class)
    return model_class


def test_reasoning_effort_is_forwarded_as_kwarg(monkeypatch):
    from smolagents import bp_cli

    _openrouter_env(monkeypatch, "max")
    model_class = _patch_model_class(monkeypatch)
    bp_cli.build_model()
    kwargs = model_class.call_args.kwargs
    assert kwargs["reasoning_effort"] == "max"
    assert "reasoning_effort" not in kwargs.get("extra_body", {})


def test_reasoning_effort_is_stripped(monkeypatch):
    from smolagents import bp_cli

    _openrouter_env(monkeypatch, "  high \n")
    model_class = _patch_model_class(monkeypatch)
    bp_cli.build_model()
    assert model_class.call_args.kwargs["reasoning_effort"] == "high"


def test_unset_or_blank_effort_sends_nothing(monkeypatch):
    from smolagents import bp_cli

    for value in (None, "", "   "):
        _openrouter_env(monkeypatch, value)
        model_class = _patch_model_class(monkeypatch)
        bp_cli.build_model()
        assert "reasoning_effort" not in model_class.call_args.kwargs


def test_reasoning_effort_reaches_completion_call(monkeypatch):
    """End to end through the real OpenAIServerModel: the kwarg lands in chat.completions.create."""
    from smolagents import bp_cli

    _openrouter_env(monkeypatch, "max")
    monkeypatch.setattr(bp_cli, "_session_id", None)
    model = bp_cli.build_model()
    assert model.kwargs["reasoning_effort"] == "max"
    completion_kwargs = model._prepare_completion_kwargs(messages=[{"role": "user", "content": "hi"}])
    assert completion_kwargs["reasoning_effort"] == "max"


def test_litellm_model_ignores_reasoning_effort(monkeypatch):
    import smolagents
    from smolagents import bp_cli

    _openrouter_env(monkeypatch, "max")
    monkeypatch.setenv("BPSA_SERVER_MODEL", "LiteLLMModel")
    model_class = MagicMock(side_effect=lambda **kwargs: MagicMock(model_id=kwargs["model_id"]))
    monkeypatch.setattr(smolagents, "LiteLLMModel", model_class)
    bp_cli.build_model()
    assert "reasoning_effort" not in model_class.call_args.kwargs


def test_show_config_row(monkeypatch):
    import io
    from types import SimpleNamespace

    from rich.console import Console

    from smolagents import bp_cli

    console = Console(file=io.StringIO(), record=True, width=300, force_terminal=False)
    monkeypatch.setattr(bp_cli, "console", console)
    agent = SimpleNamespace(
        model=SimpleNamespace(model_id="m", api_base="https://openrouter.ai/api/v1"),
        executor_type="exec", max_steps=200, planning_interval=None, compression_config=None,
        memory=SimpleNamespace(steps=[], knowledge=""),
    )

    monkeypatch.delenv("BPSA_REASONING_EFFORT", raising=False)
    bp_cli.cmd_show_config(agent)
    text = console.export_text()
    assert "Reasoning effort" in text and "(model default)" in text

    monkeypatch.setenv("BPSA_REASONING_EFFORT", "max")
    bp_cli.cmd_show_config(agent)
    assert "max" in console.export_text().split("Reasoning effort", 1)[1].splitlines()[0]
