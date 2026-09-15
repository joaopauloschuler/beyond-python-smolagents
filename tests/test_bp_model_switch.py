"""Tests for /model: cmd_model switches agent.model in place, keeps memory, and refuses a model build_model rejects."""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

from rich.console import Console

from smolagents.monitoring import Monitor


class _FakeModel:
    def __init__(self, model_id):
        self.model_id = model_id

    def generate_stream(self):
        pass


def _fake_agent(model_id="old-model"):
    model = _FakeModel(model_id)
    return SimpleNamespace(
        model=model,
        memory=SimpleNamespace(steps=[object()], knowledge="k"),
        compressor=SimpleNamespace(main_model=model),
        monitor=Monitor(tracked_model=model, logger=None),
        stream_outputs=True,
    )


def _capture(monkeypatch):
    from smolagents import bp_cli

    console = Console(file=io.StringIO(), record=True, width=300, force_terminal=False)
    monkeypatch.setattr(bp_cli, "console", console)
    return console


def test_no_args_prints_current_model_and_changes_nothing(monkeypatch):
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    agent = _fake_agent()
    old_model = agent.model
    build = MagicMock()
    monkeypatch.setattr(bp_cli, "build_model", build)
    assert bp_cli.cmd_model(agent, "   ") is None
    text = console.export_text()
    assert "Current model: old-model (_FakeModel)" in text
    assert "Usage: /model <model_id>" in text
    assert agent.model is old_model
    build.assert_not_called()


def test_switch_replaces_model_and_keeps_memory(monkeypatch):
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    agent = _fake_agent()
    memory = agent.memory
    steps = memory.steps
    new_model = _FakeModel("new-model")
    build = MagicMock(return_value=new_model)
    monkeypatch.setattr(bp_cli, "build_model", build)
    assert bp_cli.cmd_model(agent, " new-model ") is new_model
    build.assert_called_once_with(override_model_id="new-model")
    assert agent.model is new_model
    assert agent.memory is memory
    assert agent.memory.steps is steps
    assert agent.compressor.main_model is new_model
    assert agent.monitor.tracked_model is new_model
    assert agent.stream_outputs is True
    assert "Model switched: old-model -> new-model" in console.export_text()


def test_switch_to_non_streaming_model_turns_streaming_off(monkeypatch):
    from smolagents import bp_cli

    _capture(monkeypatch)
    agent = _fake_agent()
    monkeypatch.setattr(bp_cli, "build_model", lambda override_model_id=None: SimpleNamespace(model_id="plain"))
    bp_cli.cmd_model(agent, "plain")
    assert agent.stream_outputs is False


def test_switch_without_compressor_or_monitor(monkeypatch):
    from smolagents import bp_cli

    _capture(monkeypatch)
    agent = SimpleNamespace(model=_FakeModel("old"), memory=object())
    new_model = _FakeModel("new")
    monkeypatch.setattr(bp_cli, "build_model", lambda override_model_id=None: new_model)
    assert bp_cli.cmd_model(agent, "new") is new_model
    assert agent.model is new_model


def test_build_model_fail_keeps_old_model_and_does_not_exit(monkeypatch):
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    agent = _fake_agent()
    old_model = agent.model

    def rejecting_build(override_model_id=None):
        bp_cli.fail(f"Unsupported model id {override_model_id}")  # prints and raises SystemExit

    monkeypatch.setattr(bp_cli, "build_model", rejecting_build)
    assert bp_cli.cmd_model(agent, "bad-id") is None
    text = console.export_text()
    assert "Unsupported model id bad-id" in text
    assert "Model unchanged: old-model" in text
    assert agent.model is old_model
    assert agent.compressor.main_model is old_model


def test_build_model_exception_keeps_old_model(monkeypatch):
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    agent = _fake_agent()
    old_model = agent.model

    def broken_build(override_model_id=None):
        raise ValueError("boom")

    monkeypatch.setattr(bp_cli, "build_model", broken_build)
    assert bp_cli.cmd_model(agent, "x") is None
    assert "Failed to switch model, keeping old-model: boom" in console.export_text()
    assert agent.model is old_model


def test_switch_keeps_openrouter_session_id_and_extra_body(monkeypatch):
    import smolagents
    from smolagents import bp_cli

    _capture(monkeypatch)
    monkeypatch.setenv("BPSA_SERVER_MODEL", "OpenAIServerModel")
    monkeypatch.setenv("BPSA_MODEL_ID", "first-model")
    monkeypatch.setenv("BPSA_KEY_VALUE", "x")
    monkeypatch.setenv("BPSA_API_ENDPOINT", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("BPSA_PROVIDER_ORDER", "Alpha, Beta")
    monkeypatch.setenv("BPSA_HAS_SESSION_ID", "1")
    monkeypatch.setattr(bp_cli, "_session_id", None)
    model_class = MagicMock(side_effect=lambda model_id, **kwargs: _FakeModel(model_id))
    monkeypatch.setattr(smolagents, "OpenAIServerModel", model_class)

    first = bp_cli.build_model()
    agent = _fake_agent()
    agent.model = first
    assert bp_cli.cmd_model(agent, "second-model").model_id == "second-model"
    assert model_class.call_count == 2
    first_body = model_class.call_args_list[0].kwargs["extra_body"]
    second_body = model_class.call_args_list[1].kwargs["extra_body"]
    assert model_class.call_args_list[1].args[0] == "second-model"
    assert second_body["session_id"] == first_body["session_id"] == bp_cli.current_session_id()
    assert second_body["provider"] == {"order": ["Alpha", "Beta"]}
    assert second_body["usage"] == {"include": True}


def test_completer_and_help_list_model():
    from smolagents import bp_cli

    assert "/model" in bp_cli.SLASH_COMMANDS
    assert bp_cli.SLASH_COMMANDS == sorted(bp_cli.SLASH_COMMANDS)
    console = Console(file=io.StringIO(), record=True, width=300, force_terminal=False)
    original = bp_cli.console
    bp_cli.console = console
    try:
        bp_cli.print_help()
    finally:
        bp_cli.console = original
    assert "/model [id]" in console.export_text()
