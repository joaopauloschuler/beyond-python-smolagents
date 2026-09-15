"""Tests for the startup connectivity check: check_model_connectivity sends one request, names the failure cause,
skips local model classes and BPSA_SKIP_CONNECTIVITY_CHECK, and never prints the key."""

import io
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from smolagents.models import ChatMessage, MessageRole

FAKE_KEY = "sk-or-v1-0123456789abcdefFAKEKEYabcdef9876543210"


class _FakeModel:
    def __init__(self, error=None):
        self.model_id = "fake/model"
        self.generate = MagicMock(side_effect=error)


def _capture(monkeypatch):
    from smolagents import bp_cli

    console = Console(file=io.StringIO(), record=True, width=300, force_terminal=False)
    monkeypatch.setattr(bp_cli, "console", console)
    monkeypatch.delenv("BPSA_SKIP_CONNECTIVITY_CHECK", raising=False)
    monkeypatch.setenv("BPSA_KEY_VALUE", FAKE_KEY)
    return console


def _failed_check_text(monkeypatch, error) -> str:
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    with pytest.raises(SystemExit):
        bp_cli.check_model_connectivity(_FakeModel(error))
    text = console.export_text()
    assert "Startup connectivity check failed for fake/model" in text
    assert FAKE_KEY not in text
    return text


def test_success_returns_round_trip_seconds_and_sends_one_user_message(monkeypatch):
    from smolagents import bp_cli

    _capture(monkeypatch)
    model = _FakeModel()
    seconds = bp_cli.check_model_connectivity(model)
    assert isinstance(seconds, float) and seconds >= 0
    model.generate.assert_called_once()
    messages = model.generate.call_args.args[0]
    assert len(messages) == 1
    assert isinstance(messages[0], ChatMessage)
    assert messages[0].role == MessageRole.USER
    assert messages[0].content[0]["text"] == bp_cli.CONNECTIVITY_CHECK_PROMPT
    assert model.generate.call_args.kwargs == {}  # plain fake class: no max_tokens


def test_openai_model_class_gets_max_tokens(monkeypatch):
    from smolagents import bp_cli

    _capture(monkeypatch)
    model = type("OpenAIModel", (_FakeModel,), {})()
    bp_cli.check_model_connectivity(model)
    assert model.generate.call_args.kwargs == {"max_tokens": 8}


def test_401_names_the_api_key(monkeypatch):
    text = _failed_check_text(monkeypatch, RuntimeError(f"Error code: 401 - Incorrect API key provided: {FAKE_KEY}"))
    assert "API key" in text
    assert "BPSA_KEY_VALUE" in text
    assert "sk-o...3210" in text


def test_404_names_the_model_id(monkeypatch):
    text = _failed_check_text(monkeypatch, RuntimeError("Error code: 404 - The model `x` does not exist"))
    assert "model id" in text
    assert "BPSA_MODEL_ID" in text


def test_connection_error_names_the_endpoint(monkeypatch):
    text = _failed_check_text(monkeypatch, ConnectionError("Connection error."))
    assert "endpoint" in text
    assert "BPSA_API_ENDPOINT" in text


def test_other_error_reports_the_exception_text(monkeypatch):
    text = _failed_check_text(monkeypatch, ValueError("something odd happened"))
    assert "something odd happened" in text


@pytest.mark.parametrize("class_name", ["TransformersModel", "MLXModel", "VLLMModel"])
def test_local_model_classes_are_skipped(monkeypatch, class_name):
    from smolagents import bp_cli

    _capture(monkeypatch)
    model = type(class_name, (_FakeModel,), {})()
    assert bp_cli.check_model_connectivity(model) is None
    model.generate.assert_not_called()


def test_env_var_skips_the_check(monkeypatch):
    from smolagents import bp_cli

    _capture(monkeypatch)
    monkeypatch.setenv("BPSA_SKIP_CONNECTIVITY_CHECK", "1")
    model = _FakeModel()
    assert bp_cli.check_model_connectivity(model) is None
    model.generate.assert_not_called()


def test_banner_shows_round_trip_row_only_when_measured(monkeypatch):
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    bp_cli.print_banner("m", "OpenAIServerModel", 3, round_trip_seconds=0.84)
    assert "Endpoint: 0.8s round trip" in console.export_text()
    console = _capture(monkeypatch)
    bp_cli.print_banner("m", "OpenAIServerModel", 3)
    assert "round trip" not in console.export_text()
