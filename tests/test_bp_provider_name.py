"""Tests for TokenUsage.provider: extraction from OpenRouter responses and the "via <Provider>" turn-summary figure."""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rich.console import Console

from smolagents.models import (
    ChatMessage,
    ChatMessageStreamDelta,
    MessageRole,
    OpenAIModel,
    agglomerate_stream_deltas,
    extract_provider_name,
)
from smolagents.monitoring import Monitor, TokenUsage


def test_token_usage_default_and_dict():
    usage = TokenUsage(input_tokens=10, output_tokens=5)
    assert usage.provider is None
    served = TokenUsage(input_tokens=10, output_tokens=5, provider="DeepInfra")
    assert served.total_tokens == 15
    assert served.dict()["provider"] == "DeepInfra"


def test_extract_from_sdk_like_object_attribute():
    assert extract_provider_name(SimpleNamespace(provider="DeepInfra", usage=None)) == "DeepInfra"


def test_extract_from_model_extra():
    response = SimpleNamespace(model_extra={"provider": "Relace"})
    assert extract_provider_name(response) == "Relace"
    assert extract_provider_name(SimpleNamespace(provider=None, model_extra={"provider": "Relace"})) == "Relace"


def test_extract_from_dict():
    assert extract_provider_name({"provider": " Together "}) == "Together"


def test_extract_missing_none_and_non_string_give_none():
    assert extract_provider_name(None) is None
    assert extract_provider_name(SimpleNamespace(usage=None)) is None
    assert extract_provider_name({}) is None
    assert extract_provider_name(SimpleNamespace(provider=None)) is None
    assert extract_provider_name(SimpleNamespace(provider="")) is None
    assert extract_provider_name(SimpleNamespace(provider="   ")) is None
    assert extract_provider_name(SimpleNamespace(provider=42)) is None
    assert extract_provider_name(SimpleNamespace(provider={"order": ["deepseek"]})) is None
    assert extract_provider_name(SimpleNamespace(model_extra=None)) is None
    assert extract_provider_name(SimpleNamespace(model_extra={"provider": 7})) is None
    assert extract_provider_name(SimpleNamespace(model_extra="oops")) is None
    assert extract_provider_name(42) is None


def _mock_response(provider):
    response = MagicMock(spec=["choices", "usage", "provider"] if provider is not None else ["choices", "usage"])
    response.choices = [MagicMock()]
    response.choices[0].message.role = "assistant"
    response.choices[0].message.content = "hi"
    response.choices[0].message.tool_calls = None
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 2
    if provider is not None:
        response.provider = provider
    return response


def _generate_with(response):
    with patch("openai.OpenAI") as MockOpenAI:
        client = MagicMock()
        MockOpenAI.return_value = client
        client.chat.completions.create.return_value = response
        model = OpenAIModel(model_id="test-model", api_key="x")
        return model.generate([ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])])


def test_generate_populates_provider():
    assert _generate_with(_mock_response("DeepInfra")).token_usage.provider == "DeepInfra"


def test_generate_without_provider_field():
    message = _generate_with(_mock_response(None))
    assert message.token_usage.provider is None
    assert message.token_usage.input_tokens == 10


def _stream_chunk(content=None, usage=None, provider=None):
    chunk = MagicMock(spec=["choices", "usage", "provider"] if provider is not None else ["choices", "usage"])
    chunk.usage = usage
    if provider is not None:
        chunk.provider = provider
    if content is None:
        chunk.choices = []
    else:
        delta = MagicMock()
        delta.content = content
        delta.tool_calls = None
        chunk.choices = [MagicMock(delta=delta, finish_reason=None)]
    return chunk


def _generate_stream_with(chunks):
    with patch("openai.OpenAI") as MockOpenAI:
        client = MagicMock()
        MockOpenAI.return_value = client
        client.chat.completions.create.return_value = iter(chunks)
        model = OpenAIModel(model_id="test-model", api_key="x")
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])]
        return list(model.generate_stream(messages))


def test_generate_stream_carries_provider_from_earlier_chunk_to_usage_delta():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2)
    deltas = _generate_stream_with(
        [_stream_chunk("a", provider="Relace"), _stream_chunk("b"), _stream_chunk(usage=usage)]
    )
    usage_deltas = [d for d in deltas if d.token_usage]
    assert len(usage_deltas) == 1
    assert usage_deltas[0].token_usage.provider == "Relace"
    assert agglomerate_stream_deltas(deltas).token_usage.provider == "Relace"


def test_generate_stream_without_provider_field():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2)
    deltas = _generate_stream_with([_stream_chunk("a"), _stream_chunk(usage=usage)])
    usage_deltas = [d for d in deltas if d.token_usage]
    assert usage_deltas[0].token_usage.provider is None
    assert usage_deltas[0].token_usage.input_tokens == 10


def test_agglomerate_stream_deltas_keeps_last_non_none_provider():
    deltas = [
        ChatMessageStreamDelta(content="a", token_usage=TokenUsage(input_tokens=1, output_tokens=1, provider="A")),
        ChatMessageStreamDelta(content="b", token_usage=TokenUsage(input_tokens=1, output_tokens=1, provider="B")),
        ChatMessageStreamDelta(content="c", token_usage=TokenUsage(input_tokens=1, output_tokens=1)),
        ChatMessageStreamDelta(content="d"),
    ]
    message = agglomerate_stream_deltas(deltas)
    assert message.content == "abcd"
    assert message.token_usage.provider == "B"
    assert agglomerate_stream_deltas([ChatMessageStreamDelta(content="x")]).token_usage.provider is None


def _step(provider):
    return SimpleNamespace(
        timing=SimpleNamespace(duration=0.1),
        token_usage=TokenUsage(input_tokens=1, output_tokens=1, provider=provider),
    )


def test_monitor_tracks_last_provider_and_resets():
    monitor = Monitor(tracked_model=None, logger=SimpleNamespace(log=lambda *a, **k: None))
    assert monitor.last_provider is None
    monitor.update_metrics(_step("DeepInfra"))
    monitor.update_metrics(_step(None))
    assert monitor.last_provider == "DeepInfra"
    monitor.update_metrics(_step("Relace"))
    assert monitor.last_provider == "Relace"
    monitor.reset()
    assert monitor.last_provider is None


def test_get_agent_last_provider():
    from smolagents.bp_cli import get_agent_last_provider

    monitor = Monitor(tracked_model=None, logger=None)
    assert get_agent_last_provider(SimpleNamespace(monitor=monitor)) is None
    monitor.last_provider = "DeepInfra"
    assert get_agent_last_provider(SimpleNamespace(monitor=monitor)) == "DeepInfra"
    assert get_agent_last_provider(SimpleNamespace()) is None


class _FakeAgent:
    def __init__(self, provider):
        self.monitor = Monitor(tracked_model=None, logger=None)
        self.monitor.last_provider = provider
        self.memory = SimpleNamespace(steps=[], knowledge="")

    def get_context_char_size(self):
        return 0


def _capture_turn_summary(monkeypatch, agent):
    from smolagents import bp_cli

    console = Console(file=io.StringIO(), record=True, width=300, force_terminal=False)
    monkeypatch.setattr(bp_cli, "console", console)
    bp_cli.print_turn_summary(1, 2.0, 1000, 50, agent)
    return console.export_text()


def test_print_turn_summary_shows_provider(monkeypatch):
    text = _capture_turn_summary(monkeypatch, _FakeAgent("DeepInfra"))
    assert "| via DeepInfra |" in text
    assert "In: 1,000" in text


def test_print_turn_summary_without_provider(monkeypatch):
    assert "via" not in _capture_turn_summary(monkeypatch, _FakeAgent(None))
    assert "via" not in _capture_turn_summary(monkeypatch, None)


def test_print_stats_shows_last_provider(monkeypatch):
    from smolagents import bp_cli

    console = Console(file=io.StringIO(), record=True, width=300, force_terminal=False)
    monkeypatch.setattr(bp_cli, "console", console)
    stats = {"turns": 1, "total_time": 1.0, "total_input_tokens": 10, "total_output_tokens": 5}
    bp_cli.print_stats(stats, _FakeAgent("Relace"))
    text = console.export_text()
    assert "Last provider" in text
    assert "Relace" in text


def test_token_usage_session_round_trip_tolerates_old_files():
    from smolagents.bp_session import _deserialize_token_usage, _serialize_token_usage

    usage = TokenUsage(input_tokens=10, output_tokens=2, provider="DeepInfra")
    assert _deserialize_token_usage(_serialize_token_usage(usage)) == usage
    old = _deserialize_token_usage({"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 1})
    assert old.provider is None
