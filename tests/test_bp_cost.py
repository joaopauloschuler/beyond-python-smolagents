"""Tests for TokenUsage.cost_usd: OpenRouter usage.cost, the BPSA_PRICE_* estimate and the `$` turn-summary figure."""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from smolagents.models import (
    ChatMessage,
    ChatMessageStreamDelta,
    MessageRole,
    OpenAIModel,
    agglomerate_stream_deltas,
    estimate_cost_usd,
    extract_response_cost,
    request_cost_usd,
)
from smolagents.monitoring import Monitor, TokenUsage


PRICE_VARS = ("BPSA_PRICE_INPUT_PER_M", "BPSA_PRICE_OUTPUT_PER_M", "BPSA_PRICE_CACHED_INPUT_PER_M")


@pytest.fixture(autouse=True)
def _no_price_env(monkeypatch):
    for name in PRICE_VARS:
        monkeypatch.delenv(name, raising=False)


def test_token_usage_default_and_dict():
    usage = TokenUsage(input_tokens=10, output_tokens=5)
    assert usage.cost_usd == 0.0
    priced = TokenUsage(input_tokens=10, output_tokens=5, cost_usd=0.0123)
    assert priced.total_tokens == 15
    assert priced.dict()["cost_usd"] == 0.0123


def test_extract_from_sdk_like_object_and_dict():
    assert extract_response_cost(SimpleNamespace(prompt_tokens=1, cost=0.00042)) == 0.00042
    assert extract_response_cost({"cost": 0.5}) == 0.5
    assert extract_response_cost({"cost": 2}) == 2.0


def test_extract_missing_none_and_non_number_give_zero():
    assert extract_response_cost(None) == 0.0
    assert extract_response_cost(SimpleNamespace(prompt_tokens=1)) == 0.0
    assert extract_response_cost({}) == 0.0
    assert extract_response_cost(SimpleNamespace(cost=None)) == 0.0
    assert extract_response_cost(SimpleNamespace(cost="0.5")) == 0.0
    assert extract_response_cost(SimpleNamespace(cost=True)) == 0.0
    assert extract_response_cost(SimpleNamespace(cost={"total": 1})) == 0.0
    assert extract_response_cost(SimpleNamespace(cost=-1.0)) == 0.0
    assert extract_response_cost(42) == 0.0


def test_estimate_without_prices_is_zero():
    assert estimate_cost_usd(1_000_000, 1_000_000, 500_000) == 0.0


def test_estimate_needs_both_prices(monkeypatch):
    monkeypatch.setenv("BPSA_PRICE_INPUT_PER_M", "1.0")
    assert estimate_cost_usd(1_000_000, 1_000_000) == 0.0
    monkeypatch.setenv("BPSA_PRICE_OUTPUT_PER_M", "not-a-number")
    assert estimate_cost_usd(1_000_000, 1_000_000) == 0.0


def test_estimate_with_and_without_cached_price(monkeypatch):
    monkeypatch.setenv("BPSA_PRICE_INPUT_PER_M", "0.5")
    monkeypatch.setenv("BPSA_PRICE_OUTPUT_PER_M", "2.0")
    # 1M input at 0.5 (cached share priced as input) + 0.5M output at 2.0
    assert estimate_cost_usd(1_000_000, 500_000, 400_000) == pytest.approx(0.5 + 1.0)
    monkeypatch.setenv("BPSA_PRICE_CACHED_INPUT_PER_M", "0.1")
    # 0.6M uncached at 0.5 + 0.4M cached at 0.1 + 0.5M output at 2.0
    assert estimate_cost_usd(1_000_000, 500_000, 400_000) == pytest.approx(0.3 + 0.04 + 1.0)
    assert estimate_cost_usd(0, 0, 0) == 0.0


def test_request_cost_prefers_reported_cost_over_estimate(monkeypatch):
    monkeypatch.setenv("BPSA_PRICE_INPUT_PER_M", "1.0")
    monkeypatch.setenv("BPSA_PRICE_OUTPUT_PER_M", "1.0")
    reported = SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=0, cost=0.25)
    assert request_cost_usd(reported) == 0.25
    unreported = SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=0)
    assert request_cost_usd(unreported) == pytest.approx(1.0)
    assert request_cost_usd({"prompt_tokens": 500_000, "completion_tokens": 500_000}) == pytest.approx(1.0)


def test_request_cost_without_anything_is_zero():
    assert request_cost_usd(SimpleNamespace(prompt_tokens=10, completion_tokens=2)) == 0.0
    assert request_cost_usd(None) == 0.0


def _mock_response(cost):
    response = MagicMock(spec=["choices", "usage"])
    response.choices = [MagicMock()]
    response.choices[0].message.role = "assistant"
    response.choices[0].message.content = "hi"
    response.choices[0].message.tool_calls = None
    usage_fields = ["prompt_tokens", "completion_tokens"] + (["cost"] if cost is not None else [])
    response.usage = MagicMock(spec=usage_fields)
    response.usage.prompt_tokens = 1_000_000
    response.usage.completion_tokens = 500_000
    if cost is not None:
        response.usage.cost = cost
    return response


def _generate_with(response):
    with patch("openai.OpenAI") as MockOpenAI:
        client = MagicMock()
        MockOpenAI.return_value = client
        client.chat.completions.create.return_value = response
        model = OpenAIModel(model_id="test-model", api_key="x")
        return model.generate([ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])])


def test_generate_populates_reported_cost():
    assert _generate_with(_mock_response(0.0042)).token_usage.cost_usd == 0.0042


def test_generate_estimates_from_prices_when_no_cost(monkeypatch):
    monkeypatch.setenv("BPSA_PRICE_INPUT_PER_M", "0.5")
    monkeypatch.setenv("BPSA_PRICE_OUTPUT_PER_M", "2.0")
    message = _generate_with(_mock_response(None))
    assert message.token_usage.cost_usd == pytest.approx(0.5 + 1.0)
    assert message.token_usage.input_tokens == 1_000_000


def test_generate_without_cost_or_prices():
    message = _generate_with(_mock_response(None))
    assert message.token_usage.cost_usd == 0.0
    assert message.token_usage.input_tokens == 1_000_000


def _stream_chunk(content=None, usage=None):
    chunk = MagicMock(spec=["choices", "usage"])
    chunk.usage = usage
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


def test_generate_stream_populates_reported_cost():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, cost=0.00031)
    deltas = _generate_stream_with([_stream_chunk("a"), _stream_chunk("b"), _stream_chunk(usage=usage)])
    usage_deltas = [d for d in deltas if d.token_usage]
    assert len(usage_deltas) == 1
    assert usage_deltas[0].token_usage.cost_usd == 0.00031
    assert agglomerate_stream_deltas(deltas).token_usage.cost_usd == 0.00031


def test_generate_stream_estimates_from_prices(monkeypatch):
    monkeypatch.setenv("BPSA_PRICE_INPUT_PER_M", "1.0")
    monkeypatch.setenv("BPSA_PRICE_OUTPUT_PER_M", "3.0")
    usage = SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    deltas = _generate_stream_with([_stream_chunk("a"), _stream_chunk(usage=usage)])
    assert [d for d in deltas if d.token_usage][0].token_usage.cost_usd == pytest.approx(4.0)


def test_generate_stream_without_cost_or_prices():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2)
    deltas = _generate_stream_with([_stream_chunk("a"), _stream_chunk(usage=usage)])
    usage_delta = [d for d in deltas if d.token_usage][0]
    assert usage_delta.token_usage.cost_usd == 0.0
    assert usage_delta.token_usage.input_tokens == 10


def test_agglomerate_stream_deltas_sums_cost():
    deltas = [
        ChatMessageStreamDelta(content="a", token_usage=TokenUsage(input_tokens=1, output_tokens=1, cost_usd=0.01)),
        ChatMessageStreamDelta(content="b", token_usage=TokenUsage(input_tokens=1, output_tokens=1, cost_usd=0.02)),
        ChatMessageStreamDelta(content="c", token_usage=TokenUsage(input_tokens=1, output_tokens=1)),
        ChatMessageStreamDelta(content="d"),
    ]
    message = agglomerate_stream_deltas(deltas)
    assert message.content == "abcd"
    assert message.token_usage.cost_usd == pytest.approx(0.03)
    assert agglomerate_stream_deltas([ChatMessageStreamDelta(content="x")]).token_usage.cost_usd == 0.0


def _step(cost):
    return SimpleNamespace(
        timing=SimpleNamespace(duration=0.1),
        token_usage=TokenUsage(input_tokens=1, output_tokens=1, cost_usd=cost),
    )


def test_monitor_sums_cost_and_resets():
    monitor = Monitor(tracked_model=None, logger=SimpleNamespace(log=lambda *a, **k: None))
    assert monitor.total_cost_usd == 0.0
    monitor.update_metrics(_step(0.01))
    monitor.update_metrics(_step(0.0))
    monitor.update_metrics(_step(0.02))
    assert monitor.total_cost_usd == pytest.approx(0.03)
    assert monitor.get_total_token_counts().cost_usd == pytest.approx(0.03)
    monitor.reset()
    assert monitor.total_cost_usd == 0.0


def test_get_agent_cost_usd():
    from smolagents.bp_cli import get_agent_cost_usd

    monitor = Monitor(tracked_model=None, logger=None)
    assert get_agent_cost_usd(SimpleNamespace(monitor=monitor)) == 0.0
    monitor.total_cost_usd = 0.5
    assert get_agent_cost_usd(SimpleNamespace(monitor=monitor)) == 0.5
    assert get_agent_cost_usd(SimpleNamespace()) == 0.0


def test_format_cost_usd_decimals():
    from smolagents.bp_cli import format_cost_usd

    assert format_cost_usd(0.0123) == "$0.0123"
    assert format_cost_usd(1.5) == "$1.5000"
    assert format_cost_usd(0.001) == "$0.0010"
    assert format_cost_usd(0.00042) == "$0.000420"


class _FakeAgent:
    def __init__(self):
        self.monitor = Monitor(tracked_model=None, logger=None)
        self.memory = SimpleNamespace(steps=[], knowledge="")

    def get_context_char_size(self):
        return 0


def _capture_turn_summary(monkeypatch, agent, cost_usd):
    from smolagents import bp_cli

    console = Console(file=io.StringIO(), record=True, width=300, force_terminal=False)
    monkeypatch.setattr(bp_cli, "console", console)
    bp_cli.print_turn_summary(1, 2.0, 1000, 50, agent, cost_usd=cost_usd)
    return console.export_text()


def test_print_turn_summary_shows_cost(monkeypatch):
    text = _capture_turn_summary(monkeypatch, _FakeAgent(), 0.0123)
    assert "| $0.0123 |" in text
    assert "In: 1,000" in text
    assert "| $0.000420 |" in _capture_turn_summary(monkeypatch, None, 0.00042)


def test_print_turn_summary_without_cost(monkeypatch):
    assert "$" not in _capture_turn_summary(monkeypatch, _FakeAgent(), 0.0)
    assert "$" not in _capture_turn_summary(monkeypatch, None, 0.0)


def _capture_stats(monkeypatch, stats):
    from smolagents import bp_cli

    console = Console(file=io.StringIO(), record=True, width=300, force_terminal=False)
    monkeypatch.setattr(bp_cli, "console", console)
    bp_cli.print_stats(stats, _FakeAgent())
    return console.export_text()


def test_print_stats_shows_total_and_average_cost(monkeypatch):
    stats = {"turns": 4, "total_time": 1.0, "total_input_tokens": 10, "total_output_tokens": 5, "total_cost_usd": 0.2}
    text = _capture_stats(monkeypatch, stats)
    assert "Total cost" in text and "$0.2000" in text
    assert "Avg cost/turn" in text and "$0.0500" in text


def test_print_stats_without_cost(monkeypatch):
    stats = {"turns": 1, "total_time": 1.0, "total_input_tokens": 1, "total_output_tokens": 1}
    text = _capture_stats(monkeypatch, stats)
    assert "Total cost" in text and "unknown" in text
    assert "Avg cost/turn" not in text


def test_token_usage_session_round_trip_tolerates_old_files():
    from smolagents.bp_session import _deserialize_token_usage, _serialize_token_usage

    usage = TokenUsage(input_tokens=10, output_tokens=2, cost_usd=0.0042)
    assert _deserialize_token_usage(_serialize_token_usage(usage)) == usage
    old = _deserialize_token_usage({"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 1})
    assert old.cost_usd == 0.0


def _session_agent():
    from smolagents.memory import AgentMemory

    agent = _FakeAgent()
    agent.memory = AgentMemory("system prompt")
    agent._next_actionstep_id = 1
    agent._last_plan_step = 0
    return agent


def test_session_dict_round_trip_keeps_monitor_total():
    from smolagents.bp_session import load_session_from_dict, save_session_to_dict

    agent = _session_agent()
    agent.monitor.total_cost_usd = 0.75
    stats = {"turns": 1, "total_time": 1.0, "total_input_tokens": 1, "total_output_tokens": 1, "total_cost_usd": 0.75}
    payload = save_session_to_dict(agent, stats)
    assert payload["monitor_state"]["total_cost_usd"] == 0.75

    restored = _session_agent()
    assert load_session_from_dict(payload, restored)["total_cost_usd"] == 0.75
    assert restored.monitor.total_cost_usd == 0.75

    old_payload = dict(payload)
    old_payload["monitor_state"] = {"total_input_token_count": 1, "total_output_token_count": 1}
    load_session_from_dict(old_payload, restored)
    assert restored.monitor.total_cost_usd == 0.0


def _build_model_extra_body(monkeypatch, endpoint):
    import smolagents
    from smolagents import bp_cli

    monkeypatch.setenv("BPSA_SERVER_MODEL", "OpenAIServerModel")
    monkeypatch.setenv("BPSA_MODEL_ID", "test-model")
    monkeypatch.setenv("BPSA_KEY_VALUE", "x")
    monkeypatch.setenv("BPSA_API_ENDPOINT", endpoint)
    monkeypatch.delenv("BPSA_PROVIDER_ORDER", raising=False)
    monkeypatch.delenv("BPSA_HAS_SESSION_ID", raising=False)
    model_class = MagicMock()
    monkeypatch.setattr(smolagents, "OpenAIServerModel", model_class)
    bp_cli.build_model()
    model_class.assert_called_once()
    return model_class.call_args.kwargs.get("extra_body", {})


def test_build_model_sends_usage_include_to_openrouter(monkeypatch):
    extra_body = _build_model_extra_body(monkeypatch, "https://openrouter.ai/api/v1")
    assert extra_body["usage"] == {"include": True}


def test_build_model_omits_usage_include_elsewhere(monkeypatch):
    assert "usage" not in _build_model_extra_body(monkeypatch, "https://api.openai.com/v1")
