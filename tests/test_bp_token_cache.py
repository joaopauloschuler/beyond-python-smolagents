"""Tests for cached_input_tokens: extraction from provider usage objects and the Cache figure in the turn summary."""

import io
from types import SimpleNamespace

from rich.console import Console

from smolagents.models import ChatMessageStreamDelta, agglomerate_stream_deltas, extract_cached_input_tokens
from smolagents.monitoring import Monitor, TokenUsage


def test_token_usage_default_and_dict():
    usage = TokenUsage(input_tokens=10, output_tokens=5)
    assert usage.cached_input_tokens == 0
    assert usage.total_tokens == 15
    cached = TokenUsage(input_tokens=10, output_tokens=5, cached_input_tokens=8)
    assert cached.total_tokens == 15  # cached tokens are a subset of input tokens
    assert cached.dict()["cached_input_tokens"] == 8


def test_extract_openai_style_object():
    usage = SimpleNamespace(prompt_tokens=100, prompt_tokens_details=SimpleNamespace(cached_tokens=64))
    assert extract_cached_input_tokens(usage) == 64


def test_extract_openai_style_dict():
    assert extract_cached_input_tokens({"prompt_tokens_details": {"cached_tokens": 32}}) == 32


def test_extract_deepseek_style():
    usage = SimpleNamespace(prompt_tokens=100, prompt_cache_hit_tokens=70, prompt_cache_miss_tokens=30)
    assert extract_cached_input_tokens(usage) == 70
    assert extract_cached_input_tokens({"prompt_cache_hit_tokens": 12}) == 12


def test_extract_prefers_details_then_deepseek():
    usage = SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=0), prompt_cache_hit_tokens=9)
    assert extract_cached_input_tokens(usage) == 9


def test_extract_missing_none_and_wrong_type_give_zero():
    assert extract_cached_input_tokens(None) == 0
    assert extract_cached_input_tokens(SimpleNamespace(prompt_tokens=5)) == 0
    assert extract_cached_input_tokens({}) == 0
    assert extract_cached_input_tokens(SimpleNamespace(prompt_tokens_details=None)) == 0
    assert extract_cached_input_tokens(SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=None))) == 0
    assert extract_cached_input_tokens(SimpleNamespace(prompt_tokens_details="oops")) == 0
    assert extract_cached_input_tokens({"prompt_tokens_details": {"cached_tokens": "12"}}) == 0
    assert extract_cached_input_tokens({"prompt_cache_hit_tokens": True}) == 0
    assert extract_cached_input_tokens({"prompt_cache_hit_tokens": -3}) == 0
    assert extract_cached_input_tokens(42) == 0


def test_agglomerate_stream_deltas_sums_cached_tokens():
    deltas = [
        ChatMessageStreamDelta(
            content="a", token_usage=TokenUsage(input_tokens=10, output_tokens=1, cached_input_tokens=4)
        ),
        ChatMessageStreamDelta(content="b"),
        ChatMessageStreamDelta(
            content="c", token_usage=TokenUsage(input_tokens=10, output_tokens=1, cached_input_tokens=6)
        ),
    ]
    message = agglomerate_stream_deltas(deltas)
    assert message.content == "abc"
    assert message.token_usage.input_tokens == 20
    assert message.token_usage.cached_input_tokens == 10


def test_monitor_accumulates_cached_tokens():
    monitor = Monitor(tracked_model=None, logger=SimpleNamespace(log=lambda *a, **k: None))
    step = SimpleNamespace(
        timing=SimpleNamespace(duration=0.1),
        token_usage=TokenUsage(input_tokens=100, output_tokens=10, cached_input_tokens=80),
    )
    monitor.update_metrics(step)
    monitor.update_metrics(step)
    totals = monitor.get_total_token_counts()
    assert (totals.input_tokens, totals.output_tokens, totals.cached_input_tokens) == (200, 20, 160)
    monitor.reset()
    assert monitor.get_total_token_counts().cached_input_tokens == 0


def test_get_agent_token_usage_returns_cached():
    from smolagents.bp_cli import get_agent_token_usage

    monitor = Monitor(tracked_model=None, logger=None)
    monitor.total_input_token_count, monitor.total_output_token_count, monitor.total_cached_input_token_count = 7, 3, 5
    assert get_agent_token_usage(SimpleNamespace(monitor=monitor)) == (7, 3, 5)
    assert get_agent_token_usage(SimpleNamespace()) == (0, 0, 0)


def _capture_turn_summary(monkeypatch, **kwargs):
    from smolagents import bp_cli

    console = Console(file=io.StringIO(), record=True, width=300, force_terminal=False)
    monkeypatch.setattr(bp_cli, "console", console)
    bp_cli.print_turn_summary(1, 2.0, 1000, 50, None, **kwargs)
    return console.export_text()


def test_print_turn_summary_shows_cache_percentage(monkeypatch):
    text = _capture_turn_summary(monkeypatch, cached_tokens=870)
    assert "Cache: 87%" in text
    assert "In: 1,000" in text


def test_print_turn_summary_without_cached_tokens(monkeypatch):
    assert "Cache:" not in _capture_turn_summary(monkeypatch)
    assert "Cache:" not in _capture_turn_summary(monkeypatch, cached_tokens=0)


def test_token_usage_session_round_trip_tolerates_old_files():
    from smolagents.bp_session import _deserialize_token_usage, _serialize_token_usage

    usage = TokenUsage(input_tokens=10, output_tokens=2, cached_input_tokens=6)
    assert _deserialize_token_usage(_serialize_token_usage(usage)) == usage
    old = _deserialize_token_usage({"input_tokens": 10, "output_tokens": 2})
    assert old.cached_input_tokens == 0
