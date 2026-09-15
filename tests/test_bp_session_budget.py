"""Tests for the session budget: BPSA_MAX_SESSION_TOKENS / BPSA_MAX_SESSION_COST parsing, the pure
session_budget_state helper, once-per-limit warnings, the turn guard, the /show-stats and /show-config rows
and the ad-infinitum loop guard. No network, no model."""

import io
from types import SimpleNamespace

import pytest
from rich.console import Console

from smolagents.bp_compression import CompressionConfig
from smolagents.monitoring import TokenUsage


def _stats(input_tokens=0, output_tokens=0, cost=0.0, turns=1):
    return {
        "turns": turns,
        "total_time": 1.0,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_cached_input_tokens": 0,
        "total_cost_usd": cost,
    }


def _capture(monkeypatch):
    from smolagents import bp_cli

    console = Console(file=io.StringIO(), record=True, width=300, force_terminal=False)
    monkeypatch.setattr(bp_cli, "console", console)
    return console


def _fake_agent():
    return SimpleNamespace(
        model=SimpleNamespace(model_id="test/model", api_base="https://example.invalid/v1"),
        executor_type="exec",
        max_steps=200,
        planning_interval=None,
        compression_config=CompressionConfig(),
        compressor=None,
        memory=SimpleNamespace(steps=[], knowledge=""),
        monitor=_fake_monitor(),
    )


def _fake_monitor(input_tokens=0, output_tokens=0, cost=0.0):
    usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(get_total_token_counts=lambda: usage, total_cost_usd=cost)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("BPSA_MAX_SESSION_TOKENS", "BPSA_MAX_SESSION_COST"):
        monkeypatch.delenv(name, raising=False)


# --- limits from the environment -------------------------------------------------------------------------------


def test_no_limit_when_unset():
    from smolagents import bp_cli

    assert bp_cli.session_budget_limits() == {}
    assert bp_cli.session_budget_state(_stats(1_000_000, 1_000_000, 99.0)) == ("ok", "")


@pytest.mark.parametrize("value", ["0", "-5", "abc", "", "1.5", " "])
def test_invalid_token_limit_means_no_limit(monkeypatch, value):
    from smolagents import bp_cli

    monkeypatch.setenv("BPSA_MAX_SESSION_TOKENS", value)
    assert "tokens" not in bp_cli.session_budget_limits()
    assert bp_cli.session_budget_state(_stats(10**9, 10**9))[0] == "ok"


@pytest.mark.parametrize("value", ["0", "-1", "free", "", "nan"])
def test_invalid_cost_limit_means_no_limit(monkeypatch, value):
    from smolagents import bp_cli

    monkeypatch.setenv("BPSA_MAX_SESSION_COST", value)
    assert "cost" not in bp_cli.session_budget_limits()


def test_limits_parsed(monkeypatch):
    from smolagents import bp_cli

    monkeypatch.setenv("BPSA_MAX_SESSION_TOKENS", "20000")
    monkeypatch.setenv("BPSA_MAX_SESSION_COST", "1.50")
    assert bp_cli.session_budget_limits() == {"tokens": 20000, "cost": 1.5}


# --- session_budget_state ---------------------------------------------------------------------------------------


def test_state_below_warn(monkeypatch):
    from smolagents import bp_cli

    monkeypatch.setenv("BPSA_MAX_SESSION_TOKENS", "20000")
    state, message = bp_cli.session_budget_state(_stats(10_000, 5_999))
    assert state == "ok"
    assert message == "Session tokens: 15,999 of 20,000 (79%)"


def test_state_warn_at_80_percent(monkeypatch):
    from smolagents import bp_cli

    monkeypatch.setenv("BPSA_MAX_SESSION_TOKENS", "20000")
    state, message = bp_cli.session_budget_state(_stats(10_000, 6_000))
    assert state == "warn"
    assert message == "Session tokens: 16,000 of 20,000 (80%)"


def test_state_exceeded_at_100_percent(monkeypatch):
    from smolagents import bp_cli

    monkeypatch.setenv("BPSA_MAX_SESSION_TOKENS", "20000")
    assert bp_cli.session_budget_state(_stats(19_000, 1_000))[0] == "exceeded"
    assert bp_cli.session_budget_state(_stats(30_000, 0))[0] == "exceeded"


def test_state_cost_limit(monkeypatch):
    from smolagents import bp_cli

    monkeypatch.setenv("BPSA_MAX_SESSION_COST", "1.0")
    assert bp_cli.session_budget_state(_stats(cost=0.5)) == ("ok", "Session cost: $0.5000 of $1.0000 (50%)")
    assert bp_cli.session_budget_state(_stats(cost=0.8))[0] == "warn"
    assert bp_cli.session_budget_state(_stats(cost=1.0))[0] == "exceeded"
    # No cost known: the cost limit never triggers.
    assert bp_cli.session_budget_state(_stats(10**6, 10**6, cost=0.0))[0] == "ok"


def test_state_both_limits_one_exceeded(monkeypatch):
    from smolagents import bp_cli

    monkeypatch.setenv("BPSA_MAX_SESSION_TOKENS", "20000")
    monkeypatch.setenv("BPSA_MAX_SESSION_COST", "1.0")
    state, message = bp_cli.session_budget_state(_stats(1_000, 0, cost=1.25))
    assert state == "exceeded"
    assert message == "Session cost: $1.2500 of $1.0000 (125%)"
    state, message = bp_cli.session_budget_state(_stats(1_000, 0, cost=0.5))
    assert state == "ok"
    assert message == "Session tokens: 1,000 of 20,000 (5%); Session cost: $0.5000 of $1.0000 (50%)"


def test_state_tolerates_missing_fields(monkeypatch):
    from smolagents import bp_cli

    monkeypatch.setenv("BPSA_MAX_SESSION_TOKENS", "100")
    monkeypatch.setenv("BPSA_MAX_SESSION_COST", "1.0")
    assert bp_cli.session_budget_state({})[0] == "ok"
    assert bp_cli.session_budget_state({"total_input_tokens": None, "total_cost_usd": None})[0] == "ok"


# --- warnings and the turn guard -------------------------------------------------------------------------------


def test_warning_printed_once_per_limit(monkeypatch):
    from smolagents import bp_cli

    monkeypatch.setenv("BPSA_MAX_SESSION_TOKENS", "20000")
    console = _capture(monkeypatch)
    warned = set()
    assert bp_cli.warn_session_budget(_stats(16_000, 0), warned) == "warn"
    assert bp_cli.warn_session_budget(_stats(17_000, 0), warned) == "warn"
    text = console.export_text()
    assert text.count("Session budget warning") == 1
    assert "Session tokens: 16,000 of 20,000 (80%)" in text
    # Crossing the limit prints the red message once, and not again on the next call.
    assert bp_cli.warn_session_budget(_stats(20_000, 0), warned) == "exceeded"
    assert bp_cli.warn_session_budget(_stats(21_000, 0), warned) == "exceeded"
    assert console.export_text().count("Session budget exceeded") == 1


def test_warning_reset_prints_again(monkeypatch):
    from smolagents import bp_cli

    monkeypatch.setenv("BPSA_MAX_SESSION_TOKENS", "20000")
    console = _capture(monkeypatch)
    bp_cli.warn_session_budget(_stats(16_000, 0), set())
    bp_cli.warn_session_budget(_stats(16_000, 0), set())
    assert console.export_text().count("Session budget warning") == 2


def test_no_warning_without_limit_or_below(monkeypatch):
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    assert bp_cli.warn_session_budget(_stats(10**7, 10**7, cost=50.0), set()) == "ok"
    monkeypatch.setenv("BPSA_MAX_SESSION_TOKENS", "20000")
    assert bp_cli.warn_session_budget(_stats(10_000, 0), set()) == "ok"
    assert console.export_text().strip() == ""


def test_turn_guard_blocks_only_when_exceeded(monkeypatch):
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    assert bp_cli.session_budget_blocks_turn(_stats(10**7, 10**7)) is False
    monkeypatch.setenv("BPSA_MAX_SESSION_TOKENS", "20000")
    assert bp_cli.session_budget_blocks_turn(_stats(16_000, 0)) is False
    assert console.export_text().strip() == ""
    assert bp_cli.session_budget_blocks_turn(_stats(20_000, 0)) is True
    text = console.export_text()
    assert "Session budget exceeded: Session tokens: 20,000 of 20,000 (100%)" in text
    assert "/clear resets the budget" in text


# --- /show-stats and /show-config rows -------------------------------------------------------------------------


def test_print_stats_rows_only_with_limit(monkeypatch):
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    bp_cli.print_stats(_stats(1_000, 500, cost=0.2))
    assert "budget" not in console.export_text().lower()

    monkeypatch.setenv("BPSA_MAX_SESSION_TOKENS", "20000")
    monkeypatch.setenv("BPSA_MAX_SESSION_COST", "1.0")
    console = _capture(monkeypatch)
    bp_cli.print_stats(_stats(15_000, 2_000, cost=0.2))
    text = console.export_text()
    assert "Token budget" in text and "17,000 of 20,000 (85%)" in text
    assert "Cost budget" in text and "$0.2000 of $1.0000 (20%)" in text


def test_show_config_rows_only_with_limit(monkeypatch):
    from smolagents import bp_cli

    console = _capture(monkeypatch)
    bp_cli.cmd_show_config(_fake_agent(), session_stats=_stats(1_000, 0))
    assert "budget" not in console.export_text().lower()

    monkeypatch.setenv("BPSA_MAX_SESSION_COST", "2.5")
    console = _capture(monkeypatch)
    bp_cli.cmd_show_config(_fake_agent(), session_stats=_stats(1_000, 0, cost=1.0))
    text = console.export_text()
    row = [line for line in text.splitlines() if line.strip().startswith("Cost budget")]
    assert row and "$1.0000 of $2.5000 (40%)" in row[0] and "env" in row[0]
    assert "Token budget" not in text
    # Without session_stats (old callers) the row shows zero use.
    console = _capture(monkeypatch)
    bp_cli.cmd_show_config(_fake_agent())
    assert "$0.000000 of $2.5000 (0%)" in console.export_text()


# --- ad-infinitum -----------------------------------------------------------------------------------------------


def test_ad_infinitum_accumulates_and_stops(monkeypatch):
    from smolagents import bp_ad_infinitum, bp_cli

    monkeypatch.setenv("BPSA_MAX_SESSION_TOKENS", "1000")
    monkeypatch.setenv("BPSA_MAX_SESSION_COST", "1.0")
    console = _capture(monkeypatch)
    stats = {"total_input_tokens": 0, "total_output_tokens": 0, "total_cost_usd": 0.0}
    warned = set()

    agent = _fake_agent()
    agent.monitor = _fake_monitor(500, 100, 0.3)
    bp_ad_infinitum.add_agent_usage(stats, agent)
    assert stats == {"total_input_tokens": 500, "total_output_tokens": 100, "total_cost_usd": 0.3}
    assert bp_ad_infinitum.budget_stops_loop(stats, warned) is False
    assert console.export_text().strip() == ""

    bp_ad_infinitum.add_agent_usage(stats, agent)  # 1,200 tokens, $0.60
    assert bp_ad_infinitum.budget_stops_loop(stats, warned) is True
    text = console.export_text()
    assert "Session budget exceeded: Session tokens: 1,200 of 1,000 (120%). Stopping the loop." in text
    assert "slash commands" not in text
    assert "Session budget warning" not in text  # cost is at 60%
    assert bp_cli.session_budget_state(stats)[0] == "exceeded"
    assert bp_ad_infinitum.BUDGET_EXIT_CODE == 3


def test_ad_infinitum_usage_tolerates_missing_monitor():
    from smolagents import bp_ad_infinitum

    stats = {"total_input_tokens": 1, "total_output_tokens": 2, "total_cost_usd": 0.5}
    bp_ad_infinitum.add_agent_usage(stats, SimpleNamespace())
    assert stats == {"total_input_tokens": 1, "total_output_tokens": 2, "total_cost_usd": 0.5}
