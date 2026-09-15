#!/usr/bin/env python3
"""
Unit tests for steering the agent between steps (ActionStep.user_message, steering_source, _inbox.md).
No terminal and no network: the model is a fake and the listener is only tested through its guard.
"""

import os
import queue
import sys
from unittest.mock import patch

import pytest


# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smolagents import CodeAgent
from smolagents.bp_ad_infinitum import read_steering_inbox, steering_inbox_path
from smolagents.bp_cli import drain_queue, is_clean_stop, steering_available
from smolagents.bp_session import deserialize_step, serialize_step
from smolagents.memory import ActionStep, TaskStep
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import Timing
from smolagents.utils import AgentError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CountingModel(Model):
    """Runs `print(n)` for the first `steps_before_answer` calls, then final_answer; records every messages list."""

    def __init__(self, steps_before_answer: int = 2):
        super().__init__()
        self.steps_before_answer = steps_before_answer
        self.seen: list[list[ChatMessage]] = []

    def generate(self, messages, stop_sequences=None, **kwargs):
        self.seen.append(messages)
        call = len(self.seen)
        if call <= self.steps_before_answer:
            content = f"<thoughts>step {call}</thoughts>\n<runcode>\nprint({call})\n</runcode>"
        else:
            content = "<runcode>\nfinal_answer('done')\n</runcode>"
        return ChatMessage(role=MessageRole.ASSISTANT, content=content)


def user_texts(messages) -> list[str]:
    return [m.content[0]["text"] for m in messages if m.role == MessageRole.USER]


def make_agent(model, **kwargs) -> CodeAgent:
    return CodeAgent(tools=[], model=model, max_steps=6, verbosity_level=0, **kwargs)


def make_step(user_message=None, observations="Execution logs:\n1\n") -> ActionStep:
    return ActionStep(
        step_number=1,
        timing=Timing(start_time=1.0, end_time=2.0),
        model_output="<runcode>\nprint(1)\n</runcode>",
        observations=observations,
        user_message=user_message,
        actionstep_id=1,
    )


# ---------------------------------------------------------------------------
# ActionStep.user_message
# ---------------------------------------------------------------------------


class TestActionStepUserMessage:
    def test_dict_carries_the_field(self):
        assert make_step("also run the tests").dict()["user_message"] == "also run the tests"
        assert make_step().dict()["user_message"] is None

    def test_to_messages_emits_user_message_after_observation(self):
        messages = make_step("also run the tests").to_messages()
        assert messages[-1].role == MessageRole.USER
        assert messages[-1].content == [{"type": "text", "text": "also run the tests"}]
        assert messages[-2].role == MessageRole.TOOL_RESPONSE
        assert "Execution logs" in messages[-2].content[0]["text"]

    def test_to_messages_without_user_message(self):
        messages = make_step().to_messages()
        assert all(m.role != MessageRole.USER for m in messages)

    def test_to_messages_summary_mode_keeps_user_message(self):
        assert make_step("keep me").to_messages(summary_mode=True)[-1].content[0]["text"] == "keep me"


# ---------------------------------------------------------------------------
# _run_stream drains steering_source at the step boundary
# ---------------------------------------------------------------------------


class TestStepBoundaryDelivery:
    def test_prefilled_queue_lands_on_last_action_step_before_next_model_call(self):
        pending = queue.Queue()
        pending.put("also say pomegranate")
        model = CountingModel(steps_before_answer=2)
        agent = make_agent(model)
        agent.steering_source = lambda: drain_queue(pending)

        assert agent.run("count") == "done"

        action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
        assert action_steps[0].user_message == "also say pomegranate"
        assert all(s.user_message is None for s in action_steps[1:])
        # First call: no ActionStep existed, so the text was held; second call sees it as a user turn.
        assert "also say pomegranate" not in user_texts(model.seen[0])
        assert user_texts(model.seen[1])[-1] == "also say pomegranate"
        assert user_texts(model.seen[2])[-1] == "also say pomegranate"
        assert pending.empty()

    def test_two_queued_messages_join_with_newlines(self):
        pending = queue.Queue()
        pending.put("first note")
        pending.put("second note")
        model = CountingModel(steps_before_answer=1)
        agent = make_agent(model)
        agent.steering_source = lambda: drain_queue(pending)

        agent.run("count")

        first_step = [s for s in agent.memory.steps if isinstance(s, ActionStep)][0]
        assert first_step.user_message == "first note\nsecond note"
        assert user_texts(model.seen[1])[-1] == "first note\nsecond note"

    def test_steering_source_is_consulted_at_every_step(self):
        calls = []

        def source():
            calls.append(len(calls) + 1)
            return [f"note {len(calls)}"] if len(calls) == 3 else []

        model = CountingModel(steps_before_answer=3)
        agent = make_agent(model)
        agent.steering_source = source

        agent.run("count")

        # Four step boundaries: three print steps and the final_answer step.
        assert calls == [1, 2, 3, 4]
        action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
        assert [s.user_message for s in action_steps] == [None, "note 3", None, None]

    def test_blank_messages_are_ignored_and_none_without_source(self):
        model = CountingModel(steps_before_answer=1)
        agent = make_agent(model)
        agent.steering_source = lambda: ["", "   "]
        agent.run("count")
        assert all(s.user_message is None for s in agent.memory.steps if isinstance(s, ActionStep))

        agent2 = make_agent(CountingModel(steps_before_answer=1))
        assert agent2.steering_source is None
        assert agent2.run("count") == "done"

    def test_message_held_before_first_action_step_is_dropped_by_next_run(self):
        agent = make_agent(CountingModel(steps_before_answer=0))
        agent.steering_source = lambda: ["too late"]
        agent.run("count")  # one step: the text is held and never attached
        assert agent._held_steering_messages == ["too late"]
        agent.steering_source = None
        agent.run("count", reset=False)
        assert agent._held_steering_messages == []


# ---------------------------------------------------------------------------
# Clean stop at the step boundary
# ---------------------------------------------------------------------------


class TestCleanStop:
    def test_interrupt_keeps_the_finished_step_in_memory(self):
        model = CountingModel(steps_before_answer=4)
        agent = make_agent(model)

        def interrupt_after_first_step(memory_step, agent):
            if isinstance(memory_step, ActionStep):
                agent.interrupt()

        agent.step_callbacks.register(ActionStep, interrupt_after_first_step)
        with pytest.raises(AgentError) as excinfo:
            agent.run("count")

        assert "Agent interrupted" in str(excinfo.value)
        action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
        assert len(action_steps) == 1
        assert action_steps[0].observations == "Execution logs:\n1\n"
        assert len(model.seen) == 1
        assert agent.step_number == 2
        assert is_clean_stop(agent, excinfo.value)

    def test_queued_text_is_delivered_before_the_interrupt_check(self):
        pending = queue.Queue()
        agent = make_agent(CountingModel(steps_before_answer=4))
        agent.steering_source = lambda: drain_queue(pending)

        def queue_note_then_interrupt(step, agent):
            pending.put("stop note")
            agent.interrupt()

        agent.step_callbacks.register(ActionStep, queue_note_then_interrupt)
        with pytest.raises(AgentError):
            agent.run("count")
        action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
        assert action_steps[-1].user_message == "stop note"

    def test_is_clean_stop_rejects_other_errors(self):
        agent = make_agent(CountingModel())
        assert not is_clean_stop(agent, RuntimeError("boom"))
        agent.interrupt()
        assert not is_clean_stop(agent, RuntimeError("boom"))


# ---------------------------------------------------------------------------
# File inbox (ad-infinitum)
# ---------------------------------------------------------------------------


class TestFileInbox:
    def test_reads_and_truncates(self, tmp_path):
        inbox = tmp_path / "_inbox.md"
        inbox.write_text("  focus on the tests \n", encoding="utf-8")
        assert read_steering_inbox(str(inbox)) == ["focus on the tests"]
        assert inbox.read_text(encoding="utf-8") == ""
        assert read_steering_inbox(str(inbox)) == []

    def test_missing_or_empty_file(self, tmp_path):
        assert read_steering_inbox(str(tmp_path / "_inbox.md")) == []
        assert read_steering_inbox(None) == []
        (tmp_path / "_inbox.md").write_text("\n\n", encoding="utf-8")
        assert read_steering_inbox(str(tmp_path / "_inbox.md")) == []

    def test_inbox_path_only_for_folders(self, tmp_path):
        assert steering_inbox_path(str(tmp_path)) == str(tmp_path / "_inbox.md")
        task_file = tmp_path / "task.md"
        task_file.write_text("do it", encoding="utf-8")
        assert steering_inbox_path(str(task_file)) is None

    def test_inbox_feeds_the_agent(self, tmp_path):
        inbox = tmp_path / "_inbox.md"
        model = CountingModel(steps_before_answer=2)
        agent = make_agent(model)
        agent.steering_source = lambda: read_steering_inbox(str(inbox))

        def write_inbox_after_first_step(memory_step, agent):
            if isinstance(memory_step, ActionStep) and memory_step.step_number == 1:
                inbox.write_text("also count backwards", encoding="utf-8")

        agent.step_callbacks.register(ActionStep, write_inbox_after_first_step)
        agent.run("count")

        action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
        assert action_steps[0].user_message == "also count backwards"
        assert user_texts(model.seen[1])[-1] == "also count backwards"
        assert inbox.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# Session round trip
# ---------------------------------------------------------------------------


class TestSessionRoundTrip:
    def test_round_trip_with_user_message(self):
        data = serialize_step(make_step("also run the tests"))
        assert data["user_message"] == "also run the tests"
        restored = deserialize_step(data)
        assert isinstance(restored, ActionStep)
        assert restored.user_message == "also run the tests"
        assert restored.observations == "Execution logs:\n1\n"

    def test_round_trip_without_user_message(self):
        assert deserialize_step(serialize_step(make_step())).user_message is None

    def test_old_session_file_without_the_field_loads_as_none(self):
        data = serialize_step(make_step("x"))
        del data["user_message"]
        assert deserialize_step(data).user_message is None

    def test_task_step_unchanged(self):
        assert deserialize_step(serialize_step(TaskStep(task="t"))).task == "t"


# ---------------------------------------------------------------------------
# Listener guard and queue helper
# ---------------------------------------------------------------------------


class TestListenerGuard:
    def test_no_listener_when_stdin_is_not_a_tty(self):
        with patch("smolagents.bp_cli.sys") as fake_sys:
            fake_sys.stdin.isatty.return_value = False
            fake_sys.stdout.isatty.return_value = True
            assert steering_available() is False

    def test_no_listener_when_stdout_is_not_a_tty(self):
        with patch("smolagents.bp_cli.sys") as fake_sys:
            fake_sys.stdin.isatty.return_value = True
            fake_sys.stdout.isatty.return_value = False
            assert steering_available() is False

    def test_listener_allowed_on_a_terminal(self):
        with patch("smolagents.bp_cli.sys") as fake_sys:
            fake_sys.stdin.isatty.return_value = True
            fake_sys.stdout.isatty.return_value = True
            assert steering_available() is True

    def test_start_steering_listener_returns_none_without_tty(self):
        from smolagents.bp_cli import start_steering_listener, stop_steering_listener

        agent = make_agent(CountingModel())
        with patch("smolagents.bp_cli.steering_available", return_value=False):
            assert start_steering_listener(agent) is None
        assert agent.steering_source is None
        assert stop_steering_listener(None) == []

    def test_drain_queue(self):
        pending = queue.Queue()
        assert drain_queue(pending) == []
        pending.put("a")
        pending.put("b")
        assert drain_queue(pending) == ["a", "b"]
        assert pending.empty()
