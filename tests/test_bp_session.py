#!/usr/bin/env python3
"""
Unit tests for session save/load in bp_session.py.
"""

import json
import os
import sys

import pytest


# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smolagents.bp_compression import CompressedHistoryStep
from smolagents.bp_session import (
    ReconstructedError,
    deserialize_step,
    load_session,
    save_session,
    serialize_step,
)
from smolagents.memory import ActionStep, AgentMemory, MemoryStep, PlanningStep, TaskStep, ToolCall
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import Timing, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeMonitor:
    """Minimal monitor stand-in for testing."""

    def __init__(self):
        self.total_input_token_count = 0
        self.total_output_token_count = 0


class FakeAgent:
    """Minimal agent stand-in for testing save/load."""

    def __init__(self, system_prompt="You are a helpful assistant."):
        self.memory = AgentMemory(system_prompt)
        self._next_actionstep_id = 1
        self._last_plan_step = 0
        self.monitor = FakeMonitor()


def _make_pil_image(width=4, height=4, color=(255, 0, 0)):
    """Create a small PIL image for testing."""
    import PIL.Image

    img = PIL.Image.new("RGB", (width, height), color)
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRoundTripEmptySession:
    def test_empty_session(self, tmp_path):
        """Round-trip an empty session (0 steps)."""
        filepath = str(tmp_path / "empty.json")
        agent = FakeAgent()
        stats = {"turns": 0, "total_time": 0.0, "total_input_tokens": 0, "total_output_tokens": 0}

        save_session(filepath, agent, stats)
        agent2 = FakeAgent("overwritten")
        restored_stats = load_session(filepath, agent2)

        assert agent2.memory.steps == []
        assert agent2.memory.system_prompt.system_prompt == "You are a helpful assistant."
        assert restored_stats == stats


class TestRoundTripActionStep:
    def test_full_action_step(self, tmp_path):
        """Round-trip an ActionStep with all fields populated."""
        filepath = str(tmp_path / "action.json")
        agent = FakeAgent()

        img = _make_pil_image(8, 8, (0, 255, 0))
        step = ActionStep(
            step_number=3,
            timing=Timing(start_time=100.0, end_time=105.5),
            model_input_messages=[ChatMessage(role=MessageRole.USER, content="hello")],
            tool_calls=[ToolCall(name="search", arguments={"query": "test"}, id="call_1")],
            error=ReconstructedError("AgentExecutionError", "something went wrong"),
            model_output_message=ChatMessage(
                role=MessageRole.ASSISTANT, content=[{"type": "text", "text": "result"}]
            ),
            model_output="I will search",
            code_action='search(query="test")',
            observations="Found 5 results",
            observations_images=[img],
            action_output="done",
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
            is_final_answer=True,
            actionstep_id=7,
            _archived_observations="old observations",
            _archived_model_output="old model output",
        )
        agent.memory.steps.append(step)
        agent._next_actionstep_id = 8
        stats = {"turns": 1, "total_time": 5.5, "total_input_tokens": 100, "total_output_tokens": 50}

        save_session(filepath, agent, stats)

        agent2 = FakeAgent()
        load_session(filepath, agent2)

        assert len(agent2.memory.steps) == 1
        loaded = agent2.memory.steps[0]
        assert isinstance(loaded, ActionStep)
        assert loaded.step_number == 3
        assert loaded.timing.start_time == 100.0
        assert loaded.timing.end_time == 105.5
        assert abs(loaded.timing.duration - 5.5) < 0.001
        assert loaded.model_input_messages is None  # skipped on save
        assert len(loaded.tool_calls) == 1
        assert loaded.tool_calls[0].name == "search"
        assert loaded.tool_calls[0].arguments == {"query": "test"}
        assert loaded.tool_calls[0].id == "call_1"
        assert str(loaded.error) == "something went wrong"
        assert loaded.error.dict() == {"type": "AgentExecutionError", "message": "something went wrong"}
        assert loaded.model_output == "I will search"
        assert loaded.code_action == 'search(query="test")'
        assert loaded.observations == "Found 5 results"
        assert loaded.observations_images is not None
        assert len(loaded.observations_images) == 1
        assert loaded.action_output == "done"
        assert loaded.token_usage.input_tokens == 100
        assert loaded.token_usage.output_tokens == 50
        assert loaded.token_usage.total_tokens == 150
        assert loaded.is_final_answer is True
        assert loaded.actionstep_id == 7
        assert loaded._archived_observations == "old observations"
        assert loaded._archived_model_output == "old model output"


class TestRoundTripTaskStep:
    def test_task_step_without_images(self, tmp_path):
        filepath = str(tmp_path / "task.json")
        agent = FakeAgent()
        agent.memory.steps.append(TaskStep(task="Do something"))
        stats = {"turns": 0, "total_time": 0.0, "total_input_tokens": 0, "total_output_tokens": 0}

        save_session(filepath, agent, stats)
        agent2 = FakeAgent()
        load_session(filepath, agent2)

        assert len(agent2.memory.steps) == 1
        loaded = agent2.memory.steps[0]
        assert isinstance(loaded, TaskStep)
        assert loaded.task == "Do something"
        assert loaded.task_images is None

    def test_task_step_with_images(self, tmp_path):
        filepath = str(tmp_path / "task_img.json")
        agent = FakeAgent()
        img = _make_pil_image(2, 2, (0, 0, 255))
        agent.memory.steps.append(TaskStep(task="Do something", task_images=[img]))
        stats = {"turns": 0, "total_time": 0.0, "total_input_tokens": 0, "total_output_tokens": 0}

        save_session(filepath, agent, stats)
        agent2 = FakeAgent()
        load_session(filepath, agent2)

        loaded = agent2.memory.steps[0]
        assert loaded.task_images is not None
        assert len(loaded.task_images) == 1
        assert loaded.task_images[0].size == (2, 2)


class TestRoundTripPlanningStep:
    def test_planning_step(self, tmp_path):
        filepath = str(tmp_path / "plan.json")
        agent = FakeAgent()
        step = PlanningStep(
            model_input_messages=[ChatMessage(role=MessageRole.USER, content="plan this")],
            model_output_message=ChatMessage(
                role=MessageRole.ASSISTANT, content=[{"type": "text", "text": "my plan"}]
            ),
            plan="Step 1: Do X\nStep 2: Do Y",
            timing=Timing(start_time=200.0, end_time=202.0),
            token_usage=TokenUsage(input_tokens=50, output_tokens=30),
        )
        agent.memory.steps.append(step)
        stats = {"turns": 0, "total_time": 0.0, "total_input_tokens": 0, "total_output_tokens": 0}

        save_session(filepath, agent, stats)
        agent2 = FakeAgent()
        load_session(filepath, agent2)

        loaded = agent2.memory.steps[0]
        assert isinstance(loaded, PlanningStep)
        assert loaded.plan == "Step 1: Do X\nStep 2: Do Y"
        assert loaded.model_input_messages == []  # emptied on load
        assert loaded.model_output_message is not None
        assert loaded.timing.start_time == 200.0
        assert loaded.token_usage.input_tokens == 50


class TestRoundTripCompressedHistoryStep:
    def test_compressed_step(self, tmp_path):
        filepath = str(tmp_path / "compressed.json")
        agent = FakeAgent()
        step = CompressedHistoryStep(
            summary="Agent searched for X and found Y.",
            compressed_step_numbers=[1, 2, 3],
            original_step_count=3,
            timing=Timing(start_time=300.0, end_time=301.0),
            compression_token_usage=TokenUsage(input_tokens=200, output_tokens=80),
        )
        agent.memory.steps.append(step)
        stats = {"turns": 0, "total_time": 0.0, "total_input_tokens": 0, "total_output_tokens": 0}

        save_session(filepath, agent, stats)
        agent2 = FakeAgent()
        load_session(filepath, agent2)

        loaded = agent2.memory.steps[0]
        assert isinstance(loaded, CompressedHistoryStep)
        assert loaded.summary == "Agent searched for X and found Y."
        assert loaded.compressed_step_numbers == [1, 2, 3]
        assert loaded.original_step_count == 3
        assert loaded.timing.start_time == 300.0
        assert loaded.compression_token_usage.input_tokens == 200
        assert loaded.compression_token_usage.output_tokens == 80


class TestImageFidelity:
    def test_pixel_data_preserved(self, tmp_path):
        """Create a small PIL image, save/load, compare pixel data."""
        import PIL.Image

        filepath = str(tmp_path / "img_fidelity.json")
        agent = FakeAgent()
        img = PIL.Image.new("RGB", (3, 3))
        pixels = [(i * 20, i * 30, i * 10) for i in range(9)]
        img.putdata(pixels)

        agent.memory.steps.append(
            ActionStep(
                step_number=1,
                timing=Timing(start_time=0.0, end_time=1.0),
                observations_images=[img],
            )
        )
        stats = {"turns": 0, "total_time": 0.0, "total_input_tokens": 0, "total_output_tokens": 0}

        save_session(filepath, agent, stats)
        agent2 = FakeAgent()
        load_session(filepath, agent2)

        loaded_img = agent2.memory.steps[0].observations_images[0]
        assert loaded_img.size == (3, 3)
        assert loaded_img.mode == "RGB"
        loaded_pixels = list(loaded_img.getdata())
        assert loaded_pixels == pixels


class TestErrorReconstruction:
    def test_error_str_and_dict(self):
        """ReconstructedError provides str() and .dict() matching AgentError interface."""
        err = ReconstructedError("AgentExecutionError", "oops")
        assert str(err) == "oops"
        assert err.dict() == {"type": "AgentExecutionError", "message": "oops"}

    def test_action_step_error_roundtrip(self, tmp_path):
        """ActionStep with error survives save/load."""
        filepath = str(tmp_path / "error_rt.json")
        agent = FakeAgent()
        agent.memory.steps.append(
            ActionStep(
                step_number=1,
                timing=Timing(start_time=0.0, end_time=1.0),
                error=ReconstructedError("AgentGenerationError", "bad generation"),
            )
        )
        stats = {"turns": 0, "total_time": 0.0, "total_input_tokens": 0, "total_output_tokens": 0}

        save_session(filepath, agent, stats)
        agent2 = FakeAgent()
        load_session(filepath, agent2)

        loaded_err = agent2.memory.steps[0].error
        assert str(loaded_err) == "bad generation"
        assert loaded_err.dict()["type"] == "AgentGenerationError"


class TestAgentStateRestoration:
    def test_counters_restored(self, tmp_path):
        """Verify _next_actionstep_id, _last_plan_step, monitor counts."""
        filepath = str(tmp_path / "state.json")
        agent = FakeAgent()
        agent._next_actionstep_id = 42
        agent._last_plan_step = 10
        agent.monitor.total_input_token_count = 50000
        agent.monitor.total_output_token_count = 15000
        stats = {"turns": 5, "total_time": 123.4, "total_input_tokens": 50000, "total_output_tokens": 15000}

        save_session(filepath, agent, stats)
        agent2 = FakeAgent()
        load_session(filepath, agent2)

        assert agent2._next_actionstep_id == 42
        assert agent2._last_plan_step == 10
        assert agent2.monitor.total_input_token_count == 50000
        assert agent2.monitor.total_output_token_count == 15000


class TestSessionStatsRestoration:
    def test_stats_roundtrip(self, tmp_path):
        filepath = str(tmp_path / "stats.json")
        agent = FakeAgent()
        stats = {"turns": 7, "total_time": 999.9, "total_input_tokens": 12345, "total_output_tokens": 6789}

        save_session(filepath, agent, stats)
        agent2 = FakeAgent()
        restored = load_session(filepath, agent2)

        assert restored["turns"] == 7
        assert restored["total_time"] == 999.9
        assert restored["total_input_tokens"] == 12345
        assert restored["total_output_tokens"] == 6789


class TestVersionCheck:
    def test_bad_version_raises(self, tmp_path):
        filepath = str(tmp_path / "bad_version.json")
        payload = {
            "version": 99,
            "saved_at": "2025-01-01T00:00:00+00:00",
            "agent_state": {"system_prompt": "", "next_actionstep_id": 1, "last_plan_step": 0},
            "session_stats": {"turns": 0, "total_time": 0.0, "total_input_tokens": 0, "total_output_tokens": 0},
            "monitor_state": {"total_input_token_count": 0, "total_output_token_count": 0},
            "steps": [],
        }
        with open(filepath, "w") as f:
            json.dump(payload, f)

        agent = FakeAgent()
        with pytest.raises(ValueError, match="Unsupported session file version: 99"):
            load_session(filepath, agent)


class TestMixedStepTypes:
    def test_mixed_session(self, tmp_path):
        """Session with TaskStep + ActionStep + CompressedHistoryStep + PlanningStep."""
        filepath = str(tmp_path / "mixed.json")
        agent = FakeAgent()
        agent._next_actionstep_id = 5

        agent.memory.steps.append(TaskStep(task="Solve the puzzle"))
        agent.memory.steps.append(
            CompressedHistoryStep(
                summary="Earlier steps compressed.",
                compressed_step_numbers=[1, 2],
                original_step_count=2,
                timing=Timing(start_time=10.0, end_time=11.0),
            )
        )
        agent.memory.steps.append(
            ActionStep(
                step_number=3,
                timing=Timing(start_time=12.0, end_time=15.0),
                model_output="thinking...",
                observations="result: 42",
                actionstep_id=3,
                token_usage=TokenUsage(input_tokens=80, output_tokens=40),
            )
        )
        agent.memory.steps.append(
            PlanningStep(
                model_input_messages=[],
                model_output_message=ChatMessage(
                    role=MessageRole.ASSISTANT, content=[{"type": "text", "text": "plan"}]
                ),
                plan="1. Do A\n2. Do B",
                timing=Timing(start_time=16.0, end_time=17.0),
            )
        )
        agent.memory.steps.append(
            ActionStep(
                step_number=4,
                timing=Timing(start_time=18.0, end_time=20.0),
                model_output="executing plan",
                action_output="success",
                is_final_answer=True,
                actionstep_id=4,
            )
        )

        stats = {"turns": 3, "total_time": 20.0, "total_input_tokens": 300, "total_output_tokens": 100}
        save_session(filepath, agent, stats)

        agent2 = FakeAgent()
        restored_stats = load_session(filepath, agent2)

        assert len(agent2.memory.steps) == 5
        assert isinstance(agent2.memory.steps[0], TaskStep)
        assert isinstance(agent2.memory.steps[1], CompressedHistoryStep)
        assert isinstance(agent2.memory.steps[2], ActionStep)
        assert isinstance(agent2.memory.steps[3], PlanningStep)
        assert isinstance(agent2.memory.steps[4], ActionStep)
        assert agent2.memory.steps[4].is_final_answer is True
        assert agent2._next_actionstep_id == 5
        assert restored_stats["turns"] == 3


class TestSerializeUnknownStepType:
    def test_unknown_type_raises(self):
        """serialize_step should raise for unknown step types."""

        class WeirdStep(MemoryStep):
            pass

        with pytest.raises(ValueError, match="Unknown step type"):
            serialize_step(WeirdStep())

    def test_deserialize_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown step type"):
            deserialize_step({"_step_type": "FutureStep"})


class TestStringModelOutputMessage:
    def test_string_model_output_message(self, tmp_path):
        """ActionStep where model_output_message is a raw string (not ChatMessage)."""
        filepath = str(tmp_path / "str_msg.json")
        agent = FakeAgent()
        step = ActionStep(
            step_number=1,
            timing=Timing(start_time=0.0, end_time=1.0),
            model_output_message="I will help you",  # type: ignore[arg-type]
            model_output="I will help you",
        )
        agent.memory.steps.append(step)
        stats = {"turns": 1, "total_time": 1.0, "total_input_tokens": 10, "total_output_tokens": 5}

        save_session(filepath, agent, stats)
        agent2 = FakeAgent()
        load_session(filepath, agent2)

        loaded = agent2.memory.steps[0]
        assert isinstance(loaded, ActionStep)
        assert loaded.model_output_message is not None
        assert loaded.model_output_message.content == "I will help you"


class TestActionStepNoneFields:
    def test_minimal_action_step(self, tmp_path):
        """ActionStep with almost all fields None/default."""
        filepath = str(tmp_path / "minimal_action.json")
        agent = FakeAgent()
        agent.memory.steps.append(
            ActionStep(
                step_number=1,
                timing=Timing(start_time=0.0),
            )
        )
        stats = {"turns": 0, "total_time": 0.0, "total_input_tokens": 0, "total_output_tokens": 0}

        save_session(filepath, agent, stats)
        agent2 = FakeAgent()
        load_session(filepath, agent2)

        loaded = agent2.memory.steps[0]
        assert isinstance(loaded, ActionStep)
        assert loaded.step_number == 1
        assert loaded.timing.end_time is None
        assert loaded.tool_calls is None
        assert loaded.error is None
        assert loaded.model_output is None
        assert loaded.observations is None
        assert loaded.observations_images is None
        assert loaded.token_usage is None
        assert loaded.is_final_answer is False
