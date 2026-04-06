# BPSA - Beyond Python SmolAgents
# https://github.com/joaopauloschuler/beyond-python-smolagents
#
# Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and others.
# Refer to the git commit history for individual authorship.
# Licensed under the Apache License, Version 2.0

#!/usr/bin/env python
# coding=utf-8

"""
Session save/load for Beyond Python SmolAgents.

Serializes and deserializes agent memory (steps, counters, stats) to/from JSON files.
"""

import base64
import io
import json
from datetime import datetime, timezone
from logging import getLogger

from smolagents.bp_compression import CompressedHistoryStep
from smolagents.memory import ActionStep, MemoryStep, PlanningStep, SystemPromptStep, TaskStep, ToolCall
from smolagents.models import ChatMessage, MessageRole, get_dict_from_nested_dataclasses
from smolagents.monitoring import Timing, TokenUsage
from smolagents.utils import make_json_serializable


logger = getLogger(__name__)

SESSION_VERSION = 1


# ---------------------------------------------------------------------------
# ReconstructedError
# ---------------------------------------------------------------------------


class ReconstructedError:
    """Lightweight stand-in for AgentError when loading sessions.

    AgentError.__init__ requires a logger arg and logs on construction — we
    don't want that on load.  ReconstructedError provides the same interface
    used by ActionStep.to_messages() (str(error)) and ActionStep.dict() (.dict()).
    """

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message

    def __str__(self):
        return self.message

    def dict(self):
        return {"type": self.error_type, "message": self.message}


# ---------------------------------------------------------------------------
# Serialization / deserialization helpers
# ---------------------------------------------------------------------------


def _serialize_image(image) -> dict:
    """PIL Image -> dict with base64-encoded PNG data."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "_type": "pil_image",
        "mode": image.mode,
        "size": list(image.size),
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
    }


def _deserialize_image(data: dict):
    """dict with base64-encoded PNG data -> PIL Image."""
    import PIL.Image

    buf = io.BytesIO(base64.b64decode(data["data"]))
    return PIL.Image.open(buf).copy()


def _serialize_timing(timing: Timing | None) -> dict | None:
    if timing is None:
        return None
    return {"start_time": timing.start_time, "end_time": timing.end_time}


def _deserialize_timing(data: dict | None) -> Timing | None:
    if data is None:
        return None
    return Timing(start_time=data["start_time"], end_time=data.get("end_time"))


def _serialize_token_usage(usage: TokenUsage | None) -> dict | None:
    if usage is None:
        return None
    return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}


def _deserialize_token_usage(data: dict | None) -> TokenUsage | None:
    if data is None:
        return None
    return TokenUsage(input_tokens=data["input_tokens"], output_tokens=data["output_tokens"])


def _serialize_error(error) -> dict | None:
    if error is None:
        return None
    return error.dict()


def _deserialize_error(data: dict | None) -> ReconstructedError | None:
    if data is None:
        return None
    return ReconstructedError(error_type=data.get("type", "AgentError"), message=data.get("message", ""))


def _serialize_tool_call(tc: ToolCall) -> dict:
    return tc.dict()


def _deserialize_tool_call(data: dict) -> ToolCall:
    func = data.get("function", {})
    return ToolCall(
        name=func.get("name", ""),
        arguments=func.get("arguments", {}),
        id=data.get("id", ""),
    )


def _serialize_chat_message(msg: ChatMessage | None) -> dict | str | None:
    if msg is None:
        return None
    if isinstance(msg, str):
        return msg
    d = make_json_serializable(get_dict_from_nested_dataclasses(msg))
    if isinstance(d, dict):
        d.pop("raw", None)
    return d


def _deserialize_chat_message(data) -> ChatMessage | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        return ChatMessage(role=MessageRole.ASSISTANT, content=str(data))
    data.pop("raw", None)
    token_usage_data = data.pop("token_usage", None)
    token_usage = _deserialize_token_usage(token_usage_data) if token_usage_data else None
    return ChatMessage.from_dict(data, token_usage=token_usage)


# ---------------------------------------------------------------------------
# Step serialization / deserialization
# ---------------------------------------------------------------------------


def serialize_step(step: MemoryStep) -> dict:
    """Serialize a single memory step to a JSON-safe dict."""
    if isinstance(step, TaskStep):
        return {
            "_step_type": "TaskStep",
            "task": step.task,
            "task_images": [_serialize_image(img) for img in step.task_images] if step.task_images else None,
        }

    if isinstance(step, ActionStep):
        return {
            "_step_type": "ActionStep",
            "step_number": step.step_number,
            "timing": _serialize_timing(step.timing),
            "tool_calls": [_serialize_tool_call(tc) for tc in step.tool_calls] if step.tool_calls else None,
            "error": _serialize_error(step.error),
            "model_output_message": _serialize_chat_message(step.model_output_message),
            "model_output": step.model_output,
            "code_action": step.code_action,
            "observations": step.observations,
            "observations_images": (
                [_serialize_image(img) for img in step.observations_images] if step.observations_images else None
            ),
            "action_output": make_json_serializable(step.action_output),
            "token_usage": _serialize_token_usage(step.token_usage),
            "is_final_answer": step.is_final_answer,
            "actionstep_id": step.actionstep_id,
            "_archived_observations": step._archived_observations,
            "_archived_model_output": step._archived_model_output,
        }

    if isinstance(step, PlanningStep):
        return {
            "_step_type": "PlanningStep",
            "plan": step.plan,
            "timing": _serialize_timing(step.timing),
            "token_usage": _serialize_token_usage(step.token_usage),
        }

    if isinstance(step, CompressedHistoryStep):
        return {
            "_step_type": "CompressedHistoryStep",
            "summary": step.summary,
            "compressed_step_numbers": step.compressed_step_numbers,
            "original_step_count": step.original_step_count,
            "timing": _serialize_timing(step.timing),
            "compression_token_usage": _serialize_token_usage(step.compression_token_usage),
        }

    raise ValueError(f"Unknown step type: {type(step).__name__}")


def deserialize_step(data: dict) -> MemoryStep:
    """Deserialize a dict back into a MemoryStep subclass."""
    step_type = data.get("_step_type")

    if step_type == "TaskStep":
        task_images = None
        if data.get("task_images"):
            task_images = [_deserialize_image(img) for img in data["task_images"]]
        return TaskStep(task=data["task"], task_images=task_images)

    if step_type == "ActionStep":
        tool_calls = None
        if data.get("tool_calls"):
            tool_calls = [_deserialize_tool_call(tc) for tc in data["tool_calls"]]
        observations_images = None
        if data.get("observations_images"):
            observations_images = [_deserialize_image(img) for img in data["observations_images"]]
        return ActionStep(
            step_number=data["step_number"],
            timing=_deserialize_timing(data.get("timing")),
            model_input_messages=None,
            tool_calls=tool_calls,
            error=_deserialize_error(data.get("error")),
            model_output_message=_deserialize_chat_message(data.get("model_output_message")),
            model_output=data.get("model_output"),
            code_action=data.get("code_action"),
            observations=data.get("observations"),
            observations_images=observations_images,
            action_output=data.get("action_output"),
            token_usage=_deserialize_token_usage(data.get("token_usage")),
            is_final_answer=data.get("is_final_answer", False),
            actionstep_id=data.get("actionstep_id"),
            _archived_observations=data.get("_archived_observations"),
            _archived_model_output=data.get("_archived_model_output"),
        )

    if step_type == "PlanningStep":
        plan = data.get("plan", "")
        return PlanningStep(
            model_input_messages=[],
            model_output_message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": plan}],
            ),
            plan=plan,
            timing=_deserialize_timing(data.get("timing")),
            token_usage=_deserialize_token_usage(data.get("token_usage")),
        )

    if step_type == "CompressedHistoryStep":
        return CompressedHistoryStep(
            summary=data.get("summary", ""),
            compressed_step_numbers=data.get("compressed_step_numbers", []),
            original_step_count=data.get("original_step_count", 0),
            timing=_deserialize_timing(data.get("timing")),
            compression_token_usage=_deserialize_token_usage(data.get("compression_token_usage")),
        )

    raise ValueError(f"Unknown step type: {step_type}")


# ---------------------------------------------------------------------------
# Top-level save / load
# ---------------------------------------------------------------------------


def save_session_to_dict(agent, session_stats: dict) -> dict:
    """Snapshot agent state to an in-memory dict.

    Same serialization as save_session() but returns a dict instead of writing to a file.

    Args:
        agent: The CodeAgent instance whose state to save.
        session_stats: Session statistics dict (turns, time, tokens).

    Returns:
        Serialized session payload as a dict.
    """
    steps = [serialize_step(step) for step in agent.memory.steps]

    return {
        "version": SESSION_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "agent_state": {
            "system_prompt": agent.memory.system_prompt.system_prompt,
            "next_actionstep_id": agent._next_actionstep_id,
            "last_plan_step": agent._last_plan_step,
            "knowledge": getattr(agent.memory, "knowledge", ""),
        },
        "session_stats": dict(session_stats),
        "monitor_state": {
            "total_input_token_count": agent.monitor.total_input_token_count,
            "total_output_token_count": agent.monitor.total_output_token_count,
        },
        "steps": steps,
    }


def load_session_from_dict(payload: dict, agent) -> dict:
    """Restore agent state from an in-memory dict.

    Same deserialization as load_session() but reads from a dict instead of a file.

    Args:
        payload: Serialized session payload (as returned by save_session_to_dict).
        agent: The CodeAgent instance to restore into.

    Returns:
        Restored session_stats dict.

    Raises:
        ValueError: If the payload version is unsupported.
    """
    version = payload.get("version")
    if version != SESSION_VERSION:
        raise ValueError(f"Unsupported session version: {version} (expected {SESSION_VERSION})")

    # Restore agent memory
    agent_state = payload.get("agent_state", {})
    agent.memory.system_prompt = SystemPromptStep(system_prompt=agent_state.get("system_prompt", ""))
    agent.memory.steps = [deserialize_step(s) for s in payload.get("steps", [])]
    if hasattr(agent.memory, "knowledge"):
        agent.memory.knowledge = agent_state.get("knowledge", "")

    # Restore agent counters
    agent._next_actionstep_id = agent_state.get("next_actionstep_id", 1)
    agent._last_plan_step = agent_state.get("last_plan_step", 0)

    # Restore monitor token counts
    monitor_state = payload.get("monitor_state", {})
    agent.monitor.total_input_token_count = monitor_state.get("total_input_token_count", 0)
    agent.monitor.total_output_token_count = monitor_state.get("total_output_token_count", 0)

    return payload.get("session_stats", {
        "turns": 0,
        "total_time": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    })


def save_session(filepath: str, agent, session_stats: dict) -> int:
    """Save an entire agent session to a JSON file.

    Args:
        filepath: Path to the output JSON file.
        agent: The CodeAgent instance whose state to save.
        session_stats: Session statistics dict (turns, time, tokens).

    Returns:
        Number of steps saved.
    """
    steps = [serialize_step(step) for step in agent.memory.steps]

    payload = {
        "version": SESSION_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "agent_state": {
            "system_prompt": agent.memory.system_prompt.system_prompt,
            "next_actionstep_id": agent._next_actionstep_id,
            "last_plan_step": agent._last_plan_step,
            "knowledge": getattr(agent.memory, "knowledge", ""),
        },
        "session_stats": dict(session_stats),
        "monitor_state": {
            "total_input_token_count": agent.monitor.total_input_token_count,
            "total_output_token_count": agent.monitor.total_output_token_count,
        },
        "steps": steps,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return len(steps)


def load_session(filepath: str, agent) -> dict:
    """Load a session from a JSON file into an existing agent.

    Args:
        filepath: Path to the session JSON file.
        agent: The CodeAgent instance to restore into.

    Returns:
        Restored session_stats dict.

    Raises:
        ValueError: If the file version is unsupported.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    version = payload.get("version")
    if version != SESSION_VERSION:
        raise ValueError(f"Unsupported session file version: {version} (expected {SESSION_VERSION})")

    # Restore agent memory
    agent_state = payload.get("agent_state", {})
    agent.memory.system_prompt = SystemPromptStep(system_prompt=agent_state.get("system_prompt", ""))
    agent.memory.steps = [deserialize_step(s) for s in payload.get("steps", [])]
    if hasattr(agent.memory, "knowledge"):
        agent.memory.knowledge = agent_state.get("knowledge", "")

    # Restore agent counters
    agent._next_actionstep_id = agent_state.get("next_actionstep_id", 1)
    agent._last_plan_step = agent_state.get("last_plan_step", 0)

    # Restore monitor token counts
    monitor_state = payload.get("monitor_state", {})
    agent.monitor.total_input_token_count = monitor_state.get("total_input_token_count", 0)
    agent.monitor.total_output_token_count = monitor_state.get("total_output_token_count", 0)

    return payload.get("session_stats", {
        "turns": 0,
        "total_time": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    })
