#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Context compression for agent memory.

This module provides a hybrid rolling summarization system that compresses older memory steps
via LLM summarization while keeping recent steps in full detail.
"""

import time
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable

from smolagents.memory import ActionStep, MemoryStep, PlanningStep, TaskStep, SystemPromptStep
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage


if TYPE_CHECKING:
    from smolagents.agents import MultiStepAgent
    from smolagents.models import Model


__all__ = [
    "CompressionConfig",
    "CompressedHistoryStep",
    "ContextCompressor",
    "estimate_tokens",
    "estimate_step_tokens",
    "create_compression_callback",
]


logger = getLogger(__name__)

# Token estimation heuristic: ~4 characters per token (conservative estimate for English text)
CHARS_PER_TOKEN = 4


@dataclass
class CompressionConfig:
    """Configuration for context compression behavior.

    Args:
        enabled: Whether compression is enabled.
        keep_recent_steps: Number of recent steps to keep in full detail.
        step_count_threshold: Trigger compression when compressible step count exceeds this.
        estimated_token_threshold: Trigger compression when estimated tokens exceed this (0 = disabled).
        compression_model: Optional separate model for compression (None = use main model).
        max_summary_tokens: Maximum tokens for the generated summary.
        preserve_error_steps: Whether to always preserve steps with errors.
        preserve_final_answer_steps: Whether to always preserve final_answer steps.
    """

    enabled: bool = True
    keep_recent_steps: int = 5
    step_count_threshold: int = 10
    estimated_token_threshold: int = 0
    compression_model: "Model | None" = None
    max_summary_tokens: int = 500
    preserve_error_steps: bool = True
    preserve_final_answer_steps: bool = True


@dataclass
class CompressedHistoryStep(MemoryStep):
    """Represents a compressed summary of multiple historical steps.

    This step type is created when older steps are compressed via LLM summarization.
    It contains a summary of the compressed steps and metadata about the compression.

    Attributes:
        summary: The LLM-generated summary of compressed steps.
        compressed_step_numbers: List of step numbers that were compressed.
        original_step_count: Number of steps that were compressed.
        timing: Timing information for the compression operation.
        compression_token_usage: Token usage for the compression LLM call.
    """

    summary: str = ""
    compressed_step_numbers: list[int] = field(default_factory=list)
    original_step_count: int = 0
    timing: Timing | None = None
    compression_token_usage: TokenUsage | None = None

    def dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "summary": self.summary,
            "compressed_step_numbers": self.compressed_step_numbers,
            "original_step_count": self.original_step_count,
            "timing": self.timing.dict() if self.timing else None,
            "compression_token_usage": self.compression_token_usage.dict() if self.compression_token_usage else None,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        """Convert compressed history to a context message for the LLM.

        Args:
            summary_mode: If True, returns empty list (following pattern of other steps).

        Returns:
            List containing a single USER message with the compressed history summary.
        """
        if summary_mode:
            return []

        content = f"""[COMPRESSED HISTORY - {self.original_step_count} steps summarized]
{self.summary}
[END COMPRESSED HISTORY]"""

        return [
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": content}],
            )
        ]


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using character heuristic.

    Uses a conservative estimate of ~4 characters per token, which works
    reasonably well for English text across most tokenizers.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated number of tokens.
    """
    return len(text) // CHARS_PER_TOKEN


def estimate_step_tokens(step: MemoryStep) -> int:
    """Estimate token count for a memory step.

    Converts the step to messages and estimates the total token count
    based on the text content of all messages.

    Args:
        step: The memory step to estimate tokens for.

    Returns:
        Estimated number of tokens for the step.
    """
    messages = step.to_messages(summary_mode=False)
    total_chars = 0
    for msg in messages:
        if isinstance(msg.content, str):
            total_chars += len(msg.content)
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and "text" in item:
                    total_chars += len(str(item["text"]))
    return total_chars // CHARS_PER_TOKEN


def should_preserve_step(step: MemoryStep, config: CompressionConfig) -> bool:
    """Determine if a step should be preserved (not compressed).

    Certain steps should never be compressed as they contain critical context:
    - TaskStep: The original task is always needed
    - SystemPromptStep: System instructions
    - CompressedHistoryStep: Already compressed, don't re-compress
    - Error steps: Important for understanding failures (if configured)
    - Final answer steps: Important for understanding results (if configured)

    Args:
        step: The memory step to check.
        config: Compression configuration.

    Returns:
        True if the step should be preserved, False if it can be compressed.
    """
    # Always preserve TaskStep and SystemPromptStep
    if isinstance(step, (TaskStep, SystemPromptStep)):
        return True

    # Don't re-compress already compressed steps
    if isinstance(step, CompressedHistoryStep):
        return True

    # Check ActionStep-specific preservation rules
    if isinstance(step, ActionStep):
        # Preserve error steps if configured
        if config.preserve_error_steps and step.error is not None:
            return True

        # Preserve final_answer steps if configured
        if config.preserve_final_answer_steps and step.is_final_answer:
            return True

    return False


def create_compression_prompt(steps_to_compress: list[MemoryStep]) -> str:
    """Create the prompt for the compression LLM call.

    Builds a structured representation of the steps to compress and asks
    the LLM to generate a concise summary preserving key information.

    Args:
        steps_to_compress: List of memory steps to summarize.

    Returns:
        The prompt string for the compression LLM call.
    """
    step_descriptions = []

    for step in steps_to_compress:
        if isinstance(step, ActionStep):
            desc = f"Step {step.step_number}:"
            if step.model_output:
                # Truncate long outputs
                output = str(step.model_output)[:500]
                desc += f"\n  Agent thought: {output}"
            if step.observations:
                obs = str(step.observations)[:300]
                desc += f"\n  Observation: {obs}"
            if step.code_action:
                code = step.code_action[:200]
                desc += f"\n  Code executed: {code}..."
            step_descriptions.append(desc)
        elif isinstance(step, PlanningStep):
            plan = step.plan[:400] if step.plan else "No plan"
            step_descriptions.append(f"Planning step:\n  {plan}")

    steps_text = "\n\n".join(step_descriptions)

    return f"""Summarize the following agent execution history into a concise summary that preserves:
1. Key decisions and reasoning
2. Important observations and results
3. Any errors or issues encountered
4. Progress toward the goal

Be concise but preserve critical context needed for continued problem-solving.

EXECUTION HISTORY:
{steps_text}

SUMMARY:"""


class ContextCompressor:
    """Manages context compression for agent memory.

    This class monitors memory size and triggers compression when thresholds
    are exceeded. It compresses older steps into a summary while preserving
    recent steps and critical context.

    Args:
        config: Compression configuration.
        main_model: The main model used by the agent (used for compression if no separate model specified).
        agent_logger: Optional agent logger for rich logging output.
    """

    def __init__(self, config: CompressionConfig, main_model: "Model", agent_logger: AgentLogger | None = None):
        self.config = config
        self.main_model = main_model
        self.agent_logger = agent_logger
        self._compression_count = 0

    @property
    def compression_model(self) -> "Model":
        """Get the model to use for compression."""
        return self.config.compression_model or self.main_model

    def should_compress(self, steps: list[MemoryStep]) -> bool:
        """Check if compression should be triggered.

        Evaluates whether the current memory state exceeds configured thresholds
        for step count or estimated tokens.

        Args:
            steps: Current list of memory steps.

        Returns:
            True if compression should be triggered, False otherwise.
        """
        if not self.config.enabled:
            return False

        # Count compressible steps (excluding preserved ones)
        compressible_count = sum(1 for step in steps if not should_preserve_step(step, self.config))

        # Need at least keep_recent_steps + 1 to compress anything
        if compressible_count <= self.config.keep_recent_steps:
            return False

        # Check step count threshold
        if compressible_count > self.config.step_count_threshold:
            return True

        # Check token threshold if enabled
        if self.config.estimated_token_threshold > 0:
            total_tokens = sum(estimate_step_tokens(step) for step in steps)
            if total_tokens > self.config.estimated_token_threshold:
                return True

        return False

    def compress(self, steps: list[MemoryStep]) -> list[MemoryStep]:
        """Compress older steps while preserving recent and critical steps.

        This method:
        1. Identifies which steps must be preserved (TaskStep, errors, etc.)
        2. Keeps the most recent N compressible steps in full detail
        3. Compresses remaining old steps into a summary via LLM
        4. Returns a new list with compressed history

        Args:
            steps: Current list of memory steps.

        Returns:
            New list with older steps compressed into a CompressedHistoryStep.
        """
        start_time = time.time()

        if not self.should_compress(steps):
            return steps

        # Separate preserved steps and compressible steps
        preserved_indices = set()
        task_step_index = None

        for i, step in enumerate(steps):
            if isinstance(step, TaskStep):
                task_step_index = i
                preserved_indices.add(i)
            elif should_preserve_step(step, self.config):
                preserved_indices.add(i)

        # Get indices of compressible steps
        compressible_indices = [i for i in range(len(steps)) if i not in preserved_indices]

        if len(compressible_indices) <= self.config.keep_recent_steps:
            return steps  # Nothing to compress

        # Steps to keep in full detail (most recent compressible ones)
        recent_to_keep = set(compressible_indices[-self.config.keep_recent_steps :])

        # Steps to compress (older compressible ones)
        to_compress_indices = [i for i in compressible_indices if i not in recent_to_keep]

        if not to_compress_indices:
            return steps

        steps_to_compress = [steps[i] for i in to_compress_indices]

        # Calculate original character count for logging
        original_chars = sum(
            len(str(step.model_output or "")) + len(str(step.observations or ""))
            if isinstance(step, ActionStep)
            else len(str(step.plan or "")) if isinstance(step, PlanningStep)
            else 0
            for step in steps_to_compress
        )

        # Generate summary using LLM
        compression_prompt = create_compression_prompt(steps_to_compress)

        try:
            summary_message = self.compression_model.generate(
                [
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{"type": "text", "text": compression_prompt}],
                    )
                ]
            )
            summary = summary_message.content
            if isinstance(summary, list):
                summary = " ".join(item.get("text", "") for item in summary if isinstance(item, dict))

            compression_token_usage = summary_message.token_usage
        except Exception as e:
            logger.warning(f"Compression failed, keeping original steps: {e}")
            return steps

        # Build compressed step
        compressed_step_numbers = []
        for step in steps_to_compress:
            if isinstance(step, ActionStep):
                compressed_step_numbers.append(step.step_number)

        compressed_step = CompressedHistoryStep(
            summary=summary,
            compressed_step_numbers=compressed_step_numbers,
            original_step_count=len(steps_to_compress),
            timing=Timing(start_time=start_time, end_time=time.time()),
            compression_token_usage=compression_token_usage,
        )

        # Rebuild steps list maintaining logical order:
        # 1. TaskStep (if exists)
        # 2. CompressedHistoryStep
        # 3. Preserved steps (errors, etc.) that came after compressed steps
        # 4. Recent steps to keep
        new_steps = []

        # Add TaskStep first if it exists
        if task_step_index is not None:
            new_steps.append(steps[task_step_index])

        # Add compressed history
        new_steps.append(compressed_step)

        # Add preserved steps (except TaskStep, already added) in original order
        for i in sorted(preserved_indices):
            if i != task_step_index:
                new_steps.append(steps[i])

        # Add recent steps to keep in original order
        for i in sorted(recent_to_keep):
            new_steps.append(steps[i])

        self._compression_count += 1
        summary_chars = len(summary) if summary else 0

        # Log using agent logger if available
        if self.agent_logger:
            compression_ratio = (1 - summary_chars / original_chars) * 100 if original_chars > 0 else 0
            self.agent_logger.log_markdown(
                content=f"Compressed {len(steps_to_compress)} steps from {original_chars:,} chars to {summary_chars:,} chars "
                f"({compression_ratio:.1f}% reduction). Kept {len(new_steps)} steps total. (compression #{self._compression_count})",
                title="Context Compression",
                level=LogLevel.INFO,
            )
        else:
            logger.info(
                f"Compressed {len(steps_to_compress)} steps into summary "
                f"(kept {len(new_steps)} steps total, compression #{self._compression_count})"
            )

        return new_steps


def create_compression_callback(compressor: ContextCompressor) -> Callable:
    """Create a step callback that triggers compression after each step.

    This callback is registered with the agent's callback system and is
    called after each ActionStep completes. It checks if compression
    is needed and updates the agent's memory if so.

    Args:
        compressor: The ContextCompressor instance to use.

    Returns:
        A callback function compatible with the agent's callback system.
    """

    def compression_callback(step: MemoryStep, agent: "MultiStepAgent") -> None:
        """Callback that triggers compression if needed."""
        # Only trigger on ActionStep (after meaningful work)
        if not isinstance(step, ActionStep):
            return

        if compressor.should_compress(agent.memory.steps):
            agent.memory.steps = compressor.compress(agent.memory.steps)

    return compression_callback
