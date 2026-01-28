#!/usr/bin/env python
# coding=utf-8

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
from smolagents.models import CHARS_PER_TOKEN, ChatMessage, MessageRole
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
    "create_merge_prompt",
]


logger = getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for context compression behavior.

    Args:
        enabled: Whether compression is enabled.
        keep_recent_steps: Number of recent steps to keep in full detail.
        max_uncompressed_steps: Trigger compression when compressible step count exceeds this.
        estimated_token_threshold: Trigger compression when estimated tokens exceed this (0 = disabled).
        compression_model: Optional separate model for compression (None = use main model).
        max_summary_tokens: Maximum tokens for the generated summary.
        preserve_error_steps: Whether to always preserve steps with errors.
        preserve_final_answer_steps: Whether to always preserve final_answer steps.
        max_compressed_steps: Merge compressed steps when their count exceeds this (0 = disabled).
            When multiple CompressedHistoryStep instances accumulate over successive compression
            cycles, this threshold triggers merging them into a single consolidated summary.
            Note: each merge is a lossy operation (summary of summaries), so information fidelity
            decreases with each merge cycle.
        keep_compressed_steps: Number of most recent compressed steps to keep in full detail
            during a merge operation (default 0 = merge all). Analogous to keep_recent_steps
            but for the merge phase. When set, the N most recent CompressedHistoryStep instances
            are preserved and only the older ones are merged together. This helps retain higher
            fidelity in recent compressed summaries.
    """

    enabled: bool = True
    keep_recent_steps: int = 5
    max_uncompressed_steps: int = 10
    estimated_token_threshold: int = 0
    compression_model: "Model | None" = None
    max_summary_tokens: int = 50000
    preserve_error_steps: bool = False
    preserve_final_answer_steps: bool = True
    max_compressed_steps: int = 32
    keep_compressed_steps: int = 22


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
                output = str(step.model_output)
                desc += f"\n  <model_output>{output}</model_output>"
            if step.observations:
                obs = str(step.observations)
                desc += f"\n  <observations>{obs}</observations>"
            if step.code_action:
                code = step.code_action
                desc += f"\n  <code_action>{code}</code_action>"
            step_descriptions.append(desc)
        elif isinstance(step, PlanningStep):
            plan = step.plan if step.plan else "No plan"
            step_descriptions.append(f"Planning step:\n  <plan>{plan}</plan>")

    steps_text = "\n\n".join(step_descriptions)

    return f"""Summarize the following agent execution history into a concise summary that preserves:
1. Key decisions and reasoning
2. Important observations and results
3. Any errors or issues encountered
4. Progress toward the goal
5. KEEP the latest todo list or plan if present.
6. KEEP any information that is relevant for the continuation of the task.

Be concise but preserve critical context needed for continued problem-solving.

EXECUTION HISTORY:
{steps_text}

SUMMARY:"""


def create_merge_prompt(compressed_steps: list[CompressedHistoryStep]) -> str:
    """Create the prompt for merging multiple compressed history summaries.

    Unlike the initial compression prompt which works on raw action steps,
    this prompt consolidates existing summaries into a single unified summary.
    Each merge cycle is lossy (summary of summaries), so the prompt emphasizes
    preserving the most critical information.

    Args:
        compressed_steps: List of CompressedHistoryStep instances to merge.

    Returns:
        The prompt string for the merge LLM call.
    """
    summaries = []
    for i, step in enumerate(compressed_steps, 1):
        summaries.append(
            f"Summary {i} (covering {step.original_step_count} steps, "
            f"step numbers: {step.compressed_step_numbers}):\n{step.summary}"
        )

    summaries_text = "\n\n".join(summaries)
    total_steps = sum(step.original_step_count for step in compressed_steps)

    return f"""Merge the following {len(compressed_steps)} summaries of previous agent work into a single consolidated summary.
These summaries cover {total_steps} total steps of agent execution.

Preserve the most important information:
1. Key decisions, outcomes, and final results
2. Critical errors or failures encountered
3. Overall progress toward the goal
4. Any information that would be needed for continued problem-solving
5. KEEP the latest todo list or plan if present.
6. KEEP any information that is relevant for the continuation of the task.
Eliminate redundancy and consolidate overlapping information.

SUMMARIES TO MERGE:
{summaries_text}

CONSOLIDATED SUMMARY:"""


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
        if compressible_count > self.config.max_uncompressed_steps:
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

        # Always preserve the most recent PlanningStep
        for i in range(len(steps) - 1, -1, -1):
            if isinstance(steps[i], PlanningStep):
                preserved_indices.add(i)
                break

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

        # Safety check: skip compression if summary is larger than original
        summary_chars = len(summary) if summary else 0
        if summary_chars >= original_chars:
            if self.agent_logger:
                self.agent_logger.log_markdown(
                    content=f"Compression skipped: summary ({summary_chars:,} chars) is larger than original ({original_chars:,} chars)",
                    title="Context Compression Skipped",
                    level=LogLevel.INFO,
                )
            else:
                logger.info(f"Compression skipped: summary ({summary_chars} chars) >= original ({original_chars} chars)")
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

    def should_merge_compressed(self, steps: list[MemoryStep]) -> bool:
        """Check if compressed steps should be merged.

        When multiple CompressedHistoryStep instances accumulate over successive
        compression cycles, this checks whether their count exceeds the configured
        threshold for merging. Also ensures there are enough mergeable steps
        (after excluding those preserved by keep_compressed_steps).

        Args:
            steps: Current list of memory steps.

        Returns:
            True if compressed steps should be merged, False otherwise.
        """
        if not self.config.enabled or self.config.max_compressed_steps <= 0:
            return False

        compressed_count = sum(1 for step in steps if isinstance(step, CompressedHistoryStep))

        if compressed_count <= self.config.max_compressed_steps:
            return False

        # Need at least 2 mergeable steps (after excluding kept ones)
        mergeable_count = compressed_count - self.config.keep_compressed_steps
        return mergeable_count >= 2

    def merge_compressed(self, steps: list[MemoryStep]) -> list[MemoryStep]:
        """Merge multiple CompressedHistoryStep instances into one.

        Collects all compressed history steps, generates a consolidated summary
        via LLM, and replaces them with a single merged CompressedHistoryStep.
        If keep_compressed_steps is set, the most recent N compressed steps are
        preserved and only older ones are merged.

        Note: This is a lossy operation (summary of summaries). Information
        fidelity decreases with each merge cycle.

        Args:
            steps: Current list of memory steps.

        Returns:
            New list with older compressed history steps merged into one,
            preserving the most recent keep_compressed_steps instances.
        """
        start_time = time.time()

        compressed_steps = [step for step in steps if isinstance(step, CompressedHistoryStep)]

        if len(compressed_steps) <= 1:
            return steps

        # Determine which compressed steps to keep vs merge
        keep_count = self.config.keep_compressed_steps
        if keep_count > 0:
            # Keep at most len-1 to ensure at least 1 remains for potential merge
            keep_count = min(keep_count, len(compressed_steps) - 1)
            steps_to_keep = set(id(s) for s in compressed_steps[-keep_count:])
            steps_to_merge = [s for s in compressed_steps if id(s) not in steps_to_keep]
        else:
            steps_to_keep = set()
            steps_to_merge = compressed_steps

        if len(steps_to_merge) <= 1:
            return steps  # Not enough to merge

        # Build merge prompt and call LLM
        merge_prompt = create_merge_prompt(steps_to_merge)

        try:
            merge_message = self.compression_model.generate(
                [
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{"type": "text", "text": merge_prompt}],
                    )
                ]
            )
            merged_summary = merge_message.content
            if isinstance(merged_summary, list):
                merged_summary = " ".join(
                    item.get("text", "") for item in merged_summary if isinstance(item, dict)
                )

            merge_token_usage = merge_message.token_usage
        except Exception as e:
            logger.warning(f"Compressed step merge failed, keeping original steps: {e}")
            return steps

        # Safety check: skip merge if consolidated summary is larger than combined originals
        original_chars = sum(len(step.summary) for step in steps_to_merge)
        merged_chars = len(merged_summary) if merged_summary else 0
        if merged_chars >= original_chars:
            if self.agent_logger:
                self.agent_logger.log_markdown(
                    content=f"Merge skipped: consolidated summary ({merged_chars:,} chars) "
                    f"is larger than combined originals ({original_chars:,} chars)",
                    title="Compressed Step Merge Skipped",
                    level=LogLevel.INFO,
                )
            else:
                logger.info(
                    f"Merge skipped: consolidated summary ({merged_chars} chars) "
                    f">= combined originals ({original_chars} chars)"
                )
            return steps

        # Accumulate metadata from merged compressed steps
        all_step_numbers = []
        total_original_count = 0
        for step in steps_to_merge:
            all_step_numbers.extend(step.compressed_step_numbers)
            total_original_count += step.original_step_count

        merged_step = CompressedHistoryStep(
            summary=merged_summary,
            compressed_step_numbers=sorted(set(all_step_numbers)),
            original_step_count=total_original_count,
            timing=Timing(start_time=start_time, end_time=time.time()),
            compression_token_usage=merge_token_usage,
        )

        # Rebuild steps: replace merged CompressedHistoryStep instances with the single
        # merged one, while preserving kept compressed steps in their original positions
        merge_set = set(id(step) for step in steps_to_merge)
        new_steps = []
        merged_inserted = False

        for step in steps:
            if isinstance(step, CompressedHistoryStep) and id(step) in merge_set:
                if not merged_inserted:
                    new_steps.append(merged_step)
                    merged_inserted = True
                # Skip subsequent merged compressed steps (replaced by merged)
            else:
                new_steps.append(step)

        # Log
        kept_count = len(compressed_steps) - len(steps_to_merge)
        if self.agent_logger:
            compression_ratio = (1 - merged_chars / original_chars) * 100 if original_chars > 0 else 0
            kept_msg = f" Kept {kept_count} recent compressed steps." if kept_count > 0 else ""
            self.agent_logger.log_markdown(
                content=f"Merged {len(steps_to_merge)} compressed steps "
                f"({total_original_count} original steps) from {original_chars:,} chars "
                f"to {merged_chars:,} chars ({compression_ratio:.1f}% reduction).{kept_msg}",
                title="Compressed Step Merge",
                level=LogLevel.INFO,
            )
        else:
            kept_msg = f", kept {kept_count} recent" if kept_count > 0 else ""
            logger.info(
                f"Merged {len(steps_to_merge)} compressed steps "
                f"({total_original_count} original steps) into single summary{kept_msg}"
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
        """Callback that triggers compression and merging if needed."""
        # Only trigger on ActionStep (after meaningful work)
        if not isinstance(step, ActionStep):
            return

        if compressor.should_compress(agent.memory.steps):
            agent.memory.steps = compressor.compress(agent.memory.steps)

        if compressor.should_merge_compressed(agent.memory.steps):
            agent.memory.steps = compressor.merge_compressed(agent.memory.steps)

    return compression_callback
