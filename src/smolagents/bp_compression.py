#!/usr/bin/env python
# coding=utf-8

"""
Context compression for agent memory.

This module provides a hybrid rolling summarization system that compresses older memory steps
via LLM summarization while keeping recent steps in full detail.
"""

import re
import time
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable

from smolagents.memory import ActionStep, MemoryStep, PlanningStep, TaskStep, SystemPromptStep, count_messages_chars
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
    "merge_context",
    "list_xml_tag_names",
    "create_knowledge_extraction_prompt",
    "parse_compression_output",
]


logger = getLogger(__name__)

COMMON_COMPRESSION_INSTRUCTIONS="""Preserve the most important information:
1. Key decisions, outcomes, and final results
2. Critical errors or failures encountered
3. Overall progress toward the goal
4. Any information that would be needed for continued problem-solving
5. KEEP the latest todo list or plan if present.
6. KEEP any information that is relevant for the continuation of the task.
Eliminate redundancy and consolidate overlapping information.

Do not follow this bad example:
<bad-example>
# Execution Summary

**Current Working Directory:** `/home/look/fjup/content/smolagents`

**Progress:** Successfully identified the current folder location using `os.getcwd()`.

**Key Information for Continuation:** The agent is operating in the `/home/look/fjup/content/smolagents` directory. This context should be retained for any subsequent file operations or path-dependent tasks.
</bad-example>

The above example has many problems:
* It is repetitive.
* Spends tokens with section titles.

You should follow this good exanple instead:
<good-example>
I run `os.getcwd()` and found `/home/look/fjup/content/smolagents`.
</good-example>"""

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
        min_compression_chars: Minimum total character count of steps-to-compress before an LLM
            compression call is made (default 4096). If the content to compress is smaller than
            this threshold, the compression or merge is skipped to avoid wasting an LLM call on
            already-concise content. Set to 0 to disable.
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
    min_compression_chars: int = 4096


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
                role=MessageRole.ASSISTANT,
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
    return count_messages_chars(step.to_messages(summary_mode=False)) // CHARS_PER_TOKEN


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


def _build_post_steps_section(post_steps: list["MemoryStep"] | None) -> str:
    """Build a <subsequent_steps> prompt section from steps that follow the compressed batch.

    These steps are shown to the compressor as read-only context so it can avoid
    writing stale knowledge that has already been superseded by later activity.

    Args:
        post_steps: Steps occurring after the batch being compressed. May be None or empty.

    Returns:
        A formatted prompt section string, or empty string if nothing to show.
    """
    if not post_steps:
        return ""

    post_step_descs = []
    for step in post_steps:
        if isinstance(step, ActionStep):
            desc = f"Step {step.step_number}:"
            if step.model_output:
                output = str(step.model_output)[:500]
                desc += f"\n<model_output>{output}</model_output>"
            if step.observations:
                obs = str(step.observations)[:300]
                desc += f"\n<result>{obs}</result>"
            post_step_descs.append("<step>" + desc + "</step>")
        elif isinstance(step, PlanningStep):
            plan = (step.plan or "")[:400]
            post_step_descs.append("<step><plan>" + plan + "</plan></step>")
        elif isinstance(step, CompressedHistoryStep):
            summary = (step.summary or "")[:400]
            post_step_descs.append(f"<step><compressed_summary>{summary}</compressed_summary></step>")

    if not post_step_descs:
        return ""

    post_steps_text = "\n".join(post_step_descs)
    return f"""
The following steps occurred AFTER the batch you are summarizing. Use them to understand
what is still current and what has already been superseded. Do NOT summarize these steps --
they will remain in full detail. Only use them as context to avoid writing stale knowledge.
<subsequent_steps>
{post_steps_text}
</subsequent_steps>
"""


def create_compression_prompt(
    steps_to_compress: list[MemoryStep],
    knowledge: str = "",
    existing_summaries: list["CompressedHistoryStep"] | None = None,
    post_steps: list[MemoryStep] | None = None,
) -> str:
    """Create the prompt for the compression LLM call.

    Builds a structured representation of the steps to compress and asks
    the LLM to generate a concise summary preserving key information.

    The prompt provides two types of existing context to avoid duplication:
    - **Compressed history** (existing_summaries): chronological record of past events
      and changes. The new summary should complement, not repeat, this history.
    - **Knowledge** (knowledge): current beliefs and facts. The LLM can propose
      updates to knowledge when the execution history reveals corrections or
      important new information.

    Args:
        steps_to_compress: List of memory steps to summarize.
        knowledge: Current knowledge store content (tagged XML). Empty string if none.
        existing_summaries: Already-compressed history steps to avoid duplicating.
        post_steps: Steps that occurred AFTER the batch being compressed. Shown to the
            compressor so it can see what is still current vs already superseded.

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
                desc += f"\n<model_output>{output}</model_output>"
            if step.observations:
                obs = str(step.observations)
                desc += f"\n<result>{obs}</result>"
            # There is no point in compressing code action. It is already present in model_output.
            #if step.code_action:
            #    code = step.code_action
            #    desc += f"\n  <code_action>{code}</code_action>"
            step_descriptions.append('<step>'+desc+'</step>')
        elif isinstance(step, PlanningStep):
            plan = step.plan if step.plan else "No plan"
            step_descriptions.append('<step>'+f"<plan>{plan}</plan>"+'</step>')

    steps_text = "<\n>".join(step_descriptions)

    # Build compressed history section
    history_section = ""
    if existing_summaries:
        history_parts = []
        for s in existing_summaries:
            history_parts.append(s.summary)
        history_text = "\n---\n".join(history_parts)
        history_section = f"""
The following is the compressed history of earlier work (events and changes over time).
Do NOT repeat any information already captured in the compressed history.
Your summary should only describe NEW events, actions, and changes from the execution history below.

<compressed_history>
{history_text}
</compressed_history>
"""

    # Build knowledge section
    has_knowledge = knowledge and knowledge.strip()
    has_history = bool(existing_summaries)

    if has_knowledge:
        knowledge_section = f"""
The agent has a persistent knowledge store containing current beliefs and facts:
<current_knowledge>
{knowledge}
</current_knowledge>
"""
    else:
        knowledge_section = ""

    # Build subsequent steps section (steps AFTER the batch being compressed)
    post_steps_section = _build_post_steps_section(post_steps)

    # Build deduplication and output instructions
    dedup_parts = []
    if has_history:
        dedup_parts.append("the compressed history (past events)")
    if has_knowledge:
        dedup_parts.append("the knowledge store (current facts)")

    if dedup_parts:
        dedup_instruction = f"Do NOT repeat information already in {' or '.join(dedup_parts)}."
    else:
        dedup_instruction = ""

    output_instruction = f"""
{dedup_instruction}

There are two distinct stores:
- **Compressed history** captures events, changes, and what happened over time.
- **Knowledge** captures current beliefs, facts, and the latest state of things.

Episodic Memory vs. Semantic Memory
- **Compressed History** = Episodic Memory = what happened (events, actions taken). 
- **Knowledge** = Semantic Memory = what is currently true (facts, beliefs, current state).

In the Human mind: 
- **Episodic memory** = "I did X at time T." 
- **Semantic memory** = "X is true."

Your summary will be added to the compressed history (Episodic Memory). It should describe what happened
(events, actions, outcomes, changes) without repeating prior history entries.

If the execution history reveals important new facts or corrections your existing knowledge (Semantic Memory),
include a <knowledge_updates> section. Use XML tags to add, update, or delete sections:
- To ADD or UPDATE: <tag_name>new content</tag_name>
- To DELETE an obsolete section: <tag_name/>

You can add/update/delete as many <tag_name>s as you see fit. 

If no knowledge updates are needed, omit the <knowledge_updates> section entirely.
In the case that you spot any other error in the knowledge, you can fix as you see fit.

Output format:
<summary>
Your concise summary of new events and changes...
</summary>
<knowledge_updates>
...tagged updates if any...
</knowledge_updates>
"""

    return f"""Hello super-intelligence!
This task is involved in your context compression.
Please summarize the following agent execution history into a concise summary.
Note: after compression, the original steps will be permanently removed from context. Write as if the reader will never see the originals.
{COMMON_COMPRESSION_INSTRUCTIONS}
{history_section}{knowledge_section}{post_steps_section}{output_instruction}
This is the execution history to summarize:
<execution_history>
{steps_text}
</execution_history>
"""


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

    return f"""Merge the following {len(compressed_steps)} summaries (<SUMMARIES_TO_MERGE></SUMMARIES_TO_MERGE>) of previous agent work into a single consolidated summary.
These summaries cover {total_steps} total steps of agent execution.

{COMMON_COMPRESSION_INSTRUCTIONS}

<SUMMARIES_TO_MERGE>
{summaries_text}
</SUMMARIES_TO_MERGE>

Output format:
<summary>
Your consolidated summary of all events and changes...
</summary>
<knowledge_updates>
...tagged updates if any...
</knowledge_updates>
"""





def parse_compression_output(raw_output: str) -> tuple[str, str]:
    """Parse structured compression output into summary and knowledge updates.

    Expects output in the format:
        <summary>...</summary>
        <knowledge_updates>...</knowledge_updates>

    Falls back gracefully: if no <summary> tags found, treats the entire
    output as the summary with no knowledge updates.

    Args:
        raw_output: Raw LLM output from the compression call.

    Returns:
        Tuple of (summary, knowledge_updates). knowledge_updates may be empty string.
    """
    if not raw_output:
        return "", ""

    # Try to extract <summary>...</summary>
    summary_match = re.search(r'<summary>(.*?)</summary>', raw_output, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()
    else:
        # Fallback: no <summary> tags, use everything before <knowledge_updates> or entire output
        knowledge_start = raw_output.find('<knowledge_updates>')
        if knowledge_start >= 0:
            summary = raw_output[:knowledge_start].strip()
        else:
            summary = raw_output.strip()

    # Try to extract <knowledge_updates>...</knowledge_updates>
    knowledge_match = re.search(r'<knowledge_updates>(.*?)</knowledge_updates>', raw_output, re.DOTALL)
    knowledge_updates = knowledge_match.group(1).strip() if knowledge_match else ""

    return summary, knowledge_updates
def list_xml_tag_names(text: str) -> list[str]:
    """List unique top-level XML tag names found in text.

    Args:
        text: String containing XML-tagged content.

    Returns:
        Sorted list of unique tag names.
    """
    if not text or not text.strip():
        return []
    names = set()
    for m in re.finditer(r'<([\w][\w-]*)[\s>/]', text):
        names.add(m.group(1))
    return sorted(names)


def merge_context(existing: str, updates: str) -> str:
    """Merge tagged XML updates into existing context using three rules:

    1. **UPDATE**: If a tag in ``updates`` has content and exists in ``existing``, replace it.
    2. **DELETE**: If a tag in ``updates`` is self-closing (``<tag/>``) or empty
       (``<tag></tag>``), remove it from ``existing``.
    3. **APPEND**: If a tag in ``updates`` has content and does NOT exist in
       ``existing``, append it.

    Args:
        existing: The current knowledge string (tagged XML).
        updates: New tagged XML with updates, deletions, or additions.

    Returns:
        Updated knowledge string after applying all operations.
    """
    if not updates or not updates.strip():
        return existing

    result = existing if existing else ""
    processed_tags = set()

    # 1. Self-closing tags: <tag/> or <tag /> -> DELETE
    for m in re.finditer(r'<([\w][\w-]*)\s*/>', updates):
        tag = m.group(1)
        if tag in processed_tags:
            continue
        processed_tags.add(tag)
        result = re.sub(rf'<{re.escape(tag)}>.*?</{re.escape(tag)}>\s*', '', result, flags=re.DOTALL)
        result = re.sub(rf'<{re.escape(tag)}\s*/>\s*', '', result, flags=re.DOTALL)

    # 2. Content tags: <tag>content</tag>
    for m in re.finditer(r'<([\w][\w-]*)>(.*?)</\1>', updates, re.DOTALL):
        tag = m.group(1)
        content = m.group(2)
        if tag in processed_tags:
            continue
        processed_tags.add(tag)
        full_tag = f'<{tag}>{content}</{tag}>'
        if content.strip() == '':
            # Empty content -> DELETE
            result = re.sub(rf'<{re.escape(tag)}>.*?</{re.escape(tag)}>\s*', '', result, flags=re.DOTALL)
        elif re.search(rf'<{re.escape(tag)}>.*?</{re.escape(tag)}>', result, re.DOTALL):
            # Tag exists -> UPDATE
            result = re.sub(rf'<{re.escape(tag)}>.*?</{re.escape(tag)}>', full_tag, result, flags=re.DOTALL)
        else:
            # Tag doesn't exist -> APPEND
            result = result.rstrip() + '\n' + full_tag + '\n'

    return result


def create_knowledge_extraction_prompt(
    compressed_steps: list[CompressedHistoryStep],
    existing_tag_names: list[str] | None = None,
    post_steps: list[MemoryStep] | None = None,
) -> str:
    """Create a prompt for extracting knowledge from compressed summaries.

    Instead of rewriting the full knowledge, this prompt asks the LLM to produce
    a tagged XML diff: updates to existing sections, new sections, or deletions.

    Args:
        compressed_steps: List of CompressedHistoryStep instances to extract knowledge from.
        existing_tag_names: List of tag names currently in the knowledge store.
        post_steps: Steps occurring after the compressed batch. Passed as read-only context
            so the LLM avoids writing knowledge that has already been superseded.

    Returns:
        The prompt string for the knowledge extraction LLM call.
    """
    summaries = []
    for i, step in enumerate(compressed_steps, 1):
        summaries.append(
            f"Summary {i} (covering {step.original_step_count} steps, "
            f"step numbers: {step.compressed_step_numbers}):\n{step.summary}"
        )
    summaries_text = "\n\n".join(summaries)
    total_steps = sum(step.original_step_count for step in compressed_steps)

    if existing_tag_names:
        tag_list = ", ".join(existing_tag_names)
        existing_section = f"""
Existing knowledge sections: {tag_list}

Rules:
- To UPDATE an existing section, use the same tag name with new content
- To DELETE a section that is no longer relevant, use an empty self-closing tag: <tagname/>
- To ADD new information, use a new descriptive tag name
- Only output sections that are new, changed, or should be deleted
- Do NOT output sections that have not changed"""
    else:
        existing_section = """
There are no existing knowledge sections yet. Create new tagged sections for
the important information found in the summaries below.
Use descriptive tag names (e.g., <plan>, <architecture>, <key_findings>, <current_status>)."""

    post_steps_section = _build_post_steps_section(post_steps)

    return f"""Hello super-intelligence!
This task is involved in your context compression.
Note: after compression, the original summaries will be permanently removed from context. Write as if the reader will never see the originals.
Please extract key knowledge from the following {len(compressed_steps)} summaries
covering {total_steps} total steps of agent execution.
These summaries are about to be removed from the context. Therefore, updating the knowledge
with any relevant information is important. In the case that you spot any other error in
the knowledge, you can fix as you see fit.

Output the knowledge as XML-tagged sections. Each section should contain concise,
factual information that would be useful for continuing the task.
{existing_section}

{COMMON_COMPRESSION_INSTRUCTIONS}
{post_steps_section}
<SUMMARIES>
{summaries_text}
</SUMMARIES>

KNOWLEDGE UPDATE:"""


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

    def compress(self, steps: list[MemoryStep], knowledge: str = "") -> tuple[list[MemoryStep], str]:
        """Compress older steps while preserving recent and critical steps.

        This method:
        1. Identifies which steps must be preserved (TaskStep, errors, etc.)
        2. Keeps the most recent N compressible steps in full detail
        3. Compresses remaining old steps into a summary via LLM
        4. Optionally extracts knowledge updates from the same LLM call
        5. Returns a new step list and updated knowledge

        Args:
            steps: Current list of memory steps.
            knowledge: Current knowledge store content (tagged XML).

        Returns:
            Tuple of (new_steps, updated_knowledge).
        """
        start_time = time.time()

        if not self.should_compress(steps):
            return steps, knowledge

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
            return steps, knowledge  # Nothing to compress

        # Steps to keep in full detail (most recent compressible ones)
        recent_to_keep = set(compressible_indices[-self.config.keep_recent_steps :])

        # Steps to compress (older compressible ones)
        to_compress_indices = [i for i in compressible_indices if i not in recent_to_keep]

        if not to_compress_indices:
            return steps, knowledge

        steps_to_compress = [steps[i] for i in to_compress_indices]

        # Calculate original character count for logging
        original_chars = sum(
            len(str(step.model_output or "")) + len(str(step.observations or ""))
            if isinstance(step, ActionStep)
            else len(str(step.plan or "")) if isinstance(step, PlanningStep)
            else 0
            for step in steps_to_compress
        )

        # Skip compression if content is too small to be worth an LLM call
        if self.config.min_compression_chars > 0 and original_chars < self.config.min_compression_chars:
            if self.agent_logger:
                self.agent_logger.log_markdown(
                    content=f"Compression skipped: content ({original_chars:,} chars) is below minimum threshold ({self.config.min_compression_chars:,} chars)",
                    title="Context Compression Skipped",
                    level=LogLevel.INFO,
                )
            else:
                logger.info(f"Compression skipped: content ({original_chars} chars) < min_compression_chars ({self.config.min_compression_chars})")
            return steps, knowledge

        # Collect existing compressed history for deduplication
        existing_summaries = [s for s in steps if isinstance(s, CompressedHistoryStep)]

        # Steps occurring AFTER the batch being compressed (kept in full detail).
        # Pass these to the compressor so it can see what is still current and
        # avoid writing knowledge that was already superseded by later steps.
        max_to_compress_index = max(to_compress_indices)
        post_steps = [
            steps[i] for i in range(max_to_compress_index + 1, len(steps))
            if not isinstance(steps[i], (TaskStep, CompressedHistoryStep))
        ]

        # Generate summary using LLM (history + knowledge aware)
        compression_prompt = create_compression_prompt(
            steps_to_compress, knowledge, existing_summaries, post_steps=post_steps
        )

        try:
            summary_message = self.compression_model.generate(
                [
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{"type": "text", "text": compression_prompt}],
                    )
                ]
            )
            raw_output = summary_message.content
            if isinstance(raw_output, list):
                raw_output = " ".join(item.get("text", "") for item in raw_output if isinstance(item, dict))

            summary, knowledge_updates = parse_compression_output(raw_output)
            compression_token_usage = summary_message.token_usage
        except Exception as e:
            logger.warning(f"Compression failed, keeping original steps: {e}")
            return steps, knowledge

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
            return steps, knowledge

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

        # Apply knowledge updates if any were extracted
        updated_knowledge = knowledge
        if knowledge_updates:
            updated_knowledge = merge_context(knowledge, knowledge_updates)
            knowledge_chars = len(updated_knowledge) if updated_knowledge else 0
            if self.agent_logger:
                tag_names = list_xml_tag_names(updated_knowledge)
                self.agent_logger.log_markdown(
                    content=f"Knowledge updated during compression. "
                    f"Store: {knowledge_chars:,} chars, sections: {tag_names}.",
                    title="Knowledge Update (Phase 1)",
                    level=LogLevel.INFO,
                )

        return new_steps, updated_knowledge

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

    def merge_compressed(self, steps: list[MemoryStep], knowledge: str = "") -> tuple[list[MemoryStep], str]:
        """Merge older compressed history steps by extracting knowledge.

        Instead of rewriting all compressed summaries into a single prose summary,
        this method extracts tagged XML knowledge from the older compressed steps
        and merges it into the existing knowledge store using ``merge_context()``.
        The merged compressed steps are then removed from the step list.

        If keep_compressed_steps is set, the most recent N compressed steps are
        preserved and only older ones are processed.

        Args:
            steps: Current list of memory steps.
            knowledge: Current knowledge string (tagged XML).

        Returns:
            Tuple of (new_steps, updated_knowledge).
        """
        start_time = time.time()

        compressed_steps = [step for step in steps if isinstance(step, CompressedHistoryStep)]

        if len(compressed_steps) <= 1:
            return steps, knowledge

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
            return steps, knowledge  # Not enough to merge

        # Skip merge if combined content is too small to be worth an LLM call
        pre_merge_chars = sum(len(step.summary) for step in steps_to_merge)
        if self.config.min_compression_chars > 0 and pre_merge_chars < self.config.min_compression_chars:
            if self.agent_logger:
                self.agent_logger.log_markdown(
                    content=f"Merge skipped: combined summaries ({pre_merge_chars:,} chars) is below minimum threshold ({self.config.min_compression_chars:,} chars)",
                    title="Compressed Step Merge Skipped",
                    level=LogLevel.INFO,
                )
            else:
                logger.info(f"Merge skipped: combined summaries ({pre_merge_chars} chars) < min_compression_chars ({self.config.min_compression_chars})")
            return steps, knowledge

        # Build knowledge extraction prompt and call LLM
        # post_steps: everything NOT being merged (kept compressed + live recent steps)
        # so the extractor knows what is still current vs already superseded
        merge_set_ids = set(id(s) for s in steps_to_merge)
        post_steps = [
            s for s in steps
            if id(s) not in merge_set_ids and not isinstance(s, (TaskStep, SystemPromptStep))
        ]
        existing_tag_names = list_xml_tag_names(knowledge)
        merge_prompt = create_knowledge_extraction_prompt(steps_to_merge, existing_tag_names, post_steps)

        try:
            merge_message = self.compression_model.generate(
                [
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{"type": "text", "text": merge_prompt}],
                    )
                ]
            )
            knowledge_updates = merge_message.content
            if isinstance(knowledge_updates, list):
                knowledge_updates = " ".join(
                    item.get("text", "") for item in knowledge_updates if isinstance(item, dict)
                )
        except Exception as e:
            logger.warning(f"Knowledge extraction failed, keeping original steps: {e}")
            return steps, knowledge

        # Apply the tagged XML diff to the knowledge store
        original_chars = sum(len(step.summary) for step in steps_to_merge)
        updated_knowledge = merge_context(knowledge, knowledge_updates) if knowledge_updates else knowledge

        # Accumulate metadata
        total_original_count = sum(step.original_step_count for step in steps_to_merge)

        # Remove merged compressed steps from the step list
        merge_set = set(id(step) for step in steps_to_merge)
        new_steps = [step for step in steps if not (isinstance(step, CompressedHistoryStep) and id(step) in merge_set)]

        # Log
        kept_count = len(compressed_steps) - len(steps_to_merge)
        knowledge_chars = len(updated_knowledge) if updated_knowledge else 0
        elapsed = time.time() - start_time
        if self.agent_logger:
            kept_msg = f" Kept {kept_count} recent compressed steps." if kept_count > 0 else ""
            tag_names = list_xml_tag_names(updated_knowledge)
            self.agent_logger.log_markdown(
                content=f"Extracted knowledge from {len(steps_to_merge)} compressed steps "
                f"({total_original_count} original steps, {original_chars:,} chars). "
                f"Knowledge store: {knowledge_chars:,} chars, sections: {tag_names}. "
                f"Elapsed: {elapsed:.1f}s.{kept_msg}",
                title="Knowledge Extraction",
                level=LogLevel.INFO,
            )
        else:
            kept_msg = f", kept {kept_count} recent" if kept_count > 0 else ""
            logger.info(
                f"Extracted knowledge from {len(steps_to_merge)} compressed steps "
                f"({total_original_count} original steps) into knowledge store "
                f"({knowledge_chars} chars){kept_msg}"
            )

        return new_steps, updated_knowledge


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
            agent.memory.steps, agent.memory.knowledge = compressor.compress(agent.memory.steps, agent.memory.knowledge)

        if compressor.should_merge_compressed(agent.memory.steps):
            agent.memory.steps, agent.memory.knowledge = compressor.merge_compressed(
                agent.memory.steps, agent.memory.knowledge
            )

    return compression_callback
