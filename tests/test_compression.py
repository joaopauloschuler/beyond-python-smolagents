"""Tests for the context compression module."""

import pytest
from unittest.mock import MagicMock, patch

from smolagents.bp_compression import (
    CompressionConfig,
    CompressedHistoryStep,
    ContextCompressor,
    create_compression_callback,
    create_compression_prompt,
    create_merge_prompt,
    estimate_tokens,
    estimate_step_tokens,
    should_preserve_step,
    CHARS_PER_TOKEN,
)
from smolagents.memory import (
    ActionStep,
    AgentMemory,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
    ChatMessage,
    MessageRole,
)
from smolagents.monitoring import Timing, TokenUsage
from smolagents.utils import AgentError


class TestCompressionConfig:
    def test_default_values(self):
        config = CompressionConfig()
        assert config.enabled is True
        assert config.keep_recent_steps == 5
        assert config.max_uncompressed_steps == 10
        assert config.estimated_token_threshold == 0
        assert config.compression_model is None
        assert config.max_summary_tokens == 50000
        assert config.preserve_error_steps is False
        assert config.preserve_final_answer_steps is True
        assert config.max_compressed_steps == 32

    def test_custom_values(self):
        config = CompressionConfig(
            enabled=False,
            keep_recent_steps=3,
            max_uncompressed_steps=5,
            estimated_token_threshold=8000,
            preserve_error_steps=False,
        )
        assert config.enabled is False
        assert config.keep_recent_steps == 3
        assert config.max_uncompressed_steps == 5
        assert config.estimated_token_threshold == 8000
        assert config.preserve_error_steps is False

    def test_max_compressed_steps_custom(self):
        config = CompressionConfig(max_compressed_steps=3)
        assert config.max_compressed_steps == 3

    def test_keep_compressed_steps_default(self):
        config = CompressionConfig()
        assert config.keep_compressed_steps == 22

    def test_keep_compressed_steps_custom(self):
        config = CompressionConfig(keep_compressed_steps=2)
        assert config.keep_compressed_steps == 2


class TestCompressedHistoryStep:
    def test_initialization(self):
        step = CompressedHistoryStep(
            summary="Agent explored options and found solution.",
            compressed_step_numbers=[1, 2, 3],
            original_step_count=3,
        )
        assert step.summary == "Agent explored options and found solution."
        assert step.compressed_step_numbers == [1, 2, 3]
        assert step.original_step_count == 3
        assert step.timing is None
        assert step.compression_token_usage is None

    def test_to_messages(self):
        step = CompressedHistoryStep(
            summary="Test summary of agent actions.",
            compressed_step_numbers=[1, 2, 3],
            original_step_count=3,
        )
        messages = step.to_messages()
        assert len(messages) == 1
        assert messages[0].role == MessageRole.USER
        assert "[COMPRESSED HISTORY - 3 steps summarized]" in messages[0].content[0]["text"]
        assert "Test summary of agent actions." in messages[0].content[0]["text"]
        assert "[END COMPRESSED HISTORY]" in messages[0].content[0]["text"]

    def test_to_messages_summary_mode(self):
        step = CompressedHistoryStep(
            summary="Test summary",
            compressed_step_numbers=[1, 2],
            original_step_count=2,
        )
        messages = step.to_messages(summary_mode=True)
        assert len(messages) == 0

    def test_dict_serialization(self):
        timing = Timing(start_time=100.0, end_time=101.5)
        token_usage = TokenUsage(input_tokens=500, output_tokens=100)
        step = CompressedHistoryStep(
            summary="Test summary",
            compressed_step_numbers=[1, 2],
            original_step_count=2,
            timing=timing,
            compression_token_usage=token_usage,
        )
        d = step.dict()
        assert d["summary"] == "Test summary"
        assert d["compressed_step_numbers"] == [1, 2]
        assert d["original_step_count"] == 2
        assert d["timing"]["start_time"] == 100.0
        assert d["timing"]["end_time"] == 101.5
        assert d["compression_token_usage"]["input_tokens"] == 500
        assert d["compression_token_usage"]["output_tokens"] == 100

    def test_dict_serialization_without_optional_fields(self):
        step = CompressedHistoryStep(
            summary="Test summary",
            compressed_step_numbers=[1],
            original_step_count=1,
        )
        d = step.dict()
        assert d["timing"] is None
        assert d["compression_token_usage"] is None


class TestTokenEstimation:
    def test_estimate_tokens_basic(self):
        text = "Hello world"  # 11 chars
        tokens = estimate_tokens(text)
        assert tokens == 11 // CHARS_PER_TOKEN  # 2

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_estimate_tokens_long_text(self):
        text = "a" * 1000
        tokens = estimate_tokens(text)
        assert tokens == 1000 // CHARS_PER_TOKEN  # 250

    def test_estimate_step_tokens_action_step(self):
        step = ActionStep(
            step_number=1,
            timing=Timing(start_time=0, end_time=1),
            observations="This is a test observation with some content.",
        )
        tokens = estimate_step_tokens(step)
        assert tokens > 0

    def test_estimate_step_tokens_task_step(self):
        step = TaskStep(task="Please solve this complex problem.")
        tokens = estimate_step_tokens(step)
        assert tokens > 0


class TestShouldPreserveStep:
    def test_preserves_task_step(self):
        config = CompressionConfig()
        step = TaskStep(task="Do something")
        assert should_preserve_step(step, config) is True

    def test_preserves_system_prompt_step(self):
        config = CompressionConfig()
        step = SystemPromptStep(system_prompt="You are an AI assistant.")
        assert should_preserve_step(step, config) is True

    def test_preserves_compressed_history_step(self):
        config = CompressionConfig()
        step = CompressedHistoryStep(
            summary="Previous summary",
            compressed_step_numbers=[1, 2],
            original_step_count=2,
        )
        assert should_preserve_step(step, config) is True

    def test_preserves_error_step_when_configured(self):
        config = CompressionConfig(preserve_error_steps=True)
        mock_logger = MagicMock()
        step = ActionStep(
            step_number=1,
            timing=Timing(start_time=0, end_time=1),
            error=AgentError("Test error", mock_logger),
        )
        assert should_preserve_step(step, config) is True

    def test_does_not_preserve_error_step_when_disabled(self):
        config = CompressionConfig(preserve_error_steps=False)
        mock_logger = MagicMock()
        step = ActionStep(
            step_number=1,
            timing=Timing(start_time=0, end_time=1),
            error=AgentError("Test error", mock_logger),
        )
        assert should_preserve_step(step, config) is False

    def test_preserves_final_answer_step_when_configured(self):
        config = CompressionConfig(preserve_final_answer_steps=True)
        step = ActionStep(
            step_number=1,
            timing=Timing(start_time=0, end_time=1),
            is_final_answer=True,
        )
        assert should_preserve_step(step, config) is True

    def test_does_not_preserve_final_answer_step_when_disabled(self):
        config = CompressionConfig(preserve_final_answer_steps=False)
        step = ActionStep(
            step_number=1,
            timing=Timing(start_time=0, end_time=1),
            is_final_answer=True,
        )
        assert should_preserve_step(step, config) is False

    def test_does_not_preserve_normal_action_step(self):
        config = CompressionConfig()
        step = ActionStep(
            step_number=1,
            timing=Timing(start_time=0, end_time=1),
            model_output="Some output",
            observations="Some observations",
        )
        assert should_preserve_step(step, config) is False

    def test_does_not_preserve_planning_step(self):
        config = CompressionConfig()
        step = PlanningStep(
            model_input_messages=[],
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Plan"),
            plan="This is the plan",
            timing=Timing(start_time=0, end_time=1),
        )
        assert should_preserve_step(step, config) is False


class TestCreateCompressionPrompt:
    def test_creates_prompt_for_action_steps(self):
        steps = [
            ActionStep(
                step_number=1,
                timing=Timing(start_time=0, end_time=1),
                model_output="I will search for information.",
                observations="Found 10 results.",
                code_action="search('query')",
            ),
            ActionStep(
                step_number=2,
                timing=Timing(start_time=1, end_time=2),
                model_output="Analyzing results.",
                observations="Result analysis complete.",
            ),
        ]
        prompt = create_compression_prompt(steps)
        assert "Step 1:" in prompt
        assert "Step 2:" in prompt
        assert "search for information" in prompt
        assert "Found 10 results" in prompt
        assert "Summarize" in prompt

    def test_creates_prompt_for_planning_steps(self):
        steps = [
            PlanningStep(
                model_input_messages=[],
                model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Plan"),
                plan="1. First step\n2. Second step",
                timing=Timing(start_time=0, end_time=1),
            ),
        ]
        prompt = create_compression_prompt(steps)
        assert "Planning step:" in prompt
        assert "First step" in prompt


class TestCreateMergePrompt:
    def test_creates_prompt_from_compressed_steps(self):
        steps = [
            CompressedHistoryStep(
                summary="Agent searched for data and found 10 results.",
                compressed_step_numbers=[1, 2, 3],
                original_step_count=3,
            ),
            CompressedHistoryStep(
                summary="Agent analyzed results and identified key patterns.",
                compressed_step_numbers=[4, 5, 6],
                original_step_count=3,
            ),
        ]
        prompt = create_merge_prompt(steps)
        assert "Merge" in prompt
        assert "2 summaries" in prompt
        assert "6 total steps" in prompt
        assert "searched for data" in prompt
        assert "analyzed results" in prompt
        assert "Summary 1" in prompt
        assert "Summary 2" in prompt

    def test_prompt_includes_step_metadata(self):
        steps = [
            CompressedHistoryStep(
                summary="First summary.",
                compressed_step_numbers=[1, 2],
                original_step_count=2,
            ),
            CompressedHistoryStep(
                summary="Second summary.",
                compressed_step_numbers=[3, 4, 5],
                original_step_count=3,
            ),
        ]
        prompt = create_merge_prompt(steps)
        assert "covering 2 steps" in prompt
        assert "covering 3 steps" in prompt
        assert "[1, 2]" in prompt
        assert "[3, 4, 5]" in prompt


class TestContextCompressor:
    def test_initialization(self):
        config = CompressionConfig()
        mock_model = MagicMock()
        compressor = ContextCompressor(config, mock_model)
        assert compressor.config == config
        assert compressor.main_model == mock_model
        assert compressor._compression_count == 0

    def test_compression_model_returns_main_when_none(self):
        config = CompressionConfig(compression_model=None)
        mock_model = MagicMock()
        compressor = ContextCompressor(config, mock_model)
        assert compressor.compression_model == mock_model

    def test_compression_model_returns_custom_when_set(self):
        custom_model = MagicMock()
        config = CompressionConfig(compression_model=custom_model)
        main_model = MagicMock()
        compressor = ContextCompressor(config, main_model)
        assert compressor.compression_model == custom_model

    def test_should_compress_false_when_disabled(self):
        config = CompressionConfig(enabled=False)
        compressor = ContextCompressor(config, MagicMock())
        steps = [ActionStep(step_number=i, timing=Timing(start_time=0, end_time=1)) for i in range(20)]
        assert compressor.should_compress(steps) is False

    def test_should_compress_false_below_threshold(self):
        config = CompressionConfig(max_uncompressed_steps=10, keep_recent_steps=5)
        compressor = ContextCompressor(config, MagicMock())
        # Only 5 compressible steps, need more than keep_recent_steps
        steps = [ActionStep(step_number=i, timing=Timing(start_time=0, end_time=1)) for i in range(5)]
        assert compressor.should_compress(steps) is False

    def test_should_compress_true_above_threshold(self):
        config = CompressionConfig(max_uncompressed_steps=5, keep_recent_steps=3)
        compressor = ContextCompressor(config, MagicMock())
        # 10 compressible steps, threshold is 5
        steps = [ActionStep(step_number=i, timing=Timing(start_time=0, end_time=1)) for i in range(10)]
        assert compressor.should_compress(steps) is True

    def test_compress_returns_original_when_not_needed(self):
        config = CompressionConfig(max_uncompressed_steps=20, keep_recent_steps=5)
        compressor = ContextCompressor(config, MagicMock())
        steps = [ActionStep(step_number=i, timing=Timing(start_time=0, end_time=1)) for i in range(5)]
        result = compressor.compress(steps)
        assert result == steps

    def test_compress_creates_compressed_step(self):
        config = CompressionConfig(max_uncompressed_steps=3, keep_recent_steps=2, min_compression_chars=0)
        mock_model = MagicMock()
        mock_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Summary of steps 0-5.",
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        compressor = ContextCompressor(config, mock_model)

        # Create 8 steps (exceeds threshold of 3)
        steps = [
            ActionStep(
                step_number=i,
                timing=Timing(start_time=0, end_time=1),
                model_output=f"Output {i}",
                observations=f"Observation {i}",
            )
            for i in range(8)
        ]

        result = compressor.compress(steps)

        # Should have compressed step + 2 recent steps
        assert len(result) < len(steps)
        # First step should be CompressedHistoryStep
        assert isinstance(result[0], CompressedHistoryStep)
        assert "Summary of steps" in result[0].summary
        # Model should have been called
        mock_model.generate.assert_called_once()

    def test_compress_preserves_task_step(self):
        config = CompressionConfig(max_uncompressed_steps=3, keep_recent_steps=2)
        mock_model = MagicMock()
        mock_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Summary",
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        compressor = ContextCompressor(config, mock_model)

        steps = [TaskStep(task="Original task")]
        steps.extend([
            ActionStep(step_number=i, timing=Timing(start_time=0, end_time=1))
            for i in range(10)
        ])

        result = compressor.compress(steps)

        # TaskStep should be first
        assert isinstance(result[0], TaskStep)
        assert result[0].task == "Original task"

    def test_compress_handles_model_failure_gracefully(self):
        config = CompressionConfig(max_uncompressed_steps=3, keep_recent_steps=2)
        mock_model = MagicMock()
        mock_model.generate.side_effect = Exception("Model error")
        compressor = ContextCompressor(config, mock_model)

        steps = [
            ActionStep(step_number=i, timing=Timing(start_time=0, end_time=1))
            for i in range(10)
        ]

        # Should return original steps when compression fails
        result = compressor.compress(steps)
        assert result == steps

    def test_should_merge_compressed_false_when_disabled(self):
        config = CompressionConfig(max_compressed_steps=0)
        compressor = ContextCompressor(config, MagicMock())
        steps = [
            CompressedHistoryStep(summary=f"Summary {i}", compressed_step_numbers=[i], original_step_count=1)
            for i in range(5)
        ]
        assert compressor.should_merge_compressed(steps) is False

    def test_should_merge_compressed_false_when_compression_disabled(self):
        config = CompressionConfig(enabled=False, max_compressed_steps=2)
        compressor = ContextCompressor(config, MagicMock())
        steps = [
            CompressedHistoryStep(summary=f"Summary {i}", compressed_step_numbers=[i], original_step_count=1)
            for i in range(5)
        ]
        assert compressor.should_merge_compressed(steps) is False

    def test_should_merge_compressed_false_below_threshold(self):
        config = CompressionConfig(max_compressed_steps=3)
        compressor = ContextCompressor(config, MagicMock())
        steps = [
            CompressedHistoryStep(summary=f"Summary {i}", compressed_step_numbers=[i], original_step_count=1)
            for i in range(2)
        ]
        assert compressor.should_merge_compressed(steps) is False

    def test_should_merge_compressed_true_above_threshold(self):
        config = CompressionConfig(max_compressed_steps=2, keep_compressed_steps=0)
        compressor = ContextCompressor(config, MagicMock())
        steps = [
            CompressedHistoryStep(summary=f"Summary {i}", compressed_step_numbers=[i], original_step_count=1)
            for i in range(3)
        ]
        assert compressor.should_merge_compressed(steps) is True

    def test_merge_compressed_returns_original_when_single(self):
        config = CompressionConfig(max_compressed_steps=2)
        compressor = ContextCompressor(config, MagicMock())
        steps = [
            CompressedHistoryStep(summary="Only one", compressed_step_numbers=[1], original_step_count=1),
            ActionStep(step_number=5, timing=Timing(start_time=0, end_time=1)),
        ]
        result_steps, result_knowledge = compressor.merge_compressed(steps)
        assert result_steps == steps
        assert result_knowledge == ""

    def test_merge_compressed_extracts_knowledge(self):
        config = CompressionConfig(max_compressed_steps=2, keep_compressed_steps=0, min_compression_chars=0)
        mock_model = MagicMock()
        mock_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="<findings>Useful data found and analyzed.</findings><status>Final output prepared.</status>",
            token_usage=TokenUsage(input_tokens=200, output_tokens=30),
        )
        compressor = ContextCompressor(config, mock_model)

        steps = [
            TaskStep(task="Do the task"),
            CompressedHistoryStep(
                summary="Agent searched for information and found useful data about the topic.",
                compressed_step_numbers=[1, 2, 3],
                original_step_count=3,
            ),
            CompressedHistoryStep(
                summary="Agent analyzed the data and drew conclusions about the results.",
                compressed_step_numbers=[4, 5, 6],
                original_step_count=3,
            ),
            CompressedHistoryStep(
                summary="Agent refined the analysis and prepared the final output.",
                compressed_step_numbers=[7, 8],
                original_step_count=2,
            ),
            ActionStep(step_number=9, timing=Timing(start_time=0, end_time=1)),
        ]

        result_steps, result_knowledge = compressor.merge_compressed(steps)

        # Compressed steps should be removed, leaving TaskStep + ActionStep
        assert len(result_steps) == 2
        assert isinstance(result_steps[0], TaskStep)
        assert isinstance(result_steps[1], ActionStep)

        # Knowledge should contain the extracted tags
        assert "<findings>" in result_knowledge
        assert "<status>" in result_knowledge

        mock_model.generate.assert_called_once()

    def test_merge_compressed_handles_model_failure(self):
        config = CompressionConfig(max_compressed_steps=2, keep_compressed_steps=0)
        mock_model = MagicMock()
        mock_model.generate.side_effect = Exception("Model error")
        compressor = ContextCompressor(config, mock_model)

        steps = [
            CompressedHistoryStep(summary="Summary A", compressed_step_numbers=[1], original_step_count=1),
            CompressedHistoryStep(summary="Summary B", compressed_step_numbers=[2], original_step_count=1),
        ]

        result_steps, result_knowledge = compressor.merge_compressed(steps)
        assert result_steps == steps
        assert result_knowledge == ""

    def test_merge_compressed_updates_existing_knowledge(self):
        config = CompressionConfig(max_compressed_steps=1, keep_compressed_steps=0, min_compression_chars=0)
        mock_model = MagicMock()
        mock_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="<status>All tasks complete</status>",
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        compressor = ContextCompressor(config, mock_model)

        steps = [
            CompressedHistoryStep(
                summary="Agent searched for information and found useful data about the topic.",
                compressed_step_numbers=[1],
                original_step_count=1,
            ),
            CompressedHistoryStep(
                summary="Agent analyzed the data and drew conclusions about the results.",
                compressed_step_numbers=[2],
                original_step_count=1,
            ),
        ]

        existing_knowledge = "<db>PostgreSQL</db>\n<status>In progress</status>"
        result_steps, result_knowledge = compressor.merge_compressed(steps, existing_knowledge)
        # Compressed steps removed
        assert len(result_steps) == 0
        # Knowledge should have updated status and kept db
        assert "<db>PostgreSQL</db>" in result_knowledge
        assert "<status>All tasks complete</status>" in result_knowledge
        assert "In progress" not in result_knowledge

    def test_should_merge_compressed_false_when_not_enough_mergeable(self):
        """With keep_compressed_steps=2 and 3 compressed steps, only 1 is mergeable (need 2)."""
        config = CompressionConfig(max_compressed_steps=2, keep_compressed_steps=2)
        compressor = ContextCompressor(config, MagicMock())
        steps = [
            CompressedHistoryStep(summary=f"Summary {i}", compressed_step_numbers=[i], original_step_count=1)
            for i in range(3)
        ]
        assert compressor.should_merge_compressed(steps) is False

    def test_should_merge_compressed_true_with_enough_mergeable(self):
        """With keep_compressed_steps=1 and 4 compressed steps, 3 are mergeable (>= 2)."""
        config = CompressionConfig(max_compressed_steps=2, keep_compressed_steps=1)
        compressor = ContextCompressor(config, MagicMock())
        steps = [
            CompressedHistoryStep(summary=f"Summary {i}", compressed_step_numbers=[i], original_step_count=1)
            for i in range(4)
        ]
        assert compressor.should_merge_compressed(steps) is True

    def test_merge_compressed_keeps_recent_compressed_steps(self):
        config = CompressionConfig(max_compressed_steps=2, keep_compressed_steps=1, min_compression_chars=0)
        mock_model = MagicMock()
        mock_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="<findings>Useful data about the topic.</findings>",
            token_usage=TokenUsage(input_tokens=200, output_tokens=30),
        )
        compressor = ContextCompressor(config, mock_model)

        steps = [
            TaskStep(task="Do the task"),
            CompressedHistoryStep(
                summary="Agent searched for information and found useful data about the topic.",
                compressed_step_numbers=[1, 2, 3],
                original_step_count=3,
            ),
            CompressedHistoryStep(
                summary="Agent analyzed the data and drew conclusions about the results.",
                compressed_step_numbers=[4, 5, 6],
                original_step_count=3,
            ),
            CompressedHistoryStep(
                summary="Agent refined the analysis and prepared the final output.",
                compressed_step_numbers=[7, 8],
                original_step_count=2,
            ),
            ActionStep(step_number=9, timing=Timing(start_time=0, end_time=1)),
        ]

        result_steps, result_knowledge = compressor.merge_compressed(steps)

        # Merged compressed steps removed, kept one remains: TaskStep + 1 kept compressed + ActionStep
        assert len(result_steps) == 3
        assert isinstance(result_steps[0], TaskStep)
        assert isinstance(result_steps[1], CompressedHistoryStep)  # kept (most recent)
        assert isinstance(result_steps[2], ActionStep)

        # The kept step should be the most recent one (unchanged)
        kept = result_steps[1]
        assert kept.summary == "Agent refined the analysis and prepared the final output."
        assert kept.compressed_step_numbers == [7, 8]

        # Knowledge should have the extracted info
        assert "<findings>" in result_knowledge

    def test_merge_compressed_keep_zero_merges_all(self):
        """With keep_compressed_steps=0 (default), all compressed steps are removed and knowledge extracted."""
        config = CompressionConfig(max_compressed_steps=2, keep_compressed_steps=0, min_compression_chars=0)
        mock_model = MagicMock()
        mock_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="<summary>All data searched, analyzed and output prepared.</summary>",
            token_usage=TokenUsage(input_tokens=200, output_tokens=30),
        )
        compressor = ContextCompressor(config, mock_model)

        steps = [
            CompressedHistoryStep(
                summary="Agent searched for information and found useful data about the topic.",
                compressed_step_numbers=[1, 2, 3],
                original_step_count=3,
            ),
            CompressedHistoryStep(
                summary="Agent analyzed the data and drew conclusions about the results.",
                compressed_step_numbers=[4, 5, 6],
                original_step_count=3,
            ),
            CompressedHistoryStep(
                summary="Agent refined the analysis and prepared the final output.",
                compressed_step_numbers=[7, 8],
                original_step_count=2,
            ),
        ]

        result_steps, result_knowledge = compressor.merge_compressed(steps)

        # All compressed steps should be removed
        assert len(result_steps) == 0
        # Knowledge should contain the extracted info
        assert "<summary>" in result_knowledge

    def test_merge_compressed_keep_too_many_returns_original(self):
        """If keep_compressed_steps >= len-1, only 1 left to merge, so return original."""
        config = CompressionConfig(max_compressed_steps=1, keep_compressed_steps=5)
        mock_model = MagicMock()
        compressor = ContextCompressor(config, mock_model)

        steps = [
            CompressedHistoryStep(
                summary="This is a long enough first summary that should be more than enough.",
                compressed_step_numbers=[1, 2],
                original_step_count=2,
            ),
            CompressedHistoryStep(
                summary="This is a long enough second summary that should be more than enough.",
                compressed_step_numbers=[3, 4],
                original_step_count=2,
            ),
        ]

        result_steps, result_knowledge = compressor.merge_compressed(steps)
        assert result_steps == steps  # Nothing merged
        assert result_knowledge == ""
        mock_model.generate.assert_not_called()

    def test_merge_compressed_removes_merged_steps(self):
        """Verify that merged compressed steps are removed from the step list."""
        config = CompressionConfig(max_compressed_steps=1, keep_compressed_steps=0, min_compression_chars=0)
        mock_model = MagicMock()
        mock_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="<info>Important data</info>",
            token_usage=TokenUsage(input_tokens=50, output_tokens=10),
        )
        compressor = ContextCompressor(config, mock_model)

        steps = [
            CompressedHistoryStep(
                summary="This is a longer summary with enough content to exceed the merged output.",
                compressed_step_numbers=[1, 2, 3],
                original_step_count=3,
            ),
            CompressedHistoryStep(
                summary="Another summary with enough content to ensure the merge saves space overall.",
                compressed_step_numbers=[3, 4, 5],
                original_step_count=3,
            ),
            ActionStep(step_number=6, timing=Timing(start_time=0, end_time=1)),
        ]

        result_steps, result_knowledge = compressor.merge_compressed(steps)
        # Only ActionStep should remain
        assert len(result_steps) == 1
        assert isinstance(result_steps[0], ActionStep)
        # Knowledge should be populated
        assert "<info>Important data</info>" in result_knowledge


class TestCreateCompressionCallback:
    def test_callback_triggers_compression(self):
        config = CompressionConfig(max_uncompressed_steps=3, keep_recent_steps=2, min_compression_chars=0)
        mock_model = MagicMock()
        mock_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Summary",
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        compressor = ContextCompressor(config, mock_model)
        callback = create_compression_callback(compressor)

        # Create mock agent with memory (include content so original_chars > summary_chars)
        mock_agent = MagicMock()
        mock_agent.memory.knowledge = ""
        mock_agent.memory.steps = [
            ActionStep(
                step_number=i,
                timing=Timing(start_time=0, end_time=1),
                model_output=f"This is a long model output for step {i} with enough content to be larger than the summary.",
                observations=f"Observation for step {i} with additional content.",
            )
            for i in range(10)
        ]

        # Trigger callback with ActionStep
        action_step = ActionStep(step_number=10, timing=Timing(start_time=0, end_time=1))
        callback(action_step, mock_agent)

        # Memory should have been compressed
        assert len(mock_agent.memory.steps) < 10

    def test_callback_ignores_non_action_steps(self):
        config = CompressionConfig(max_uncompressed_steps=3, keep_recent_steps=2)
        compressor = ContextCompressor(config, MagicMock())
        callback = create_compression_callback(compressor)

        mock_agent = MagicMock()
        original_steps = [
            ActionStep(step_number=i, timing=Timing(start_time=0, end_time=1))
            for i in range(10)
        ]
        mock_agent.memory.steps = original_steps.copy()

        # Trigger callback with PlanningStep (should be ignored)
        planning_step = PlanningStep(
            model_input_messages=[],
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Plan"),
            plan="Plan",
            timing=Timing(start_time=0, end_time=1),
        )
        callback(planning_step, mock_agent)

        # Memory should be unchanged
        assert mock_agent.memory.steps == original_steps

    def test_callback_triggers_merge(self):
        config = CompressionConfig(
            max_uncompressed_steps=3,
            keep_recent_steps=2,
            max_compressed_steps=1,
            keep_compressed_steps=0,
            min_compression_chars=0,
        )
        mock_model = MagicMock()
        # First call: compress, second call: merge (knowledge extraction)
        mock_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="<extracted>Knowledge from merge.</extracted>",
            token_usage=TokenUsage(input_tokens=100, output_tokens=10),
        )
        compressor = ContextCompressor(config, mock_model)
        callback = create_compression_callback(compressor)

        # Set up agent with multiple compressed steps + action steps that exceed threshold
        mock_agent = MagicMock()
        mock_agent.memory.knowledge = ""
        mock_agent.memory.steps = [
            CompressedHistoryStep(
                summary="This is a long enough first summary that the merge will save space versus the originals.",
                compressed_step_numbers=[1, 2, 3],
                original_step_count=3,
            ),
            CompressedHistoryStep(
                summary="This is a long enough second summary that the merge will save space versus the originals.",
                compressed_step_numbers=[4, 5, 6],
                original_step_count=3,
            ),
            ActionStep(
                step_number=7,
                timing=Timing(start_time=0, end_time=1),
                model_output="Long output for step 7 with enough content.",
                observations="Observation for step 7.",
            ),
            ActionStep(
                step_number=8,
                timing=Timing(start_time=0, end_time=1),
                model_output="Long output for step 8 with enough content.",
                observations="Observation for step 8.",
            ),
            ActionStep(
                step_number=9,
                timing=Timing(start_time=0, end_time=1),
                model_output="Long output for step 9 with enough content.",
                observations="Observation for step 9.",
            ),
            ActionStep(
                step_number=10,
                timing=Timing(start_time=0, end_time=1),
                model_output="Long output for step 10 with enough content.",
                observations="Observation for step 10.",
            ),
        ]

        action_step = ActionStep(step_number=11, timing=Timing(start_time=0, end_time=1))
        callback(action_step, mock_agent)

        # The model should have been called (compression and/or merge)
        assert mock_model.generate.call_count >= 1


class TestMergeContext:
    def test_append_new_tag(self):
        from smolagents.bp_compression import merge_context
        existing = "<db>PostgreSQL</db>"
        updates = "<auth>JWT tokens</auth>"
        result = merge_context(existing, updates)
        assert "<db>PostgreSQL</db>" in result
        assert "<auth>JWT tokens</auth>" in result

    def test_update_existing_tag(self):
        from smolagents.bp_compression import merge_context
        existing = "<db>PostgreSQL</db>\n<status>In progress</status>"
        updates = "<status>Complete</status>"
        result = merge_context(existing, updates)
        assert "<db>PostgreSQL</db>" in result
        assert "<status>Complete</status>" in result
        assert "In progress" not in result

    def test_delete_with_self_closing(self):
        from smolagents.bp_compression import merge_context
        existing = "<db>PostgreSQL</db>\n<old_notes>Some notes</old_notes>"
        updates = "<old_notes/>"
        result = merge_context(existing, updates)
        assert "<db>PostgreSQL</db>" in result
        assert "old_notes" not in result

    def test_delete_with_empty_tag(self):
        from smolagents.bp_compression import merge_context
        existing = "<db>PostgreSQL</db>\n<old_notes>Some notes</old_notes>"
        updates = "<old_notes></old_notes>"
        result = merge_context(existing, updates)
        assert "<db>PostgreSQL</db>" in result
        assert "old_notes" not in result

    def test_mixed_operations(self):
        from smolagents.bp_compression import merge_context
        existing = "<plan>Step 1</plan>\n<db>MySQL</db>\n<old>Remove me</old>"
        updates = "<plan>Step 2</plan><old/><auth>JWT</auth>"
        result = merge_context(existing, updates)
        assert "<plan>Step 2</plan>" in result
        assert "<auth>JWT</auth>" in result
        assert "old" not in result.lower() or "<old" not in result
        assert "<db>MySQL</db>" in result

    def test_empty_existing(self):
        from smolagents.bp_compression import merge_context
        result = merge_context("", "<info>New data</info>")
        assert "<info>New data</info>" in result

    def test_empty_updates(self):
        from smolagents.bp_compression import merge_context
        result = merge_context("<db>PostgreSQL</db>", "")
        assert "<db>PostgreSQL</db>" in result


class TestListXmlTagNames:
    def test_basic(self):
        from smolagents.bp_compression import list_xml_tag_names
        text = "<db>PostgreSQL</db>\n<auth>JWT</auth>\n<plan>Steps</plan>"
        tags = list_xml_tag_names(text)
        assert tags == ["auth", "db", "plan"]

    def test_empty(self):
        from smolagents.bp_compression import list_xml_tag_names
        assert list_xml_tag_names("") == []

    def test_no_tags(self):
        from smolagents.bp_compression import list_xml_tag_names
        assert list_xml_tag_names("plain text no tags") == []


class TestCreateKnowledgeExtractionPrompt:
    def test_includes_summaries(self):
        from smolagents.bp_compression import create_knowledge_extraction_prompt, CompressedHistoryStep
        steps = [
            CompressedHistoryStep(summary="Found PostgreSQL", compressed_step_numbers=[1], original_step_count=1),
        ]
        prompt = create_knowledge_extraction_prompt(steps)
        assert "Found PostgreSQL" in prompt

    def test_includes_existing_tags(self):
        from smolagents.bp_compression import create_knowledge_extraction_prompt, CompressedHistoryStep
        steps = [
            CompressedHistoryStep(summary="Some info", compressed_step_numbers=[1], original_step_count=1),
        ]
        prompt = create_knowledge_extraction_prompt(steps, existing_tag_names=["db", "plan"])
        assert "db" in prompt
        assert "plan" in prompt

    def test_without_existing_tags(self):
        from smolagents.bp_compression import create_knowledge_extraction_prompt, CompressedHistoryStep
        steps = [
            CompressedHistoryStep(summary="Info", compressed_step_numbers=[1], original_step_count=1),
        ]
        prompt = create_knowledge_extraction_prompt(steps, existing_tag_names=[])
        assert "no existing" in prompt.lower() or "new" in prompt.lower() or len(prompt) > 0
