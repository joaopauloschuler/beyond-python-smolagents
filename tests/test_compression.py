"""Tests for the context compression module."""

import pytest
from unittest.mock import MagicMock, patch

from smolagents.compression import (
    CompressionConfig,
    CompressedHistoryStep,
    ContextCompressor,
    create_compression_callback,
    create_compression_prompt,
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
        assert config.step_count_threshold == 10
        assert config.estimated_token_threshold == 0
        assert config.compression_model is None
        assert config.max_summary_tokens == 500
        assert config.preserve_error_steps is True
        assert config.preserve_final_answer_steps is True

    def test_custom_values(self):
        config = CompressionConfig(
            enabled=False,
            keep_recent_steps=3,
            step_count_threshold=5,
            estimated_token_threshold=8000,
            preserve_error_steps=False,
        )
        assert config.enabled is False
        assert config.keep_recent_steps == 3
        assert config.step_count_threshold == 5
        assert config.estimated_token_threshold == 8000
        assert config.preserve_error_steps is False


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
        config = CompressionConfig(step_count_threshold=10, keep_recent_steps=5)
        compressor = ContextCompressor(config, MagicMock())
        # Only 5 compressible steps, need more than keep_recent_steps
        steps = [ActionStep(step_number=i, timing=Timing(start_time=0, end_time=1)) for i in range(5)]
        assert compressor.should_compress(steps) is False

    def test_should_compress_true_above_threshold(self):
        config = CompressionConfig(step_count_threshold=5, keep_recent_steps=3)
        compressor = ContextCompressor(config, MagicMock())
        # 10 compressible steps, threshold is 5
        steps = [ActionStep(step_number=i, timing=Timing(start_time=0, end_time=1)) for i in range(10)]
        assert compressor.should_compress(steps) is True

    def test_compress_returns_original_when_not_needed(self):
        config = CompressionConfig(step_count_threshold=20, keep_recent_steps=5)
        compressor = ContextCompressor(config, MagicMock())
        steps = [ActionStep(step_number=i, timing=Timing(start_time=0, end_time=1)) for i in range(5)]
        result = compressor.compress(steps)
        assert result == steps

    def test_compress_creates_compressed_step(self):
        config = CompressionConfig(step_count_threshold=3, keep_recent_steps=2)
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
        config = CompressionConfig(step_count_threshold=3, keep_recent_steps=2)
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
        config = CompressionConfig(step_count_threshold=3, keep_recent_steps=2)
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


class TestCreateCompressionCallback:
    def test_callback_triggers_compression(self):
        config = CompressionConfig(step_count_threshold=3, keep_recent_steps=2)
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
        config = CompressionConfig(step_count_threshold=3, keep_recent_steps=2)
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
