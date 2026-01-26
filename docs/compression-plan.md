# Context Compression Implementation Plan

## Overview
Add a hybrid rolling summarization system to smolagents that compresses older memory steps via LLM summarization while keeping recent steps in full detail.

## Files to Create/Modify

### 1. CREATE: `src/smolagents/bp_compression.py`
New module containing all compression logic:

```python
@dataclass
class CompressionConfig:
    enabled: bool = True
    keep_recent_steps: int = 5          # Recent steps to keep in full
    step_count_threshold: int = 10      # Compress when exceeds this
    estimated_token_threshold: int = 0  # Token-based trigger (0=disabled)
    compression_model: Model | None = None  # Cheaper model for compression
    preserve_error_steps: bool = True
    preserve_final_answer_steps: bool = True

@dataclass
class CompressedHistoryStep(MemoryStep):
    summary: str
    compressed_step_numbers: list[int]
    original_step_count: int
    # Implements to_messages() returning summary as USER message

class ContextCompressor:
    def should_compress(steps) -> bool
    def compress(steps) -> list[MemoryStep]  # Returns new list with compressed history
```

Key functions:
- `estimate_tokens(text)` - Character-based heuristic (~4 chars/token)
- `should_preserve_step(step, config)` - Check if step must be kept
- `create_compression_prompt(steps)` - Build LLM prompt for summarization
- `create_compression_callback(compressor)` - Callback for automatic triggering

### 2. MODIFY: `src/smolagents/agents.py`
Add to `MultiStepAgent.__init__` (around line 360):
- New parameter: `compression_config: CompressionConfig | None = None`
- New method: `_setup_compression()` that registers compression callback

Integration via existing callback system (lines 425-443) - no changes to core agent loop.

### 3. MODIFY: `src/smolagents/__init__.py`
Add exports:
```python
from .bp_compression import CompressionConfig, CompressedHistoryStep, ContextCompressor
```

### 4. CREATE: `tests/test_compression.py`
Tests for:
- `CompressedHistoryStep.to_messages()` and `dict()` serialization
- Token estimation functions
- `should_preserve_step()` logic
- `ContextCompressor.should_compress()` threshold behavior
- Integration test with mock model

## Implementation Sequence

1. Create `bp_compression.py` with all classes and functions
2. Modify `MultiStepAgent.__init__` to accept `compression_config`
3. Add `_setup_compression()` method to register callback
4. Update `__init__.py` exports
5. Create test file
6. Run tests to verify

## Usage Example
```python
from smolagents import CodeAgent, CompressionConfig, LiteLLMModel

config = CompressionConfig(
    keep_recent_steps=5,
    step_count_threshold=8,
    compression_model=LiteLLMModel(model_id="gpt-4o-mini"),  # Cheap model
)

agent = CodeAgent(
    tools=[...],
    model=main_model,
    compression_config=config,
)
```

## Design Decisions
- **New file vs existing**: New `bp_compression.py` keeps related logic together, follows pattern of `monitoring.py`
- **Callback-based**: Uses existing callback system for clean integration without modifying agent loop
- **Token estimation**: Character heuristic (4 chars/token) since no proactive token counting exists
- **Graceful fallback**: If compression LLM call fails, keep original steps and log warning

## Verification
1. Run existing tests: `pytest tests/test_memory.py tests/test_agents.py`
2. Run new tests: `pytest tests/test_compression.py`
3. Manual test: Create agent with compression enabled, run multi-step task, verify memory gets compressed
