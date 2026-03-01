
# Context Compression & Knowledge Extraction

## Overview
A hybrid rolling summarization system for smolagents that compresses older memory steps via LLM summarization while keeping recent steps in full detail. When compressed summaries accumulate, they are further distilled into a persistent knowledge store using tagged XML.

## Architecture

### Two-Phase Compression Pipeline

**Phase 1 — Step Compression:** Older action steps are summarized by the LLM into `CompressedHistoryStep` instances. Recent steps are kept in full detail.

**Phase 2 — Knowledge Extraction:** When compressed steps accumulate beyond a threshold, older ones are merged into a persistent `memory.knowledge` store as tagged XML. The merged compressed steps are then removed entirely.

```
Steps accumulate → compress older steps → CompressedHistoryStep summaries
                                              ↓ (when too many accumulate)
                                    Extract knowledge via LLM
                                              ↓
                                    merge_context() into memory.knowledge
                                              ↓
                                    Old compressed steps removed
                                              ↓
                                    Knowledge injected into LLM context
                                    as <knowledge>...</knowledge> message
```

## Files

### `src/smolagents/bp_compression.py`
All compression and knowledge logic:

```python
@dataclass
class CompressionConfig:
    enabled: bool = True
    keep_recent_steps: int = 5            # Recent steps to keep in full
    max_uncompressed_steps: int = 10      # Compress when exceeds this
    estimated_token_threshold: int = 0    # Token-based trigger (0=disabled)
    compression_model: Model | None = None  # Separate model for compression (None=use main)
    max_summary_tokens: int = 50000       # Max tokens for generated summary
    preserve_error_steps: bool = False    # Keep error steps uncompressed
    preserve_final_answer_steps: bool = True  # Keep final_answer steps uncompressed
    max_compressed_steps: int = 32        # Merge compressed steps when exceeds this
    keep_compressed_steps: int = 22       # Recent compressed steps to keep during merge
    min_compression_chars: int = 4096     # Skip compression if content below this

@dataclass
class CompressedHistoryStep(MemoryStep):
    summary: str
    compressed_step_numbers: list[int]
    original_step_count: int
    timing: Timing | None
    compression_token_usage: TokenUsage | None
    # to_messages() renders as [COMPRESSED HISTORY - N steps summarized]

class ContextCompressor:
    def should_compress(steps) -> bool
    def compress(steps) -> list[MemoryStep]
    def should_merge_compressed(steps) -> bool
    def merge_compressed(steps, knowledge) -> tuple[list[MemoryStep], str]
```

Key functions:
- `estimate_tokens(text)` — Character-based heuristic (~4 chars/token)
- `estimate_step_tokens(step)` — Token estimate for a memory step
- `should_preserve_step(step, config)` — Check if step must be kept
- `create_compression_prompt(steps)` — Build LLM prompt for step summarization
- `create_knowledge_extraction_prompt(steps, tag_names)` — Build LLM prompt for knowledge extraction
- `create_merge_prompt(steps)` — Build prompt for merging compressed steps
- `list_xml_tag_names(text)` — Extract XML tag names from a string
- `merge_context(existing, updates)` — Apply tagged XML diff (add/update/delete)
- `create_compression_callback(compressor)` — Callback for automatic triggering

### `src/smolagents/agents.py`
Integration in `MultiStepAgent`:
- `__init__` accepts `compression_config: CompressionConfig | None = None`
- `_setup_compression()` registers the compression callback
- `write_memory_to_messages()` injects `memory.knowledge` as a `<knowledge>` message just before the last message in context
- System prompt log line shows Context and Knowledge char counts

### `src/smolagents/memory.py`
- `AgentMemory.knowledge: str = ""` — Persistent knowledge store (tagged XML)
- Reset on `memory.reset()`

### `src/smolagents/bp_tools.py`
- `UpdateKnowledge` tool — Allows the agent to explicitly update its knowledge store via `update_knowledge(updates='<tag>content</tag>')`

### `src/smolagents/bp_cli.py`
- `print_turn_summary()` shows Context and Knowledge char counts
- Environment variable configuration (see below)

### `tests/test_compression.py`
Tests for:
- `CompressedHistoryStep.to_messages()` and `dict()` serialization
- Token estimation functions
- `should_preserve_step()` logic
- `ContextCompressor.should_compress()` threshold behavior
- `merge_context()` add/update/delete operations
- `list_xml_tag_names()` extraction
- Integration test with mock model

## Knowledge Store

The knowledge store (`memory.knowledge`) is a plain string of tagged XML:

```xml
<plan>1. Setup done
2. Now implementing API</plan>
<key_findings>The database uses PostgreSQL 14 with pgvector extension</key_findings>
<current_status>API endpoints implemented, testing in progress</current_status>
```

**Two sources of updates:**
1. **Automatic:** `merge_compressed()` extracts knowledge from old compressed summaries (Phase 2)
2. **Manual:** The `update_knowledge` tool lets the agent explicitly add/update/delete sections

**`merge_context(existing, updates)` applies three operations:**
- `<tag>content</tag>` where tag exists → **UPDATE** (replace content)
- `<tag>content</tag>` where tag is new → **APPEND**
- `<tag/>` or `<tag></tag>` (self-closing/empty) → **DELETE**

**Injection:** Knowledge is inserted as a `<knowledge>...</knowledge>` USER message just before the last message in the LLM context, giving it high attention weight.

## BPSA CLI Configuration

Environment variables (with defaults used by the CLI):

| Variable | Default | Description |
|---|---|---|
| `BPSA_COMPRESSION_ENABLED` | `1` | Enable compression |
| `BPSA_COMPRESSION_KEEP_RECENT_STEPS` | `40` | Recent steps to keep uncompressed |
| `BPSA_COMPRESSION_MAX_UNCOMPRESSED_STEPS` | `50` | Trigger threshold for compression |
| `BPSA_COMPRESSION_KEEP_COMPRESSED_STEPS` | `80` | Compressed steps to keep on merge |
| `BPSA_COMPRESSION_MAX_COMPRESSED_STEPS` | `120` | Trigger threshold for merge |
| `BPSA_COMPRESSION_TOKEN_THRESHOLD` | `0` | Token-based trigger (0=disabled) |
| `BPSA_COMPRESSION_MODEL` | same as main | Model ID for compression |
| `BPSA_COMPRESSION_MAX_SUMMARY_TOKENS` | `50000` | Max tokens in summary |
| `BPSA_COMPRESSION_PRESERVE_ERROR_STEPS` | `0` | Keep error steps uncompressed |
| `BPSA_COMPRESSION_PRESERVE_FINAL_ANSWER_STEPS` | `1` | Keep final_answer steps |
| `BPSA_COMPRESSION_MIN_CHARS` | `4096` | Min chars before compressing |

Note: The CLI defaults differ from `CompressionConfig` defaults to suit interactive use (more steps kept).

## Usage Example

### Programmatic
```python
from smolagents import CodeAgent, CompressionConfig, LiteLLMModel

config = CompressionConfig(
    keep_recent_steps=5,
    max_uncompressed_steps=10,
    compression_model=LiteLLMModel(model_id="gpt-4o-mini"),  # Cheaper model
    max_compressed_steps=32,
    keep_compressed_steps=22,
)

agent = CodeAgent(
    tools=[...],
    model=main_model,
    compression_config=config,
)
```

### BPSA CLI
```bash
export BPSA_COMPRESSION_ENABLED=1
export BPSA_COMPRESSION_KEEP_RECENT_STEPS=40
export BPSA_COMPRESSION_MAX_UNCOMPRESSED_STEPS=50
bpsa
```

## Design Decisions
- **New file vs existing:** `bp_compression.py` keeps all compression/knowledge logic together, follows pattern of `monitoring.py`
- **Callback-based:** Uses existing callback system for clean integration without modifying the agent loop
- **Token estimation:** Character heuristic (4 chars/token) since no proactive token counting exists
- **Graceful fallback:** If compression or knowledge extraction LLM call fails, keep original steps and log warning
- **Two-phase design:** Step compression (lossy but retains prose summaries) feeds into knowledge extraction (structured XML) for long-term retention
- **Tagged XML for knowledge:** Simple, parseable format that supports incremental updates via diff operations
- **Knowledge placement:** Injected near end of context for high attention weight in transformer models
- **Min chars threshold:** Avoids wasting LLM calls on already-concise content

## Verification
1. Run existing tests: `pytest tests/test_memory.py tests/test_agents.py`
2. Run compression tests: `pytest tests/test_compression.py`
3. Manual test: Create agent with compression enabled, run multi-step task, verify memory gets compressed and knowledge accumulates
