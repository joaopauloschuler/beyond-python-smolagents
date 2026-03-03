

# Context Compression & Knowledge Extraction

## Overview
A hybrid rolling summarization system for BPSA that compresses older memory steps via LLM summarization while keeping recent steps in full detail. Knowledge is extracted incrementally during compression and further refined when compressed summaries accumulate.

## Inspirations from Biology

The two-phase compression pipeline was designed from first principles, yet it converges
strikingly closely on the **Standard Model of Memory Consolidation** — the dominant
neuroscientific theory of how biological brains move experiences from short-term storage
into long-term knowledge. The parallels are not superficial; they appear to reflect deep
structural constraints that any system managing finite working memory over unbounded
experience must eventually solve. This convergence is a hypothesis, not a proven fact —
but the hypothesis is a strong one: finite capacity + unbounded experience + the need for
generalisation are universal pressures, and similar pressures tend to produce similar
solutions regardless of substrate.

### The Deepest Parallel

The entire two-phase design mirrors the **Standard Model of Memory Consolidation**:

```
Experience → Hippocampus (short-lived, detailed)
                ↓  (sleep / Phase 1)
           Compressed replay → early neocortex
                ↓  (deeper sleep / Phase 2)
           Abstract semantic knowledge → late neocortex
                ↓
           Hippocampus no longer needed for retrieval
```

Replace hippocampus with "action steps", early neocortex with "CompressedHistoryStep",
late neocortex with "knowledge store" — and you have BPSA's compression pipeline almost
exactly.

---

### 1. Working Memory vs. Long-Term Memory
* **BPSA:** Recent steps are kept in **full detail** (`keep_recent_steps`). Older steps are compressed into summaries.
* **Human mind:** The **prefrontal cortex** holds a small working memory buffer (~7±2 items, Miller 1956) in full resolution. Older experiences are consolidated and compressed by the hippocampus over time.

> *"Keep 40 recent steps in full" is literally what your brain does right now — you remember today in detail, last Tuesday as a blur.*

---

### 2. Sleep Consolidation → Phase 1 + Phase 2
* **BPSA:** Two-phase pipeline — Phase 1 compresses live steps + extracts knowledge; Phase 2 merges accumulated compressed steps into deeper knowledge.
* **Human mind:** Sleep has **two consolidation phases** — slow-wave sleep (SWS) replays episodic memories from hippocampus to neocortex (Phase 1 analog), and REM sleep is associated with abstracting and integrating those replays into semantic knowledge (Phase 2 analog). The analogy is functional: SWS and REM differ in their underlying neural mechanisms (sharp-wave ripples vs. theta oscillations) rather than being a clean "first pass / second pass" distinction, but the broad directionality — from detailed episodic replay toward abstract semantic integration — maps well onto BPSA's two phases.

> *Phase 2 in BPSA ("merge_compressed when they accumulate") is functionally analogous to the later sleep stages that refine, consolidate, and eventually render raw episodic traces unnecessary for retrieval.*

---

### 3. Episodic vs. Semantic Memory
* **BPSA:** `CompressedHistoryStep` = what happened (events, actions taken). `knowledge` store = what is currently true (facts, beliefs, current state).
* **Human mind:** **Episodic memory** = "I did X at time T." **Semantic memory** = "X is true." The brain explicitly separates these. Old episodic memories gradually convert to semantic ones — exactly what Phase 2 does.

> *"Compressed history = events/changes over time; knowledge = current beliefs/facts" — this is straight from cognitive psychology textbooks.*

---

### 4. Schemas / Semantic Networks → Tagged XML Knowledge
* **BPSA:** Knowledge stored as tagged XML sections (`<plan>`, `<key_findings>`, `<current_status>`). Sections (tags) can be added, updated, or deleted via diff operations.
* **Human mind:** Cognitive psychologists call these **schemas** — organised clusters of knowledge with labels and relationships, updated incrementally as new information arrives. The `merge_context()` add/update/delete operations mirror how schemas are revised.

---

### 5. Deliberate Belief Revision → Agent-Driven Knowledge Updates
* **BPSA:** The `update_knowledge` tool lets the *agent itself* explicitly revise its knowledge store at any point during live execution.
* **Human mind:** **Deliberate belief revision** — the conscious, intentional process of updating one's own knowledge when new evidence or reasoning warrants it. This is distinct from *metacognition* in the strict cognitive science sense (which additionally involves monitoring uncertainty and regulating reasoning strategies); what the agent does here is closer to deliberate note-taking and self-correction — updating a belief because a step's outcome has changed what is known to be true.


## Architecture

### Two-Phase Compression Pipeline

**Phase 1 — Step Compression + Knowledge Extraction:** Older action steps are summarized by the LLM into `CompressedHistoryStep` instances. The same LLM call also extracts knowledge updates, which are applied to the persistent knowledge store immediately. The LLM receives both the full compressed history (past events) and the full knowledge store (current facts) so it can avoid all duplication and propose corrections. Recent steps are kept in full detail.

**Phase 2 — Knowledge Refinement:** When compressed steps accumulate beyond a threshold, older ones are merged into the knowledge store via a separate LLM call. The merged compressed steps are then removed entirely. This phase refines and consolidates knowledge that may have been partially captured in Phase 1.

```
Steps accumulate → Phase 1: compress older steps
                     ↓
                   LLM produces <summary> + optional <knowledge_updates>
                     ↓                              ↓
                   CompressedHistoryStep    merge_context() → memory.knowledge
                     ↓
                   (when too many compressed steps accumulate)
                     ↓
                   Phase 2: extract knowledge from old compressed steps
                     ↓
                   merge_context() → memory.knowledge
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
    def compress(steps, knowledge) -> tuple[list[MemoryStep], str]
    def should_merge_compressed(steps) -> bool
    def merge_compressed(steps, knowledge) -> tuple[list[MemoryStep], str]
```

Key functions:
- `estimate_tokens(text)` — Character-based heuristic (~4 chars/token)
- `estimate_step_tokens(step)` — Token estimate for a memory step
- `should_preserve_step(step, config)` — Check if step must be kept
- `create_compression_prompt(steps, knowledge, existing_summaries)` — Build LLM prompt for step summarization with full context: existing compressed history (to avoid duplicating events) and knowledge store (current facts, updatable). Requests structured `<summary>` + optional `<knowledge_updates>` output
- `parse_compression_output(raw_output)` — Parse structured LLM output into `(summary, knowledge_updates)` with graceful fallback for unstructured output
- `create_knowledge_extraction_prompt(steps, tag_names)` — Build LLM prompt for Phase 2 knowledge extraction
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
- `/compress` command handles tuple return from `compress()`
- Environment variable configuration (see below)

### `tests/test_compression.py`
Tests for:
- `CompressedHistoryStep.to_messages()` and `dict()` serialization
- Token estimation functions
- `should_preserve_step()` logic
- `ContextCompressor.should_compress()` threshold behavior
- `ContextCompressor.compress()` — tuple return, knowledge extraction, fallback for unstructured output
- `parse_compression_output()` — structured output, summary-only, fallback, empty/None input
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

**Three sources of updates:**
1. **Phase 1 (automatic):** `compress()` extracts `<knowledge_updates>` from the same LLM call that produces the summary — knowledge starts accumulating from the very first compression cycle
2. **Phase 2 (automatic):** `merge_compressed()` extracts knowledge from old compressed summaries when they accumulate beyond the threshold — refines and consolidates
3. **Manual:** The `update_knowledge` tool lets the agent explicitly add/update/delete sections at any time

**`merge_context(existing, updates)` applies three operations:**
- `<tag>content</tag>` where tag exists → **UPDATE** (replace content)
- `<tag>content</tag>` where tag is new → **APPEND**
- `<tag/>` or `<tag></tag>` (self-closing/empty) → **DELETE**

**Injection:** Knowledge is inserted as a `<knowledge>...</knowledge>` USER message just before the last message in the LLM context, giving it high attention weight.

### Phase 1 Knowledge Extraction

During Phase 1 compression, the LLM receives:
- The full current knowledge store as `<current_knowledge>` context
- Instructions to output structured format:

```
<summary>
Concise summary of compressed steps...
</summary>
<knowledge_updates>
<tag>new or updated content</tag>
<obsolete_tag/>
</knowledge_updates>
```

The `parse_compression_output()` function handles parsing with graceful fallback:
- If `<summary>` tags present → extract summary and knowledge_updates separately
- If no `<summary>` tags → entire output becomes the summary (backwards compatible)
- If no `<knowledge_updates>` → no knowledge changes applied

This design means:
- **Zero extra LLM calls** — knowledge extraction piggybacks on the existing compression call
- **Higher fidelity** — Phase 1 has access to full original steps (not lossy summaries)
- **Immediate availability** — knowledge accumulates from the first compression, not after 32+ steps

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
- **Graceful fallback:** If compression LLM call fails, keep original steps and log warning. If LLM doesn't follow structured format, entire output becomes the summary with no knowledge changes.
- **Combined summary + knowledge in Phase 1:** Single LLM call produces both summary and knowledge updates. The LLM sees the full compressed history AND knowledge store so it can avoid all duplication. The prompt explains the distinction: compressed history = events/changes over time, knowledge = current beliefs/facts. Zero extra cost.
- **Two-phase design:** Phase 1 extracts knowledge from full original steps (high fidelity). Phase 2 refines/consolidates from compressed summaries when they accumulate. Both phases use `merge_context()` for consistent tagged XML operations.
- **Tagged XML for knowledge:** Simple, parseable format that supports incremental updates via diff operations
- **Knowledge placement:** Injected near end of context for high attention weight in transformer models
- **Min chars threshold:** Avoids wasting LLM calls on already-concise content

## Verification
1. Run existing tests: `pytest tests/test_memory.py tests/test_agents.py`
2. Run compression tests: `pytest tests/test_compression.py`
3. Manual test: Create agent with compression enabled, run multi-step task, verify memory gets compressed and knowledge accumulates from Phase 1


