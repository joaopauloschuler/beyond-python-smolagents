# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Beyond Python Smolagents is a fork of HuggingFace's smolagents (v1.23.0) that extends the original framework with multi-language support and advanced problem-solving capabilities. The main additions are:
- `fast_solver`: Multi-agent parallel problem-solving that generates 3 independent solutions and synthesizes them
- `evolutive_problem_solver`: Iterative evolutionary refinement through analysis, comparison, mixing, and improvement cycles
- Extended tooling for file operations, source code analysis, and multi-language code execution
- Context compression: Automatic LLM-based summarization of older memory steps to manage context window size
- Browser integration: Playwright-based headed Chromium browser controllable from agent `<runcode>` blocks via `--browser` CLI flag

## Build & Development Commands

```bash
# Install package (with LiteLLM support)
pip install ./smolagents[litellm]

# Install for development
pip install ./smolagents[dev]

# Run all tests
make test
# Or directly:
pytest ./tests/

# Run a single test file
pytest ./tests/test_agents.py

# Run a specific test
pytest ./tests/test_agents.py::test_function_name

# Check code quality (linting + formatting)
make quality

# Auto-fix code style issues
make style
```

## Code Architecture

### Source Layout
The project uses a `src/smolagents/` layout. All source code is under `src/smolagents/`.

### Core Modules (inherited from smolagents)
- `agents.py` - Agent implementations: `CodeAgent`, `ToolCallingAgent`, `AgentRunner`
- `models.py` - LLM abstractions: `OpenAIModel`, `InferenceClientModel`, `LiteLLMModel`, `TransformersModel`
- `tools.py` - Tool framework: `Tool` base class, `@tool` decorator, JSON schema generation
- `local_python_executor.py` - Sandboxed Python code execution
- `remote_executors.py` - Remote execution: E2B, Docker, Modal, Blaxel, Wasm
- `default_tools.py` - Built-in tools: `FinalAnswerTool`, `DuckDuckGoSearchTool`, `VisitWebpageTool`
- `memory.py` - Agent memory: `AgentMemory`, step types (`ActionStep` with `actionstep_id`, `TaskStep`, `PlanningStep`)
- `bp_compression.py` - Context compression: `CompressionConfig`, `CompressedHistoryStep`, `ContextCompressor`
- `prompts/` - YAML prompt templates for different agent types

### Beyond Python (BP) Extensions
- `bp_cli.py` - Interactive CLI (`bpsa`): REPL and one-shot modes with CodeAgent, slash commands, token tracking, shell escapes (`!`, `!!`, `!!!`), `/alias`, `/redo`, auto-save
- `bp_tools.py` - Extended tool library: file I/O, source code analysis, OS commands, multi-language support (24+ languages including Pascal, PHP, C++, Java, Go, Rust)
- `bp_thinkers.py` - `Thinker` agent with multi-step reasoning (thoughts/plans/code sections)
- `bp_executors.py` - `LocalExecExecutor` for direct Python execution via `exec()`
- `bp_tools_browser.py` - Playwright browser integration: `BrowserManager`, `navigate`, `get_page_html`, `get_page_markdown`, `click`, `type_text`
- `bp_ad_infinitum.py` - Ad-infinitum CLI: autonomous task cycling from folders of `.md` (agent prompts), `.py`, and `.sh` (direct execution) files
- `bp_session.py` - Session persistence: save/load agent memory, counters, and stats to/from JSON files
- `bp_utils.py` - Utilities: code validation, tag fixing, file operations

### Agent Types
- **CodeAgent**: Generates Python code to solve tasks
- **ToolCallingAgent**: Uses structured tool calls (OpenAI function-calling style)
- **Thinker**: BP extension with reasoning framework supporting thoughts, plans, and code sections

### Tool System
Tools are functions decorated with `@tool` or classes inheriting from `Tool`. Key patterns:
- Tools define input/output schemas via type hints
- Sub-assistant classes (`Summarize`, `CoderSubassistant`, `InternetSearchSubassistant`) wrap agents as tools for delegation
- `add_base_tools=True` gives agents default tools (web search, file ops, Python interpreter)
- Context manipulation tools (`MoveActionStepToMemory`, `RetrieveActionStepFromMemory`, `SummarizeActionStep`) allow the agent to manually manage its own context by archiving, restoring, or LLM-summarizing individual step content by `actionstep_id`

### Execution Model
1. Agent receives task via `.run(task)`
2. Agent generates code/tool calls based on prompt template
3. Executor runs code (local sandboxed, local exec, or remote)
4. Results fed back to agent for next step
5. Agent calls `final_answer()` when complete

### Context Compression
Agents support automatic context compression to manage memory size during long-running tasks:
```python
from smolagents import CodeAgent, CompressionConfig

config = CompressionConfig(
    keep_recent_steps=5,       # Keep last N steps in full detail
    max_uncompressed_steps=10, # Compress when step count exceeds this
    max_compressed_steps=32,   # Merge compressed summaries when count exceeds this (0=disabled)
    keep_compressed_steps=22,  # Keep last N compressed summaries during merge
    compression_model=None,    # Optional: use cheaper model for compression
    min_compression_chars=4096,# Skip compression if content is below this (0=disabled)
)

agent = CodeAgent(tools=[...], model=model, compression_config=config)
```
When enabled, older steps are automatically summarized via LLM while preserving recent steps and critical context (task, errors, final answers). When compressed summaries accumulate beyond `max_compressed_steps`, the older ones are merged while `keep_compressed_steps` most recent summaries are preserved at full fidelity.

## Key Files for Common Tasks

| Task | Files |
|------|-------|
| Add new tool | `src/smolagents/bp_tools.py` or `src/smolagents/default_tools.py` |
| Modify agent behavior | `src/smolagents/agents.py` |
| Change prompts | `src/smolagents/prompts/*.yaml` |
| Add LLM provider | `src/smolagents/models.py` |
| Modify Python execution | `src/smolagents/local_python_executor.py` |
| Add remote executor | `src/smolagents/remote_executors.py` |
| Configure context compression | `src/smolagents/bp_compression.py` |
| Modify CLI (`bpsa`) | `src/smolagents/bp_cli.py` |
| Modify browser integration | `src/smolagents/bp_tools_browser.py` |
| Modify ad-infinitum task cycling | `src/smolagents/bp_ad_infinitum.py` |
| Modify session save/load | `src/smolagents/bp_session.py` |
| Modify context manipulation tools | `src/smolagents/bp_tools.py` (`MoveActionStepToMemory`, `RetrieveActionStepFromMemory`, `SummarizeActionStep`) |

## Testing

Tests are in `./tests/`. Key test files:
- `test_agents.py` - Agent behavior tests
- `test_bp_context_tools.py` - Beyond Python tools tests
- `test_bp_session.py` - Session save/load tests
- `test_compression.py` - Context compression tests
- `test_local_python_executor.py` - Python execution tests
- `test_models.py` - LLM integration tests
- `test_tools.py` - Tool framework tests

## Code Style

- Line length: 119 characters
- Linter: Ruff (rules: E, F, I, W)
- Import sorting: isort via Ruff, `smolagents` as first-party
- Pre-commit hooks configured for Ruff

## Branch Information

- Main branch for PRs: `v1.23-bp`
- Development happens on `development*` branches
