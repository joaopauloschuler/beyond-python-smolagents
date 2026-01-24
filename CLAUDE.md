# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Beyond Python Smolagents is a fork of HuggingFace's smolagents (v1.23.0) that extends the original framework with multi-language support and advanced problem-solving capabilities. The main additions are:
- `fast_solver`: Multi-agent parallel problem-solving that generates 3 independent solutions and synthesizes them
- `evolutive_problem_solver`: Iterative evolutionary refinement through analysis, comparison, mixing, and improvement cycles
- Extended tooling for file operations, source code analysis, and multi-language code execution

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
- `prompts/` - YAML prompt templates for different agent types

### Beyond Python (BP) Extensions
- `bp_tools.py` - Extended tool library: file I/O, source code analysis, OS commands, multi-language support (24+ languages including Pascal, PHP, C++, Java, Go, Rust)
- `bp_thinkers.py` - `Thinker` agent with multi-step reasoning (thoughts/plans/code sections)
- `bp_executors.py` - `LocalExecExecutor` for direct Python execution via `exec()`
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

### Execution Model
1. Agent receives task via `.run(task)`
2. Agent generates code/tool calls based on prompt template
3. Executor runs code (local sandboxed, local exec, or remote)
4. Results fed back to agent for next step
5. Agent calls `final_answer()` when complete

## Key Files for Common Tasks

| Task | Files |
|------|-------|
| Add new tool | `src/smolagents/bp_tools.py` or `src/smolagents/default_tools.py` |
| Modify agent behavior | `src/smolagents/agents.py` |
| Change prompts | `src/smolagents/prompts/*.yaml` |
| Add LLM provider | `src/smolagents/models.py` |
| Modify Python execution | `src/smolagents/local_python_executor.py` |
| Add remote executor | `src/smolagents/remote_executors.py` |

## Testing

Tests are in `./tests/`. Key test files:
- `test_agents.py` - Agent behavior tests
- `test_bp_context_tools.py` - Beyond Python tools tests
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
