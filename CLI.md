# CLI Design Specification for `bpsa`

## Overview

`bpsa` is the CLI interface for Beyond Python SmolAgents. It defaults to an interactive REPL with multi-turn conversation using a persistent `CodeAgent`.

## Command

```bash
bpsa              # Interactive REPL (default)
bpsa run "task"   # One-shot mode
echo "task" | bpsa  # Piped input (one-shot, detected via stdin.isatty())
```

Entry point in `pyproject.toml`:
```toml
[project.scripts]
bpsa = "smolagents.bp_cli:main"
```

## Agent

- **CodeAgent** only.
- `additional_authorized_imports=['*']`
- `add_base_tools=True`

## Tools

All tools from `DEFAULT_THINKER_TOOLS` (defined in `bp_thinkers.py`) plus all default tools from `tools.py` (via `add_base_tools=True`):

**DEFAULT_THINKER_TOOLS:**
`copy_file`, `is_file`, `print_source_code_lines`, `get_line_from_file`, `get_file_lines`, `read_file_range`, `insert_lines_into_file`, `replace_line_in_file`, `remove_pascal_comments_from_string`, `pascal_interface_to_string`, `source_code_to_string`, `string_to_source_code`, `run_os_command`, `replace_on_file`, `replace_on_file_with_files`, `get_file_size`, `load_string_from_file`, `save_string_to_file`, `append_string_to_file`, `list_directory_tree`, `search_in_files`, `get_file_info`, `list_directory`, `extract_function_signatures`, `compare_files`, `count_lines_of_code`, `mkdir`, `delete_file`, `delete_directory`, `compare_folders`, `read_first_n_lines`, `read_last_n_lines`, `delete_lines_from_file`

## Compression

Uses `DEFAULT_THINKER_COMPRESSION` from `bp_thinkers.py`:

```python
CompressionConfig(
    keep_recent_steps=22,        # DEFAULT_THINKER_PLANNING_INTERVAL
    max_uncompressed_steps=32,   # planning_interval + 10
    keep_compressed_steps=44,    # planning_interval * 2
    max_compressed_steps=66,     # planning_interval * 3
    preserve_error_steps=False
)
```

## Environment Variables

All prefixed with `BPSA_`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BPSA_MODEL_ID` | Yes | - | Model identifier (e.g., `Gemini-2.5-Flash`) |
| `BPSA_SERVER_MODEL` | No | `OpenAIServerModel` | Model class to use (see supported models below) |
| `BPSA_API_ENDPOINT` | No | - | API endpoint URL (e.g., `https://api.poe.com/v1`) |
| `BPSA_KEY_VALUE` | No | - | API key for authentication |
| `BPSA_POSTPEND_STRING` | No | `''` | String set on `model.postpend_string` |
| `BPSA_GLOBAL_EXECUTOR` | No | `exec` | Executor type (`exec`, `local`, `e2b`, etc.) |
| `BPSA_MAX_TOKENS` | No | `64000` | Max tokens for model responses |
| `BPSA_VERBOSE` | No | `0` | Verbose output (`0` or `1`) |

### Supported Model Classes (`BPSA_SERVER_MODEL`)

`OpenAIServerModel`, `LiteLLMModel`, `LiteLLMRouterModel`, `InferenceClientModel`, `TransformersModel`, `AzureOpenAIServerModel`, `AmazonBedrockModel`, `VLLMModel`, `MLXModel`, `GoogleColabModel`

### Example Configuration

```bash
export BPSA_SERVER_MODEL=OpenAIServerModel
export BPSA_API_ENDPOINT="https://api.poe.com/v1"
export BPSA_KEY_VALUE='MY_API_KEY'
export BPSA_MODEL_ID="Gemini-2.5-Flash"
export BPSA_POSTPEND_STRING=''
export BPSA_GLOBAL_EXECUTOR='exec'
export BPSA_MAX_TOKENS=64000
```

## Startup Behavior

**Fail fast.** On startup, before showing the banner:

1. Check `BPSA_MODEL_ID` is set. If not: `Error: BPSA_MODEL_ID is not set. Export it with: export BPSA_MODEL_ID=<value>`
2. Validate `BPSA_SERVER_MODEL` is a supported model class.
3. Load all tools. If any fail, report which ones and exit.

## Banner

Functional banner shown after successful startup:

```
╭─────────────────────────────────────────╮
│ Beyond Python SmolAgents v1.23-bp       │
│ Model: Gemini-2.5-Flash (OpenAIServer…) │
│ Tools: 38 loaded                        │
╰─────────────────────────────────────────╯
Type /help for commands, /exit to quit.
>
```

## REPL

### Input

Use `prompt_toolkit` for:
- Command history (persisted across sessions)
- Multiline editing
- Ctrl+C handling (cancel current input, not exit)
- Autocomplete (slash commands)

### Step Display

- Show intermediate code/thoughts as they happen (streaming)
- Dim or prefix intermediate steps so the final answer stands out

### Slash Commands

| Command | Description |
|---------|-------------|
| `/auto-approve [on\|off]` | Toggle or set auto-approve for tag execution |
| `/cd <dir>` | Change working directory |
| `/clear` | Clear screen, reset agent and conversation history |
| `/compress [N]` | Force compression now, or compress a specific step N |
| `/compression [on\|off]` | Toggle compression on/off |
| `/compression-keep-recent-steps <N>` | Change keep_recent_steps |
| `/compression-max-uncompressed-steps <N>` | Change max_uncompressed_steps |
| `/compression-model <model>` | Switch compression model |
| `/exit` | Exit the REPL |
| `/file <path>` | Load a file's content as the prompt |
| `/help` | Show available commands and brief descriptions |
| `/load-instructions` | Load agent instruction files into next prompt |
| `/plan [on\|off\|N]` | Toggle or set planning interval (default: 22) |
| `/pwd` | Show current working directory |
| `/repeat <N> <prompt>` | Run the same prompt N times, each on a fresh agent with current context |
| `/repeat-prompt <N> <path>` | Run a prompt file N times, each on a fresh agent with current context |
| `/run <script.py>` | Execute a Python script |
| `/save <filename>` | Save the last answer to a file |
| `/save-step <N> <file>` | Save full content of step N to a file |
| `/session-load <file>` | Load a session from a JSON file |
| `/session-save <file>` | Save entire session to a JSON file |
| `/show-compression-stats` | Show compression config and stats |
| `/show-memory-stats` | Show memory breakdown: steps, tokens, compressed vs uncompressed |
| `/show-stats` | Show session statistics (token usage, time) |
| `/show-step <N>` | Show full content of a specific step |
| `/show-steps` | Show one-line summary of all memory steps |
| `/steps <N>` | Change max_steps for the agent |
| `/show-tools` | List all loaded tools |
| `/undo-steps [N]` | Remove last N steps from memory (default: 1) |
| `/verbose` | Toggle verbose output |

## Configuration Layering

Priority (highest to lowest):

1. CLI flags (e.g., `--model`, `--verbose`)
2. Environment variables (`BPSA_*`)
3. Config file (`~/.bpsa.yaml`)
4. Built-in defaults

## Dependencies

- `prompt_toolkit` - REPL input handling (optional, falls back to basic `input()`)
- `rich` - Output formatting (already a project dependency)
- `argparse` - CLI argument parsing (stdlib)
