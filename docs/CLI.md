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
`copy_file`, `is_file`, `print_file_lines`, `get_line_from_file`, `count_file_lines`, `read_file_range`, `insert_lines_into_file`, `replace_line_in_file`, `remove_pascal_comments_from_string`, `pascal_interface_to_string`, `source_code_to_string`, `string_to_source_code`, `run_os_command`, `replace_in_file`, `replace_in_file_from_files`, `get_file_size`, `load_string_from_file`, `save_string_to_file`, `append_string_to_file`, `list_directory_tree`, `search_in_files`, `get_file_info`, `list_directory`, `extract_function_signatures`, `compare_files`, `count_lines_of_code`, `mkdir`, `delete_file`, `delete_directory`, `compare_folders`, `read_first_n_lines`, `read_last_n_lines`, `delete_lines_from_file`

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
| `BPSA_SYSTEM_PROMPT_FIRST` | No | `false` | Place system prompt before memory steps instead of after |
| `BPSA_INJECT_FOLDER` | No | `true` | Inject directory tree (`false`, `true` = cwd, or a path) |
| `BPSA_MCP` | No | `''` | Newline-separated list of MCP servers (URLs or stdio commands). Merged with `--mcp` CLI flags. |
| `BPSA_COPILOT_MODEL_ID` | No | - | When set, enables the **GitHub Copilot** tool (`GitHubCopilotCoder`). Value is the Copilot model ID to use (e.g. `claude-sonnet-4.6`). Requires the `github-copilot-sdk` package. |

### Tool-Enabling Environment Variables

These variables enable optional tool sets. Each corresponds to a CLI flag. Setting the variable to `true` or `1` enables the tools without needing the flag.

| Variable | CLI Flag | Description |
|----------|----------|-------------|
| `BPSA_BROWSER` | `--browser` | Enable Playwright browser integration (navigate, click, type, etc.) |
| `BPSA_GUI` | `--gui-x11` | Enable native GUI interaction tools (screenshot, click, type via xdotool/ImageMagick on X11) |
| `BPSA_IMAGE` | `--image` | Enable image analysis and drawing tools (diff_images, screen_ocr, canvas drawing) |

### Context Compression Variables

All optional. Configure `CompressionConfig` without touching code:

| Variable | Default | Description |
|----------|---------|-------------|
| `BPSA_COMPRESSION_ENABLED` | `1` | Enable/disable compression (`0` or `1`) |
| `BPSA_COMPRESSION_KEEP_RECENT_STEPS` | `40` | Recent steps kept in full detail |
| `BPSA_COMPRESSION_MAX_UNCOMPRESSED_STEPS` | `50` | Trigger compression when uncompressed step count exceeds this |
| `BPSA_COMPRESSION_KEEP_COMPRESSED_STEPS` | `20` | Compressed steps to keep during a merge |
| `BPSA_COMPRESSION_MAX_COMPRESSED_STEPS` | `25` | Trigger merge when compressed step count exceeds this |
| `BPSA_COMPRESSION_TOKEN_THRESHOLD` | `0` | Token-based compression trigger (`0` = disabled) |
| `BPSA_COMPRESSION_MODEL` | _(main model)_ | Model ID for compression (uses main model if unset) |
| `BPSA_COMPRESSION_MAX_SUMMARY_TOKENS` | `50000` | Max tokens in a generated summary |
| `BPSA_COMPRESSION_PRESERVE_ERROR_STEPS` | `0` | Keep steps with errors uncompressed (`0` or `1`) |
| `BPSA_COMPRESSION_PRESERVE_FINAL_ANSWER_STEPS` | `1` | Keep final_answer steps uncompressed (`0` or `1`) |
| `BPSA_COMPRESSION_MIN_CHARS` | `4096` | Min characters of content before an LLM compression call is made |

### Dictation Input Variables

Requires `pip install bpsa[dictation]`.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BPSA_DICTATION_TRANSCRIBER` | Yes (for `/dictation`) | - | Transcriber name: `whisper` or `elevenlabs` |
| `BPSA_DICTATION_MODEL` | No | `base.en` (`whisper`) or `scribe_v2` (`elevenlabs`) | Model name passed to the transcriber (whisper only) |
| `ELEVENLABS_API_KEY` | Yes (for `elevenlabs`) | - | API key for ElevenLabs Scribe API |

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
│ Beyond Python SmolAgents v1.24-bp       │
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

### Shell Escapes

| Command | Description |
|---------|-------------|
| `!<command>` | Run an OS command directly (agent does not see the output) |
| `!!<command>` | Run an OS command; output is appended to the next prompt sent to the agent |
| `!!!<command>` | Run an OS command and immediately send the output to the agent for analysis |

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
| `/help` | Show available commands and brief descriptions |
| `/instructions-load` | Load agent instruction files into next prompt |
| `/plan [on\|off\|N]` | Toggle or set planning interval (default: 22) |
| `/pwd` | Show current working directory |
| `/repeat <N> <prompt>` | Run the same prompt N times, each on a fresh agent with current context |
| `/repeat-prompt <N> <path>` | Run a prompt file N times, each on a fresh agent with current context |
| `/run-prompt <path>` | Load a file's content as the prompt |
| `/run-py <script.py>` | Execute a Python script |
| `/save <filename>` | Save the last answer to a file |
| `/save-step <N> <file>` | Save full content of step N to a file |
| `/session-load <file>` | Load a session from a JSON file |
| `/session-save <file>` | Save entire session to a JSON file |
| `/show-compression-stats` | Show compression config and stats |
| `/show-memory-stats` | Show memory breakdown: steps, tokens, compressed vs uncompressed |
| `/show-stats` | Show session statistics (token usage, time) |
| `/show-step <N>` | Show full content of a specific step |
| `/show-steps` | Show one-line summary of all memory steps |
| `/set-max-steps <N>` | Change max_steps for the agent |
| `/show-tools` | List all loaded tools |
| `/undo-steps [N]` | Remove last N steps from memory (default: 1) |
| `/verbose` | Toggle verbose output |
| `/dictation [on\|off]` | Toggle dictation (requires `BPSA_DICTATION_TRANSCRIBER`) |

## MCP Server Integration

The `--mcp` flag connects [Model Context Protocol](https://modelcontextprotocol.io) servers as additional tool sources. Tools exposed by MCP servers are automatically available to the agent alongside the built-in tools.

```bash
# HTTP-based MCP server (Streamable HTTP transport)
bpsa --mcp http://localhost:8000/mcp

# stdio-based MCP server (shell command)
bpsa --mcp 'npx -y @modelcontextprotocol/server-filesystem /'

# Multiple MCP servers (flag can be repeated)
bpsa --mcp http://server1/mcp --mcp http://server2/mcp
```

The flag can be repeated to connect multiple servers simultaneously. Each server's tools are merged into the agent's tool list. MCP connections are automatically closed when the session ends.

You can also set MCP servers persistently via the `BPSA_MCP` environment variable (one entry per line):

```bash
export BPSA_MCP="http://localhost:8000/mcp"
# or multiple servers:
export BPSA_MCP="http://server1/mcp
npx -y @modelcontextprotocol/server-filesystem /"
```

CLI `--mcp` entries and `BPSA_MCP` entries are merged, so both can be used simultaneously.

## Configuration Layering

Priority (highest to lowest):

1. CLI flags (e.g., `--model`, `--verbose`)
2. Environment variables (`BPSA_*`)
3. Config file (`~/.bpsa.yaml`)
4. Built-in defaults

## Utility CLIs

Beyond the main `bpsa` REPL, the project ships lightweight CLI wrappers around commonly used tools from `bp_tools.py`. All are installed automatically via `pip install` and support `--help`.

### Quick Reference

| Command | Description |
|---------|-------------|
| `bptree` | Directory tree with line counts and optional function signatures |
| `bpgrep` | Search for text patterns in files with extension filtering |
| `bppack` | Pack a folder's source code into a single tagged string |
| `bpunpack` | Reconstruct source files from a packed string |
| `bploc` | Lines-of-code summary broken down by file type |
| `bpdiff` | Compare two files or two folders (unified diff) |
| `bpsig` | Extract function/class signatures from a source file |
| `bppas` | Extract Pascal unit interface sections from a folder |

Entry points in `pyproject.toml`:
```toml
[project.scripts]
bptree  = "smolagents.bp_tree_cli:main"
bpgrep  = "smolagents.bp_grep_cli:main"
bppack  = "smolagents.bp_pack_cli:main"
bpunpack = "smolagents.bp_unpack_cli:main"
bploc   = "smolagents.bp_loc_cli:main"
bpdiff  = "smolagents.bp_diff_cli:main"
bpsig   = "smolagents.bp_sig_cli:main"
bppas   = "smolagents.bp_pas_cli:main"
```

---

### `bptree` — Directory Tree

Displays a tree view of a directory with source-file line counts and optional function signatures. Skips common build/artifact folders by default.

```bash
bptree [FOLDER] [-d DEPTH] [--no-files] [-s] [--skip-dirs DIRS]
```

| Flag | Description |
|------|-------------|
| `FOLDER` | Root folder (default: `.`) |
| `-d`, `--depth N` | Max depth to traverse (default: 6) |
| `--no-files` | Show only directories, hide files |
| `-s`, `--signatures` | Show function/class signatures inline |
| `--skip-dirs d1,d2` | Comma-separated dirs to show but not traverse |

```bash
# Basic tree of src/ with depth 2
bptree src/ -d 2

# Tree with function signatures
bptree . -s

# Directories only, skip vendor
bptree . --no-files --skip-dirs vendor,tmp
```

---

### `bpgrep` — Search in Files

Searches for a text pattern in files within a folder (or a single file). Returns matching lines with file paths and line numbers.

```bash
bpgrep PATTERN [PATH] [-e EXTS] [-c] [-m MAX]
```

| Flag | Description |
|------|-------------|
| `PATTERN` | Text pattern to search for |
| `PATH` | Folder or file to search (default: `.`) |
| `-e`, `--extensions` | Comma-separated extensions to filter (e.g. `py,js,ts`) |
| `-c`, `--case-sensitive` | Case-sensitive search (default: case-insensitive) |
| `-m`, `--max-results N` | Maximum results to return (default: 50) |

```bash
# Search for "def main" in Python files
bpgrep "def main" src/ -e py

# Case-sensitive search with limit
bpgrep "TODO" . -c -m 20
```

---

### `bppack` — Pack Source Code

Packs all source files in a folder into a single string with `<file filename="...">...</file>` XML-like tags. Useful for feeding entire projects to LLMs or creating context dumps.

```bash
bppack FOLDER [-e EXTS] [--strip-pascal-comments] [--exclude ITEMS] [-o FILE]
```

| Flag | Description |
|------|-------------|
| `FOLDER` | Root folder to scan |
| `-e`, `--extensions` | Comma-separated extensions to include (default: common source extensions) |
| `--strip-pascal-comments` | Remove Pascal comments from `.pas`/`.inc` files |
| `--exclude` | Comma-separated files/folders to exclude |
| `-o`, `--output FILE` | Write to a file instead of stdout |

```bash
# Pack src/ to a file
bppack src/ -o context.txt

# Pack only Python and JS files
bppack . -e py,js --exclude __pycache__,node_modules
```

---

### `bpunpack` — Unpack Source Code

Reconstructs source files from a packed string produced by `bppack`. Reads from a file or stdin.

```bash
bpunpack [INPUT] [-o DIR] [--no-overwrite] [-q]
```

| Flag | Description |
|------|-------------|
| `INPUT` | Packed file to read (default: stdin) |
| `-o`, `--output-dir DIR` | Base directory to write into (default: `.`) |
| `--no-overwrite` | Skip files that already exist |
| `-q`, `--quiet` | Suppress status messages |

```bash
# Unpack from file into output/
bpunpack context.txt -o output/

# Pipe from bppack
bppack src/ | bpunpack -o copy/

# Unpack without overwriting existing files
bpunpack context.txt --no-overwrite
```

---

### `bploc` — Lines of Code

Counts lines of code broken down by file extension. Skips hidden files and directories.

```bash
bploc [FOLDER] [-e EXTS]
```

| Flag | Description |
|------|-------------|
| `FOLDER` | Root folder to analyze (default: `.`) |
| `-e`, `--extensions` | Comma-separated extensions to count (default: `py,js,java,cpp,c,php,rb`) |

```bash
# Count lines in current project
bploc

# Count only Python and TypeScript
bploc src/ -e py,ts
```

Example output:
```
  .py     23985 lines
  total   23985 lines
```

---

### `bpdiff` — Compare Files or Folders

Compares two files or two folders and shows differences in unified diff format. For folders, only source code files are compared.

```bash
bpdiff PATH1 PATH2 [-c CONTEXT]
```

| Flag | Description |
|------|-------------|
| `PATH1` | First file or folder |
| `PATH2` | Second file or folder |
| `-c`, `--context N` | Context lines around differences (default: 3) |

```bash
# Compare two files
bpdiff old_version.py new_version.py

# Compare two folders with more context
bpdiff project_v1/ project_v2/ -c 5
```

---

### `bpsig` — Function Signatures

Extracts function and class signatures from a source file without showing the implementation. Supports Python, JavaScript/TypeScript, Java, PHP, Pascal, C/C++, Markdown (section headers), and a generic fallback.

```bash
bpsig FILE [-l LANGUAGE]
```

| Flag | Description |
|------|-------------|
| `FILE` | Source file to extract signatures from |
| `-l`, `--language` | Programming language (auto-detected from extension if omitted) |

```bash
# Auto-detect language
bpsig src/main.py

# Explicit language
bpsig utils.inc -l pascal
```

---

### `bppas` — Pascal Interfaces

Extracts the interface sections (between `interface` and `implementation` keywords) from Pascal source files in a folder. Output uses `<pascal_interface filename="...">` tags.

```bash
bppas FOLDER [--strip-comments] [-o FILE]
```

| Flag | Description |
|------|-------------|
| `FOLDER` | Root folder to scan for `.pas`, `.inc`, `.pp`, `.lpr`, `.dpr` files |
| `--strip-comments` | Remove Pascal comments from extracted interfaces |
| `-o`, `--output FILE` | Write to a file instead of stdout |

```bash
# Extract interfaces
bppas pascal_src/

# Strip comments and save to file
bppas lib/ --strip-comments -o interfaces.txt
```

---

## Dependencies

- `prompt_toolkit` - REPL input handling (optional, falls back to basic `input()`)
- `rich` - Output formatting (already a project dependency)
- `argparse` - CLI argument parsing (stdlib)
