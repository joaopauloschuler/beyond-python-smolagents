# CLI Design Specification for `bpsa`

## Overview

`bpsa` is the CLI interface for Beyond Python SmolAgents. It defaults to an interactive REPL with multi-turn conversation using a persistent `CodeAgent`.

## Command

```bash
bpsa              # Interactive REPL (default)
bpsa run "task"   # One-shot mode
echo "task" | bpsa  # Piped input (one-shot, detected via stdin.isatty())
bpsa listen       # No keyboard: run on agent-to-agent messages (see Agent-to-Agent Messages)
```

Unless auto-approve is on, `bpsa` asks before it runs code or writes files,
in every mode. On Linux and macOS, when stdin is not a terminal:

- If stdout is a terminal, `bpsa` reads your answer from the terminal itself
  (`/dev/tty`), so `echo "task" | bpsa` can still be approved at the keyboard.
- Otherwise nobody can answer (cron, CI, or a program that captures the
  output of `bpsa`), so `bpsa run` and piped mode stop at startup with an
  error and exit status 1. To run without asking, use `--auto-approve on` or
  set `BPSA_AUTO_APPROVE=on`.

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
    enabled=True,                  # BPSA_COMPRESSION_ENABLED
    keep_recent_steps=40,          # BPSA_COMPRESSION_KEEP_RECENT_STEPS
    max_uncompressed_steps=50,     # BPSA_COMPRESSION_MAX_UNCOMPRESSED_STEPS
    keep_compressed_steps=20,      # BPSA_COMPRESSION_KEEP_COMPRESSED_STEPS
    max_compressed_steps=25,       # BPSA_COMPRESSION_MAX_COMPRESSED_STEPS
    estimated_token_threshold=0,   # BPSA_COMPRESSION_TOKEN_THRESHOLD (0 = off; 75% of the context length when known)
    max_summary_tokens=50000,      # BPSA_COMPRESSION_MAX_SUMMARY_TOKENS
    preserve_error_steps=False,    # BPSA_COMPRESSION_PRESERVE_ERROR_STEPS
    preserve_final_answer_steps=True,  # BPSA_COMPRESSION_PRESERVE_FINAL_ANSWER_STEPS
)
```

Each value is the built-in default; the named environment variable overrides it (see the Context Compression Variables table below).

## Environment Variables

Set these in the shell or in a `.env` file in the working directory, which
`bpsa` loads at startup (a variable already set in the shell takes precedence).
All are prefixed with `BPSA_`.

All on/off settings follow one rule. `1`, `true`, `yes` or `on` turns the setting on;
`0`, `false`, `no`, `off` or an empty value turns it off. Case and outer spaces do not
matter. For any other value, `bpsa` prints a warning and uses the setting's default.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BPSA_MODEL_ID` | Yes | - | Model identifier (e.g., `Gemini-2.5-Flash`) |
| `BPSA_SERVER_MODEL` | No | `OpenAIServerModel` | Model class to use (see supported models below) |
| `BPSA_API_ENDPOINT` | No | - | API endpoint URL (e.g., `https://api.poe.com/v1`) |
| `BPSA_KEY_VALUE` | No | - | API key for authentication |
| `BPSA_POSTPEND_STRING` | No | `''` | String set on `model.postpend_string` |
| `BPSA_GLOBAL_EXECUTOR` | No | `exec` | Executor type (`exec`, `local`, `e2b`, etc.) |
| `BPSA_MAX_TOKENS` | No | `64000` | Max tokens for model responses |
| `BPSA_PROVIDER_ORDER` | No | - | Comma-separated OpenRouter provider order (e.g., `openai,together`). Sent as `provider.order` in the request body. `OpenAIServerModel` and `LiteLLMModel` only. |
| `BPSA_HAS_SESSION_ID` | No | `0` | `1`, `true` or `on` to send a random `session_id` in the request body. OpenRouter uses it as a sticky-routing key so requests of the same conversation stay on the same provider and keep the prompt cache warm. A new id is generated on `/clear`. Note: `BPSA_PROVIDER_ORDER` overrides sticky routing on OpenRouter. `OpenAIServerModel` and `LiteLLMModel` only. |
| `BPSA_REASONING_EFFORT` | No | - | Value of the `reasoning_effort` request field sent with every model call, e.g. `low`, `medium`, `high`, `xhigh` or `max` (the accepted set depends on the model; OpenAI and OpenRouter document `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`). Unset means the provider's default. High values spend most of `BPSA_MAX_TOKENS` on reasoning, so raise it together. Non-reasoning models may reject the field. `OpenAIServerModel` and `LiteLLMModel` only. |
| `BPSA_REASONING_TAGS` | No | by model | `on` or `off`. `on` keeps the `<thoughts>`, `<plans>`, `<observations>` and `<freewill>` sections in the agent's system prompt. Default: `off` when the model id contains `claude-fable`, `claude-opus`, `claude-sonnet` or `claude-mythos`, because those models refuse such prompts (`stop_reason` `refusal`, category `reasoning_extraction`); `on` for every other model. |
| `BPSA_PRICE_INPUT_PER_M` | No | - | USD per million input tokens. With `BPSA_PRICE_OUTPUT_PER_M` it lets `OpenAIModel` estimate the cost of each request when the response carries no `usage.cost` (OpenAI, DeepSeek). |
| `BPSA_PRICE_OUTPUT_PER_M` | No | - | USD per million output tokens. Both price variables must be set for an estimate; otherwise no cost is shown. |
| `BPSA_PRICE_CACHED_INPUT_PER_M` | No | _(input price)_ | USD per million cached input tokens, applied to the cached share of the input when the provider reports it. |
| `BPSA_VERBOSE` | No | `0` | Verbose output (`0` or `1`). The `--verbose on\|off` CLI flag overrides it when given. Applies to interactive, `run` and piped modes. |
| `BPSA_AUTO_APPROVE` | No | `off` | Auto-approve tag execution (`runcode`, `savetofile`, `appendtofile`) without asking (`on` or `off`). The `--auto-approve on\|off` CLI flag overrides it when given, and `/auto-approve` changes it mid-session. Applies to interactive, `run` and piped modes. |
| `BPSA_SYSTEM_PROMPT_FIRST` | No | `true` | Place system prompt before memory steps. Set to `0` to place it after the memory steps instead. |
| `BPSA_SKIP_CONNECTIVITY_CHECK` | No | `0` | `1`, `true` or `on` to skip the startup connectivity check (one tiny request that verifies the key, endpoint and model id; see Startup Behavior). Local model classes (`TransformersModel`, `MLXModel`, `VLLMModel`) always skip it. |
| `BPSA_CONTEXT_LENGTH` | No | `0` | Model context window in integer tokens. `0` or unset: when `BPSA_API_ENDPOINT` contains `openrouter`, `resolve_context_length` asks `GET <endpoint>/models` once at startup (and once per `/model`) for the model id's `context_length`; otherwise the window is unknown. When known, the banner shows `Context: N tokens`, the turn summary shows `Context: NN%` (last model call's input tokens over the window; yellow from 70%, red from 90%) instead of `Context: N chars`, and `BPSA_COMPRESSION_TOKEN_THRESHOLD` defaults to 75% of it. |
| `BPSA_MAX_SESSION_TOKENS` | No | `0` | Session token budget: input + output tokens over the whole session (`session_stats`). At 80% `warn_session_budget` prints one yellow warning; at 100% the REPL prints a red message and starts no further agent turn (plain prompts, `!!!`, `/redo`, `/run-prompt`, `/repeat`, `/repeat-prompt`) while slash commands keep working. `/clear` resets the totals and so the budget; `/session-load` counts the loaded totals. `ad-infinitum` sums every prompt task and exits with code 3 at 100%. `0`, unset or an invalid value = no limit. |
| `BPSA_MAX_SESSION_COST` | No | `0` | Session cost budget in USD, same rule, compared against the session's `total_cost_usd`. It only works when the model reports a cost (OpenRouter `usage.cost`) or the `BPSA_PRICE_*` variables allow an estimate; without a known cost the total stays 0 and the limit never triggers. |
| `BPSA_INJECT_FOLDER` | No | `true` | Inject directory tree (`false`, `true` = cwd, or a path) |
| `BPSA_MCP` | No | `''` | Newline-separated list of MCP servers (URLs or stdio commands). Merged with `--mcp` CLI flags. |
| `BPSA_AGENT_NAME` | No | - | This process's agent name for agent-to-agent messages: 1 to 64 lowercase ASCII letters, digits or `-`, not starting with `-`, not `all`. Unset or empty turns messages off, and the three variables below are then ignored. An invalid value stops `bpsa` at startup with one error line. When it is set, the agent has the tool `send_message(to, text, folder?)`: `to` is an agent name or `all`, `folder` the label of one of the message folders (needed only with several), `text` at most 100,000 characters (the `send` command below and every other writer have the same limit, and a reader shows a message file above 600,000 bytes as one bracketed line). It also has `wait_for_messages(timeout_seconds)`. Unless `BPSA_MAX_GROUP_MEMBERS` is `0`, it also has `start_agent(folder?, working_directory?, count?)`, which starts one `bpsa listen` process (or `count` of them at the same time) named `<own name>-<k>` on one of its folders and returns each one's name, process id and log file, and `stop_agent(name)` (`all` = every agent it started). |
| `BPSA_MESSAGE_FOLDERS` | With `BPSA_AGENT_NAME` | - | Message folders, one existing folder path per line (as `BPSA_MCP`). The last path component labels each folder (`/x/y/leaders` is `leaders`); two folders with the same label are an error. |
| `BPSA_MESSAGE_POLL_SECONDS` | No | `60` | Seconds between message folder checks; any number above 0, fractions allowed. |
| `BPSA_MAX_GROUP_MEMBERS` | No | `9` | How many agents started with `start_agent` this agent may have running at once; a whole number, 0 or more. `start_agent(count=N)` starts none when the running agents plus `N` exceed it, or when `N` x 260 MB exceeds the machine's available memory (`MemAvailable` in `/proc/meminfo`). With `0` the tools `start_agent` and `stop_agent` do not exist; `start_agent` sets `0` for every agent it starts. Each started agent logs to `<name>.log` in its message folder and is recorded in `.<name>.agent.json` there (process id, log path, start time, working directory). An agent that ends in an orderly way (`/exit`, end of `bpsa run`, `bpsa listen` stopped by SIGTERM) first stops the agents it started. |
| `BPSA_COPILOT_MODEL_ID` | No | - | When set, enables the **GitHub Copilot** tool (`GitHubCopilotCoder`). Value is the Copilot model ID to use (e.g. `claude-sonnet-4.6`). Requires the `github-copilot-sdk` package. |

### Tool-Enabling Environment Variables

These variables enable optional tool sets. Each corresponds to a CLI flag. Setting the variable to `true` or `1` enables the tools without needing the flag.

| Variable | CLI Flag | Description |
|----------|----------|-------------|
| `BPSA_BROWSER` | `--browser` | Enable Playwright browser integration (navigate, click, type, etc.) |
| `BPSA_GUI` | `--gui-x11` | Enable native GUI interaction tools (screenshot, click, type via xdotool/ImageMagick on X11) |
| `BPSA_IMAGE` | `--image` | Enable image analysis and drawing tools (diff_images, screen_ocr, canvas drawing) |
| `BPSA_TMUX` | `--tmux` | Enable tmux multi-screen tools (create, send, read, list, destroy, wait) |

### Context Compression Variables

All optional. Configure `CompressionConfig` without touching code:

| Variable | Default | Description |
|----------|---------|-------------|
| `BPSA_COMPRESSION_ENABLED` | `1` | Enable/disable compression (`0` or `1`) |
| `BPSA_COMPRESSION_KEEP_RECENT_STEPS` | `40` | Recent steps kept in full detail |
| `BPSA_COMPRESSION_MAX_UNCOMPRESSED_STEPS` | `50` | Trigger compression when uncompressed step count exceeds this |
| `BPSA_COMPRESSION_KEEP_COMPRESSED_STEPS` | `20` | Compressed steps to keep during a merge |
| `BPSA_COMPRESSION_MAX_COMPRESSED_STEPS` | `25` | Trigger merge when compressed step count exceeds this |
| `BPSA_COMPRESSION_TOKEN_THRESHOLD` | `0` | Token-based compression trigger (`0` = disabled). Tokens are estimated as characters / 4 (`estimate_tokens`). When the variable is unset and the context length is known (`BPSA_CONTEXT_LENGTH` or OpenRouter, see the Environment Variables table), `apply_context_length` sets it to 75% of the context length; an explicit value, including `0`, always wins. |
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
4. Send one tiny request through the model (`Reply with OK.`, `max_tokens` 8 where the model class accepts it)
   and exit with `Error: Startup connectivity check failed for <model id>: ...` naming the likely cause: a
   rejected API key (401/403), an unknown model id (404 / model not found) or an unreachable endpoint
   (connection, DNS or timeout errors). Set `BPSA_SKIP_CONNECTIVITY_CHECK=1` to skip this request; local
   model classes (`TransformersModel`, `MLXModel`, `VLLMModel`) never send it. The banner shows the
   round-trip time of this request as endpoint latency. `/model` does not repeat the check.
5. Resolve the model's context length: `BPSA_CONTEXT_LENGTH` when positive, else one `GET <endpoint>/models`
   request on OpenRouter endpoints (5 s timeout; the model id is matched exactly, then without a leading `~`).
   Any failure leaves the window unknown and keeps the `Context: N chars` figure in the turn summary.
   `/model` resolves it again for the new model; `/clear` reuses the stored value.

## Banner

Functional banner shown after successful startup:

```
╭─────────────────────────────────────────╮
│ Beyond Python SmolAgents v1.24-bp       │
│ Model: Gemini-2.5-Flash (OpenAIServer…) │
│ Reasoning effort: high                  │
│ Endpoint: 0.8s round trip               │
│ Context: 1,048,576 tokens               │
│ Tools: 38 loaded                        │
╰─────────────────────────────────────────╯
Type /help for commands, /exit to quit.
>
```

`print_banner` shows the `Reasoning effort` line only when `BPSA_REASONING_EFFORT`
is set and `build_model` sends it for the model class (`OpenAIServerModel`,
`AzureOpenAIServerModel` or `LiteLLMModel`). The line shows the requested value;
the provider may ignore it.

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

After each turn `print_turn_summary` prints one dim line with the turn's input,
output and total tokens. When the provider reports cached prompt tokens the line
also shows `Cache: NN%`, the share of that turn's input tokens served from the
provider's prompt cache (cached / input, rounded). `OpenAIModel` reads
`usage.prompt_tokens_details.cached_tokens` (OpenAI, OpenRouter) or
`usage.prompt_cache_hit_tokens` (DeepSeek); when neither is present the figure
is left out. `/show-stats` shows the session total as "Total cached input
tokens". A non-zero figure confirms `BPSA_HAS_SESSION_ID` and prompt ordering
are hitting the cache.

The same line ends with `via <Provider>` (for example `via DeepInfra`) when the
last model call of the turn reported which upstream provider served it. Today
only OpenRouter sends this: `OpenAIModel` reads the top-level `provider` string
of the response (`extract_provider_name`), the agent's `Monitor` keeps the value
from the most recent step, and `print_turn_summary` prints it. OpenAI and
DeepSeek endpoints do not send the field, so the line has no `via` part there.
`/show-stats` shows the same value as "Last provider". Use it to check that
`BPSA_PROVIDER_ORDER` and `BPSA_HAS_SESSION_ID` route where you expect.

The line also shows the turn's cost, for example `$0.0123` (four decimals, six
below one tenth of a cent), when a cost is known. On OpenRouter `build_model`
adds `usage: {"include": true}` to the request body (only when
`BPSA_API_ENDPOINT` contains `openrouter`, because OpenAI rejects unknown body
fields) and `OpenAIModel` reads the exact `usage.cost` the response then
carries (`extract_response_cost`). For endpoints that do not report a cost, set
`BPSA_PRICE_INPUT_PER_M` and `BPSA_PRICE_OUTPUT_PER_M` (USD per million tokens)
and `OpenAIModel` estimates it from the token counts (`estimate_cost_usd`);
cached input tokens are charged at `BPSA_PRICE_CACHED_INPUT_PER_M` when set,
else at the input price. When neither a reported cost nor both price variables
are available, the line has no `$` part. `/show-stats` shows "Total cost" and
"Avg cost/turn"; `/session-save` and `/session-load` keep the totals.

### Session Budget

`BPSA_MAX_SESSION_TOKENS` (input + output tokens) and `BPSA_MAX_SESSION_COST`
(USD) cap a session. After each turn `warn_session_budget` compares the
`session_stats` totals with each limit: at 80% it prints
`Session budget warning: Session tokens: 16,500 of 20,000 (82%)` once per
limit; at 100% it prints a red `Session budget exceeded:` message, and from
then on `session_budget_blocks_turn` refuses every new agent turn (plain
prompts, `!!!`, `/redo`, `/run-prompt`, `/repeat`, `/repeat-prompt`) with the
same red message. Slash commands keep working, so `/session-save`,
`/show-stats` and `/show-config` remain available. `/show-stats` and
`/show-config` show a "Token budget" / "Cost budget" row (used of limit,
percent) only when the limit is set. `/clear` resets `session_stats` and
therefore the budget; `/session-load` restores the saved totals, which count
against the budget, and prints the warnings again from those totals. Turns
interrupted with Ctrl+C and `/repeat` cycles are not added to `session_stats`
today, so they do not count. `ad-infinitum` applies the same limits to the
sum over every prompt task of the run (`add_agent_usage` in
`bp_ad_infinitum.py`); when a limit is reached it stops after the current
task and exits with code 3 (`BUDGET_EXIT_CODE`).

### Steering the agent while it runs

While `agent.run` is working the REPL keeps the spinner line on screen and
shows a plain `> ` input line under it. Type a message and press Enter: the
line is echoed dimmed as `queued: <text>` and the spinner goes on. At the next
step boundary `MultiStepAgent._deliver_steering_messages` (called at the top
of every step in `_run_stream`, right before the `interrupt_switch` check)
drains the queue into the `user_message` field of the ActionStep that just
finished, and `ActionStep.to_messages` sends that text to the model as one
user-role message placed right after the step's execution logs. Several
queued lines join with newlines into that one field. The system prompt tells
the model that such a message takes priority over the original task.
`/show-step N` prints the text under "User message:", `/session-save` and
`/session-load` keep it, and context compression hands it to the summariser
together with the step's run output. A message queued before the first step
of a turn has finished is held until that step exists; a line typed after the
agent's last step boundary is printed back as `Not delivered (the agent
finished first): <text>`.

Keys while the agent runs:

| Key | Effect |
|-----|--------|
| Enter | Queue the typed line for the next step boundary |
| Esc | Clean stop: `agent.interrupt()`; the current step finishes, the REPL prints `Stopped after step N; memory kept.` and returns to the prompt with every finished step in memory |
| Ctrl+C | Hard abort, exactly as before: the in-flight step is lost, `Interrupted.` is printed. As before, a Ctrl+C that lands inside a model call is swallowed by the retry loop in `CodeAgent._step_stream` (bare `except:` plus a 30 s sleep); press it again |

Mechanics: `SteeringListener` in `bp_cli.py` is one thread that owns stdin
for the whole turn with its own `prompt_toolkit` `PromptSession` inside
`patch_stdout`, so step output printed by the main thread scrolls above the
input line. The Rich `Live` spinner overwrites that input line, so while the
listener is active `Spinner.start` does nothing and the listener draws the
spinner text (with its frame) as the first line of its two-line prompt.
`interactive_approval_callback` reads a key with `_getch`, so it calls
`SteeringListener.pause()` before printing the approval box and `resume()`
after a `y`; the listener does not run in one-shot or piped mode
(`steering_available` requires stdin and stdout to be terminals), during
`/repeat` cycles, and cannot cancel a model call in progress.

`ad-infinitum` reads the same kind of message from a file: write text into
`_inbox.md` in the task folder while a prompt task runs; at the next step
boundary `read_steering_inbox` reads it, truncates the file to empty and
hands the text to the agent exactly as a typed line. A single task file has
no inbox.

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
| `/model [id]` | Switch the main model mid-session (memory is kept); no id shows the current one |
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
| `/show-config` | Show the effective settings and where each came from (env, `.env`, default, a CLI flag, or a slash command) |
| `/show-memory-stats` | Show memory breakdown: steps, tokens, compressed vs uncompressed |
| `/show-stats` | Show session statistics (token usage, time) |
| `/show-step <N>` | Show full content of a specific step |
| `/show-steps` | Show one-line summary of all memory steps |
| `/set-max-steps <N>` | Change max_steps for the agent |
| `/show-tools` | List all loaded tools |
| `/undo-steps [N]` | Remove last N steps from memory (default: 1) |
| `/verbose` | Toggle verbose output |
| `/dictation [on\|off]` | Toggle dictation (requires `BPSA_DICTATION_TRANSCRIBER`) |

`/model <id>` rebuilds the main model with `build_model`, so the new model uses
the same `BPSA_SERVER_MODEL` class, endpoint, key, provider order and
OpenRouter session id as the old one; only the model id changes. The
conversation history, the compressor's fallback model and the token counters
stay in place, so you can draft with a cheap model and finish with a strong
one. A model id that `build_model` rejects leaves the old model in place; an id
the endpoint does not know fails on the next request, as at startup.

`/show-config` prints one table with the model class, model id, endpoint, the
API key masked as its first 4 and last 4 characters, provider order, the
current OpenRouter session id, the system-prompt position, max tokens, the
context length (source `env`, `openrouter` or `unknown`), the
price variables when set, executor, max steps, planning interval,
auto-approve, the compression thresholds and model, and which optional tool
sets (browser, GUI, image, tmux, MCP) are enabled. The Source column says
`env` when the process environment set the variable, `.env` when
`try_load_dotenv` loaded it from `./.env` at startup, `default` when neither
did, the command-line flag that set it (for example `--auto-approve`), or
the slash command that changed the value mid-session. `bp_cli.py` never
reads the `~/.bpsa.yaml` file listed under Configuration Layering; only these
sources apply.

## Agent-to-Agent Messages

Several `bpsa` processes can exchange messages through plain files in shared
folders. Nothing else connects them: no server, no socket. The feature is off
unless `BPSA_AGENT_NAME` is set (see Environment Variables for the four
settings).

### Message files

A message is one UTF-8 text file named `from_to_id.txt` in a message folder,
for example `leader1_all_000003.txt`:

- `from` and `to` are agent names (1 to 64 lowercase ASCII letters, digits or
  `-`). `to` may also be `all`: every agent that reads the folder receives it,
  except the sender.
- `id` counts the sender's messages in that folder, starting at `000001`.

A reset request (see Reset below) is an empty file named `from_to_id.reset`,
for example `boss_agent31_000004.reset`. Its `id` comes from the same counter
as the sender's messages. Only the file name makes it a reset request: the
`send` command and `send_message` always write `.txt`, so no message text can
reset an agent. A bpsa version from before reset requests ignores such a file.

A folder is addressed by its label, the last component of its path. Besides
messages, a folder holds hidden files per agent: `.<name>.lock`,
`.<name>.read.json` (the last id read per sender) and, for agents started with
`start_agent`, `.<name>.agent.json` and the log `<name>.log`. Each agent name
must belong to one process per folder.

Message files (`*.txt`) and reset requests (`*.reset`) that were read may be deleted at any time, also while
agents run; the hidden files must stay, because `.<name>.lock` holds the
sender's highest id, so its ids never go down. To reset a folder completely,
stop its agents and delete everything in it, hidden files included. Deleting a
sender's `.<name>.lock` together with its newest messages while a reader's
`.<reader>.read.json` stays makes that sender's ids start lower again, and the
reader then skips its messages up to the last id it had read.

### What an agent has

| | |
|---|---|
| `send_message(to, text, folder?)` | Writes one message. `folder` is needed only by an agent with several folders. |
| `start_agent(folder?, working_directory?, count?)` | Starts one `bpsa listen` process named `<own name>-<k>` on that folder and returns when it listens (about 30 s). With `count=N` it starts `N` processes at the same time, with consecutive `k`, and returns when each listens or has failed (measured on one processor: 29 s for one, 41 s for two, 57 s for three), with one line per agent; those that failed are stopped, the others stay. It starts none when `N` exceeds the limit or the available memory. A new agent cannot start agents itself. |
| `stop_agent(name)` | Stops an agent that the caller started; `all` stops each of them. |
| `reset_agent(name, folder?)` | Writes a reset request to the agent `name`, or with `all` to every `bpsa listen` agent in that folder except the caller (see Reset). It returns at once; each agent that reset answers with a message starting `Reset done:`. Only agents that may start agents (`BPSA_MAX_GROUP_MEMBERS` above 0) have this tool, as `start_agent` and `stop_agent`; an agent started with `start_agent` has no `reset_agent` tool. |
| `wait_for_messages(timeout_seconds)` | Blocks the code step until at least one message has arrived or `timeout_seconds` (above 0, at most 3,600; any other value raises an exception) have passed; meanwhile the folders are checked every second. It prints the new messages into the step's execution log and returns how many it printed; it returns 0 only when none arrived in that time. The tool prints only into the first 10,000 characters of the execution log (a log above 20,000 characters keeps its first and last 10,000) and in the order of arrival. When a message has arrived that does not fit there, the tool raises an exception instead of returning: the code step ends, its error text says how many messages wait, and they are delivered right after the step (with the next task when that step was the last one the run may take). A further call in the same code step raises again at once. With `BPSA_GLOBAL_EXECUTOR=local` the tool cannot print into the execution log, so it raises for every message; that executor at present ends the run on any exception of a code step, and the messages are then delivered with the next task. A printed message is recorded as read once its code step has ended, the step's output or error text holds the message and the step is in the agent's memory: before the next step, or when the run ends (REPL and `bpsa listen`; `bpsa run` does not record the messages printed in its last step). After an interrupt (Ctrl-C, SIGINT, SIGTERM) or a failure before that point the message is delivered again: with the next task or after a restart. |
| Incoming messages | Checked every `BPSA_MESSAGE_POLL_SECONDS` without a model call. They reach the model between two steps of a run, or with the next task when the agent is idle. A message counts as read when it is handed to a run. |

The model sees each message as a block; every line of the text starts with
`|`, so a message cannot imitate the header of another:

```
=== message 1 from leader1 to agent31 in folder group1 ===
| Please write one haiku about rivers.
=== end of message 1 from leader1 ===
```

The system prompt tells the agent to treat messages as requests from its team,
and to do what the user asked where a message conflicts with it.

### `bpsa listen`

`bpsa listen` waits for messages, runs the agent on them, and waits again. It
keeps its memory between runs and makes no model call while it waits. It never
reads stdin and needs auto-approve on (`--auto-approve on` or
`BPSA_AUTO_APPROVE=on`); it writes no file in the home directory. Redirect its
output to a log file. It is up when the log shows a line starting
`Listening for messages: agent <name>, folders`. SIGTERM ends it with status 0
after it has stopped the agents it started; a reset request (see Reset) clears
its memory and keeps the process. Nobody reads its final answers:
the agent must reply with `send_message`. The messages of a failed or
interrupted run are not delivered again, except those that
`wait_for_messages` printed in the code step that was in progress.

### Reset

A reset gives a `bpsa listen` agent a fresh start without stopping its process
(a start takes about 30 s): for a new topic, or for an agent that is stuck.
It is sent with `reset_agent` or with the `reset` command below, to one agent
name or to `all` (every `bpsa listen` agent that reads the folder, except the
sender). No model takes part: the listener finds the request at its next
folder check (every `BPSA_MESSAGE_POLL_SECONDS`; every second while it is in
`wait_for_messages`) and then:

1. leaves the task in progress at once, also in the middle of a model call or
   a code step (as SIGTERM does), and nothing of that task runs afterwards;
2. replaces its agent by a new one, as `/clear` does in the interactive
   session;
3. answers the sender with a message written by code:
   `Reset done: agent <name> cleared its conversation and its Python variables on reset <id> from <sender> (<N> unread messages discarded). Its next task starts in a fresh context.`
   When several requests are found before the agent is replaced (from
   several senders or folders), one replacement serves them all, each sender
   gets its own message, and `<N>` is the total of that replacement:
   `(<N> unread messages discarded in total for the <K> reset requests served by this one replacement)`.

A process that has just started has a fresh context: the reset requests it
finds at its first check of a folder (old requests to its name or to `all`)
are recorded as read and neither carried out, refused nor answered. Requests
written after that check are handled as described here.

Cleared: the conversation, the knowledge notepad, the plan, the Python
variables of the code executor; the browser, GUI and MCP connections of the
old agent are closed and opened again. Kept: the process, the agent name, the
folders, the last id read per sender (no old message is delivered again), and
the session token and cost totals with `BPSA_MAX_SESSION_TOKENS` and
`BPSA_MAX_SESSION_COST`, so a reset is no way around a budget. Not undone:
files, the working directory of the process, imported modules, environment
variables, tmux sessions and child processes of the old context.

Messages around a reset:

- Messages that the listener had queued when it found the request are
  discarded: recorded as read and never shown. With several pending requests
  these are the messages queued before the first of them was found.
- Messages of the request's sender with an id below the request (in the same
  folder) are discarded too. So `send`, `reset`, `send` in this order delivers
  only the last one.
- Every other message reaches the fresh context: the sender's messages with an
  id above the request, other senders' messages found together with it or
  later, also those of another folder.

Agents that the reset agent had started keep running and keep their own
context. When some run, the first task after the reset starts with one line:
`Before your context was reset you started these agents, and they still run: leader1-1 (folder group1). They kept their own context and know nothing of your reset; stop those you no longer need with stop_agent.`
A folder that cannot be listed then (removed, for example) adds no agent to
that line; the listener prints one line naming it and runs the task.

Only `bpsa listen` resets. An interactive session or a `bpsa run` with the
same name keeps its context, shows one line and answers
`Reset refused: agent <name> is not a 'bpsa listen' process, so reset <id> from <sender> changed nothing. Only a 'bpsa listen' process resets.`
It records the request as read, so no later process under that name refuses it again.
The `new` and `wait` commands pass over a reset request without output. A
name that nobody listens under gives no answer. The sender is not checked:
any agent that can write to the folder can reset any listener in it.

Measured with a 5 s poll interval, from the request file to the confirmation
file: 2.0 to 2.9 s for an idle listener, 0.5 s during `wait_for_messages`,
0.3 s during a `time.sleep` code step and 4.4 s during a model call (the poll
wait is part of each).

### Commands for a person or an outside program

```bash
python -m smolagents.bp_messages send --folder F --name ME --to TARGET "text"   # text from stdin when absent or "-"
python -m smolagents.bp_messages reset --folder F --name ME --to TARGET         # reset one `bpsa listen` agent, or `all`; prints the request's file name
python -m smolagents.bp_messages new  --folder F --name ME [--peek]             # print unread messages
python -m smolagents.bp_messages wait --folder F --name ME [--timeout 300]      # block until one arrives; exit 3 on timeout
```

`--folder` and `--name` default to `BPSA_MESSAGE_FOLDERS` (when it has one
line) and `BPSA_AGENT_NAME`.

### Example: one leader and two members

A person acts as `boss` and talks to `leader1` in the folder `leaders`;
`leader1` talks to the members it starts in the folder `group1`.

```bash
S=/path/to/trial
mkdir -p $S/leaders $S/group1 $S/work && cd $S/work
BPSA_AGENT_NAME=leader1 \
BPSA_MESSAGE_FOLDERS="$S/leaders
$S/group1" \
BPSA_AUTO_APPROVE=on BPSA_MESSAGE_POLL_SECONDS=5 BPSA_MAX_GROUP_MEMBERS=2 \
  nohup bpsa --verbose on listen < /dev/null > $S/leader1.log 2>&1 &
echo $! > $S/leader1.pid
# wait until:  grep -c '^Listening for messages: agent leader1' $S/leader1.log   prints 1

python -m smolagents.bp_messages send --folder $S/leaders --name boss --to leader1 \
  "Start two members in your group folder. Ask the first for a haiku about rivers and the second for one about mountains. Send me both, their authors and your choice. Then stop both members."
python -m smolagents.bp_messages wait --folder $S/leaders --name boss --timeout 420

kill -TERM $(cat $S/leader1.pid)    # stops the leader and any member still running
```

In a run on 2026-11-11 the report from `leader1` arrived 2 min 25 s after the
task was sent: 5 s until the leader's poll, 70 s to start the two members one
after the other, 9 s for both members to answer, and about 55 s that the
leader spent in a `time.sleep(60)` it chose to wait for the replies.
`--verbose on` makes the leader's tool calls visible in its log.

### Limits

- A leader that is killed (SIGKILL) or dies from an unhandled error leaves the
  agents it started running; their process ids are in the `.agent.json` files.
- `start_agent` needs `/proc` (Linux).
- Without `fcntl` (Windows) the per-sender lock is absent; one process per
  name is still safe.
- An agent started with `start_agent` has no `start_agent` tool, but the code
  it runs is not sandboxed.

## Tmux Multi-Screen Tools

The `--tmux` flag (or `BPSA_TMUX=1` env var) enables tools that let the agent create and operate multiple independent shell sessions concurrently via tmux. This is useful for running long-running processes (servers, builds, file watchers) in parallel while the agent continues other work.

**Requires:** `tmux` installed on the system.

```bash
bpsa --tmux

# Or via environment variable:
export BPSA_TMUX=1
bpsa
```

### Available Tools

| Tool | Description |
|------|-------------|
| `tmux_create(session_name, command?)` | Create a new named screen session (default command: `bash`) |
| `tmux_send(session_name, text, press_enter?)` | Send keystrokes to a session (default: presses Enter after text) |
| `tmux_read(session_name, lines?)` | Read the current screen content / scrollback (default: 100 lines, max: 2000) |
| `tmux_list()` | List all active agent screen sessions |
| `tmux_destroy(session_name)` | Kill a session and all its processes |
| `tmux_wait(session_name, pattern, timeout?, interval?)` | Poll until a text pattern appears (default: 60s timeout, 1s interval) |

### Example Agent Workflow

```
tmux_create("server")
tmux_send("server", "python app.py")
tmux_wait("server", "Listening on port 8080")

tmux_create("tests")
tmux_send("tests", "pytest tests/ -v")
tmux_wait("tests", "passed")
tmux_read("tests")

tmux_destroy("server")
tmux_destroy("tests")
```

### Session Namespacing

All sessions are automatically prefixed with `bpsa_` internally to avoid collisions with user tmux sessions. The agent uses short names (e.g., `server`) while tmux sees `bpsa_server`. Sessions are automatically cleaned up when the process exits.

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

1. CLI flags (`--load-instructions`, `--auto-approve`, `--verbose`, `--browser`, `--gui-x11`, `--image`, `--tmux`, `--mcp`); there is no `--model` flag, use `/model <id>` inside the REPL
2. Environment variables (`BPSA_*`)
3. Config file (`~/.bpsa.yaml`) - documented but not implemented: `bp_cli.py` does not read it
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
- `tmux` - Multi-screen session management (optional, required for `--tmux`)
