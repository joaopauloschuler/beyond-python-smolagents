# Task List — Feature & Enhancement Ideas

A running list of work for BPSA (Beyond Python SmolAgents). Each entry names
the problem, the change, and the code it touches, so the next agent can act
on it without re-deriving the context. Mark an entry `[x]` when it lands and
append a short "LANDED" note with what was verified.

## Conventions for every task below

- **Never break the application on a missing field.** Every task that reads
  optional data from a provider response (`usage.prompt_tokens_details`,
  `provider`, `cost`, `context_length`, ...) must treat the field as absent
  by default. A provider that omits it, returns `None`, or returns an
  unexpected type must produce the same behaviour as today: no exception,
  no changed output except the new figure being left out. Test each reader
  with a response object that lacks the field.
- **One env var per knob, prefixed `BPSA_`.** Document each new variable in
  `docs/CLI.md` (Environment Variables table) and in the `--help` text in
  `src/smolagents/bp_cli.py`.
- **`OpenAIServerModel` is the primary target.** OpenRouter, DeepSeek, Poe and
  OpenAI all speak this protocol. Other model classes must keep working
  unchanged; a feature that cannot be implemented for them is simply skipped
  for them.
- **Follow `../neural-api/docs/CODING-AND-COMMUNICATION-GUIDE.md`**: look for
  existing code before adding any, name the actor in every report, and do not
  write code unless asked.

## Baseline (existing test suite)

- Commit: `a8fb19f` (branch `a1`), date 2026-09-14.
- Command: `cd /home/bpsa/app/bpsa && python -m pytest ./tests/ -q -p no:cacheprovider`
- Result: **not run**. `python` resolves to `/home/bpsa/x/bin/python`
  (venv, Python 3.12.3) and printed `No module named pytest`; the same holds
  for `/usr/bin/python3`. No `pytest` or `_pytest` package exists under any
  `site-packages` on this machine. The baseline agent was told not to
  install packages, so it stopped here.
- Counts: passed / failed / skipped / errors / xfailed = unknown; wall time
  unknown.
- [ ] Install the test extras (`pip install -e '.[test]'` inside
  `/home/bpsa/x`, which is what `pyproject.toml` `[project.optional-dependencies].test`
  lists: pytest, pytest-datadir, pytest-timeout, pandas, ...) and rerun the
  command above to fill in the counts and the list of pre-existing failures.

Coding agents must run the same pytest command, compare their counts against
the numbers recorded here, and leave any pre-existing failure alone; until the
counts above are filled in, an agent must run the suite before and after its
change and report both results.

## Provider visibility (turn summary and `/show-stats`)

These share one blind spot: `TokenUsage` in `src/smolagents/monitoring.py`
holds only `input_tokens` and `output_tokens`, so the REPL cannot show whether
`BPSA_HAS_SESSION_ID`, `BPSA_PROVIDER_ORDER` and `BPSA_SYSTEM_PROMPT_FIRST`
are paying off. `OpenAIModel.generate` and `OpenAIModel.generate_stream` in
`src/smolagents/models.py` already hold the full response object and drop
these fields when they build `TokenUsage`.

- [ ] **Cache hit rate per turn.** OpenRouter returns
      `usage.prompt_tokens_details.cached_tokens`, DeepSeek returns
      `usage.prompt_cache_hit_tokens` and `usage.prompt_cache_miss_tokens`,
      OpenAI returns `usage.prompt_tokens_details.cached_tokens`. Add a
      `cached_input_tokens: int = 0` field to `TokenUsage`, fill it in
      `OpenAIModel.generate` / `generate_stream` from whichever field is
      present (0 when none is), and carry it through
      `agglomerate_stream_deltas` and `Monitor.get_total_token_counts`.
      `print_turn_summary` in `bp_cli.py` then prints `Cache: 87%`
      (cached / input) when the turn had any cached tokens. This lets the
      user see immediately whether `BPSA_HAS_SESSION_ID` is working.
      Must not break when `usage` is `None` or lacks the detail object.
- [ ] **Show which provider served the request.** OpenRouter puts a
      `provider` string in every response body (`response.provider` on the
      OpenAI SDK object, reachable via `getattr` or `model_extra`). Store it
      on the `ChatMessage` (or on `TokenUsage` as `provider: str | None`)
      and print `via DeepInfra` in the turn summary. Verifies
      `BPSA_PROVIDER_ORDER` and sticky routing without guessing. When the
      field is absent (OpenAI, DeepSeek), print nothing.
- [ ] **Cost per turn and per session.** OpenRouter reports exact cost when
      the request body contains `usage: {"include": true}` — one more entry
      in the `extra_body` that `build_model` in `bp_cli.py` already
      assembles; the response then carries `usage.cost`. For DeepSeek and
      OpenAI add two optional env vars, `BPSA_PRICE_INPUT_PER_M` and
      `BPSA_PRICE_OUTPUT_PER_M` (USD per million tokens), and estimate cost
      from `TokenUsage`; count cached input tokens at the input price unless
      a `BPSA_PRICE_CACHED_INPUT_PER_M` is given. Add `cost_usd: float`
      to `TokenUsage`, accumulate it in `session_stats` next to
      `total_input_tokens`, show it in `print_turn_summary` and
      `print_stats`, and persist it through `/session-save` /
      `/session-load`. Money is what users actually budget by. When
      neither the response cost nor the price env vars are available, show
      no cost line.

## Configuration ergonomics

- [ ] **`/model <id>` to switch the main model mid-session.**
      `build_model(override_model_id=...)` in `bp_cli.py` already accepts an
      override id, and `cmd_compression_model` already does this for the
      compression model. Rebuild the main model with the new id, keep
      `agent.memory` intact, assign `agent.model`, and update the banner /
      `session_stats` model name. Lets a user draft with a cheap model and
      finish with a strong one. `/model` with no argument prints the current
      model id. Add the command to the completer list, `/help` table and
      `docs/CLI.md`.
- [ ] **`/show-config`.** Print the effective settings in one table: model
      class, endpoint, model id, masked API key (first 4 and last 4
      characters), provider order, current OpenRouter session id
      (`current_session_id()`), system-prompt position
      (`BPSA_SYSTEM_PROMPT_FIRST`), executor, max steps, planning interval,
      compression thresholds (`agent.compression_config`), and which
      optional tool sets are enabled (browser, gui, image, tmux, mcp). Right
      now the only way to check any of this is to inspect the environment by
      hand. Note that `docs/CLI.md` promises a `~/.bpsa.yaml` config layer
      that `bp_cli.py` never reads; `/show-config` should state the actual
      source of each value (env, `.env`, default).
- [ ] **Startup connectivity check.** After `build_model` and before the
      banner, send one tiny request (a single short user message with
      `max_tokens` around 8) and fail with a clear message on a bad key,
      wrong endpoint, or unknown model id. This matches the "fail fast" rule
      in `docs/CLI.md` (Startup Behavior). Today the first failure appears
      only after the user has typed a full prompt. Skip the check for local
      model classes (`TransformersModel`, `MLXModel`, `VLLMModel`) and allow
      `BPSA_SKIP_CONNECTIVITY_CHECK=1` to bypass it. Report the round-trip
      time in the banner so the user also sees endpoint latency.

## Context awareness and budgets

- [ ] **Know the model's context length.** OpenRouter's `GET /models`
      endpoint returns `context_length` per model id; OpenAI-compatible
      endpoints that do not serve it fall back to a `BPSA_CONTEXT_LENGTH`
      env var. With that number `print_turn_summary` can show `Context: 61%`
      (last-turn `input_tokens` over the window) instead of a raw character
      count, and `BPSA_COMPRESSION_TOKEN_THRESHOLD` (currently `0` = off)
      can default to a fraction of the real window. Fetch once at startup,
      never on the hot path; on any failure keep the current character
      display.
- [ ] **Session budget.** `BPSA_MAX_SESSION_TOKENS` and
      `BPSA_MAX_SESSION_COST` (USD, requires the cost task above). The REPL
      warns once at 80% and stops accepting new agent turns at 100%
      (slash commands still work so the user can `/session-save`).
      `ad-infinitum` (`bp_ad_infinitum.py`) stops the cycle loop at 100% and
      exits with a non-zero code. Protects unattended runs from an open-ended
      bill. `0` or unset means no limit.

## Interaction while the agent is working

- [ ] **Steer while it runs.** A background thread reads lines while the
      agent works and puts them in a queue. At the next step boundary the
      agent drains the queue into memory, and the model sees
      "user update: also run the tests" at its next call. Non-blocking for
      the user, no step lost. This is the interaction feature the user
      misses most.
      Mechanics: today the REPL thread sits inside `agent.run()` and
      `prompt_toolkit` is not running, so typed text goes nowhere and the
      only escape is Ctrl+C (a hard abort that loses the in-flight step).
      `_run_stream` in `src/smolagents/agents.py` already checks
      `interrupt_switch` at the top of every step; drain the queue there.
      `write_memory_to_messages` rebuilds the context before every model
      call, so anything appended to memory between steps is seen on the
      next call with no other change.
      Decisions: (a) no new step class. When a message is queued, the
      agent drains it at the step boundary and appends it to the messages
      sent to the model as a normal user-role message, placed at the end,
      after the run output of the step that just finished. To keep it in
      context on later calls, store the text on the finished `ActionStep`
      (a `user_message` field next to `observations`) and have
      `ActionStep.to_messages` in `src/smolagents/memory.py` emit it as one
      `MessageRole.USER` message right after the observation message; the
      step's `dict()` then carries it through compression and
      `/session-save` with no other change;
      (b) use `prompt_toolkit`'s `patch_stdout` so the input line stays
      pinned while step output scrolls above it; (c) one listener thread
      owns stdin and hands lines to whichever consumer is waiting (the
      steering queue, `interactive_approval_callback`, a future
      `ask_user` tool), otherwise two readers lose keystrokes;
      (d) add one sentence to the system prompt saying user updates may
      arrive between steps and take priority over the original task;
      (e) disable the listener when stdin is not a tty; (f) let
      `ad-infinitum` feed the same queue from a file such as
      `tasks/_inbox.md`, which also makes the mechanism testable without a
      terminal.
