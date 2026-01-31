## Guidance for AI coding agents working on Beyond Python Smolagents

This file gives concise, repository-specific instructions so an AI coding agent can be productive quickly.

Highlights

- Big picture: this repo is an extended fork of HuggingFace's `smolagents`. Core logic lives under `src/smolagents/` (agents, tools, thinkers) and higher-level examples/tests live in `examples/` and `tests/`.

- Read first (fast path): `src/smolagents/agents.py` (agent lifecycle, tools, memory), `src/smolagents/bp_thinkers.py` (multi-agent solvers: `evolutive_problem_solver` / `fast_solver`), `src/smolagents/bp_tools.py` (repo-specific tools and file/OS helpers), `src/smolagents/bp_cli.py` (interactive CLI `bpsa`: REPL, one-shot mode, slash commands, token tracking), `src/smolagents/cli.py` (original smolagents CLI).

Essential repo conventions (do not change without checking usage)

- Tools are Python functions decorated with `@tool` in `bp_tools.py` (or default tools in `default_tools.py`). They must return simple JSON-serializable values and keep stable `name` attributes.
- Many orchestration flows are file-based. Example filenames used repeatedly: `solution1(.py)`, `solution2`, `solution3`, `best_solution.best`, and `advices.notes`. Avoid renaming these without updating every call site in `bp_thinkers.py` and tests.
- Agents can be constructed with `add_base_tools=True` (injects `TOOL_MAPPING`) or with an explicit tool list. Prefer explicit lists for scoped runs and testing.
- Several tools execute arbitrary OS commands or write files (`run_os_command`, `save_string_to_file`, `replace_on_file`, `force_directories`). Treat these as privileged operations; do not run untrusted PR code in CI or shared runners.

Developer workflows & examples

- Install for dev and tests: pip install -e .[test] (Python >= 3.10). Tests are run with pytest from the repo root (see `pyproject.toml` optional deps).
- Run the BP CLI (`bpsa`): configure via `BPSA_MODEL_ID`, `BPSA_SERVER_MODEL`, `BPSA_API_ENDPOINT`, `BPSA_KEY_VALUE` env vars (or a `.env` file), then run `bpsa` for interactive REPL or `bpsa run "task"` for one-shot. See `bp_cli.py` for implementation.
- Run the original CLI: python -m smolagents.cli "<prompt>" --model-type LiteLLMModel --model-id <id> --api-key <key>. `cli.py` shows how models (LiteLLMModel, TransformersModel, InferenceClientModel, OpenAIServerModel) and tools are resolved.
- Small example: create a CodeAgent with explicit tools (see `bp_thinkers.get_local_agent`) and call agent.run(prompt). Use `source_code_to_string()` to ingest files for documentation tasks.

Integration points and tests to check when changing behavior

- Models: `src/smolagents/models.py` — changing model APIs requires updating `tests/test_models.py`.
- HuggingFace Hub: code uses `huggingface_hub` (create_repo, upload_folder, snapshot_download) — guard network operations behind CLI flags and env vars.
- Heavy/optional deps (docker, torch, vllm, e2b, gradio, litellm) are declared in `pyproject.toml` and gated with pytest markers in `tests/utils/markers.py`.

Concrete examples to reference

- File tools: see `bp_tools.save_string_to_file`, `load_string_from_file`, `replace_on_file_with_files` for safe file edits and encodings.
- OS execution: `bp_tools.run_os_command` uses subprocess (prlimit wrapper) — returns combined stdout/stderr string and handles timeouts.
- Line helpers: `print_source_code_lines`, `get_line_from_file`, `replace_line_in_file` are intended for debugging compiler errors in solution files.

Quick validation checklist

- Run unit tests: pytest -q (from repo root).
- If you change a tool signature: run `tests/test_tools.py` and `tests/test_tool_validation.py`.
- If you change solvers in `bp_thinkers.py`: run a local smoke run with a small model (or mock Model) and verify the expected files (`solution1`, `best_solution.best`, `advices.notes`) are created/updated.

Security note

- This project intentionally exposes file and command execution tools. Do not run arbitrary code from untrusted contributors in shared CI/workers. Prefer isolated sandboxes or mock executions in tests.