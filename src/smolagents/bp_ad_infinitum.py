# BPSA - Beyond Python SmolAgents
# https://github.com/joaopauloschuler/beyond-python-smolagents
#
# Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and others.
# Refer to the git commit history for individual authorship.
# Licensed under the Apache License, Version 2.0

#!/usr/bin/env python
# coding=utf-8

"""
Ad-Infinitum CLI for Beyond Python SmolAgents.

Autonomous agent cycling: loads tasks from a folder of task files (.md, .py, .sh)
or a single file and runs them repeatedly.

- .md files are treated as agent prompts (run via agent.run())
- .py files are executed directly via the Python interpreter (subprocess)
- .sh files are executed directly via bash (subprocess)

Folder convention:
    tasks/
    +-- _preamble.md          (optional) prepended to ALL prompt tasks
    +-- 01-setup-env.sh       script: install deps, create/clean dirs
    +-- 02-implement.md       prompt: agent does the work
    +-- 03-validate.py        script: programmatic validation
    +-- 04-refine.md          prompt: agent fixes issues
    +-- _postamble.md         (optional) appended to ALL prompt tasks
    +-- _inbox.md             (optional) steering: text written here mid-run reaches the model at the next step

Files starting with '_' are modifiers, not tasks. All other task files are
loaded in alphabetical order. Each becomes one element in the task array.

Environment variables (same BPSA_* as bpsa — including BPSA_SYSTEM_PROMPT_FIRST — plus):
    BPSA_CYCLES         - Number of cycles, 0 = infinite (default: 1)
    BPSA_PLAN_INTERVAL  - Planning interval (default: None = off)
    BPSA_MAX_STEPS      - Max steps per agent run (default: 200)
    BPSA_COOLDOWN       - Seconds to wait between cycles (default: 0)
    BPSA_INJECT_FOLDER  - Inject directory tree (default: true = cwd, false = off, or a path)
    BPSA_BROWSER        - Enable Playwright browser integration (default: false)
    BPSA_GUI            - Enable native GUI interaction tools (default: false)
    BPSA_MAX_SESSION_TOKENS / BPSA_MAX_SESSION_COST - Budget over every prompt task of the run (input + output
                          tokens; USD when the model reports or estimates a cost): warn once at 80%, stop the loop
                          at 100% and exit with code BUDGET_EXIT_CODE (3). 0/unset = no limit.

    Context compression parameters (see bpsa --help or CompressionConfig for details):
    BPSA_COMPRESSION_ENABLED, BPSA_COMPRESSION_KEEP_RECENT_STEPS,
    BPSA_COMPRESSION_MAX_UNCOMPRESSED_STEPS, BPSA_COMPRESSION_KEEP_COMPRESSED_STEPS,
    BPSA_COMPRESSION_MAX_COMPRESSED_STEPS, BPSA_COMPRESSION_TOKEN_THRESHOLD,
    BPSA_COMPRESSION_MODEL, BPSA_COMPRESSION_MAX_SUMMARY_TOKENS,
    BPSA_COMPRESSION_PRESERVE_ERROR_STEPS, BPSA_COMPRESSION_PRESERVE_FINAL_ANSWER_STEPS,
    BPSA_COMPRESSION_MIN_CHARS
"""

import glob
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from smolagents.bp_utils import get_env, get_env_bool, get_env_int


console = Console()

# Exit code of `ad-infinitum` when BPSA_MAX_SESSION_TOKENS or BPSA_MAX_SESSION_COST stops the loop.
BUDGET_EXIT_CODE = 3

_EXTENSION_TO_KIND = {".md": "prompt", ".py": "python", ".sh": "shell"}


@dataclass
class TaskItem:
    """A single task: either an agent prompt or an executable script."""

    name: str  # display name (file basename)
    kind: str  # "prompt" | "python" | "shell"
    content: str  # assembled prompt text (prompts) or raw content (scripts)
    path: str | None  # original file path (needed for script execution)


def _file_kind(filepath: str) -> str | None:
    """Return task kind for a file extension, or None if unsupported."""
    _, ext = os.path.splitext(filepath)
    return _EXTENSION_TO_KIND.get(ext.lower())


# Graceful shutdown flag
_stop_requested = False


def _signal_handler(signum, frame):
    global _stop_requested
    if _stop_requested:
        console.print("\n[bold red]Double Ctrl+C: aborting immediately.[/]")
        sys.exit(1)
    _stop_requested = True
    console.print("\n[yellow]Ctrl+C received. Will stop after current task finishes.[/]")


def fail(msg: str):
    console.print(f"[bold red]Error:[/] {msg}")
    sys.exit(1)


def load_tasks(path: str) -> list[TaskItem]:
    """Load tasks from a folder of task files (.md, .py, .sh) or a single file.

    Folder mode:
        - _preamble.md and _postamble.md are optional wrappers (prompt tasks only)
        - All other *.md, *.py, *.sh files are tasks, sorted alphabetically
        - Each prompt task = preamble + file content + postamble
        - Script tasks (.py, .sh) are executed directly via subprocess

    File mode:
        - Returns a single-element list with a TaskItem.
    """
    if os.path.isdir(path):
        # Collect all supported files, sorted alphabetically
        all_files = sorted(
            f
            for ext in ("*.md", "*.py", "*.sh")
            for f in glob.glob(os.path.join(path, ext))
        )
        if not all_files:
            fail(f"No task files (.md, .py, .sh) found in {path}")

        preamble = ""
        postamble = ""
        task_files = []

        for f in all_files:
            basename = os.path.basename(f)
            if basename == "_preamble.md":
                with open(f, "r", encoding="utf-8") as fh:
                    preamble = fh.read().strip() + "\n\n"
                console.print(f"  [green]Preamble:[/] {basename}")
            elif basename == "_postamble.md":
                with open(f, "r", encoding="utf-8") as fh:
                    postamble = "\n\n" + fh.read().strip()
                console.print(f"  [green]Postamble:[/] {basename}")
            elif not basename.startswith("_"):
                task_files.append(f)

        if not task_files:
            fail(f"No task files found in {path} (files starting with '_' are modifiers, not tasks)")

        tasks = []
        for f in task_files:
            basename = os.path.basename(f)
            kind = _file_kind(f)
            with open(f, "r", encoding="utf-8") as fh:
                content = fh.read().strip()

            if kind == "prompt":
                content = preamble + content + postamble
                console.print(f"  [cyan]Task:[/] {basename}")
            else:
                console.print(f"  [magenta]Script ({kind}):[/] {basename}")

            tasks.append(TaskItem(name=basename, kind=kind, content=content, path=f))

        return tasks

    elif os.path.isfile(path):
        kind = _file_kind(path)
        if kind is None:
            fail(f"Unsupported file type: {path} (expected .md, .py, or .sh)")
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        if not content:
            fail(f"File is empty: {path}")
        basename = os.path.basename(path)
        if kind == "prompt":
            console.print(f"  [cyan]Task:[/] {basename}")
        else:
            console.print(f"  [magenta]Script ({kind}):[/] {basename}")
        return [TaskItem(name=basename, kind=kind, content=content, path=path)]

    else:
        fail(f"Path not found: {path}")

def inject_tree(folder: str) -> str:
    """Generate directory tree string to append to task prompts."""
    from smolagents.bp_tools import inject_tree as _inject_tree
    return _inject_tree(folder)


STEERING_INBOX_NAME = "_inbox.md"


def read_steering_inbox(inbox_path: str | None) -> list[str]:
    """Return the inbox file's text as one message and truncate the file; [] when missing, empty or unreadable."""
    if not inbox_path or not os.path.isfile(inbox_path):
        return []
    try:
        with open(inbox_path, "r+", encoding="utf-8") as fh:
            text = fh.read().strip()
            fh.seek(0)
            fh.truncate()
    except OSError:
        return []
    return [text] if text else []


def steering_inbox_path(task_source: str) -> str | None:
    """Path of the _inbox.md steering file for a task folder; None for a single task file."""
    if not os.path.isdir(task_source):
        return None
    return os.path.join(task_source, STEERING_INBOX_NAME)


def run_script(task: TaskItem) -> subprocess.CompletedProcess:
    """Execute a .py or .sh script directly via subprocess."""
    if task.kind == "python":
        cmd = [sys.executable, task.path]
    elif task.kind == "shell":
        cmd = ["bash", task.path]
    else:
        raise ValueError(f"Unknown script kind: {task.kind}")
    return subprocess.run(cmd)


def add_agent_usage(budget_stats: dict, agent) -> None:
    """Add the agent's monitor token counts and cost to the run totals the session budget compares."""
    from smolagents.bp_cli import get_agent_cost_usd, get_agent_token_usage

    input_tokens, output_tokens, _ = get_agent_token_usage(agent)
    budget_stats["total_input_tokens"] += input_tokens
    budget_stats["total_output_tokens"] += output_tokens
    budget_stats["total_cost_usd"] += get_agent_cost_usd(agent)


def budget_stops_loop(budget_stats: dict, warned: set) -> bool:
    """Print the once-per-limit budget warnings for the run totals; True when a limit is reached."""
    from smolagents.bp_cli import warn_session_budget

    return warn_session_budget(budget_stats, warned, stop_hint="Stopping the loop.") == "exceeded"


def print_banner(config: dict):
    cycles_str = str(config["cycles"]) if config["cycles"] > 0 else "infinite"
    plan_str = str(config["plan_interval"]) if config["plan_interval"] else "off"
    tree_str = config["tree_folder"] if config["tree_folder"] else "off"

    browser_str = "[green]on[/]" if config.get("browser") else "off"
    gui_str = "[green]on[/]" if config.get("gui") else "off"
    mcp_count = len(config.get("mcp") or [])
    mcp_str = f"[green]{mcp_count} server(s)[/]" if mcp_count else "off"

    console.print(
        Panel.fit(
            f"[bold]AD-INFINITUM[/] - Autonomous Agents\n"
            f"Model: [cyan]{config['model_id']}[/] ({config['server_model']})\n"
            f"Tasks: [green]{config['task_count']}[/] | "
            f"Cycles: [green]{cycles_str}[/] | "
            f"Steps/run: [green]{config['max_steps']}[/]\n"
            f"Planning: {plan_str} | "
            f"Inject folder: {tree_str} | "
            f"Cooldown: {config['cooldown']}s\n"
            f"Browser: {browser_str} | "
            f"GUI: {gui_str} | "
            f"MCP: {mcp_str}",
            border_style="blue",
        )
    )
    console.print(
        Panel.fit(
            "[bold red]EXTREME SECURITY RISK[/]\n"
            "Running autonomously with full system access.\n"
            "Only run inside a securely isolated environment.\n"
            "[bold]USE AT YOUR OWN RISK.[/]",
            border_style="red",
        )
    )
    console.print("[dim]Press Ctrl+C to stop after current task. Double Ctrl+C to abort.[/]\n")


def run_loop(model, tasks, cycles, max_steps, plan_interval, tree_folder, cooldown,
             browser_enabled=False, gui_enabled=False, image_enabled=False, mcp_servers=None, inbox_path=None):
    """Core autonomous loop: cycles x tasks, fresh agent per task. Returns True when the session budget stopped it.
    inbox_path (the task folder's _inbox.md) feeds each prompt task's agent.steering_source."""
    from smolagents.bp_cli import (
        _shutdown_browser, _shutdown_gui, _shutdown_mcp, build_agent, get_agent_token_usage, session_budget_state,
    )

    original_dir = os.getcwd()
    total_start = time.time()
    cycle = 0
    total_tasks_run = 0
    budget_stats = {"total_input_tokens": 0, "total_output_tokens": 0, "total_cost_usd": 0.0}
    budget_warned = set()
    budget_exceeded = False

    while cycles == 0 or cycle < cycles:
        cycle += 1
        cycle_label = f"{cycle}" if cycles > 0 else f"{cycle}"
        cycle_limit = f"/{cycles}" if cycles > 0 else ""

        console.print(Rule(f"[bold]Cycle {cycle_label}{cycle_limit}[/]", style="blue"))

        for task_idx, task in enumerate(tasks):
            if _stop_requested:
                break

            os.chdir(original_dir)

            task_label = f"Task {task_idx + 1}/{len(tasks)} ({task.name})"
            console.print(f"[dim]{task_label} starting...[/]")

            task_start = time.time()

            if task.kind == "prompt":
                # Inject directory tree if configured
                prompt = task.content
                if tree_folder:
                    prompt += inject_tree(tree_folder)

                agent = build_agent(model, browser_enabled=browser_enabled, gui_enabled=gui_enabled, image_enabled=image_enabled, mcp_servers=mcp_servers)
                if plan_interval:
                    agent.planning_interval = plan_interval
                if inbox_path:
                    agent.steering_source = lambda: read_steering_inbox(inbox_path)

                try:
                    agent.run(prompt, reset=True)
                    elapsed = time.time() - task_start
                    total_tasks_run += 1

                    in_tok, out_tok, _ = get_agent_token_usage(agent)
                    console.print(
                        f"[green]OK[/] {task_label} | {elapsed:.1f}s | "
                        f"In: {in_tok:,} | Out: {out_tok:,}"
                    )
                except KeyboardInterrupt:
                    console.print(f"[yellow]{task_label} interrupted.[/]")
                    break
                except Exception as e:
                    elapsed = time.time() - task_start
                    total_tasks_run += 1
                    console.print(f"[red]FAIL[/] {task_label} | {elapsed:.1f}s | {e}")
                finally:
                    add_agent_usage(budget_stats, agent)
                    _shutdown_mcp(agent)
                    _shutdown_browser(agent)
                    _shutdown_gui(agent)
                if budget_stops_loop(budget_stats, budget_warned):
                    budget_exceeded = True
                    break

            else:
                # Script execution (python or shell)
                try:
                    result = run_script(task)
                    elapsed = time.time() - task_start
                    total_tasks_run += 1

                    if result.returncode == 0:
                        console.print(f"[green]OK[/] {task_label} | {elapsed:.1f}s | exit 0")
                    else:
                        console.print(
                            f"[red]FAIL[/] {task_label} | {elapsed:.1f}s | exit {result.returncode}"
                        )
                except KeyboardInterrupt:
                    console.print(f"[yellow]{task_label} interrupted.[/]")
                    break
                except Exception as e:
                    elapsed = time.time() - task_start
                    total_tasks_run += 1
                    console.print(f"[red]FAIL[/] {task_label} | {elapsed:.1f}s | {e}")

        if budget_exceeded:
            console.print(f"\n[bold red]Session budget reached: stopped in cycle {cycle}.[/]")
            break
        if _stop_requested:
            console.print(f"\n[yellow]Stopped after cycle {cycle}.[/]")
            break

        # Cooldown between cycles
        if cooldown > 0 and (cycles == 0 or cycle < cycles):
            console.print(f"[dim]Cooldown: {cooldown}s...[/]")
            time.sleep(cooldown)

    # Session summary
    total_elapsed = time.time() - total_start
    os.chdir(original_dir)
    console.print()
    console.print(Rule("[bold]Session Summary[/]", style="green"))
    console.print(f"  Cycles completed: [green]{cycle}[/]")
    console.print(f"  Tasks run: [green]{total_tasks_run}[/]")
    console.print(f"  Total time: [green]{total_elapsed:.1f}s[/]")
    budget_message = session_budget_state(budget_stats)[1]
    if budget_message:
        console.print(f"  Budget: {budget_message}")
    return budget_exceeded


def _resolve_tree_folder(tree_folder):
    """Resolve tree_folder parameter: None=use env, False=off, True=cwd, str=path."""
    if tree_folder is not None:
        if tree_folder is False:
            return None
        if tree_folder is True:
            return os.getcwd()
        return tree_folder
    # Fall back to env var
    tree_folder_raw = get_env("BPSA_INJECT_FOLDER")
    if tree_folder_raw is not None and tree_folder_raw.lower() == "false":
        return None
    if tree_folder_raw is None or tree_folder_raw.lower() == "true":
        return os.getcwd()
    return tree_folder_raw


def run_ad_infinitum(
    task_source,
    cycles=None,
    max_steps=None,
    plan_interval=None,
    tree_folder=None,
    cooldown=None,
    model=None,
    browser_enabled=None,
    gui_enabled=None,
    image_enabled=None,
    mcp_servers=None,
    banner=True,
):
    """Run ad-infinitum programmatically.

    Args:
        task_source: Folder of task files (.md, .py, .sh) or a single task file path.
        cycles: Number of cycles (0=infinite). Default: BPSA_CYCLES or 1.
        max_steps: Max steps per agent run. Default: BPSA_MAX_STEPS or 200.
        plan_interval: Planning interval. Default: BPSA_PLAN_INTERVAL or None.
        tree_folder: Inject directory tree. None=use env, False=off, True=cwd, str=path.
        cooldown: Seconds between cycles. Default: BPSA_COOLDOWN or 0.
        model: Pre-built model instance. If None, builds from BPSA_* env vars.
        browser_enabled: Enable browser. Default: BPSA_BROWSER or False.
        gui_enabled: Enable GUI. Default: BPSA_GUI or False.
        image_enabled: Enable image tools. Default: BPSA_IMAGE or False.
        mcp_servers: List of MCP server specs (URLs or commands).
        banner: Whether to print the startup banner. Default: True.

    Returns BUDGET_EXIT_CODE when BPSA_MAX_SESSION_TOKENS or BPSA_MAX_SESSION_COST stopped the loop, else 0.

    Example::

        from smolagents.bp_ad_infinitum import run_ad_infinitum
        run_ad_infinitum("./tasks/", cycles=3, cooldown=10)
    """
    from smolagents.bp_cli import build_model, check_required_env, try_load_dotenv

    # Install Ctrl+C handler
    signal.signal(signal.SIGINT, _signal_handler)

    # Load .env and validate
    try_load_dotenv()
    check_required_env()

    # Resolve parameters from kwargs or env vars
    cycles = cycles if cycles is not None else get_env_int("BPSA_CYCLES", 1)
    max_steps = max_steps if max_steps is not None else get_env_int("BPSA_MAX_STEPS", 200)
    cooldown = cooldown if cooldown is not None else get_env_int("BPSA_COOLDOWN", 0)
    if plan_interval is None:
        plan_interval_val = get_env("BPSA_PLAN_INTERVAL")
        plan_interval = int(plan_interval_val) if plan_interval_val else None
    tree_folder = _resolve_tree_folder(tree_folder)
    browser_enabled = browser_enabled if browser_enabled is not None else get_env_bool("BPSA_BROWSER")
    gui_enabled = gui_enabled if gui_enabled is not None else get_env_bool("BPSA_GUI")
    image_enabled = image_enabled if image_enabled is not None else get_env_bool("BPSA_IMAGE")

    # Load tasks
    console.print("[dim]Loading tasks...[/]")
    tasks = load_tasks(task_source)

    if banner:
        config = {
            "model_id": get_env("BPSA_MODEL_ID"),
            "server_model": get_env("BPSA_SERVER_MODEL", "OpenAIServerModel"),
            "task_count": len(tasks),
            "cycles": cycles,
            "max_steps": max_steps,
            "plan_interval": plan_interval,
            "tree_folder": tree_folder,
            "cooldown": cooldown,
            "browser": browser_enabled,
            "gui": gui_enabled,
            "image": image_enabled,
            "mcp": mcp_servers,
        }
        print_banner(config)

    # Build model if not provided
    if model is None:
        model = build_model()

    # Run the loop
    budget_exceeded = run_loop(
        model, tasks, cycles, max_steps, plan_interval, tree_folder, cooldown,
        browser_enabled=browser_enabled, gui_enabled=gui_enabled, image_enabled=image_enabled,
        mcp_servers=mcp_servers, inbox_path=steering_inbox_path(task_source),
    )
    return BUDGET_EXIT_CODE if budget_exceeded else 0


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="ad-infinitum",
        description="Ad-Infinitum: Autonomous agent cycling for Beyond Python SmolAgents",
    )
    parser.add_argument(
        "task_source",
        help="Folder of task files (.md, .py, .sh) or a single task file",
    )
    parser.add_argument(
        "-c", "--cycles",
        type=int,
        default=None,
        help="Number of cycles, 0 = infinite (overrides BPSA_CYCLES, default: 1)",
    )
    parser.add_argument(
        "--browser", action="store_true", default=None,
        help="Enable Playwright browser integration (overrides BPSA_BROWSER)",
    )
    parser.add_argument(
        "--gui-x11", action="store_true", default=None,
        help="Enable native GUI interaction tools (overrides BPSA_GUI)",
    )
    parser.add_argument(
        "--image", action="store_true", default=None,
        help="Enable image analysis and drawing tools (overrides BPSA_IMAGE)",
    )
    parser.add_argument(
        "--mcp", action="append", metavar="URL_OR_CMD", dest="mcp",
        help="MCP server to connect (URL or shell command); repeatable",
    )
    args = parser.parse_args()

    from smolagents.bp_cli import _parse_mcp_servers
    mcp_servers = _parse_mcp_servers(args.mcp or []) or None

    exit_code = run_ad_infinitum(
        task_source=args.task_source,
        cycles=args.cycles,
        browser_enabled=args.browser if args.browser else None,
        gui_enabled=args.gui_x11 if args.gui_x11 else None,
        image_enabled=args.image if args.image else None,
        mcp_servers=mcp_servers,
    )
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
