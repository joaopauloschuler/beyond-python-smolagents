
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

Files starting with '_' are modifiers, not tasks. All other task files are
loaded in alphabetical order. Each becomes one element in the task array.

Environment variables (same BPSA_* as bpsa, plus):
    BPSA_CYCLES         - Number of cycles, 0 = infinite (default: 1)
    BPSA_PLAN_INTERVAL  - Planning interval (default: None = off)
    BPSA_MAX_STEPS      - Max steps per agent run (default: 200)
    BPSA_COOLDOWN       - Seconds to wait between cycles (default: 0)
    BPSA_INJECT_FOLDER  - Inject directory tree (default: true = cwd, false = off, or a path)
    BPSA_BROWSER        - Enable Playwright browser integration (default: false)
    BPSA_GUI            - Enable native GUI interaction tools (default: false)

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


def run_script(task: TaskItem) -> subprocess.CompletedProcess:
    """Execute a .py or .sh script directly via subprocess."""
    if task.kind == "python":
        cmd = [sys.executable, task.path]
    elif task.kind == "shell":
        cmd = ["bash", task.path]
    else:
        raise ValueError(f"Unknown script kind: {task.kind}")
    return subprocess.run(cmd)


def print_banner(config: dict):
    cycles_str = str(config["cycles"]) if config["cycles"] > 0 else "infinite"
    plan_str = str(config["plan_interval"]) if config["plan_interval"] else "off"
    tree_str = config["tree_folder"] if config["tree_folder"] else "off"

    browser_str = "[green]on[/]" if config.get("browser") else "off"
    gui_str = "[green]on[/]" if config.get("gui") else "off"

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
            f"GUI: {gui_str}",
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
             browser_enabled=False, gui_enabled=False):
    """Core autonomous loop: cycles x tasks, fresh agent per task."""
    from smolagents.bp_cli import _shutdown_browser, _shutdown_gui, build_agent

    original_dir = os.getcwd()
    total_start = time.time()
    cycle = 0
    total_tasks_run = 0

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

                agent = build_agent(model, browser_enabled=browser_enabled, gui_enabled=gui_enabled)
                if plan_interval:
                    agent.planning_interval = plan_interval

                try:
                    agent.run(prompt, reset=True)
                    elapsed = time.time() - task_start
                    total_tasks_run += 1

                    # Get token usage
                    try:
                        usage = agent.monitor.get_total_token_counts()
                        in_tok, out_tok = usage.input_tokens, usage.output_tokens
                    except Exception:
                        in_tok, out_tok = 0, 0

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
                    _shutdown_browser(agent)
                    _shutdown_gui(agent)

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
        "--gui", action="store_true", default=None,
        help="Enable native GUI interaction tools (overrides BPSA_GUI)",
    )
    args = parser.parse_args()

    # Install Ctrl+C handler
    signal.signal(signal.SIGINT, _signal_handler)

    # Load .env
    from smolagents.bp_cli import build_model, check_required_env, try_load_dotenv
    try_load_dotenv()
    check_required_env()

    # Read config from env
    cycles = args.cycles if args.cycles is not None else get_env_int("BPSA_CYCLES", 1)
    plan_interval_val = get_env("BPSA_PLAN_INTERVAL")
    plan_interval = int(plan_interval_val) if plan_interval_val else None
    max_steps = get_env_int("BPSA_MAX_STEPS", 200)
    cooldown = get_env_int("BPSA_COOLDOWN", 0)
    tree_folder_raw = get_env("BPSA_INJECT_FOLDER")
    if tree_folder_raw is not None and tree_folder_raw.lower() == "false":
        tree_folder = None
    elif tree_folder_raw is None or tree_folder_raw.lower() == "true":
        tree_folder = os.getcwd()
    else:
        tree_folder = tree_folder_raw

    browser_enabled = args.browser if args.browser else get_env_bool("BPSA_BROWSER")
    gui_enabled = args.gui if args.gui else get_env_bool("BPSA_GUI")

    # Load tasks
    console.print("[dim]Loading tasks...[/]")
    tasks = load_tasks(args.task_source)

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
    }
    print_banner(config)

    # Build model (reused across all cycles)
    model = build_model()

    # Run the loop
    run_loop(model, tasks, cycles, max_steps, plan_interval, tree_folder, cooldown,
             browser_enabled=browser_enabled, gui_enabled=gui_enabled)


if __name__ == "__main__":
    main()
