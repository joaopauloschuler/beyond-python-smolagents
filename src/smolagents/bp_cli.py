#!/usr/bin/env python
# coding=utf-8

"""
Beyond Python SmolAgents CLI (bpsa)

Interactive REPL and one-shot CLI for CodeAgent with DEFAULT_THINKER_TOOLS.

Environment variables:
    BPSA_SERVER_MODEL   - Model class name (default: OpenAIServerModel)
    BPSA_API_ENDPOINT   - API endpoint URL
    BPSA_KEY_VALUE      - API key
    BPSA_MODEL_ID       - Model identifier (required)
    BPSA_POSTPEND_STRING - String to append to model outputs (default: '')
    BPSA_GLOBAL_EXECUTOR - Executor type (default: exec)
    BPSA_MAX_TOKENS     - Max tokens for model (default: 64000)
    BPSA_VERBOSE        - Verbose output (0 or 1, default: 1)
"""

import os
import re
import sys
import threading
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.markdown import Markdown
from rich.table import Table


VERSION = "1.23-bp"
console = Console()

# Mutable state for verbose mode, accessible by the step callback
_verbose = True

MODEL_CLASS_MAP = {
    "OpenAIServerModel": "OpenAIServerModel",
    "OpenAIModel": "OpenAIServerModel",
    "LiteLLMModel": "LiteLLMModel",
    "LiteLLMRouterModel": "LiteLLMRouterModel",
    "InferenceClientModel": "InferenceClientModel",
    "TransformersModel": "TransformersModel",
    "AzureOpenAIServerModel": "AzureOpenAIServerModel",
    "AzureOpenAIModel": "AzureOpenAIServerModel",
    "AmazonBedrockModel": "AmazonBedrockModel",
    "VLLMModel": "VLLMModel",
    "MLXModel": "MLXModel",
    "GoogleColabModel": "GoogleColabModel",
}


AGENT_INSTRUCTION_FILES = [
    "CLAUDE.md",
    "AGENTS.md",
    ".github/copilot-instructions.md",
]


# Required env vars per model class (beyond BPSA_MODEL_ID which is always required)
MODEL_REQUIRED_VARS = {
    "OpenAIServerModel": ["BPSA_KEY_VALUE", "BPSA_API_ENDPOINT"],
    "AzureOpenAIServerModel": ["BPSA_KEY_VALUE", "BPSA_API_ENDPOINT"],
    "LiteLLMModel": ["BPSA_KEY_VALUE"],
    "LiteLLMRouterModel": ["BPSA_KEY_VALUE"],
    "InferenceClientModel": ["BPSA_KEY_VALUE"],
    "TransformersModel": [],
    "AmazonBedrockModel": [],
    "VLLMModel": [],
    "MLXModel": [],
    "GoogleColabModel": [],
}


class Spinner:
    """A simple threaded spinner that doesn't conflict with agent output."""
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Thinking..."):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        i = 0
        while not self._stop_event.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            print(f"\r{frame} {self.message}", end="", flush=True)
            i += 1
            self._stop_event.wait(0.1)
        # Clear the spinner line
        print("\r" + " " * (len(self.message) + 4) + "\r", end="", flush=True)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            self._thread = None


# Global spinner instance
_spinner = Spinner("Agent is thinking...")


def _extract_thoughts(model_output: str) -> str:
    """Extract content from <thoughts> tags in model output."""
    if not model_output:
        return ""
    match = re.search(r"<thoughts>(.*?)</thoughts>", model_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _compact_step_callback(step):
    """Step callback that prints a compact one-liner when verbose is off."""
    if _verbose:
        return

    _spinner.stop()

    step_num = getattr(step, "step_number", "?")
    duration = ""
    if hasattr(step, "timing") and step.timing and step.timing.duration is not None:
        duration = f"{step.timing.duration:.1f}s"

    tokens = ""
    if hasattr(step, "token_usage") and step.token_usage:
        total = step.token_usage.input_tokens + step.token_usage.output_tokens
        tokens = f"{total:,} tok"

    # Extract thoughts for a brief summary
    thoughts = ""
    model_output = getattr(step, "model_output", None)
    if isinstance(model_output, str):
        thoughts = _extract_thoughts(model_output)
    if len(thoughts) > 80:
        thoughts = thoughts[:77] + "..."

    is_final = getattr(step, "is_final_answer", False)
    has_error = getattr(step, "error", None) is not None

    # Build the status icon
    if has_error:
        icon = "[red]✗[/]"
    elif is_final:
        icon = "[green]★[/]"
    else:
        icon = "[green]✓[/]"

    # Build the line
    parts = [f"{icon} [bold]Step {step_num}[/]"]
    if duration:
        parts.append(f"[dim]{duration}[/]")
    if tokens:
        parts.append(f"[dim]{tokens}[/]")
    if is_final:
        parts.append("[bold green]Final answer[/]")
    elif has_error:
        error_msg = str(step.error)[:60]
        parts.append(f"[red]{error_msg}[/]")
    elif thoughts:
        parts.append(f"[dim italic]{thoughts}[/]")

    console.print(" | ".join(parts))


def fail(msg: str):
    console.print(f"[bold red]Error:[/] {msg}", highlight=False)
    sys.exit(1)


def get_env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def try_load_dotenv():
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(env_path):
        load_dotenv(env_path)
        console.print(f"[green]Loaded .env from:[/] {env_path}")
    else:
        console.print("[dim]No .env file found in current directory.[/]")


def check_required_env():
    """Check all required env vars and report missing ones."""
    model_id = get_env("BPSA_MODEL_ID")
    server_model = get_env("BPSA_SERVER_MODEL", "OpenAIServerModel")

    missing = []
    if not model_id:
        missing.append("BPSA_MODEL_ID")

    if server_model not in MODEL_CLASS_MAP:
        supported = ", ".join(sorted(MODEL_CLASS_MAP.keys()))
        fail(f"Unsupported BPSA_SERVER_MODEL: {server_model}. Supported: {supported}")

    canonical_name = MODEL_CLASS_MAP[server_model]
    for var in MODEL_REQUIRED_VARS.get(canonical_name, []):
        if not get_env(var):
            missing.append(var)

    if missing:
        console.print(f"[bold red]Error:[/] Missing required environment variables for {server_model}:\n")
        for var in missing:
            console.print(f"  [bold red]- {var}[/] is not set")
        console.print("\n[bold]All available BPSA variables:[/]\n")
        console.print("  [bold]BPSA_MODEL_ID[/]        (required) Model identifier, e.g. Gemini-2.5-Flash")
        console.print("  [bold]BPSA_SERVER_MODEL[/]     Model class (default: OpenAIServerModel)")
        console.print("  [bold]BPSA_API_ENDPOINT[/]     API endpoint URL")
        console.print("  [bold]BPSA_KEY_VALUE[/]        API key for authentication")
        console.print("  [bold]BPSA_POSTPEND_STRING[/]  String set on model.postpend_string (default: '')")
        console.print("  [bold]BPSA_GLOBAL_EXECUTOR[/]  Executor type (default: exec)")
        console.print("  [bold]BPSA_MAX_TOKENS[/]       Max tokens for model (default: 64000)")
        console.print("  [bold]BPSA_VERBOSE[/]          Verbose output, 0 or 1 (default: 0)")
        console.print("\nExample:")
        console.print("  export BPSA_MODEL_ID=Gemini-2.5-Flash")
        console.print("  export BPSA_SERVER_MODEL=OpenAIServerModel")
        console.print("  export BPSA_API_ENDPOINT=https://api.poe.com/v1")
        console.print("  export BPSA_KEY_VALUE=your_api_key")
        console.print("\nOr create a .env file in your working directory with these variables.")
        sys.exit(1)


def build_model():
    server_model = get_env("BPSA_SERVER_MODEL", "OpenAIServerModel")
    model_id = get_env("BPSA_MODEL_ID")
    api_key = get_env("BPSA_KEY_VALUE")
    api_endpoint = get_env("BPSA_API_ENDPOINT")
    postpend_string = get_env("BPSA_POSTPEND_STRING", "")
    max_tokens = int(get_env("BPSA_MAX_TOKENS", "64000"))

    if server_model not in MODEL_CLASS_MAP:
        supported = ", ".join(sorted(MODEL_CLASS_MAP.keys()))
        fail(f"Unsupported BPSA_SERVER_MODEL: {server_model}. Supported: {supported}")

    canonical_name = MODEL_CLASS_MAP[server_model]

    import smolagents

    model_class = getattr(smolagents, canonical_name, None)
    if model_class is None:
        fail(f"Model class {canonical_name} not found in smolagents. Check your installation.")

    # Build kwargs based on model type
    if canonical_name in ("OpenAIServerModel", "AzureOpenAIServerModel"):
        model = model_class(model_id, api_key=api_key, max_tokens=max_tokens, api_base=api_endpoint)
    elif canonical_name == "LiteLLMModel":
        model = model_class(model_id=model_id, api_key=api_key, api_base=api_endpoint, max_tokens=max_tokens)
    elif canonical_name == "LiteLLMRouterModel":
        model = model_class(model_id=model_id, api_key=api_key, api_base=api_endpoint, max_tokens=max_tokens)
    elif canonical_name == "InferenceClientModel":
        model = model_class(model_id=model_id, token=api_key, max_tokens=max_tokens)
    elif canonical_name == "TransformersModel":
        model = model_class(model_id=model_id, device_map="auto", max_tokens=max_tokens)
    elif canonical_name == "AmazonBedrockModel":
        model = model_class(model_id=model_id, max_tokens=max_tokens)
    elif canonical_name == "VLLMModel":
        model = model_class(model_id=model_id, max_tokens=max_tokens)
    elif canonical_name == "MLXModel":
        model = model_class(model_id=model_id, max_tokens=max_tokens)
    elif canonical_name == "GoogleColabModel":
        model = model_class(model_id=model_id, max_tokens=max_tokens)
    else:
        fail(f"No constructor logic for model class: {canonical_name}")

    model.postpend_string = postpend_string
    return model


def build_agent(model):
    from smolagents import CodeAgent
    from smolagents.bp_thinkers import (
        DEFAULT_THINKER_COMPRESSION, DEFAULT_THINKER_MAX_STEPS,
        DEFAULT_THINKER_PLANNING_INTERVAL, DEFAULT_THINKER_TOOLS,
    )

    executor_type = get_env("BPSA_GLOBAL_EXECUTOR", default="exec")

    agent = CodeAgent(
        tools=DEFAULT_THINKER_TOOLS,
        model=model,
        additional_authorized_imports=["*"],
        add_base_tools=True,
        max_steps=DEFAULT_THINKER_MAX_STEPS,
        executor_type=executor_type,
        compression_config=DEFAULT_THINKER_COMPRESSION,
        planning_interval=DEFAULT_THINKER_PLANNING_INTERVAL,
        step_callbacks=[_compact_step_callback],
    )
    return agent


def load_agent_instructions() -> str | None:
    """Load agent instructions from common instruction files (CLAUDE.md, AGENTS.md, etc.)."""
    instructions = []
    for filename in AGENT_INSTRUCTION_FILES:
        filepath = os.path.join(os.getcwd(), filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, "r") as f:
                    content = f.read().strip()
                if content:
                    instructions.append(f"# Instructions from {filename}\n\n{content}")
                    console.print(f"  [green]Loaded:[/] {filename}")
            except OSError:
                pass
    if instructions:
        return "\n\n---\n\n".join(instructions)
    return None


def count_tools(agent) -> int:
    return len(agent.tools)


def format_tokens(n: int) -> str:
    """Format token count with color based on magnitude."""
    if n < 10_000:
        return f"[green]{n:,}[/]"
    elif n < 50_000:
        return f"[yellow]{n:,}[/]"
    else:
        return f"[red]{n:,}[/]"


def get_agent_token_usage(agent):
    """Get current total token usage from the agent's monitor."""
    try:
        usage = agent.monitor.get_total_token_counts()
        return usage.input_tokens, usage.output_tokens
    except Exception:
        return 0, 0


def print_turn_summary(turn_num: int, elapsed: float, input_tokens: int, output_tokens: int):
    """Print a one-line summary after each turn."""
    total = input_tokens + output_tokens
    console.print(
        f"[dim]Turn {turn_num} | {elapsed:.1f}s | "
        f"In: {format_tokens(input_tokens)} | Out: {format_tokens(output_tokens)} | "
        f"Total: {format_tokens(total)}[/]"
    )


def print_banner(model_id: str, server_model: str, tool_count: int):
    console.print(
        Panel.fit(
            f"[bold]Beyond Python SmolAgents[/] v{VERSION}\n"
            f"Model: [cyan]{model_id}[/] ({server_model})\n"
            f"Tools: [green]{tool_count}[/] loaded",
            border_style="blue",
        )
    )
    console.print(
        Panel.fit(
            "[bold red]EXTREME SECURITY RISK[/]\n"
            "This agent has extensive access and control over the environment in which it runs,\n"
            "including file system access, running arbitrary OS commands, and executing code.\n"
            "Only run inside a securely isolated environment (VM or container).\n"
            "[bold]USE AT YOUR OWN RISK.[/]",
            border_style="red",
        )
    )
    console.print("[dim]Type /help for commands, /verbose to toggle verbosity, /exit to quit.[/]\n")


def print_help():
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")
    table.add_row("/help", "Show this help message")
    table.add_row("/exit", "Exit the REPL")
    table.add_row("/reset", "Clear conversation history and reset the agent")
    table.add_row("/tools", "List all loaded tools")
    table.add_row("/verbose", "Toggle verbose output")
    table.add_row("/save <filename>", "Save the last answer to a file")
    table.add_row("/stats", "Show session statistics")
    console.print(table)
    console.print()


def print_tools(agent):
    table = Table(title="Loaded Tools", show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    for name in sorted(agent.tools.keys()):
        tool = agent.tools[name]
        desc = getattr(tool, "description", "")
        first_line = desc.split("\n")[0].strip() if desc else ""
        if len(first_line) > 80:
            first_line = first_line[:77] + "..."
        table.add_row(name, first_line)
    console.print(table)
    console.print()


def save_answer(last_answer, args: str):
    filename = args.strip()
    if not filename:
        console.print("[yellow]Usage: /save <filename>[/]")
        return
    if last_answer is None:
        console.print("[yellow]No answer to save yet.[/]")
        return
    try:
        with open(filename, "w") as f:
            f.write(str(last_answer))
        console.print(f"[green]Saved to {filename}[/]")
    except OSError as e:
        console.print(f"[red]Error saving: {e}[/]")


def print_stats(session_stats: dict):
    console.print(Rule("[bold]Session Statistics", style="blue"))
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")
    table.add_row("Turns", str(session_stats["turns"]))
    table.add_row("Total time", f"{session_stats['total_time']:.1f}s")
    table.add_row("Total input tokens", f"{session_stats['total_input_tokens']:,}")
    table.add_row("Total output tokens", f"{session_stats['total_output_tokens']:,}")
    total_tokens = session_stats["total_input_tokens"] + session_stats["total_output_tokens"]
    table.add_row("Total tokens", f"{total_tokens:,}")
    if session_stats["turns"] > 0:
        avg_time = session_stats["total_time"] / session_stats["turns"]
        avg_tokens = total_tokens // session_stats["turns"]
        table.add_row("Avg time/turn", f"{avg_time:.1f}s")
        table.add_row("Avg tokens/turn", f"{avg_tokens:,}")
    console.print(table)
    console.print()


def prepend_instructions(task: str, instructions: str | None) -> str:
    if instructions:
        return f"{instructions}\n\n---\n\n# Task\n\n{task}"
    return task


def run_one_shot(task: str):
    try_load_dotenv()
    check_required_env()
    model = build_model()
    agent = build_agent(model)
    console.print("[dim]Loading agent instructions...[/]")
    instructions = load_agent_instructions()
    _spinner.start()
    result = agent.run(prepend_instructions(task, instructions))
    _spinner.stop()
    console.print(Markdown(str(result)))


def run_repl():
    try_load_dotenv()
    check_required_env()

    model = build_model()
    agent = build_agent(model)
    model_id = get_env("BPSA_MODEL_ID")
    server_model = get_env("BPSA_SERVER_MODEL", default="OpenAIServerModel")
    tool_count = count_tools(agent)
    global _verbose
    verbose = get_env("BPSA_VERBOSE", default="1") == "1"
    _verbose = verbose

    print_banner(model_id, server_model, tool_count)

    console.print("[dim]Loading agent instructions...[/]")
    instructions = load_agent_instructions()
    if not instructions:
        console.print("  [dim]No instruction files found.[/]")
    console.print()

    # Try to use prompt_toolkit, fall back to basic input
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        history_file = os.path.expanduser("~/.bpsa_history")
        session = PromptSession(history=FileHistory(history_file))

        def get_input():
            try:
                console.print(Rule(style="dim"))
                return session.prompt("> ")
            except EOFError:
                return None
            except KeyboardInterrupt:
                print()
                return ""
    except ImportError:
        def get_input():
            try:
                console.print(Rule(style="dim"))
                return input("> ")
            except EOFError:
                return None
            except KeyboardInterrupt:
                print()
                return ""

    last_answer = None
    session_stats = {
        "turns": 0,
        "total_time": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }
    first_turn = True

    while True:
        user_input = get_input()
        if user_input is None:
            console.print("[dim]Goodbye![/]")
            break

        text = user_input.strip()
        if not text:
            continue

        # Handle slash commands
        if text.startswith("/"):
            cmd_parts = text.split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if cmd == "/exit":
                if session_stats["turns"] > 0:
                    console.print()
                    print_stats(session_stats)
                console.print("[dim]Goodbye![/]")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/reset":
                agent = build_agent(model)
                session_stats = {
                    "turns": 0,
                    "total_time": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                }
                last_answer = None
                first_turn = True
                console.print("[green]Agent reset. Conversation history cleared.[/]")
            elif cmd == "/tools":
                print_tools(agent)
            elif cmd == "/verbose":
                verbose = not verbose
                _verbose = verbose
                console.print(f"[cyan]Verbose mode: {'on' if verbose else 'off (compact)'}[/]")
            elif cmd == "/save":
                save_answer(last_answer, cmd_args)
            elif cmd == "/stats":
                print_stats(session_stats)
            else:
                console.print(f"[yellow]Unknown command: {cmd}. Type /help for available commands.[/]")
            continue

        # Visual separator before agent run
        turn_num = session_stats["turns"] + 1
        console.print(Rule(style="cyan"))

        # Capture token counts before this turn
        input_before, output_before = get_agent_token_usage(agent)

        # Run agent
        try:
            from smolagents.monitoring import LogLevel

            start_time = time.time()
            if verbose:
                agent.logger.level = LogLevel.INFO
            else:
                agent.logger.level = LogLevel.ERROR
            _spinner.start()
            task_text = prepend_instructions(text, instructions) if first_turn else text
            first_turn = False
            result = agent.run(task_text, reset=False)
            _spinner.stop()
            elapsed = time.time() - start_time

            # Calculate token usage for this turn
            input_after, output_after = get_agent_token_usage(agent)
            turn_input = input_after - input_before
            turn_output = output_after - output_before

            session_stats["turns"] += 1
            session_stats["total_time"] += elapsed
            session_stats["total_input_tokens"] += turn_input
            session_stats["total_output_tokens"] += turn_output
            last_answer = result

            console.print()
            console.print(Markdown(str(result)))
            console.print()

            # Per-turn summary line
            print_turn_summary(turn_num, elapsed, turn_input, turn_output)
            console.print()

        except KeyboardInterrupt:
            _spinner.stop()
            console.print("\n[yellow]Interrupted.[/]")
        except Exception as e:
            _spinner.stop()
            console.print(f"\n[bold red]Error:[/] {e}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="bpsa",
        description="Beyond Python SmolAgents - Interactive AI agent CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a one-shot task")
    run_parser.add_argument("task", type=str, help="The task to execute")

    args = parser.parse_args()

    # Piped input detection
    if not sys.stdin.isatty() and args.command is None:
        task = sys.stdin.read().strip()
        if task:
            run_one_shot(task)
        else:
            fail("No input provided via pipe.")
        return

    if args.command == "run":
        run_one_shot(args.task)
    else:
        run_repl()


if __name__ == "__main__":
    main()
