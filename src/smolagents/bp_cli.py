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

    Context compression parameters (see CompressionConfig for details):
    BPSA_COMPRESSION_ENABLED                  - Enable compression (default: 1)
    BPSA_COMPRESSION_KEEP_RECENT_STEPS        - Recent steps to keep uncompressed (default: 40)
    BPSA_COMPRESSION_MAX_UNCOMPRESSED_STEPS   - Trigger threshold for compression (default: 50)
    BPSA_COMPRESSION_KEEP_COMPRESSED_STEPS    - Compressed steps to keep on merge (default: 80)
    BPSA_COMPRESSION_MAX_COMPRESSED_STEPS     - Trigger threshold for merge (default: 120)
    BPSA_COMPRESSION_TOKEN_THRESHOLD          - Token-based trigger (default: 0 = disabled)
    BPSA_COMPRESSION_MODEL                    - Model ID for compression (default: same as main)
    BPSA_COMPRESSION_MAX_SUMMARY_TOKENS       - Max tokens in summary (default: 50000)
    BPSA_COMPRESSION_PRESERVE_ERROR_STEPS     - Keep error steps uncompressed (default: 0)
    BPSA_COMPRESSION_PRESERVE_FINAL_ANSWER_STEPS - Keep final_answer steps (default: 1)
    BPSA_COMPRESSION_MIN_CHARS                - Min chars before compressing (default: 4096)

    Voice input (requires `pip install bpsa[voice]`):
    BPSA_VOICE_TRANSCRIBER    - Transcriber name: 'whisper' or 'elevenlabs' (required for /voice)
    BPSA_VOICE_MODEL          - Model name passed to transcriber (optional, whisper only)
    ELEVENLABS_API_KEY        - API key for ElevenLabs transcriber (required when using elevenlabs)
"""

import os
import queue
import re
import subprocess
import sys
import time

from dotenv import load_dotenv
from smolagents.utils import truncate_content
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from smolagents.bp_utils import get_env


VERSION = "1.23-bp"
console = Console()

# Mutable state for verbose mode, accessible by the step callback
_verbose = True
# Mutable state for auto-approve mode
_auto_approve = False

TRUNCATE_PREVIEW_LINES = 5

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

BPSA_DEFAULT_VOICE_MODEL = 'base.en'

class Spinner:
    """Improved spinner using Rich library for better UX and reliability."""
    
    def __init__(self, message: str = "Thinking...", style: str = "dots", color: str = "cyan"):
        """Initialize spinner with Rich components.
        
        Args:
            message: The message to display next to the spinner
            style: Rich spinner style (dots, line, pipe, etc.)
            color: Color for the message text
        """
        from rich.spinner import Spinner as RichSpinner
        from rich.live import Live
        
        self.message = message
        self.style = style
        self.color = color
        self.spinner = RichSpinner(style, text=f"[{color}]{message}[/{color}]\n")
        self.live = None
    
    def start(self):
        """Start the spinner animation."""
        if not self.live:
            from rich.live import Live
            self.live = Live(
                self.spinner,
                console=console,
                refresh_per_second=10,
                transient=True  # Spinner disappears when stopped
            )
            self.live.start()
    
    def stop(self):
        """Stop the spinner animation cleanly."""
        if self.live:
            self.live.stop()
            console.print()  # Add line feed after spinner stops
            self.live = None
    
    def update(self, message: str):
        """Update the spinner message while it's running.
        
        Args:
            message: New message to display
        """
        self.message = message
        self.spinner.text = f"[{self.color}]{message}[/{self.color}]"
        if self.live:
            self.live.update(self.spinner)

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
    server_model = get_env("BPSA_SERVER_MODEL", None)

    missing = []

    supported = ", ".join(sorted(MODEL_CLASS_MAP.keys()))
        
    if server_model is None:
        fail(f"BPSA_SERVER_MODEL is not set. Supported models: {supported}")

    if not model_id:
        missing.append("BPSA_MODEL_ID")

    if server_model not in MODEL_CLASS_MAP:
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


def build_model(override_model_id=None):
    server_model = get_env("BPSA_SERVER_MODEL", None)
    model_id = override_model_id or get_env("BPSA_MODEL_ID")
    api_key = get_env("BPSA_KEY_VALUE")
    api_endpoint = get_env("BPSA_API_ENDPOINT")
    postpend_string = get_env("BPSA_POSTPEND_STRING", "")
    max_tokens = int(get_env("BPSA_MAX_TOKENS", "64000"))
    supported = ", ".join(sorted(MODEL_CLASS_MAP.keys()))
    
    if server_model is None:
        fail(f"BPSA_SERVER_MODEL is not set. Supported models: {supported}")
    
    if server_model not in MODEL_CLASS_MAP:     
        fail(f"Unsupported BPSA_SERVER_MODEL: {server_model}. Supported: {supported}")

    canonical_name = MODEL_CLASS_MAP[server_model]

    import smolagents

    model_class = getattr(smolagents, canonical_name, None)
    if model_class is None:
        fail(f"Model class {canonical_name} not found. Supported models are: {supported}")

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


def build_agent(model, approval_callback=None, browser_enabled=False, gui_enabled=False):
    from smolagents import CodeAgent
    from smolagents.bp_thinkers import (
        DEFAULT_THINKER_COMPRESSION, DEFAULT_THINKER_MAX_STEPS,
        DEFAULT_THINKER_PLANNING_INTERVAL, DEFAULT_THINKER_TOOLS,
    )

    executor_type = get_env("BPSA_GLOBAL_EXECUTOR", default="exec")

    tools = list(DEFAULT_THINKER_TOOLS)
    browser_manager = None
    gui_manager = None

    # Image tools — always available (Pillow only; tesseract optional for OCR)
    from smolagents.bp_tools import LoadImageTool, load_image_callback
    load_image_tool = LoadImageTool()
    tools.append(load_image_tool)

    from smolagents.bp_tools_image import create_image_tools
    tools.extend(create_image_tools())

    if browser_enabled:
        from smolagents.bp_tools_browser import create_browser_tools
        browser_manager, browser_tools = create_browser_tools()
        tools.extend(browser_tools)

    if gui_enabled:
        from smolagents.bp_tools_gui import create_gui_tools
        gui_manager, gui_tools = create_gui_tools()
        tools.extend(gui_tools)

    step_cbs = [_compact_step_callback, load_image_callback]
    if gui_manager:
        from smolagents.bp_tools_gui import gui_screenshot_callback
        step_cbs.append(gui_screenshot_callback)

    # Resolve compression model (may be a separate model via BPSA_COMPRESSION_MODEL)
    compression_cfg = DEFAULT_THINKER_COMPRESSION
    compression_model_id = get_env("BPSA_COMPRESSION_MODEL")
    if compression_model_id:
        from copy import copy
        compression_model = build_model(override_model_id=compression_model_id)
        compression_cfg = copy(DEFAULT_THINKER_COMPRESSION)
        compression_cfg.compression_model = compression_model

    agent = CodeAgent(
        tools=tools,
        model=model,
        additional_authorized_imports=["*"],
        add_base_tools=True,
        max_steps=DEFAULT_THINKER_MAX_STEPS,
        executor_type=executor_type,
        compression_config=compression_cfg,
        planning_interval=None,
        step_callbacks=step_cbs,
        approval_callback=approval_callback,
    )
    agent.add_planning_tool()

    agent._load_image_tool = load_image_tool

    if browser_manager:
        agent._browser_manager = browser_manager

    if gui_manager:
        agent._gui_manager = gui_manager

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
                    instructions.append(f"# Content from {filename}\n\n<filecontent>{content}</filecontent>")
                    console.print(f"  [green]Loaded:[/] {filename}")
            except OSError:
                pass
    if instructions:
        return """The folder contains some files with notes.
You are not required to follow these notes. The notes in <filenotes> are there for your information.
The notes are given in the tags <filenotes></filenotes>. These are the notes:
<filenotes>
"""+"\n\n".join(instructions)+"""
</filenotes>
"""
    # Fallback: try to load a README file for general project context
    readme_content = _load_readme_fallback()
    if readme_content:
        return readme_content
    return None


README_MAX_CHARS = 20000

# Preferred extensions in priority order for README fallback
_README_EXTENSIONS_PRIORITY = [".md", ".txt", ".rst", ""]


def _load_readme_fallback() -> str | None:
    """Search for a README file (case-insensitive) and return its content as context."""
    cwd = os.getcwd()
    try:
        entries = os.listdir(cwd)
    except OSError:
        return None
    # Group readme files by extension priority
    candidates = {}
    for entry in entries:
        lower = entry.lower()
        if not lower.startswith("readme"):
            continue
        # Determine extension
        _, ext = os.path.splitext(lower)
        if ext not in _README_EXTENSIONS_PRIORITY:
            continue
        path = os.path.join(cwd, entry)
        if not os.path.isfile(path):
            continue
        # Keep first match per extension priority
        if ext not in candidates:
            candidates[ext] = (entry, path)
    # Pick best candidate by extension priority
    for ext in _README_EXTENSIONS_PRIORITY:
        if ext in candidates:
            filename, filepath = candidates[ext]
            try:
                with open(filepath, "r") as f:
                    content = f.read().strip()
            except OSError:
                continue
            if not content:
                continue
            truncated = ""
            if len(content) > README_MAX_CHARS:
                content = content[:README_MAX_CHARS]
                truncated = " (truncated to 20000 chars)"
            console.print(f"  [green]Loaded README:[/] {filename}{truncated}")
            return f"""The folder contains a README file, provided for general context about the project.
This is not an instruction file. Use it only as background information.
<readme>
{content}
</readme>
"""
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


def get_compression_stats(agent):
    """Get context compression statistics from the agent's memory."""
    try:
        from smolagents.bp_compression import CompressedHistoryStep
        steps = agent.memory.steps
        total_steps = len(steps)
        compressed_count = sum(1 for s in steps if isinstance(s, CompressedHistoryStep))
        compressed_original = sum(s.original_step_count for s in steps if isinstance(s, CompressedHistoryStep))
        return total_steps, compressed_count, compressed_original
    except Exception:
        return 0, 0, 0


def print_turn_summary(turn_num: int, elapsed: float, input_tokens: int, output_tokens: int, agent=None):
    """Print a one-line summary after each turn."""
    total = input_tokens + output_tokens
    line = (
        f"[dim]Turn {turn_num} | {elapsed:.1f}s | "
        f"In: {format_tokens(input_tokens)} | Out: {format_tokens(output_tokens)} | "
        f"Total: {format_tokens(total)}"
    )
    if agent is not None:
        total_steps, compressed_count, compressed_original = get_compression_stats(agent)
        if compressed_count > 0:
            line += f" | Compressed: {compressed_count} (from {compressed_original} steps)"
        line += f" | Memory: {total_steps} steps"
        ctx_chars = agent.get_context_char_size()
        if ctx_chars > 0:
            line += f" | Context: {format_tokens(ctx_chars)} chars"
    line += f" | Auto-approve: {'on' if _auto_approve else 'off'}"
    line += "[/]"
    console.print(line)


def print_banner(model_id: str, server_model: str, tool_count: int):
    console.print(
        Panel.fit(
            f"[bold]BPSA - Beyond Python SmolAgents[/] v{VERSION}\n"
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


def _run_shell_streaming(shell_cmd: str) -> str:
    """Run a shell command, streaming output to the terminal and returning the full output."""
    output_lines = []
    try:
        proc = subprocess.Popen(shell_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            output_lines.append(line)
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        console.print("\n[dim]Command interrupted.[/]")
    return "".join(output_lines)


ALIASES_FILE = os.path.expanduser("~/.bpsa_aliases")


def _load_aliases() -> dict:
    """Load aliases from ~/.bpsa_aliases."""
    aliases = {}
    if os.path.isfile(ALIASES_FILE):
        with open(ALIASES_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        aliases[parts[0]] = parts[1]
    return aliases


def _save_aliases(aliases: dict):
    """Save aliases to ~/.bpsa_aliases."""
    with open(ALIASES_FILE, "w") as f:
        for name, value in sorted(aliases.items()):
            f.write(f"{name} {value}\n")


SLASH_COMMANDS = [
    "/alias", "/auto-approve", "/cd", "/clear", "/compress", "/compression",
    "/compression-keep-recent-steps", "/compression-max-uncompressed-steps",
    "/compression-model", "/exit", "/help",
    "/load-instructions", "/plan", "/pwd", "/redo", "/repeat", "/repeat-prompt", "/run-prompt", "/run-py", "/save",
    "/session-load", "/session-save",
    "/show-compression-stats", "/show-memory-stats", "/show-stats",
    "/save-step", "/set-max-steps", "/show-step", "/show-steps", "/show-tools", "/undo-steps", "/verbose",
    "/voice",
]


def print_help():
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")
    table.add_row("!<command>", "Run an OS command directly (agent does not see the output)")
    table.add_row("!!<command>", "Run an OS command; output is appended to the next prompt sent to the agent")
    table.add_row("!!!<command>", "Run an OS command and immediately send the output to the agent for analysis")
    table.add_row("/alias <name> <value>", "Define alias (saved to ~/.bpsa_aliases). No args=list, -d <name>=delete")
    table.add_row("/auto-approve \[on|off]", "Toggle or set auto-approve for tag execution")
    table.add_row("/cd <dir>", "Change working directory")
    table.add_row("/clear", "Clear screen, reset agent and conversation history")
    table.add_row("/compress \[N]", "Force compression now, or compress a specific step N")
    table.add_row("/compression \[on|off]", "Toggle compression on/off")
    table.add_row("/compression-keep-recent-steps <N>", "Change keep_recent_steps")
    table.add_row("/compression-max-uncompressed-steps <N>", "Change max_uncompressed_steps")
    table.add_row("/compression-model <model>", "Switch compression model")
    table.add_row("/exit", "Exit the REPL")
    table.add_row("/help", "Show this help message")
    table.add_row("/load-instructions", "Load agent instruction files into next prompt")
    table.add_row("/plan \[on|off|N]", "Toggle or set planning interval (default: 22)")
    table.add_row("/pwd", "Show current working directory")
    table.add_row("/redo", "Re-run the last prompt (undo last steps and run again)")
    table.add_row("/repeat <N> <prompt>", "Run the same prompt N times, each on a fresh agent with current context")
    table.add_row("/repeat-prompt <N> <path>", "Run a prompt file N times, each on a fresh agent with current context")
    table.add_row("/run-prompt <path>", "Load a file's content as the prompt")
    table.add_row("/run-py <script.py>", "Execute a Python script in the agent's executor")
    table.add_row("/save <filename>", "Save the last answer to a file")
    table.add_row("/save-step <N> <file>", "Save full content of step N to a file")
    table.add_row("/session-load <file>", "Load a session from a JSON file")
    table.add_row("/session-save <file>", "Save entire session to a JSON file")
    table.add_row("/show-compression-stats", "Show compression config and stats")
    table.add_row("/show-memory-stats", "Show memory breakdown: steps, tokens, compressed vs uncompressed")
    table.add_row("/show-step <N>", "Show full content of a specific step")
    table.add_row("/show-steps", "Show one-line summary of all memory steps")
    table.add_row("/show-stats", "Show session statistics")
    table.add_row("/set-max-steps <N>", "Change max_steps for the agent")
    table.add_row("/show-tools", "List all loaded tools")
    table.add_row("/undo-steps \[N]", "Remove last N steps from memory (default: 1)")
    table.add_row("/verbose", "Toggle verbose output")
    table.add_row(r"/voice \[on|off]", "Toggle voice dictation (requires BPSA_VOICE_TRANSCRIBER)")
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



def _getch():
    """Read a single keypress without requiring Enter. Cross-platform."""
    try:
        # Try Windows first
        import msvcrt
        key = msvcrt.getch()
        # msvcrt returns bytes, decode to string
        return key.decode('utf-8', errors='ignore').lower()
    except ImportError:
        # Unix/Linux/Mac
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
            return key.lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ── Voice input ──────────────────────────────────────────────────────────────

_voice_listener = None
_voice_queue = queue.Queue()
_voice_prompt_active = False
_prompt_session = None  # Set to PromptSession instance in run_repl()

_VOICE_TRANSCRIBERS = {"whisper", "elevenlabs"}


def _voice_start():
    """Start the voice listener. Returns an error message string on failure, or None on success."""
    global _voice_listener
    if _voice_listener is not None:
        return "Voice input is already active."
    try:
        from voicelistener import VoiceListener
    except ImportError:
        return "Voice input requires the voicelistener package. Install with: pip install bpsa[voice]"

    transcriber_name = get_env("BPSA_VOICE_TRANSCRIBER", default="")
    if not transcriber_name:
        return (
            "Set BPSA_VOICE_TRANSCRIBER environment variable to enable voice input"
            f" (available transcribers: {', '.join(sorted(_VOICE_TRANSCRIBERS))})"
        )
    transcriber_name = transcriber_name.lower().strip()
    if transcriber_name not in _VOICE_TRANSCRIBERS:
        return (
            f"Unknown transcriber '{transcriber_name}'."
            f" Available transcribers: {', '.join(sorted(_VOICE_TRANSCRIBERS))}"
        )

    model = get_env("BPSA_VOICE_MODEL", default=BPSA_DEFAULT_VOICE_MODEL)

    if transcriber_name == "whisper":
        from voicelistener import WhisperTranscriber
        kwargs = {}
        if model:
            kwargs["model"] = model
        transcriber = WhisperTranscriber(**kwargs)
    elif transcriber_name == "elevenlabs":
        from voicelistener import ElevenLabsTranscriber
        transcriber = ElevenLabsTranscriber()

    def _on_transcription(text):
        _voice_queue.put(text)
        # Force prompt_toolkit to re-render so the text appears immediately
        if _prompt_session is not None and _prompt_session.app is not None:
            _prompt_session.app.invalidate()

    _voice_listener = VoiceListener(transcriber=transcriber, on_transcription=_on_transcription)
    _voice_listener.start()
    return None


def _voice_stop():
    """Stop the voice listener."""
    global _voice_listener
    if _voice_listener is None:
        return "Voice input is not active."
    _voice_listener.stop()
    _voice_listener = None
    # Drain any remaining items
    while not _voice_queue.empty():
        try:
            _voice_queue.get_nowait()
        except queue.Empty:
            break
    return None


def _shutdown_voice():
    """Safe cleanup for voice listener."""
    if _voice_listener is not None:
        _voice_stop()


def _drain_voice_queue_into_buffer(buffer):
    """Drain voice transcriptions into a prompt_toolkit buffer at cursor position."""
    if not _voice_prompt_active or _voice_listener is None:
        return
    while not _voice_queue.empty():
        try:
            text = _voice_queue.get_nowait()
        except queue.Empty:
            break
        # Prepend space if buffer has content and doesn't end with whitespace
        doc = buffer.document
        if doc.text and not doc.text_before_cursor.endswith((" ", "\n", "\t")):
            text = " " + text
        buffer.insert_text(text)


def interactive_approval_callback(tag_type: str, content: str) -> bool:
    """Interactive approval callback for tag execution. Returns True if approved."""
    global _auto_approve
    if _auto_approve:
        console.print(f"[dim]Auto-approved: {tag_type}[/]")
        return True

    # Stop spinner while prompting
    _spinner.stop()

    # Show tag type header
    tag_label = tag_type.upper()
    console.print(f"\n[bold yellow]{'═' * 50}[/]")
    console.print(f"[bold yellow]  Approval required: {tag_label}[/]")
    console.print(f"[bold yellow]{'═' * 50}[/]")

    # Show truncated preview
    lines = content.split('\n')
    preview = '\n'.join(lines[:TRUNCATE_PREVIEW_LINES])
    console.print(preview)
    truncated = len(lines) > TRUNCATE_PREVIEW_LINES
    if truncated:
        console.print(f"[dim]... ({len(lines) - TRUNCATE_PREVIEW_LINES} more lines, press 'f' to see full content)[/]")
    console.print(f"[bold yellow]{'─' * 50}[/]")

    # Prompt loop with single-key input
    while True:
        try:
            prompt_text = "Approve? (y)es / (n)o / (f)ull: " if truncated else "Approve? (y)es / (n)o: "
            console.print(prompt_text, end='', style="bold cyan")
            response = _getch()
            console.print(response)  # Echo the key pressed
        except (EOFError, KeyboardInterrupt):
            console.print()  # New line
            return False
        if response == 'f' and truncated:
            console.print(content)
            continue
        if response in ('y',):
            _spinner.start()
            return True
        if response in ('n',):
            return False
        console.print("[yellow]Please press y, n" + (", or f" if truncated else "") + "[/]")


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


def load_file_as_prompt(args: str) -> str | None:
    """Load a file's content to use as a prompt."""
    filepath = args.strip()
    if not filepath:
        console.print("[yellow]Usage: /run-prompt <path>[/]")
        return None
    filepath = os.path.expanduser(filepath)
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.getcwd(), filepath)
    if not os.path.isfile(filepath):
        console.print(f"[red]File not found: {filepath}[/]")
        return None
    try:
        with open(filepath, "r") as f:
            content = f.read()
        console.print(f"[green]Loaded {len(content):,} chars from {filepath}[/]")
        return content
    except OSError as e:
        console.print(f"[red]Error reading file: {e}[/]")
        return None


def change_steps(agent, args: str):
    """Change the agent's max_steps."""
    args = args.strip()
    if not args:
        console.print(f"[cyan]Current max_steps: {agent.max_steps}[/]")
        console.print("[dim]Usage: /set-max-steps <N>[/]")
        return
    try:
        n = int(args)
        if n < 1:
            console.print("[red]Steps must be at least 1.[/]")
            return
        agent.max_steps = n
        console.print(f"[green]max_steps set to {n}[/]")
    except ValueError:
        console.print("[red]Invalid number. Usage: /set-max-steps <N>[/]")


def run_script(agent, args: str):
    """Execute a Python script in the agent's executor."""
    filepath = args.strip()
    if not filepath:
        console.print("[yellow]Usage: /run-py <script.py>[/]")
        return
    filepath = os.path.expanduser(filepath)
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.getcwd(), filepath)
    if not os.path.isfile(filepath):
        console.print(f"[red]File not found: {filepath}[/]")
        return
    try:
        with open(filepath, "r") as f:
            code = f.read()
        console.print(f"[dim]Running {filepath}...[/]")
        result = subprocess.run(
            [sys.executable, filepath],
            capture_output=True, text=True, timeout=3600,
        )
        if result.stdout:
            console.print(result.stdout)
        if result.stderr:
            console.print(f"[red]{result.stderr}[/]")
        if result.returncode == 0:
            console.print(f"[green]Script finished (exit code 0)[/]")
        else:
            console.print(f"[yellow]Script finished (exit code {result.returncode})[/]")
    except subprocess.TimeoutExpired:
        console.print("[red]Script timed out (3600s limit).[/]")
    except OSError as e:
        console.print(f"[red]Error running script: {e}[/]")


def change_directory(args: str):
    """Change the working directory."""
    dirpath = args.strip()
    if not dirpath:
        console.print("[yellow]Usage: /cd <dir>[/]")
        return
    dirpath = os.path.expanduser(dirpath)
    if not os.path.isabs(dirpath):
        dirpath = os.path.join(os.getcwd(), dirpath)
    try:
        os.chdir(dirpath)
        console.print(f"[green]Changed to {os.getcwd()}[/]")
    except OSError as e:
        console.print(f"[red]Error: {e}[/]")

def _get_compression_config(agent):
    """Get compression config from agent, or print a warning and return None."""
    config = getattr(agent, "compression_config", None)
    if config is None:
        console.print("[yellow]No compression config set on this agent.[/]")
    return config


def _get_compressor(agent):
    """Get compressor from agent, or print a warning and return None."""
    compressor = getattr(agent, "compressor", None)
    if compressor is None:
        console.print("[yellow]No compressor configured on this agent.[/]")
    return compressor



def cmd_compression_stats(agent):
    """Show current compression config and stats."""
    from smolagents.bp_compression import CompressedHistoryStep, estimate_step_tokens

    compressor = _get_compressor(agent)
    config = _get_compression_config(agent)
    if config is None:
        return

    console.print(Rule("[bold]Compression Configuration", style="blue"))
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Setting", style="bold")
    table.add_column("Value", style="cyan")
    table.add_row("enabled", str(config.enabled))
    table.add_row("keep_recent_steps", str(config.keep_recent_steps))
    table.add_row("max_uncompressed_steps", str(config.max_uncompressed_steps))
    table.add_row("estimated_token_threshold", str(config.estimated_token_threshold))
    table.add_row("max_summary_tokens", str(config.max_summary_tokens))
    table.add_row("preserve_error_steps", str(config.preserve_error_steps))
    table.add_row("preserve_final_answer_steps", str(config.preserve_final_answer_steps))
    table.add_row("max_compressed_steps", str(config.max_compressed_steps))
    table.add_row("keep_compressed_steps", str(config.keep_compressed_steps))
    comp_model = config.compression_model
    table.add_row("compression_model", str(getattr(comp_model, "model_id", "same as main")) if comp_model else "same as main")
    console.print(table)

    # Stats
    steps = agent.memory.steps
    total_steps = len(steps)
    compressed_count = sum(1 for s in steps if isinstance(s, CompressedHistoryStep))
    compressed_original = sum(s.original_step_count for s in steps if isinstance(s, CompressedHistoryStep))
    total_chars = sum(len(s.summary) for s in steps if isinstance(s, CompressedHistoryStep))
    compression_count = compressor._compression_count if compressor else 0

    console.print()
    console.print(Rule("[bold]Compression Stats", style="blue"))
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value", style="cyan")
    stats_table.add_row("Total memory steps", str(total_steps))
    stats_table.add_row("Compressed summary steps", str(compressed_count))
    stats_table.add_row("Original steps compressed", str(compressed_original))
    stats_table.add_row("Compression runs", str(compression_count))
    stats_table.add_row("Compressed summary chars", f"{total_chars:,}")
    console.print(stats_table)
    console.print()


def cmd_memory_stats(agent):
    """Show memory breakdown: total steps, compressed vs uncompressed, estimated token usage."""
    from smolagents.bp_compression import CompressedHistoryStep, estimate_step_tokens
    from smolagents.memory import ActionStep, PlanningStep, TaskStep, SystemPromptStep

    steps = agent.memory.steps
    total_steps = len(steps)

    type_counts = {}
    total_tokens = 0
    total_chars = 0

    for step in steps:
        type_name = type(step).__name__
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        tokens = estimate_step_tokens(step)
        total_tokens += tokens
        # Estimate chars from step content
        msgs = step.to_messages(summary_mode=False)
        for msg in msgs:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and "text" in item:
                        total_chars += len(str(item["text"]))

    console.print(Rule("[bold]Memory Stats", style="blue"))
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")
    table.add_row("Total memory steps", str(total_steps))
    for type_name, count in sorted(type_counts.items()):
        table.add_row(f"  {type_name}", str(count))
    table.add_row("Total chars", f"{total_chars:,}")
    table.add_row("Estimated tokens", f"{total_tokens:,}")
    console.print(table)
    console.print()


def cmd_compress(agent, args: str):
    """Force immediate compression, or compress a specific step."""
    from smolagents.bp_compression import CompressedHistoryStep

    compressor = _get_compressor(agent)
    if compressor is None:
        return

    args = args.strip()
    if args:
        # Compress a specific step (1-based memory index, same as /show-steps and /show-step)
        try:
            step_num = int(args)
        except ValueError:
            console.print("[red]Invalid step number. Usage: /compress [N][/]")
            return

        steps = agent.memory.steps
        if step_num < 1 or step_num > len(steps):
            console.print(f"[red]Step {step_num} not found (valid range: 1-{len(steps)}). Use /show-steps to list all.[/]")
            return

        target_idx = step_num - 1
        target = steps[target_idx]

        if isinstance(target, CompressedHistoryStep):
            console.print(f"[yellow]Step {step_num} is already compressed.[/]")
            return

        from smolagents.memory import ActionStep
        if not isinstance(target, ActionStep):
            console.print(f"[yellow]Step {step_num} is a {type(target).__name__}, only ActionSteps can be compressed.[/]")
            return

        # Compress just this step
        from smolagents.bp_compression import create_compression_prompt
        from smolagents.models import ChatMessage, MessageRole
        import time

        start_time = time.time()
        prompt = create_compression_prompt([target])
        try:
            result = compressor.compression_model.generate(
                [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])]
            )
            summary = result.content
            if isinstance(summary, list):
                summary = " ".join(item.get("text", "") for item in summary if isinstance(item, dict))

            original_step_number = target.step_number if hasattr(target, "step_number") else step_num
            compressed = CompressedHistoryStep(
                summary=summary,
                compressed_step_numbers=[original_step_number],
                original_step_count=1,
                timing=__import__("smolagents.monitoring", fromlist=["Timing"]).Timing(
                    start_time=start_time, end_time=time.time()
                ),
                compression_token_usage=result.token_usage,
            )
            agent.memory.steps[target_idx] = compressed
            console.print(f"[green]Step {step_num} compressed successfully.[/]")
        except Exception as e:
            console.print(f"[red]Compression failed: {e}[/]")
    else:
        # Force compression of all eligible steps
        old_threshold = compressor.config.max_uncompressed_steps
        compressor.config.max_uncompressed_steps = 0  # Force trigger
        original_len = len(agent.memory.steps)
        agent.memory.steps = compressor.compress(agent.memory.steps)
        compressor.config.max_uncompressed_steps = old_threshold
        new_len = len(agent.memory.steps)
        if new_len < original_len:
            console.print(f"[green]Compression done: {original_len} steps → {new_len} steps.[/]")
        else:
            console.print("[yellow]No steps were eligible for compression.[/]")


def cmd_compression_toggle(agent, args: str):
    """Toggle compression on/off."""
    config = _get_compression_config(agent)
    if config is None:
        return

    arg = args.strip().lower()
    if arg == "on":
        config.enabled = True
    elif arg == "off":
        config.enabled = False
    elif arg == "":
        config.enabled = not config.enabled
    else:
        console.print("[yellow]Usage: /compression [on|off][/]")
        return
    console.print(f"[cyan]Compression: {'on' if config.enabled else 'off'}[/]")


def cmd_compression_keep_recent(agent, args: str):
    """Change keep_recent_steps."""
    config = _get_compression_config(agent)
    if config is None:
        return
    args = args.strip()
    if not args:
        console.print(f"[cyan]Current keep_recent_steps: {config.keep_recent_steps}[/]")
        console.print("[dim]Usage: /compression-keep-recent-steps <N>[/]")
        return
    try:
        n = int(args)
        if n < 0:
            raise ValueError
        config.keep_recent_steps = n
        console.print(f"[green]keep_recent_steps set to {n}[/]")
    except ValueError:
        console.print("[red]Invalid number. Usage: /compression-keep-recent-steps <N>[/]")


def cmd_compression_max_uncompressed(agent, args: str):
    """Change max_uncompressed_steps."""
    config = _get_compression_config(agent)
    if config is None:
        return
    args = args.strip()
    if not args:
        console.print(f"[cyan]Current max_uncompressed_steps: {config.max_uncompressed_steps}[/]")
        console.print("[dim]Usage: /compression-max-uncompressed-steps <N>[/]")
        return
    try:
        n = int(args)
        if n < 1:
            raise ValueError
        config.max_uncompressed_steps = n
        console.print(f"[green]max_uncompressed_steps set to {n}[/]")
    except ValueError:
        console.print("[red]Invalid number. Usage: /compression-max-uncompressed-steps <N>[/]")


def cmd_compression_model(agent, args: str):
    """Switch compression model."""
    config = _get_compression_config(agent)
    if config is None:
        return
    args = args.strip()
    if not args:
        comp_model = config.compression_model
        current = getattr(comp_model, "model_id", "same as main") if comp_model else "same as main"
        console.print(f"[cyan]Current compression model: {current}[/]")
        console.print("[dim]Usage: /compression-model <model_id>[/]")
        return

    # Build a new model with the given model_id, using the same class/config as the main model
    try:
        main_model = agent.model
        model_class = type(main_model)
        # Try to create with same constructor pattern
        import copy
        new_model = copy.copy(main_model)
        new_model.model_id = args
        config.compression_model = new_model
        console.print(f"[green]Compression model set to {args}[/]")
    except Exception as e:
        console.print(f"[red]Failed to set compression model: {e}[/]")


def cmd_show_step(agent, args: str):
    """Show a specific step's full content."""
    from smolagents.bp_compression import CompressedHistoryStep
    from smolagents.memory import ActionStep, PlanningStep, TaskStep, SystemPromptStep

    args = args.strip()
    if not args:
        console.print("[yellow]Usage: /show-step <N>[/]")
        return
    try:
        step_num = int(args)
    except ValueError:
        console.print("[red]Invalid step number.[/]")
        return

    # Step 0 = system prompt; steps 1..N = memory steps
    steps = agent.memory.steps
    if step_num == 0:
        if agent.memory.system_prompt:
            _print_step_detail(agent.memory.system_prompt)
        else:
            console.print("[yellow]No system prompt stored.[/]")
    elif 1 <= step_num <= len(steps):
        _print_step_detail(steps[step_num - 1])
    else:
        console.print(f"[red]Step {step_num} not found (valid range: 0-{len(steps)}). Use /show-steps to list all.[/]")


def cmd_session_save(agent, session_stats: dict, args: str):
    """Save entire session to a JSON file."""
    from smolagents.bp_session import save_session

    filename = args.strip()
    if not filename:
        console.print("[yellow]Usage: /session-save <filename>[/]")
        return
    if not filename.endswith(".json"):
        filename += ".json"
    try:
        count = save_session(filename, agent, session_stats)
        console.print(f"[green]Session saved to {filename} ({count} steps).[/]")
    except Exception as e:
        console.print(f"[red]Failed to save session: {e}[/]")


def cmd_session_load(agent, args: str) -> dict | None:
    """Load a session from a JSON file. Returns restored session_stats or None on failure."""
    import os

    from smolagents.bp_session import load_session

    filename = args.strip()
    if not filename:
        console.print("[yellow]Usage: /session-load <filename>[/]")
        return None
    if not os.path.isfile(filename):
        console.print(f"[red]File not found: {filename}[/]")
        return None
    try:
        stats = load_session(filename, agent)
        step_count = len(agent.memory.steps)
        console.print(f"[green]Session loaded from {filename} ({step_count} steps).[/]")
        return stats
    except Exception as e:
        console.print(f"[red]Failed to load session: {e}[/]")
        return None


def cmd_save_step(agent, args: str):
    """Save a specific step's full content to a file (no truncation)."""
    from smolagents.bp_compression import CompressedHistoryStep
    from smolagents.memory import ActionStep, PlanningStep, TaskStep, SystemPromptStep

    parts = args.strip().split(None, 1)
    if len(parts) < 2:
        console.print("[yellow]Usage: /save-step <N> <filename>[/]")
        return
    try:
        step_num = int(parts[0])
    except ValueError:
        console.print("[red]Invalid step number.[/]")
        return
    filename = parts[1]

    # Resolve the step
    steps = agent.memory.steps
    if step_num == 0:
        if agent.memory.system_prompt:
            step = agent.memory.system_prompt
        else:
            console.print("[yellow]No system prompt stored.[/]")
            return
    elif 1 <= step_num <= len(steps):
        step = steps[step_num - 1]
    else:
        console.print(f"[red]Step {step_num} not found (valid range: 0-{len(steps)}). Use /show-steps to list all.[/]")
        return

    # Build full text content
    lines = [f"Step {step_num} — {type(step).__name__}", ""]
    if isinstance(step, ActionStep):
        if step.timing and step.timing.duration is not None:
            lines.append(f"Duration: {step.timing.duration:.1f}s")
        if step.token_usage:
            lines.append(f"Tokens: in={step.token_usage.input_tokens:,} out={step.token_usage.output_tokens:,}")
        if step.error:
            lines.append(f"Error: {step.error}")
        if step.model_output:
            lines += ["", "=== Model output ===", str(step.model_output)]
        if step.code_action:
            lines += ["", "=== Code action ===", str(step.code_action)]
        if step.observations:
            lines += ["", "=== Run output ===", str(step.observations)]
    elif isinstance(step, CompressedHistoryStep):
        lines.append(f"Original steps compressed: {step.original_step_count}")
        lines.append(f"Step numbers: {step.compressed_step_numbers}")
        if step.timing and step.timing.duration is not None:
            lines.append(f"Compression duration: {step.timing.duration:.1f}s")
        lines += ["", "=== Summary ===", step.summary]
    elif isinstance(step, PlanningStep):
        lines += ["=== Plan ===", step.plan or ""]
    elif isinstance(step, TaskStep):
        lines += ["=== Task ===", step.task or ""]
    elif isinstance(step, SystemPromptStep):
        lines += ["=== System prompt ===", step.system_prompt or ""]
    else:
        lines.append(str(step))

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        console.print(f"[green]Step {step_num} saved to {filename}[/]")
    except OSError as e:
        console.print(f"[red]Failed to write {filename}: {e}[/]")


def _print_step_detail(step):
    """Print full detail of a single step."""
    from smolagents.bp_compression import CompressedHistoryStep
    from smolagents.memory import ActionStep, PlanningStep, TaskStep, SystemPromptStep

    type_name = type(step).__name__
    console.print(Rule(f"[bold]{type_name}", style="blue"))

    if isinstance(step, ActionStep):
        if step.timing and step.timing.duration is not None:
            console.print(f"[bold]Duration:[/] {step.timing.duration:.1f}s")
        if step.token_usage:
            console.print(f"[bold]Tokens:[/] in={step.token_usage.input_tokens:,} out={step.token_usage.output_tokens:,}")
        if step.error:
            console.print(f"[bold red]Error:[/] {step.error}")
        # if step.is_final_answer:
        #    console.print("[bold green]Final answer[/]")
        if step.model_output:
            console.print(f"[bold]Model output:[/]")
            console.print(str(step.model_output)[:2000])
        # if step.code_action:
        #    console.print(f"[bold]Code action:[/]")
        #    console.print(step.code_action[:2000])
        if step.observations:
            console.print(f"[bold]Run output:[/]")
            console.print(str(step.observations)[:2000])
    elif isinstance(step, CompressedHistoryStep):
        console.print(f"[bold]Original steps compressed:[/] {step.original_step_count}")
        console.print(f"[bold]Step numbers:[/] {step.compressed_step_numbers}")
        if step.timing and step.timing.duration is not None:
            console.print(f"[bold]Compression duration:[/] {step.timing.duration:.1f}s")
        console.print(f"[bold]Summary:[/]")
        console.print(step.summary[:3000])
    elif isinstance(step, PlanningStep):
        console.print(f"[bold]Plan:[/]")
        console.print(step.plan[:2000])
    elif isinstance(step, TaskStep):
        console.print(f"[bold]Task:[/]")
        console.print(step.task[:2000])
    elif isinstance(step, SystemPromptStep):
        console.print(f"[bold]System prompt:[/]")
        console.print(step.system_prompt[:2000])
    else:
        console.print(str(step)[:2000])
    console.print()


def cmd_show_steps(agent):
    """Show all steps with one-line summaries."""
    from smolagents.bp_compression import CompressedHistoryStep
    from smolagents.memory import ActionStep, PlanningStep, TaskStep, SystemPromptStep

    steps = agent.memory.steps
    if not steps:
        console.print("[yellow]No steps in memory.[/]")
        return

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Step", style="bold", justify="right")
    table.add_column("Type", style="cyan")
    table.add_column("Compressed", justify="center")
    table.add_column("Size", justify="right")
    table.add_column("Preview")

    if agent.memory.system_prompt:
        sp = agent.memory.system_prompt
        content = sp.system_prompt or ""
        chars = len(content)
        preview = content.replace("\n", " ").strip()
        if len(preview) > 80:
            preview = preview[:77] + "..."
        table.add_row("0", "SystemPrompt", "[dim]no[/]", f"{chars:,} chars", preview)

    for i, step in enumerate(steps, 1):
        type_name = type(step).__name__

        label = str(i)

        # Get content and size
        if isinstance(step, ActionStep):
            type_name = f"ActionStep(#{step.actionstep_id})"
            content = str(step.model_output or step.observations or "")
            chars = len(str(step.model_output or "")) + len(str(step.observations or ""))
        elif isinstance(step, CompressedHistoryStep):
            type_name = f"Compressed({step.original_step_count})"
            content = step.summary
            chars = len(step.summary)
        elif isinstance(step, PlanningStep):
            content = step.plan or ""
            chars = len(content)
        elif isinstance(step, TaskStep):
            content = step.task or ""
            chars = len(content)
        elif isinstance(step, SystemPromptStep):
            content = step.system_prompt or ""
            chars = len(content)
        else:
            content = str(step)
            chars = len(content)

        # Build preview: first line, truncated
        preview = content.replace("\n", " ").strip()
        if len(preview) > 80:
            preview = preview[:77] + "..."

        compressed = "[green]yes[/]" if isinstance(step, CompressedHistoryStep) else "[dim]no[/]"
        table.add_row(label, type_name, compressed, f"{chars:,} chars", preview)

    console.print(table)
    console.print()


def cmd_undo(agent, args: str):
    """Remove the last N steps from agent memory. Default N=1."""
    from smolagents.memory import SystemPromptStep

    steps = agent.memory.steps
    if not steps:
        console.print("[yellow]No steps in memory.[/]")
        return

    args = args.strip()
    if args:
        try:
            n = int(args)
            if n < 1:
                console.print("[red]N must be at least 1.[/]")
                return
        except ValueError:
            console.print("[red]Invalid number. Usage: /undo [N][/]")
            return
    else:
        n = 1

    # Count removable steps from the end (skip SystemPromptStep)
    removable = 0
    for step in reversed(steps):
        if isinstance(step, SystemPromptStep):
            break
        removable += 1

    if removable == 0:
        console.print("[yellow]No removable steps (only system prompt steps remain).[/]")
        return

    actual = min(n, removable)
    removed = steps[-actual:]
    agent.memory.steps = steps[:-actual]

    for step in removed:
        type_name = type(step).__name__
        preview = ""
        if hasattr(step, "model_output") and step.model_output:
            preview = str(step.model_output).replace("\n", " ")[:60]
        elif hasattr(step, "summary"):
            preview = step.summary.replace("\n", " ")[:60]
        elif hasattr(step, "plan"):
            preview = step.plan.replace("\n", " ")[:60]
        if preview:
            console.print(f"  [red]Removed:[/] {type_name} - {preview}...")
        else:
            console.print(f"  [red]Removed:[/] {type_name}")

    console.print(f"[green]Undone {actual} step{'s' if actual != 1 else ''}. {len(agent.memory.steps)} steps remain.[/]")
    if actual < n:
        console.print(f"[yellow]Only {actual} of {n} requested steps were removable (protected system prompt steps).[/]")


def cmd_repeat(agent, model, n, prompt_text, session_stats, verbose, instructions, first_turn, browser_enabled, gui_enabled=False):
    """Run a prompt N times, each on a fresh agent with snapshotted context."""
    from smolagents.bp_session import load_session_from_dict, save_session_to_dict
    from smolagents.monitoring import LogLevel

    from smolagents.bp_tools import inject_tree

    # Snapshot current agent state (in-memory, not to file)
    snapshot = save_session_to_dict(agent, session_stats)

    # Save current working directory
    original_folder = os.getcwd()

    # Resolve BPSA_INJECT_FOLDER (default: true = cwd)
    tree_folder_raw = get_env("BPSA_INJECT_FOLDER")
    if tree_folder_raw is not None and tree_folder_raw.lower() == "false":
        tree_folder = None
    elif tree_folder_raw is None or tree_folder_raw.lower() == "true":
        tree_folder = original_folder
    else:
        tree_folder = tree_folder_raw

    completed = 0
    errors = 0

    for i in range(1, n + 1):
        console.print(Rule(f"[bold cyan] Cycle {i}/{n} [/]", style="cyan"))
        try:
            # Restore working directory
            os.chdir(original_folder)

            # Create fresh agent and restore snapshot
            cycle_agent = build_agent(model, approval_callback=interactive_approval_callback, browser_enabled=browser_enabled, gui_enabled=gui_enabled)
            load_session_from_dict(snapshot, cycle_agent)

            # Prepare prompt (prepend instructions on first_turn only)
            task_text = prepend_instructions(prompt_text, instructions) if first_turn else prompt_text

            # Inject directory tree with function signatures if configured
            if tree_folder:
                task_text += inject_tree(tree_folder)

            # Run
            _spinner.start()
            start_time = time.time()
            if verbose:
                cycle_agent.logger.level = LogLevel.INFO
            else:
                cycle_agent.logger.level = LogLevel.ERROR
            result = cycle_agent.run(task_text, reset=False)
            _spinner.stop()
            elapsed = time.time() - start_time

            completed += 1
            console.print()
            console.print(Markdown(str(result)))
            console.print()
            console.print(f"[dim]Cycle {i}/{n} completed in {elapsed:.1f}s[/]")

        except KeyboardInterrupt:
            _spinner.stop()
            console.print(f"\n[yellow]Interrupted at cycle {i}/{n}.[/]")
            break
        except Exception as e:
            _spinner.stop()
            errors += 1
            console.print(f"\n[bold red]Cycle {i}/{n} error:[/] {e}")

    # Summary
    console.print(Rule(style="cyan"))
    console.print(f"[bold]Repeat summary:[/] {completed} completed, {errors} errors out of {n} cycles")


def _shutdown_browser(agent):
    """Shut down the browser manager if one exists on the agent."""
    manager = getattr(agent, "_browser_manager", None)
    if manager:
        manager.shutdown()


def _shutdown_gui(agent):
    """Shut down the GUI manager if one exists on the agent."""
    manager = getattr(agent, "_gui_manager", None)
    if manager:
        manager.shutdown()


def prepend_instructions(task: str, instructions: str | None) -> str:
    if instructions:
        return instructions+"""
The above should be treated as information only. What the user is asking (what you need to reply to) is the following:
"""+task
    return task


def run_one_shot(task: str, skip_instructions: bool = False, auto_approve: bool = True, browser_enabled: bool = False, gui_enabled: bool = False):
    global _auto_approve
    _auto_approve = auto_approve
    try_load_dotenv()
    check_required_env()
    model = build_model()
    agent = build_agent(model, approval_callback=interactive_approval_callback, browser_enabled=browser_enabled, gui_enabled=gui_enabled)
    instructions = None
    if not skip_instructions:
        console.print("[dim]Loading agent instructions...[/]")
        instructions = load_agent_instructions()
    _spinner.start()
    try:
        result = agent.run(prepend_instructions(task, instructions))
        _spinner.stop()
        console.print(Markdown(str(result)))
    except Exception as e:
        _spinner.stop()
        from smolagents.utils import AgentExecutionRejected
        if isinstance(e, AgentExecutionRejected):
            console.print("\n[yellow]Execution rejected by user.[/]")
        else:
            raise
    finally:
        manager = getattr(agent, "_browser_manager", None)
        if manager:
            manager.shutdown()
        _shutdown_gui(agent)


def run_repl(skip_instructions: bool = False, auto_approve: bool = True, browser_enabled: bool = False, gui_enabled: bool = False):
    global _auto_approve
    _auto_approve = auto_approve
    try_load_dotenv()
    check_required_env()

    model = build_model()
    agent = build_agent(model, approval_callback=interactive_approval_callback, browser_enabled=browser_enabled, gui_enabled=gui_enabled)
    model_id = get_env("BPSA_MODEL_ID")
    server_model = get_env("BPSA_SERVER_MODEL", default="OpenAIServerModel")
    tool_count = count_tools(agent)
    global _verbose
    verbose = get_env("BPSA_VERBOSE", default="1") == "1"
    _verbose = verbose

    console.clear()
    print_banner(model_id, server_model, tool_count)

    instructions = None
    if not skip_instructions:
        console.print("[dim]Loading agent instructions...[/]")
        instructions = load_agent_instructions()
        if not instructions:
            console.print("  [dim]No instruction files found.[/]")
    else:
        console.print("[dim]Skipping agent instruction files.[/]")
    console.print()

    # Try to use prompt_toolkit, fall back to basic input
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings

        history_file = os.path.expanduser("~/.bpsa_history")
        completer = WordCompleter(SLASH_COMMANDS, sentence=True)

        # Key bindings: Enter submits, Alt+Enter and Shift+Enter insert newline
        bindings = KeyBindings()

        @bindings.add("escape", "enter")  # Alt+Enter
        def handle_alt_enter(event):
            event.current_buffer.insert_text("\n")


        global _prompt_session
        session = PromptSession(
            history=FileHistory(history_file),
            completer=completer,
            key_bindings=bindings,
        )
        _prompt_session = session

        _has_prompt_toolkit = True

        def get_input():
            global _voice_prompt_active
            try:
                console.print(Rule(style="dim"))
                voice_on = _voice_listener is not None
                hint = "[dim]Enter to submit, Alt+Enter for newline"
                if voice_on:
                    hint += ", voice active"
                hint += "[/]"
                console.print(hint)
                prompt_str = "[mic] > " if voice_on else "> "
                # Clear stale voice transcriptions from agent execution time
                while not _voice_queue.empty():
                    try:
                        _voice_queue.get_nowait()
                    except queue.Empty:
                        break
                _voice_prompt_active = True
                try:
                    return session.prompt(prompt_str, pre_run=_setup_voice_before_render)
                finally:
                    _voice_prompt_active = False
            except EOFError:
                return None
            except KeyboardInterrupt:
                print()
                return ""

        def _setup_voice_before_render():
            """Hook called once when prompt_toolkit app starts; registers the voice renderer."""
            app = session.app
            if not hasattr(app, "_voice_renderer_registered"):
                app.before_render += _voice_before_render
                app._voice_renderer_registered = True

        def _voice_before_render(app):
            """Called before each prompt_toolkit render; drains voice queue into buffer."""
            _drain_voice_queue_into_buffer(app.current_buffer)

    except ImportError:
        _has_prompt_toolkit = False

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
    last_prompt = None
    pending_shell_outputs = []
    aliases = _load_aliases()
    autosave_interval = int(get_env("BPSA_AUTOSAVE_INTERVAL", default="5"))
    autosave_file = os.path.expanduser("~/.bpsa_autosave.json")
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
            _shutdown_voice()
            _shutdown_browser(agent)
            _shutdown_gui(agent)
            console.print("[dim]Goodbye![/]")
            break

        text = user_input.strip()
        if not text:
            continue

        # Expand aliases: check if first word matches an alias
        first_word = text.split(None, 1)[0] if text else ""
        if first_word in aliases:
            rest = text[len(first_word):].lstrip()
            text = aliases[first_word] + (" " + rest if rest else "")

        # Handle !!! shell escape: run OS command and immediately send to agent
        if text.startswith("!!!"):
            shell_cmd = text[3:].strip()
            if shell_cmd:
                output = _run_shell_streaming(shell_cmd)
                shell_context = f"<shell>\n<cmd>{shell_cmd}</cmd>\n<output>\n{truncate_content(output)}</output>\n</shell>"
                text = f"Analyze the output of the command above.\n{shell_context}"
                # Fall through to agent run below
            else:
                continue

        # Handle !! shell escape: run OS command, output appended to next prompt
        elif text.startswith("!!"):
            shell_cmd = text[2:].strip()
            if shell_cmd:
                output = _run_shell_streaming(shell_cmd)
                pending_shell_outputs.append((shell_cmd, truncate_content(output)))
            continue

        # Handle ! shell escape: run OS command directly (agent doesn't see it)
        if text.startswith("!"):
            shell_cmd = text[1:].strip()
            if shell_cmd:
                try:
                    subprocess.run(shell_cmd, shell=True)
                except KeyboardInterrupt:
                    console.print("\n[dim]Command interrupted.[/]")
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
                _shutdown_voice()
                _shutdown_browser(agent)
                _shutdown_gui(agent)
                console.print("[dim]Goodbye![/]")
                break
            elif cmd == "/help":
                print_help()
                continue
            elif cmd == "/alias":
                args = cmd_args.strip()
                if not args:
                    if aliases:
                        for name, value in sorted(aliases.items()):
                            console.print(f"  [cyan]{name}[/] = {value}")
                    else:
                        console.print("[dim]No aliases defined. Usage: /alias <name> <value>[/]")
                elif args.startswith("-d "):
                    alias_name = args[3:].strip()
                    if alias_name in aliases:
                        del aliases[alias_name]
                        _save_aliases(aliases)
                        console.print(f"[cyan]Alias '{alias_name}' deleted.[/]")
                    else:
                        console.print(f"[yellow]Alias '{alias_name}' not found.[/]")
                else:
                    parts = args.split(None, 1)
                    if len(parts) < 2:
                        console.print("[yellow]Usage: /alias <name> <value> or /alias -d <name>[/]")
                    else:
                        aliases[parts[0]] = parts[1]
                        _save_aliases(aliases)
                        console.print(f"[cyan]{parts[0]}[/] = {parts[1]}")
                continue
            elif cmd == "/redo":
                if last_prompt is None:
                    console.print("[yellow]No previous prompt to redo.[/]")
                    continue
                # Undo the steps from the last turn, then re-run
                cmd_undo(agent, "")
                text = last_prompt
                # Fall through to agent run below
            elif cmd == "/clear":
                _shutdown_browser(agent)
                _shutdown_gui(agent)
                agent = build_agent(model, approval_callback=interactive_approval_callback, browser_enabled=browser_enabled, gui_enabled=gui_enabled)
                session_stats = {
                    "turns": 0,
                    "total_time": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                }
                last_answer = None
                first_turn = True
                console.clear()
                print_banner(model_id, server_model, count_tools(agent))
                continue
            elif cmd == "/show-tools":
                print_tools(agent)
                continue
            elif cmd == "/verbose":
                verbose = not verbose
                _verbose = verbose
                console.print(f"[cyan]Verbose mode: {'on' if verbose else 'off (compact)'}[/]")
                continue
            elif cmd == "/save":
                save_answer(last_answer, cmd_args)
                continue
            elif cmd == "/show-stats":
                print_stats(session_stats)
                continue
            elif cmd == "/run-prompt":
                file_content = load_file_as_prompt(cmd_args)
                if file_content:
                    text = file_content
                    # Fall through to agent run
                else:
                    continue
            elif cmd == "/set-max-steps":
                change_steps(agent, cmd_args)
                continue
            elif cmd == "/run-py":
                run_script(agent, cmd_args)
                continue
            elif cmd == "/cd":
                change_directory(cmd_args)
                continue
            elif cmd == "/pwd":
                console.print(f"[cyan]{os.getcwd()}[/]")
                continue
            elif cmd == "/load-instructions":
                console.print("[dim]Loading agent instructions...[/]")
                instructions = load_agent_instructions()
                if instructions:
                    first_turn = True
                    console.print("[green]Instructions loaded. They will be included in the next prompt.[/]")
                else:
                    console.print("[yellow]No instruction files found.[/]")
                continue
            elif cmd == "/plan":
                arg = cmd_args.strip().lower()
                if arg == "":
                    interval = getattr(agent, "planning_interval", None)
                    status = f"on (every {interval} steps)" if interval else "off"
                    console.print(f"[cyan]Planning: {status}[/]")
                elif arg == "on":
                    agent.planning_interval = 22
                    console.print("[cyan]Planning: on (every 22 steps)[/]")
                elif arg == "off":
                    agent.planning_interval = None
                    console.print("[cyan]Planning: off[/]")
                else:
                    try:
                        n = int(arg)
                        if n < 1:
                            raise ValueError
                        agent.planning_interval = n
                        console.print(f"[cyan]Planning: on (every {n} steps)[/]")
                    except ValueError:
                        console.print("[yellow]Usage: /plan [on|off|N][/]")
                continue
            elif cmd == "/show-compression-stats":
                cmd_compression_stats(agent)
                continue
            elif cmd == "/show-memory-stats":
                cmd_memory_stats(agent)
                continue
            elif cmd == "/compress":
                cmd_compress(agent, cmd_args)
                continue
            elif cmd == "/compression":
                cmd_compression_toggle(agent, cmd_args)
                continue
            elif cmd == "/compression-keep-recent-steps":
                cmd_compression_keep_recent(agent, cmd_args)
                continue
            elif cmd == "/compression-max-uncompressed-steps":
                cmd_compression_max_uncompressed(agent, cmd_args)
                continue
            elif cmd == "/compression-model":
                cmd_compression_model(agent, cmd_args)
                continue
            elif cmd == "/save-step":
                cmd_save_step(agent, cmd_args)
                continue
            elif cmd == "/show-step":
                cmd_show_step(agent, cmd_args)
                continue
            elif cmd == "/show-steps":
                cmd_show_steps(agent)
                continue
            elif cmd == "/undo-steps":
                cmd_undo(agent, cmd_args)
                continue
            elif cmd == "/repeat":
                parts = cmd_args.strip().split(None, 1)
                if len(parts) < 2:
                    console.print("[yellow]Usage: /repeat <N> <prompt>[/]")
                    continue
                try:
                    repeat_n = int(parts[0])
                    if repeat_n < 1:
                        raise ValueError
                except ValueError:
                    console.print("[red]N must be a positive integer. Usage: /repeat <N> <prompt>[/]")
                    continue
                cmd_repeat(agent, model, repeat_n, parts[1], session_stats, verbose, instructions, first_turn, browser_enabled, gui_enabled=gui_enabled)
                continue
            elif cmd == "/repeat-prompt":
                parts = cmd_args.strip().split(None, 1)
                if len(parts) < 2:
                    console.print("[yellow]Usage: /repeat-prompt <N> <path>[/]")
                    continue
                try:
                    repeat_n = int(parts[0])
                    if repeat_n < 1:
                        raise ValueError
                except ValueError:
                    console.print("[red]N must be a positive integer. Usage: /repeat-prompt <N> <path>[/]")
                    continue
                file_content = load_file_as_prompt(parts[1])
                if file_content is None:
                    continue
                cmd_repeat(agent, model, repeat_n, file_content, session_stats, verbose, instructions, first_turn, browser_enabled, gui_enabled=gui_enabled)
                continue
            elif cmd == "/session-save":
                cmd_session_save(agent, session_stats, cmd_args)
                continue
            elif cmd == "/session-load":
                result = cmd_session_load(agent, cmd_args)
                if result is not None:
                    session_stats = result
                    first_turn = False
                continue
            elif cmd == "/auto-approve":
                arg = cmd_args.strip().lower()
                if arg == "on":
                    _auto_approve = True
                elif arg == "off":
                    _auto_approve = False
                elif arg == "":
                    _auto_approve = not _auto_approve
                else:
                    console.print("[yellow]Usage: /auto-approve [on|off][/]")
                    continue
                console.print(f"[cyan]Auto-approve: {'on' if _auto_approve else 'off'}[/]")
                continue
            elif cmd == "/voice":
                arg = cmd_args.strip().lower()
                if arg == "on":
                    console.print("[cyan]Loading voice support.[/]")
                    if not _has_prompt_toolkit:
                        console.print("[red]Voice input requires voicelistener. Install with: pip install voicelistener[/]")
                    else:
                        err = _voice_start()
                        if err:
                            console.print(f"[red]{err}[/]")
                        else:
                            console.print("[cyan][mic] Voice input active[/]")
                elif arg == "off":
                    err = _voice_stop()
                    if err:
                        console.print(f"[yellow]{err}[/]")
                    else:
                        console.print("[cyan]Voice input deactivated[/]")
                elif arg == "":
                    if _voice_listener is not None:
                        transcriber = get_env("BPSA_VOICE_TRANSCRIBER", default="(unknown)")
                        model = get_env("BPSA_VOICE_MODEL", default=BPSA_DEFAULT_VOICE_MODEL)
                        console.print(f"[cyan]Voice: on | transcriber: {transcriber} | model: {model}[/]")
                    else:
                        console.print("[dim]Voice: off[/]")
                else:
                    console.print("[yellow]Usage: /voice [on|off][/]")
                continue
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
            last_prompt = text
            task_text = prepend_instructions(text, instructions) if first_turn else text
            first_turn = False
            if pending_shell_outputs:
                shell_context = "\n".join(
                    f"<shell>\n<cmd>{cmd}</cmd>\n<output>\n{out}</output>\n</shell>"
                    for cmd, out in pending_shell_outputs
                )
                task_text = task_text + "\n" + shell_context
                pending_shell_outputs.clear()
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
            print_turn_summary(turn_num, elapsed, turn_input, turn_output, agent)
            console.print()

            # Auto-save session periodically
            if autosave_interval > 0 and session_stats["turns"] % autosave_interval == 0:
                try:
                    from smolagents.bp_session import save_session
                    save_session(autosave_file, agent, session_stats)
                    console.print(f"[dim]Auto-saved session to {autosave_file}[/]")
                except Exception:
                    pass

        except KeyboardInterrupt:
            _spinner.stop()
            console.print("\n[yellow]Interrupted.[/]")
        except Exception as e:
            _spinner.stop()
            from smolagents.utils import AgentExecutionRejected
            if isinstance(e, AgentExecutionRejected):
                console.print("\n[yellow]Execution rejected by user.[/]")
            else:
                console.print(f"\n[bold red]Error:[/] {e}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="bpsa",
        description="Beyond Python SmolAgents - Interactive AI agent CLI",
    )
    parser.add_argument(
        "--load-instructions", action="store_true",
        help="Load agent instruction files (CLAUDE.md, AGENTS.md, etc.) at startup",
    )
    parser.add_argument(
        "--auto-approve", choices=["on", "off"], default="off",
        help="Auto-approve tag execution (runcode, savetofile, appendtofile). Default: off",
    )
    parser.add_argument(
        "--browser", action="store_true",
        help="Enable Playwright browser integration (navigate, click, type_text, etc. in runcode blocks)",
    )
    parser.add_argument(
        "--gui-x11", action="store_true",
        help="Enable native GUI interaction tools (screenshot, click, type, key via xdotool/ImageMagick on X11)",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a one-shot task")
    run_parser.add_argument("task", type=str, help="The task to execute")

    args = parser.parse_args()
    skip_instructions = not args.load_instructions
    auto_approve = args.auto_approve == "on"
    from smolagents.bp_utils import get_env_bool
    browser_enabled = args.browser or get_env_bool("BPSA_BROWSER")
    gui_enabled = args.gui_x11 or get_env_bool("BPSA_GUI")

    # Piped input detection
    if not sys.stdin.isatty() and args.command is None:
        task = sys.stdin.read().strip()
        if task:
            run_one_shot(task, skip_instructions=skip_instructions, auto_approve=auto_approve, browser_enabled=browser_enabled, gui_enabled=gui_enabled)
        else:
            fail("No input provided via pipe.")
        return

    if args.command == "run":
        run_one_shot(args.task, skip_instructions=skip_instructions, auto_approve=auto_approve, browser_enabled=browser_enabled, gui_enabled=gui_enabled)
    else:
        run_repl(skip_instructions=skip_instructions, auto_approve=auto_approve, browser_enabled=browser_enabled, gui_enabled=gui_enabled)


if __name__ == "__main__":
    main()
