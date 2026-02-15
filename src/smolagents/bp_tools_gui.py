"""
Native GUI application interaction tools for BPSA CLI.

Provides Tool classes for launching GUI applications and interacting with them
via screenshot capture (ImageMagick ``import``) and mouse/keyboard automation
(``xdotool``) on the real X11 display.

System dependencies: xdotool, imagemagick, a running X11 display ($DISPLAY).
"""

import os
import shutil
import signal
import subprocess
import time
from io import BytesIO

from .memory import ActionStep
from .tools import Tool


class GuiManager:
    """Manages a launched GUI process and its X11 window.

    Uses ``xdotool`` for window discovery and input simulation, and
    ImageMagick ``import`` for screenshot capture.
    """

    def __init__(self):
        self._process = None
        self._window_id = None
        self._should_screenshot = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def launch(self, binary_path: str, args: list[str] | None = None, working_dir: str | None = None) -> dict:
        """Launch a GUI binary and wait for its X11 window to appear.

        Returns a dict with ``pid`` and ``window_id``.
        """
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError(
                "A GUI process is already running (PID {pid}). "
                "Close it first with gui_close.".format(pid=self._process.pid)
            )

        # Resolve relative binary paths against working_dir (Popen's cwd
        # only affects the child process, not executable lookup).
        resolved_path = binary_path
        if working_dir and not os.path.isabs(binary_path):
            candidate = os.path.join(working_dir, binary_path)
            if os.path.isfile(candidate):
                resolved_path = candidate
        cmd = [resolved_path] + (args or [])
        self._process = subprocess.Popen(
            cmd,
            cwd=working_dir or None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._window_id = self._wait_for_window(timeout=15)
        return {"pid": self._process.pid, "window_id": self._window_id}

    def close(self):
        """Terminate the managed process (SIGTERM then SIGKILL) and reset state."""
        if self._process is None:
            return
        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        except OSError:
            pass
        self._process = None
        self._window_id = None

    def shutdown(self):
        """Alias for :meth:`close` (matches BrowserManager API)."""
        self.close()

    # ------------------------------------------------------------------
    # Window discovery
    # ------------------------------------------------------------------

    def _wait_for_window(self, timeout: float = 15) -> str:
        """Poll ``xdotool search --pid`` until the window appears or *timeout* expires."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"GUI process exited prematurely with code {self._process.returncode}"
                )
            try:
                result = subprocess.run(
                    ["xdotool", "search", "--pid", str(self._process.pid)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                wids = result.stdout.strip().splitlines()
                if wids:
                    return wids[0]
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            time.sleep(0.3)
        raise TimeoutError(
            f"No X11 window found for PID {self._process.pid} within {timeout}s"
        )

    def _ensure_window(self):
        """Validate the process is alive and the window still exists."""
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("No GUI process is running. Launch one first with gui_launch.")
        if self._window_id is None:
            raise RuntimeError("No window ID recorded. Launch a GUI process first.")

    def _ensure_process(self):
        """Validate the process is alive (window may be any)."""
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("No GUI process is running. Launch one first with gui_launch.")

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def list_windows(self) -> list[dict]:
        """List all X11 windows belonging to the managed process.

        Returns a list of dicts with ``window_id``, ``name``, ``geometry``,
        and ``is_current`` (whether it is the currently targeted window).
        """
        self._ensure_process()
        result = subprocess.run(
            ["xdotool", "search", "--pid", str(self._process.pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        wids = [w.strip() for w in result.stdout.strip().splitlines() if w.strip()]
        windows = []
        for wid in wids:
            info = {"window_id": wid, "is_current": wid == self._window_id}
            # Get window name
            try:
                name_result = subprocess.run(
                    ["xdotool", "getwindowname", wid],
                    capture_output=True, text=True, timeout=3,
                )
                info["name"] = name_result.stdout.strip()
            except Exception:
                info["name"] = ""
            # Get window geometry
            try:
                geo_result = subprocess.run(
                    ["xdotool", "getwindowgeometry", wid],
                    capture_output=True, text=True, timeout=3,
                )
                info["geometry"] = geo_result.stdout.strip()
            except Exception:
                info["geometry"] = ""
            windows.append(info)
        return windows

    def focus_window(self, window_id: str):
        """Switch the targeted window to *window_id*."""
        self._ensure_process()
        # Verify the window belongs to this process
        result = subprocess.run(
            ["xdotool", "search", "--pid", str(self._process.pid)],
            capture_output=True, text=True, timeout=5,
        )
        wids = [w.strip() for w in result.stdout.strip().splitlines() if w.strip()]
        if window_id not in wids:
            raise ValueError(
                f"Window {window_id} does not belong to PID {self._process.pid}. "
                f"Available windows: {wids}"
            )
        self._window_id = window_id

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def capture_screenshot(self):
        """Capture a screenshot of the managed window and return a PIL Image."""
        import PIL.Image

        self._ensure_window()
        # Activate the window so it is on top
        subprocess.run(
            ["xdotool", "windowactivate", "--sync", self._window_id],
            capture_output=True,
            timeout=5,
        )
        time.sleep(0.15)  # brief settle time after activation
        # Use ImageMagick import to capture the window
        result = subprocess.run(
            ["import", "-window", self._window_id, "png:-"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Screenshot capture failed (exit {result.returncode}): {result.stderr.decode(errors='replace')}"
            )
        image = PIL.Image.open(BytesIO(result.stdout))
        return image.copy()  # detach from BytesIO buffer

    def click(self, x: int, y: int, button: int = 1):
        """Click at (x, y) coordinates relative to the managed window."""
        self._ensure_window()
        subprocess.run(
            ["xdotool", "mousemove", "--window", self._window_id, str(x), str(y)],
            capture_output=True,
            timeout=5,
        )
        subprocess.run(
            ["xdotool", "click", "--window", self._window_id, str(button)],
            capture_output=True,
            timeout=5,
        )

    def type_text(self, text: str):
        """Type a text string into the managed window."""
        self._ensure_window()
        subprocess.run(
            ["xdotool", "type", "--window", self._window_id, "--clearmodifiers", text],
            capture_output=True,
            timeout=10,
        )

    def send_key(self, key: str):
        """Send a key press (e.g. 'Return', 'Tab', 'ctrl+s') to the managed window."""
        self._ensure_window()
        subprocess.run(
            ["xdotool", "key", "--window", self._window_id, key],
            capture_output=True,
            timeout=5,
        )


# ======================================================================
# Tool classes
# ======================================================================


class GuiLaunchTool(Tool):
    """Launch a GUI binary and wait for its X11 window."""

    name = "gui_launch"
    description = (
        "Launch a compiled GUI binary and wait for its X11 window to appear. "
        "Returns PID and window info. Only one GUI process can be managed at a time; "
        "close the current one before launching another."
    )
    inputs = {
        "binary_path": {"type": "string", "description": "Path to the executable to launch"},
        "args": {
            "type": "string",
            "description": "Space-separated command-line arguments (optional)",
            "nullable": True,
        },
        "working_dir": {
            "type": "string",
            "description": "Working directory for the process (optional)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self, binary_path: str, args: str | None = None, working_dir: str | None = None) -> str:
        arg_list = args.split() if args and args.strip() else None
        info = self.gui_manager.launch(binary_path, args=arg_list, working_dir=working_dir or None)
        return f"Launched GUI: PID={info['pid']}, window_id={info['window_id']}"


class GuiScreenshotTool(Tool):
    """Request a screenshot of the GUI window (captured after the step)."""

    name = "gui_screenshot"
    description = (
        "Request a screenshot of the running GUI window. The screenshot is captured "
        "after the current code step finishes and will be visible as an image in the "
        "next turn. Returns confirmation that the screenshot was requested."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self) -> str:
        self.gui_manager._ensure_window()
        self.gui_manager._should_screenshot = True
        return "Screenshot requested — it will be captured after this step completes and visible in the next turn."


class GuiClickTool(Tool):
    """Click at (x, y) coordinates in the GUI window."""

    name = "gui_click"
    description = (
        "Click at pixel coordinates (x, y) relative to the top-left corner of the "
        "GUI window. Use gui_screenshot first to see the window and determine coordinates."
    )
    inputs = {
        "x": {"type": "integer", "description": "X coordinate (pixels from left edge of window)"},
        "y": {"type": "integer", "description": "Y coordinate (pixels from top edge of window)"},
        "button": {
            "type": "integer",
            "description": "Mouse button: 1=left (default), 2=middle, 3=right",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self, x: int, y: int, button: int | None = None) -> str:
        self.gui_manager.click(x, y, button=button or 1)
        return f"Clicked at ({x}, {y}) with button {button or 1}"


class GuiTypeTool(Tool):
    """Type text into the GUI window."""

    name = "gui_type"
    description = (
        "Type a text string into the currently focused widget of the GUI window. "
        "Click on the target widget first if needed."
    )
    inputs = {
        "text": {"type": "string", "description": "The text to type"},
    }
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self, text: str) -> str:
        self.gui_manager.type_text(text)
        return f"Typed {len(text)} characters"


class GuiKeyTool(Tool):
    """Send a key press to the GUI window."""

    name = "gui_key"
    description = (
        "Send a key press or key combination to the GUI window. "
        "Examples: 'Return', 'Tab', 'Escape', 'ctrl+s', 'alt+F4', 'shift+Tab'."
    )
    inputs = {
        "key": {
            "type": "string",
            "description": "Key name or combination (e.g. 'Return', 'ctrl+s', 'alt+F4')",
        },
    }
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self, key: str) -> str:
        self.gui_manager.send_key(key)
        return f"Sent key: {key}"


class GuiListWindowsTool(Tool):
    """List all X11 windows belonging to the managed GUI process."""

    name = "gui_list_windows"
    description = (
        "List all X11 windows belonging to the running GUI process. "
        "Returns each window's ID, name, geometry, and whether it is the "
        "currently targeted window. Useful for discovering tooltips, dialogs, "
        "popups, and child windows that appear on top of the main window."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self) -> str:
        windows = self.gui_manager.list_windows()
        if not windows:
            return "No windows found for the managed process."
        lines = []
        for w in windows:
            marker = " [CURRENT]" if w["is_current"] else ""
            lines.append(
                f"  id={w['window_id']}{marker}  name={w['name']!r}  {w['geometry']}"
            )
        return f"Found {len(windows)} window(s):\n" + "\n".join(lines)


class GuiFocusWindowTool(Tool):
    """Switch the targeted window for GUI interactions."""

    name = "gui_focus_window"
    description = (
        "Switch which window receives screenshots, clicks, typing, and key presses. "
        "Use gui_list_windows first to see available window IDs. "
        "Useful for interacting with tooltips, dialogs, or popup windows."
    )
    inputs = {
        "window_id": {
            "type": "string",
            "description": "The X11 window ID to target (from gui_list_windows output)",
        },
    }
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self, window_id: str) -> str:
        window_id = window_id.strip()
        self.gui_manager.focus_window(window_id)
        return f"Now targeting window {window_id}. All GUI interactions will use this window."


class GuiCloseTool(Tool):
    """Close the managed GUI process."""

    name = "gui_close"
    description = "Close (terminate) the currently running GUI process and clean up."
    inputs = {}
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self) -> str:
        if self.gui_manager._process is None:
            return "No GUI process is running."
        pid = self.gui_manager._process.pid
        self.gui_manager.close()
        return f"GUI process (PID {pid}) terminated."


# ======================================================================
# Step callback
# ======================================================================


def gui_screenshot_callback(memory_step, agent=None):
    """Step callback: capture a GUI screenshot when the flag is set.

    Follows the same pattern as the vision_web_browser ``save_screenshot``
    callback — clears old screenshots to save context, stores the new one
    in ``memory_step.observations_images``.
    """
    manager = getattr(agent, "_gui_manager", None) if agent else None
    if manager is None or not manager._should_screenshot:
        return

    manager._should_screenshot = False

    try:
        image = manager.capture_screenshot()
    except Exception as e:
        info = f"\n[GUI screenshot failed: {e}]"
        memory_step.observations = (
            info if memory_step.observations is None else memory_step.observations + info
        )
        return

    # Clear screenshots from older steps to keep context lean
    if agent is not None:
        current_step = memory_step.step_number
        for previous_step in agent.memory.steps:
            if isinstance(previous_step, ActionStep) and previous_step.step_number <= current_step - 2:
                previous_step.observations_images = None

    memory_step.observations_images = [image]
    size_info = f"\n[GUI screenshot captured: {image.size[0]}x{image.size[1]} pixels]"
    memory_step.observations = (
        size_info if memory_step.observations is None else memory_step.observations + size_info
    )


# ======================================================================
# Factory
# ======================================================================


def _check_gui_dependencies():
    """Verify that required system tools and X11 display are available."""
    missing = []
    if not shutil.which("xdotool"):
        missing.append("xdotool")
    if not shutil.which("import"):
        missing.append("imagemagick (provides the 'import' command)")
    if missing:
        raise EnvironmentError(
            "Missing system dependencies for --gui: {deps}. "
            "Install with: sudo apt install xdotool imagemagick".format(
                deps=", ".join(missing)
            )
        )
    if not os.environ.get("DISPLAY"):
        raise EnvironmentError(
            "No X11 display found ($DISPLAY is not set). "
            "The --gui flag requires a running X11 display."
        )


def create_gui_tools() -> tuple[GuiManager, list[Tool]]:
    """Create a GuiManager and return ``(manager, list_of_tools)``."""
    _check_gui_dependencies()
    manager = GuiManager()
    tools = [
        GuiLaunchTool(manager),
        GuiScreenshotTool(manager),
        GuiClickTool(manager),
        GuiTypeTool(manager),
        GuiKeyTool(manager),
        GuiListWindowsTool(manager),
        GuiFocusWindowTool(manager),
        GuiCloseTool(manager),
    ]
    return manager, tools
