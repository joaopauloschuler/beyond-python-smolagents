"""
Native GUI application interaction tools for BPSA CLI.

Provides Tool classes for launching GUI applications and interacting with them
via screenshot capture (ImageMagick ``import``) and mouse/keyboard automation
(``xdotool``) on the real X11 display.

Multiple GUI apps can be managed simultaneously, each identified by an
agent-chosen name assigned at launch time.

System dependencies: xdotool, imagemagick, a running X11 display ($DISPLAY).
"""

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from io import BytesIO

from .memory import ActionStep
from .tools import Tool


@dataclass
class _AppState:
    """Internal state for a single managed GUI application."""

    process: subprocess.Popen
    window_id: str
    name: str


class GuiManager:
    """Manages launched GUI processes and their X11 windows.

    Uses ``xdotool`` for window discovery and input simulation, and
    ImageMagick ``import`` for screenshot capture.  Supports multiple
    simultaneous apps, each identified by a unique name.
    """

    def __init__(self):
        self._apps: dict[str, _AppState] = {}
        self._current: str | None = None
        self._should_screenshot: bool = False
        self._screenshot_app: str | None = None

    # ------------------------------------------------------------------
    # App resolution
    # ------------------------------------------------------------------

    def _resolve_app(self, app: str | None = None) -> _AppState:
        """Resolve an app name to its ``_AppState``.

        * If *app* is given, look it up directly.
        * If *app* is ``None`` and only one app exists, return it.
        * If *app* is ``None`` and ``_current`` is set, return that.
        * Otherwise raise.

        Dead processes are cleaned up automatically.
        """
        # Normalize empty string to None
        if app is not None and not app.strip():
            app = None

        if app is not None:
            state = self._apps.get(app)
            if state is None:
                available = list(self._apps.keys()) or ["(none)"]
                raise ValueError(
                    f"No app named {app!r}. Running apps: {', '.join(available)}"
                )
        elif len(self._apps) == 1:
            state = next(iter(self._apps.values()))
        elif self._current is not None and self._current in self._apps:
            state = self._apps[self._current]
        elif not self._apps:
            raise RuntimeError("No GUI process is running. Launch one first with gui_launch.")
        else:
            available = list(self._apps.keys())
            raise ValueError(
                f"Multiple apps are running ({', '.join(available)}). "
                "Specify which app to target with the 'app' parameter."
            )

        # Validate the process is still alive
        if state.process.poll() is not None:
            del self._apps[state.name]
            if self._current == state.name:
                self._current = next(iter(self._apps), None)
            raise RuntimeError(
                f"GUI process {state.name!r} (PID {state.process.pid}) has exited. "
                "It has been removed from the tracked apps."
            )

        return state

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def launch(
        self, name: str, binary_path: str, args: list[str] | None = None, working_dir: str | None = None
    ) -> dict:
        """Launch a GUI binary and wait for its X11 window to appear.

        *name* is a mandatory identifier chosen by the caller (e.g. ``"editor"``).
        Returns a dict with ``pid``, ``window_id``, and ``name``.
        """
        if name in self._apps:
            raise RuntimeError(
                f"An app named {name!r} is already running (PID {self._apps[name].process.pid}). "
                "Close it first with gui_close or choose a different name."
            )

        # Resolve relative binary paths against working_dir (Popen's cwd
        # only affects the child process, not executable lookup).
        resolved_path = binary_path
        if working_dir and not os.path.isabs(binary_path):
            candidate = os.path.join(working_dir, binary_path)
            if os.path.isfile(candidate):
                resolved_path = candidate
        cmd = [resolved_path] + (args or [])
        process = subprocess.Popen(
            cmd,
            cwd=working_dir or None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        window_id = self._wait_for_window(process, timeout=15)
        state = _AppState(process=process, window_id=window_id, name=name)
        self._apps[name] = state
        self._current = name
        return {"pid": process.pid, "window_id": window_id, "name": name}

    def close(self, app: str | None = None):
        """Terminate the resolved app (SIGTERM then SIGKILL) and remove it."""
        if not self._apps:
            return
        state = self._resolve_app(app)
        try:
            state.process.terminate()
            try:
                state.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                state.process.kill()
                state.process.wait(timeout=3)
        except OSError:
            pass
        del self._apps[state.name]
        if self._current == state.name:
            self._current = next(iter(self._apps), None)

    def shutdown(self):
        """Close all managed apps."""
        for name in list(self._apps.keys()):
            self.close(app=name)
        self._current = None

    # ------------------------------------------------------------------
    # Window discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _wait_for_window(process: subprocess.Popen, timeout: float = 15) -> str:
        """Poll ``xdotool search --pid`` until the window appears or *timeout* expires."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"GUI process exited prematurely with code {process.returncode}"
                )
            try:
                result = subprocess.run(
                    ["xdotool", "search", "--pid", str(process.pid)],
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
            f"No X11 window found for PID {process.pid} within {timeout}s"
        )

    def _ensure_window(self, app: str | None = None) -> _AppState:
        """Validate the resolved app is alive and has a window_id."""
        state = self._resolve_app(app)
        if state.window_id is None:
            raise RuntimeError(f"No window ID recorded for app {state.name!r}.")
        return state

    def _ensure_process(self, app: str | None = None) -> _AppState:
        """Validate the resolved app is alive (window may be any)."""
        return self._resolve_app(app)

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def list_windows(self, app: str | None = None) -> list[dict]:
        """List all X11 windows belonging to the resolved app.

        Returns a list of dicts with ``window_id``, ``name``, ``geometry``,
        and ``is_current`` (whether it is the currently targeted window).
        """
        state = self._ensure_process(app)
        result = subprocess.run(
            ["xdotool", "search", "--pid", str(state.process.pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        wids = [w.strip() for w in result.stdout.strip().splitlines() if w.strip()]
        windows = []
        for wid in wids:
            info = {"window_id": wid, "is_current": wid == state.window_id}
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

    def focus_window(self, window_id: str, app: str | None = None):
        """Switch the targeted window to *window_id* within the resolved app."""
        state = self._ensure_process(app)
        # Verify the window belongs to this process
        result = subprocess.run(
            ["xdotool", "search", "--pid", str(state.process.pid)],
            capture_output=True, text=True, timeout=5,
        )
        wids = [w.strip() for w in result.stdout.strip().splitlines() if w.strip()]
        if window_id not in wids:
            raise ValueError(
                f"Window {window_id} does not belong to PID {state.process.pid}. "
                f"Available windows: {wids}"
            )
        state.window_id = window_id
        self._current = state.name

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def capture_screenshot(self, app: str | None = None):
        """Capture a screenshot of the resolved app's window and return a PIL Image."""
        import PIL.Image

        state = self._ensure_window(app)
        # Activate the window so it is on top
        subprocess.run(
            ["xdotool", "windowactivate", "--sync", state.window_id],
            capture_output=True,
            timeout=5,
        )
        time.sleep(0.15)  # brief settle time after activation
        # Use ImageMagick import to capture the window
        result = subprocess.run(
            ["import", "-window", state.window_id, "png:-"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Screenshot capture failed (exit {result.returncode}): {result.stderr.decode(errors='replace')}"
            )
        image = PIL.Image.open(BytesIO(result.stdout))
        return image.copy()  # detach from BytesIO buffer

    def click(self, x: int, y: int, button: int = 1, app: str | None = None):
        """Click at (x, y) coordinates relative to the resolved app's window."""
        state = self._ensure_window(app)
        subprocess.run(
            ["xdotool", "mousemove", "--window", state.window_id, str(x), str(y)],
            capture_output=True,
            timeout=5,
        )
        subprocess.run(
            ["xdotool", "click", "--window", state.window_id, str(button)],
            capture_output=True,
            timeout=5,
        )

    def type_text(self, text: str, app: str | None = None):
        """Type a text string into the resolved app's window."""
        state = self._ensure_window(app)
        subprocess.run(
            ["xdotool", "type", "--window", state.window_id, "--clearmodifiers", text],
            capture_output=True,
            timeout=10,
        )

    def send_key(self, key: str, app: str | None = None):
        """Send a key press (e.g. 'Return', 'Tab', 'ctrl+s') to the resolved app's window."""
        state = self._ensure_window(app)
        subprocess.run(
            ["xdotool", "key", "--window", state.window_id, key],
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
        "Returns PID and window info. Multiple GUI apps can be managed "
        "simultaneously; each must be given a unique name."
    )
    inputs = {
        "name": {
            "type": "string",
            "description": "A unique name for this app instance (e.g. 'editor', 'browser')",
        },
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

    def forward(
        self, name: str, binary_path: str, args: str | None = None, working_dir: str | None = None
    ) -> str:
        arg_list = args.split() if args and args.strip() else None
        info = self.gui_manager.launch(name, binary_path, args=arg_list, working_dir=working_dir or None)
        return f"Launched GUI {info['name']!r}: PID={info['pid']}, window_id={info['window_id']}"


class GuiScreenshotTool(Tool):
    """Request a screenshot of a GUI window (captured after the step)."""

    name = "gui_screenshot"
    description = (
        "Request a screenshot of a running GUI window. The screenshot is captured "
        "after the current code step finishes and will be visible as an image in the "
        "next turn. Returns confirmation that the screenshot was requested."
    )
    inputs = {
        "app": {
            "type": "string",
            "description": "Name of the app to target (from gui_launch). Optional if only one app is running.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self, app: str | None = None) -> str:
        state = self.gui_manager._ensure_window(app)
        self.gui_manager._should_screenshot = True
        self.gui_manager._screenshot_app = state.name
        return (
            f"Screenshot of {state.name!r} requested — it will be captured after this step "
            "completes and visible in the next turn."
        )


class GuiClickTool(Tool):
    """Click at (x, y) coordinates in a GUI window."""

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
        "app": {
            "type": "string",
            "description": "Name of the app to target (from gui_launch). Optional if only one app is running.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self, x: int, y: int, button: int | None = None, app: str | None = None) -> str:
        self.gui_manager.click(x, y, button=button or 1, app=app)
        return f"Clicked at ({x}, {y}) with button {button or 1}"


class GuiTypeTool(Tool):
    """Type text into a GUI window."""

    name = "gui_type"
    description = (
        "Type a text string into the currently focused widget of the GUI window. "
        "Click on the target widget first if needed."
    )
    inputs = {
        "text": {"type": "string", "description": "The text to type"},
        "app": {
            "type": "string",
            "description": "Name of the app to target (from gui_launch). Optional if only one app is running.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self, text: str, app: str | None = None) -> str:
        self.gui_manager.type_text(text, app=app)
        return f"Typed {len(text)} characters"


class GuiKeyTool(Tool):
    """Send a key press to a GUI window."""

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
        "app": {
            "type": "string",
            "description": "Name of the app to target (from gui_launch). Optional if only one app is running.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self, key: str, app: str | None = None) -> str:
        self.gui_manager.send_key(key, app=app)
        return f"Sent key: {key}"


class GuiListWindowsTool(Tool):
    """List all X11 windows belonging to a managed GUI process."""

    name = "gui_list_windows"
    description = (
        "List all X11 windows belonging to a running GUI process. "
        "Returns each window's ID, name, geometry, and whether it is the "
        "currently targeted window. Useful for discovering tooltips, dialogs, "
        "popups, and child windows that appear on top of the main window."
    )
    inputs = {
        "app": {
            "type": "string",
            "description": "Name of the app to target (from gui_launch). Optional if only one app is running.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self, app: str | None = None) -> str:
        windows = self.gui_manager.list_windows(app=app)
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
        "app": {
            "type": "string",
            "description": "Name of the app to target (from gui_launch). Optional if only one app is running.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self, window_id: str, app: str | None = None) -> str:
        window_id = window_id.strip()
        self.gui_manager.focus_window(window_id, app=app)
        return f"Now targeting window {window_id}. All GUI interactions will use this window."


class GuiCloseTool(Tool):
    """Close a managed GUI process."""

    name = "gui_close"
    description = (
        "Close (terminate) a running GUI process and clean up. "
        "Specify which app to close by name, or omit to close the current/only app."
    )
    inputs = {
        "app": {
            "type": "string",
            "description": "Name of the app to close (from gui_launch). Optional if only one app is running.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, gui_manager: GuiManager):
        self.gui_manager = gui_manager
        super().__init__()

    def forward(self, app: str | None = None) -> str:
        if not self.gui_manager._apps:
            return "No GUI apps are running."
        state = self.gui_manager._resolve_app(app)
        name = state.name
        pid = state.process.pid
        self.gui_manager.close(app=name)
        return f"GUI app {name!r} (PID {pid}) terminated."


# ======================================================================
# Step callback
# ======================================================================


def gui_screenshot_callback(memory_step, agent=None):
    """Step callback: capture a GUI screenshot when the flag is set.

    Follows the same pattern as the vision_web_browser ``save_screenshot``
    callback -- clears old screenshots to save context, stores the new one
    in ``memory_step.observations_images``.
    """
    manager = getattr(agent, "_gui_manager", None) if agent else None
    if manager is None or not manager._should_screenshot:
        return

    app_name = manager._screenshot_app
    manager._should_screenshot = False
    manager._screenshot_app = None

    try:
        image = manager.capture_screenshot(app=app_name)
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
    label = f" of {app_name!r}" if app_name else ""
    size_info = f"\n[GUI screenshot{label} captured: {image.size[0]}x{image.size[1]} pixels]"
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
