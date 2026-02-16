#!/usr/bin/env python3
"""
Unit tests for GUI interaction tools in bp_tools_gui.py.

All subprocess calls are mocked — no real xdotool/ImageMagick/X11 required.
"""

import os
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smolagents.bp_tools_gui import (
    GuiManager,
    _AppState,
    GuiLaunchTool,
    GuiScreenshotTool,
    GuiClickTool,
    GuiTypeTool,
    GuiKeyTool,
    GuiListWindowsTool,
    GuiFocusWindowTool,
    GuiCloseTool,
    gui_screenshot_callback,
    create_gui_tools,
    _check_gui_dependencies,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_proc(pid=100, alive=True):
    """Create a mock subprocess.Popen."""
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None if alive else 0
    proc.wait.return_value = 0
    return proc


def _make_ready_manager(name="app1", pid=100, window_id="555"):
    """Create a GuiManager with one app already 'launched'."""
    mgr = GuiManager()
    proc = _make_proc(pid=pid)
    state = _AppState(process=proc, window_id=window_id, name=name)
    mgr._apps[name] = state
    mgr._current = name
    return mgr


# ======================================================================
# GuiManager tests
# ======================================================================

class TestGuiManager:

    def test_initial_state(self):
        mgr = GuiManager()
        assert mgr._apps == {}
        assert mgr._current is None
        assert mgr._should_screenshot is False
        assert mgr._screenshot_app is None

    @patch("smolagents.bp_tools_gui.subprocess.Popen")
    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_launch_success(self, mock_run, mock_popen):
        """Launch finds a window via xdotool."""
        proc = _make_proc(pid=12345)
        mock_popen.return_value = proc

        # xdotool search returns a window id
        mock_run.return_value = MagicMock(stdout="67890\n", returncode=0)

        mgr = GuiManager()
        result = mgr.launch("editor", "/usr/bin/myapp", args=["--flag"])

        assert result["pid"] == 12345
        assert result["window_id"] == "67890"
        assert result["name"] == "editor"
        assert "editor" in mgr._apps
        assert mgr._apps["editor"].window_id == "67890"
        assert mgr._current == "editor"
        mock_popen.assert_called_once()

    @patch("smolagents.bp_tools_gui.subprocess.Popen")
    def test_launch_duplicate_name_raises(self, mock_popen):
        """Cannot launch two apps with the same name."""
        mgr = _make_ready_manager(name="editor")

        with pytest.raises(RuntimeError, match="already running"):
            mgr.launch("editor", "/usr/bin/myapp")

    @patch("smolagents.bp_tools_gui.subprocess.Popen")
    @patch("smolagents.bp_tools_gui.subprocess.run")
    @patch("smolagents.bp_tools_gui.time.monotonic")
    def test_launch_process_exits_prematurely(self, mock_time, mock_run, mock_popen):
        """If process exits before window appears, raise RuntimeError."""
        proc = _make_proc(pid=99, alive=False)
        proc.returncode = 1
        mock_popen.return_value = proc

        mock_time.side_effect = [0, 1]  # first call for deadline, second in loop

        mgr = GuiManager()
        with pytest.raises(RuntimeError, match="exited prematurely"):
            mgr.launch("badapp", "/usr/bin/badapp")

    @patch("smolagents.bp_tools_gui.subprocess.Popen")
    @patch("smolagents.bp_tools_gui.subprocess.run")
    @patch("smolagents.bp_tools_gui.time.monotonic")
    @patch("smolagents.bp_tools_gui.time.sleep")
    def test_launch_timeout(self, mock_sleep, mock_time, mock_run, mock_popen):
        """If no window found within timeout, raise TimeoutError."""
        proc = _make_proc(pid=42)
        mock_popen.return_value = proc

        # Simulate time passing beyond timeout
        mock_time.side_effect = [0, 1, 5, 10, 16]
        mock_run.return_value = MagicMock(stdout="", returncode=1)

        mgr = GuiManager()
        with pytest.raises(TimeoutError, match="No X11 window"):
            mgr.launch("slowapp", "/usr/bin/slowapp")

    def test_ensure_window_no_apps(self):
        mgr = GuiManager()
        with pytest.raises(RuntimeError, match="No GUI process"):
            mgr._ensure_window()

    def test_ensure_window_process_dead(self):
        mgr = GuiManager()
        proc = _make_proc(alive=False)
        state = _AppState(process=proc, window_id="123", name="app1")
        mgr._apps["app1"] = state
        mgr._current = "app1"
        with pytest.raises(RuntimeError, match="has exited"):
            mgr._ensure_window()
        # Dead app should be removed
        assert "app1" not in mgr._apps

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_click(self, mock_run):
        mgr = _make_ready_manager()

        mgr.click(100, 200, button=3)

        assert mock_run.call_count == 2
        # First call: mousemove
        args0 = mock_run.call_args_list[0][0][0]
        assert "mousemove" in args0
        assert "100" in args0
        assert "200" in args0
        # Second call: click
        args1 = mock_run.call_args_list[1][0][0]
        assert "click" in args1
        assert "3" in args1

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_type_text(self, mock_run):
        mgr = _make_ready_manager()

        mgr.type_text("hello world")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "type" in args
        assert "hello world" in args

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_send_key(self, mock_run):
        mgr = _make_ready_manager()

        mgr.send_key("ctrl+s")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "key" in args
        assert "ctrl+s" in args

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_capture_screenshot(self, mock_run):
        """Screenshot returns a PIL Image."""
        import PIL.Image
        from io import BytesIO

        # Create a small valid PNG for the mock
        img = PIL.Image.new("RGB", (100, 50), color="red")
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        mgr = _make_ready_manager()

        # First call: windowactivate, second call: import
        mock_run.side_effect = [
            MagicMock(returncode=0),  # windowactivate
            MagicMock(returncode=0, stdout=png_bytes, stderr=b""),  # import
        ]

        with patch("smolagents.bp_tools_gui.time.sleep"):
            result = mgr.capture_screenshot()

        assert isinstance(result, PIL.Image.Image)
        assert result.size == (100, 50)

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_capture_screenshot_failure(self, mock_run):
        mgr = _make_ready_manager()

        mock_run.side_effect = [
            MagicMock(returncode=0),  # windowactivate
            MagicMock(returncode=1, stdout=b"", stderr=b"import error"),  # import failed
        ]

        with patch("smolagents.bp_tools_gui.time.sleep"):
            with pytest.raises(RuntimeError, match="Screenshot capture failed"):
                mgr.capture_screenshot()

    def test_close(self):
        mgr = _make_ready_manager()

        mgr.close()

        assert mgr._apps == {}
        assert mgr._current is None

    def test_close_no_process(self):
        """Closing when nothing is running is a no-op."""
        mgr = GuiManager()
        mgr.close()  # should not raise

    def test_shutdown_closes_all(self):
        mgr = _make_ready_manager(name="app1", pid=1, window_id="100")
        proc2 = _make_proc(pid=2)
        mgr._apps["app2"] = _AppState(process=proc2, window_id="200", name="app2")

        mgr.shutdown()

        assert mgr._apps == {}
        assert mgr._current is None


# ======================================================================
# _resolve_app tests
# ======================================================================

class TestResolveApp:

    def test_explicit_name(self):
        mgr = _make_ready_manager(name="editor")
        state = mgr._resolve_app("editor")
        assert state.name == "editor"

    def test_explicit_name_not_found(self):
        mgr = _make_ready_manager(name="editor")
        with pytest.raises(ValueError, match="No app named"):
            mgr._resolve_app("browser")

    def test_single_app_implicit(self):
        mgr = _make_ready_manager(name="only_one")
        mgr._current = None  # unset current
        state = mgr._resolve_app()
        assert state.name == "only_one"

    def test_current_app_implicit(self):
        mgr = _make_ready_manager(name="app1")
        proc2 = _make_proc(pid=2)
        mgr._apps["app2"] = _AppState(process=proc2, window_id="200", name="app2")
        mgr._current = "app2"
        state = mgr._resolve_app()
        assert state.name == "app2"

    def test_multiple_apps_no_current_raises(self):
        mgr = _make_ready_manager(name="app1")
        proc2 = _make_proc(pid=2)
        mgr._apps["app2"] = _AppState(process=proc2, window_id="200", name="app2")
        mgr._current = None
        with pytest.raises(ValueError, match="Multiple apps"):
            mgr._resolve_app()

    def test_no_apps_raises(self):
        mgr = GuiManager()
        with pytest.raises(RuntimeError, match="No GUI process"):
            mgr._resolve_app()

    def test_dead_process_cleaned_up(self):
        mgr = _make_ready_manager(name="dead_app")
        mgr._apps["dead_app"].process.poll.return_value = 1  # dead
        with pytest.raises(RuntimeError, match="has exited"):
            mgr._resolve_app("dead_app")
        assert "dead_app" not in mgr._apps

    def test_empty_string_treated_as_none(self):
        mgr = _make_ready_manager(name="solo")
        state = mgr._resolve_app("")
        assert state.name == "solo"


# ======================================================================
# Multi-app tests
# ======================================================================

class TestMultiApp:

    @patch("smolagents.bp_tools_gui.subprocess.Popen")
    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_launch_two_apps(self, mock_run, mock_popen):
        """Can launch two apps; _current is last launched."""
        mgr = GuiManager()

        proc1 = _make_proc(pid=10)
        proc2 = _make_proc(pid=20)
        mock_popen.side_effect = [proc1, proc2]
        mock_run.side_effect = [
            MagicMock(stdout="100\n", returncode=0),
            MagicMock(stdout="200\n", returncode=0),
        ]

        mgr.launch("editor", "/usr/bin/editor")
        mgr.launch("browser", "/usr/bin/browser")

        assert len(mgr._apps) == 2
        assert "editor" in mgr._apps
        assert "browser" in mgr._apps
        assert mgr._current == "browser"

    def test_close_specific_app(self):
        """Close one app among multiple; other remains."""
        mgr = _make_ready_manager(name="app1", pid=1, window_id="100")
        proc2 = _make_proc(pid=2)
        mgr._apps["app2"] = _AppState(process=proc2, window_id="200", name="app2")
        mgr._current = "app1"

        mgr.close(app="app1")

        assert "app1" not in mgr._apps
        assert "app2" in mgr._apps
        # _current should update to remaining app
        assert mgr._current == "app2"

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_interaction_with_explicit_app(self, mock_run):
        """Interaction tools work with explicit app name."""
        mgr = _make_ready_manager(name="app1", pid=1, window_id="100")
        proc2 = _make_proc(pid=2)
        mgr._apps["app2"] = _AppState(process=proc2, window_id="200", name="app2")

        mgr.click(10, 20, app="app2")

        # Should use window_id "200" from app2
        args0 = mock_run.call_args_list[0][0][0]
        assert "200" in args0

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_focus_window_sets_current(self, mock_run):
        """focus_window updates _current to the app."""
        mgr = _make_ready_manager(name="app1", pid=1, window_id="100")
        proc2 = _make_proc(pid=2)
        mgr._apps["app2"] = _AppState(process=proc2, window_id="200", name="app2")
        mgr._current = "app1"

        mock_run.return_value = MagicMock(stdout="200\n300\n", returncode=0)
        mgr.focus_window("300", app="app2")

        assert mgr._apps["app2"].window_id == "300"
        assert mgr._current == "app2"


# ======================================================================
# Tool forward() delegation tests
# ======================================================================

class TestGuiTools:

    @patch("smolagents.bp_tools_gui.subprocess.Popen")
    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_launch_tool(self, mock_run, mock_popen):
        proc = _make_proc(pid=50)
        mock_popen.return_value = proc
        mock_run.return_value = MagicMock(stdout="999\n", returncode=0)

        mgr = GuiManager()
        tool = GuiLaunchTool(mgr)
        result = tool.forward("myapp", "/tmp/myapp", args="--x --y")

        assert "PID=50" in result
        assert "999" in result
        assert "'myapp'" in result

    @patch("smolagents.bp_tools_gui.subprocess.Popen")
    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_launch_tool_empty_strings(self, mock_run, mock_popen):
        """LLMs often pass empty strings instead of None for optional params."""
        proc = _make_proc(pid=60)
        mock_popen.return_value = proc
        mock_run.return_value = MagicMock(stdout="888\n", returncode=0)

        mgr = GuiManager()
        tool = GuiLaunchTool(mgr)
        result = tool.forward("myapp", "/tmp/myapp", args="", working_dir="")

        assert "PID=60" in result
        # working_dir="" should become None, not be passed to Popen
        popen_kwargs = mock_popen.call_args[1]
        assert popen_kwargs.get("cwd") is None

    @patch("smolagents.bp_tools_gui.subprocess.Popen")
    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_launch_relative_path_resolved(self, mock_run, mock_popen, tmp_path):
        """Relative binary path is resolved against working_dir before Popen."""
        # Create a fake binary file so os.path.isfile returns True
        fake_bin = tmp_path / "myapp"
        fake_bin.touch()

        proc = _make_proc(pid=70)
        mock_popen.return_value = proc
        mock_run.return_value = MagicMock(stdout="777\n", returncode=0)

        mgr = GuiManager()
        tool = GuiLaunchTool(mgr)
        result = tool.forward("myapp", "myapp", working_dir=str(tmp_path))

        assert "PID=70" in result
        # The resolved path should be the joined absolute path
        popen_cmd = mock_popen.call_args[0][0]
        assert popen_cmd[0] == str(tmp_path / "myapp")

    @patch("smolagents.bp_tools_gui.subprocess.Popen")
    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_launch_relative_path_not_found_keeps_original(self, mock_run, mock_popen, tmp_path):
        """If relative path doesn't exist under working_dir, keep original."""
        proc = _make_proc(pid=71)
        mock_popen.return_value = proc
        mock_run.return_value = MagicMock(stdout="778\n", returncode=0)

        mgr = GuiManager()
        tool = GuiLaunchTool(mgr)
        # "nonexistent_app" doesn't exist in tmp_path
        result = tool.forward("myapp", "nonexistent_app", working_dir=str(tmp_path))

        assert "PID=71" in result
        # Original path kept since file not found at working_dir/nonexistent_app
        popen_cmd = mock_popen.call_args[0][0]
        assert popen_cmd[0] == "nonexistent_app"

    def test_screenshot_tool_sets_flag(self):
        mgr = _make_ready_manager()
        tool = GuiScreenshotTool(mgr)
        result = tool.forward()
        assert mgr._should_screenshot is True
        assert mgr._screenshot_app == "app1"
        assert "requested" in result.lower()

    def test_screenshot_tool_with_explicit_app(self):
        mgr = _make_ready_manager(name="app1")
        proc2 = _make_proc(pid=2)
        mgr._apps["app2"] = _AppState(process=proc2, window_id="200", name="app2")

        tool = GuiScreenshotTool(mgr)
        result = tool.forward(app="app2")
        assert mgr._screenshot_app == "app2"
        assert "'app2'" in result

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_click_tool(self, mock_run):
        mgr = _make_ready_manager()
        tool = GuiClickTool(mgr)
        result = tool.forward(50, 75)
        assert "(50, 75)" in result

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_click_tool_with_app(self, mock_run):
        mgr = _make_ready_manager(name="editor", window_id="999")
        tool = GuiClickTool(mgr)
        result = tool.forward(10, 20, app="editor")
        assert "(10, 20)" in result
        # Should use window_id "999"
        args0 = mock_run.call_args_list[0][0][0]
        assert "999" in args0

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_type_tool(self, mock_run):
        mgr = _make_ready_manager()
        tool = GuiTypeTool(mgr)
        result = tool.forward("abc")
        assert "3" in result  # 3 characters

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_key_tool(self, mock_run):
        mgr = _make_ready_manager()
        tool = GuiKeyTool(mgr)
        result = tool.forward("Return")
        assert "Return" in result

    def test_close_tool(self):
        mgr = _make_ready_manager()
        tool = GuiCloseTool(mgr)
        result = tool.forward()
        assert "PID" in result
        assert "terminated" in result.lower()

    def test_close_tool_no_process(self):
        mgr = GuiManager()
        tool = GuiCloseTool(mgr)
        result = tool.forward()
        assert "No GUI" in result

    def test_close_tool_specific_app(self):
        mgr = _make_ready_manager(name="app1", pid=1, window_id="100")
        proc2 = _make_proc(pid=2)
        mgr._apps["app2"] = _AppState(process=proc2, window_id="200", name="app2")

        tool = GuiCloseTool(mgr)
        result = tool.forward(app="app1")
        assert "'app1'" in result
        assert "app1" not in mgr._apps
        assert "app2" in mgr._apps

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_list_windows_tool(self, mock_run):
        mgr = _make_ready_manager()
        # xdotool search returns 2 windows
        mock_run.side_effect = [
            MagicMock(stdout="555\n777\n", returncode=0),  # search --pid
            MagicMock(stdout="Main Window", returncode=0),  # getwindowname 555
            MagicMock(stdout="Window 555:\n  Position: 100,200\n  Geometry: 400x300", returncode=0),
            MagicMock(stdout="Tooltip", returncode=0),  # getwindowname 777
            MagicMock(stdout="Window 777:\n  Position: 150,250\n  Geometry: 100x30", returncode=0),
        ]
        tool = GuiListWindowsTool(mgr)
        result = tool.forward()

        assert "2 window(s)" in result
        assert "555" in result
        assert "777" in result
        assert "[CURRENT]" in result  # 555 is current
        assert "Main Window" in result
        assert "Tooltip" in result

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_focus_window_tool(self, mock_run):
        mgr = _make_ready_manager()
        # search returns both windows so 777 is valid
        mock_run.return_value = MagicMock(stdout="555\n777\n", returncode=0)

        tool = GuiFocusWindowTool(mgr)
        result = tool.forward("777")

        assert mgr._apps["app1"].window_id == "777"
        assert "777" in result

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_focus_window_tool_invalid_id(self, mock_run):
        mgr = _make_ready_manager()
        mock_run.return_value = MagicMock(stdout="555\n", returncode=0)

        tool = GuiFocusWindowTool(mgr)
        with pytest.raises(ValueError, match="does not belong"):
            tool.forward("999")


# ======================================================================
# GuiManager.list_windows / focus_window
# ======================================================================

class TestGuiManagerWindows:

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_list_windows(self, mock_run):
        mgr = _make_ready_manager(name="app1", pid=42, window_id="100")

        mock_run.side_effect = [
            MagicMock(stdout="100\n200\n", returncode=0),
            MagicMock(stdout="MainWin", returncode=0),
            MagicMock(stdout="geom1", returncode=0),
            MagicMock(stdout="Dialog", returncode=0),
            MagicMock(stdout="geom2", returncode=0),
        ]

        windows = mgr.list_windows()
        assert len(windows) == 2
        assert windows[0]["window_id"] == "100"
        assert windows[0]["is_current"] is True
        assert windows[0]["name"] == "MainWin"
        assert windows[1]["window_id"] == "200"
        assert windows[1]["is_current"] is False

    def test_list_windows_no_process(self):
        mgr = GuiManager()
        with pytest.raises(RuntimeError, match="No GUI process"):
            mgr.list_windows()

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_focus_window_valid(self, mock_run):
        mgr = _make_ready_manager(name="app1", pid=42, window_id="100")

        mock_run.return_value = MagicMock(stdout="100\n200\n", returncode=0)

        mgr.focus_window("200")
        assert mgr._apps["app1"].window_id == "200"

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_focus_window_invalid(self, mock_run):
        mgr = _make_ready_manager(name="app1", pid=42, window_id="100")

        mock_run.return_value = MagicMock(stdout="100\n", returncode=0)

        with pytest.raises(ValueError, match="does not belong"):
            mgr.focus_window("999")

    def test_focus_window_no_process(self):
        mgr = GuiManager()
        with pytest.raises(RuntimeError, match="No GUI process"):
            mgr.focus_window("123")


# ======================================================================
# create_gui_tools()
# ======================================================================

class TestCreateGuiTools:

    @patch("smolagents.bp_tools_gui._check_gui_dependencies")
    def test_shape(self, mock_check):
        manager, tools = create_gui_tools()
        assert isinstance(manager, GuiManager)
        assert len(tools) == 8
        names = {t.name for t in tools}
        assert names == {
            "gui_launch", "gui_screenshot", "gui_click", "gui_type", "gui_key",
            "gui_list_windows", "gui_focus_window", "gui_close",
        }
        mock_check.assert_called_once()


# ======================================================================
# Dependency checks
# ======================================================================

class TestDependencyCheck:

    @patch("smolagents.bp_tools_gui.shutil.which", return_value="/usr/bin/stub")
    @patch.dict(os.environ, {"DISPLAY": ":0"})
    def test_all_present(self, mock_which):
        _check_gui_dependencies()  # should not raise

    @patch("smolagents.bp_tools_gui.shutil.which", return_value=None)
    @patch.dict(os.environ, {"DISPLAY": ":0"})
    def test_missing_xdotool_and_imagemagick(self, mock_which):
        with pytest.raises(EnvironmentError, match="xdotool"):
            _check_gui_dependencies()

    @patch("smolagents.bp_tools_gui.shutil.which", side_effect=lambda cmd: "/usr/bin/xdotool" if cmd == "xdotool" else None)
    @patch.dict(os.environ, {"DISPLAY": ":0"})
    def test_missing_imagemagick_only(self, mock_which):
        with pytest.raises(EnvironmentError, match="imagemagick"):
            _check_gui_dependencies()

    @patch("smolagents.bp_tools_gui.shutil.which", return_value="/usr/bin/stub")
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_display(self, mock_which):
        with pytest.raises(EnvironmentError, match="DISPLAY"):
            _check_gui_dependencies()


# ======================================================================
# gui_screenshot_callback
# ======================================================================

class TestGuiScreenshotCallback:

    def test_noop_when_no_manager(self):
        """Callback does nothing when agent has no _gui_manager."""
        step = MagicMock()
        step.observations_images = None
        agent = MagicMock(spec=[])  # no _gui_manager attribute
        gui_screenshot_callback(step, agent=agent)
        # Should not touch step
        assert step.observations_images is None

    def test_noop_when_flag_not_set(self):
        """Callback does nothing when _should_screenshot is False."""
        mgr = GuiManager()
        mgr._should_screenshot = False

        step = MagicMock()
        agent = MagicMock()
        agent._gui_manager = mgr
        gui_screenshot_callback(step, agent=agent)

    def test_captures_when_flag_set(self):
        """Callback captures screenshot and clears flag."""
        import PIL.Image

        mgr = GuiManager()
        mgr._should_screenshot = True
        mgr._screenshot_app = "editor"
        fake_img = PIL.Image.new("RGB", (200, 100), "blue")
        mgr.capture_screenshot = MagicMock(return_value=fake_img)

        step = MagicMock()
        step.step_number = 5
        step.observations = None

        agent = MagicMock()
        agent._gui_manager = mgr
        agent.memory.steps = []  # no old steps

        gui_screenshot_callback(step, agent=agent)

        assert mgr._should_screenshot is False
        assert mgr._screenshot_app is None
        assert step.observations_images == [fake_img]
        assert "200x100" in step.observations
        assert "'editor'" in step.observations
        # Verify capture_screenshot was called with the app name
        mgr.capture_screenshot.assert_called_once_with(app="editor")

    def test_clears_old_screenshots(self):
        """Callback clears observations_images from steps older than current-2."""
        import PIL.Image
        from smolagents.memory import ActionStep

        mgr = GuiManager()
        mgr._should_screenshot = True
        mgr._screenshot_app = "app1"
        fake_img = PIL.Image.new("RGB", (10, 10), "green")
        mgr.capture_screenshot = MagicMock(return_value=fake_img)

        # Create old steps with images
        old_step = MagicMock(spec=ActionStep)
        old_step.step_number = 1
        old_step.observations_images = [fake_img]

        recent_step = MagicMock(spec=ActionStep)
        recent_step.step_number = 4
        recent_step.observations_images = [fake_img]

        current_step = MagicMock()
        current_step.step_number = 5
        current_step.observations = None

        agent = MagicMock()
        agent._gui_manager = mgr
        agent.memory.steps = [old_step, recent_step]

        gui_screenshot_callback(current_step, agent=agent)

        # Old step (5 - 1 >= 2) should be cleared
        assert old_step.observations_images is None
        # Recent step (5 - 4 < 2) should keep its images
        assert recent_step.observations_images == [fake_img]

    def test_handles_screenshot_failure(self):
        """Callback records error in observations on capture failure."""
        mgr = GuiManager()
        mgr._should_screenshot = True
        mgr._screenshot_app = "broken"
        mgr.capture_screenshot = MagicMock(side_effect=RuntimeError("X11 gone"))

        step = MagicMock()
        step.step_number = 3
        step.observations = "existing"

        agent = MagicMock()
        agent._gui_manager = mgr

        gui_screenshot_callback(step, agent=agent)

        assert mgr._should_screenshot is False
        assert mgr._screenshot_app is None
        assert "screenshot failed" in step.observations
        assert "X11 gone" in step.observations

    def test_callback_without_app_name(self):
        """Callback works when _screenshot_app is None (label omitted)."""
        import PIL.Image

        mgr = GuiManager()
        mgr._should_screenshot = True
        mgr._screenshot_app = None
        fake_img = PIL.Image.new("RGB", (50, 50), "green")
        mgr.capture_screenshot = MagicMock(return_value=fake_img)

        step = MagicMock()
        step.step_number = 1
        step.observations = None

        agent = MagicMock()
        agent._gui_manager = mgr
        agent.memory.steps = []

        gui_screenshot_callback(step, agent=agent)

        assert "50x50" in step.observations
        # No app name label when _screenshot_app is None
        assert "of " not in step.observations
