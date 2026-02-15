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
    GuiLaunchTool,
    GuiScreenshotTool,
    GuiClickTool,
    GuiTypeTool,
    GuiKeyTool,
    GuiCloseTool,
    gui_screenshot_callback,
    create_gui_tools,
)


# ======================================================================
# GuiManager tests
# ======================================================================

class TestGuiManager:

    def test_initial_state(self):
        mgr = GuiManager()
        assert mgr._process is None
        assert mgr._window_id is None
        assert mgr._should_screenshot is False

    @patch("smolagents.bp_tools_gui.subprocess.Popen")
    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_launch_success(self, mock_run, mock_popen):
        """Launch finds a window via xdotool."""
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None
        mock_popen.return_value = proc

        # xdotool search returns a window id
        mock_run.return_value = MagicMock(stdout="67890\n", returncode=0)

        mgr = GuiManager()
        result = mgr.launch("/usr/bin/myapp", args=["--flag"])

        assert result["pid"] == 12345
        assert result["window_id"] == "67890"
        assert mgr._window_id == "67890"
        mock_popen.assert_called_once()

    @patch("smolagents.bp_tools_gui.subprocess.Popen")
    def test_launch_double_raises(self, mock_popen):
        """Cannot launch twice without closing first."""
        proc = MagicMock()
        proc.pid = 1
        proc.poll.return_value = None
        mock_popen.return_value = proc

        mgr = GuiManager()
        mgr._process = proc  # simulate already running

        with pytest.raises(RuntimeError, match="already running"):
            mgr.launch("/usr/bin/myapp")

    @patch("smolagents.bp_tools_gui.subprocess.Popen")
    @patch("smolagents.bp_tools_gui.subprocess.run")
    @patch("smolagents.bp_tools_gui.time.monotonic")
    def test_launch_process_exits_prematurely(self, mock_time, mock_run, mock_popen):
        """If process exits before window appears, raise RuntimeError."""
        proc = MagicMock()
        proc.pid = 99
        proc.poll.return_value = 1  # already exited
        proc.returncode = 1
        mock_popen.return_value = proc

        mock_time.side_effect = [0, 1]  # first call for deadline, second in loop

        mgr = GuiManager()
        with pytest.raises(RuntimeError, match="exited prematurely"):
            mgr.launch("/usr/bin/badapp")

    @patch("smolagents.bp_tools_gui.subprocess.Popen")
    @patch("smolagents.bp_tools_gui.subprocess.run")
    @patch("smolagents.bp_tools_gui.time.monotonic")
    @patch("smolagents.bp_tools_gui.time.sleep")
    def test_launch_timeout(self, mock_sleep, mock_time, mock_run, mock_popen):
        """If no window found within timeout, raise TimeoutError."""
        proc = MagicMock()
        proc.pid = 42
        proc.poll.return_value = None
        mock_popen.return_value = proc

        # Simulate time passing beyond timeout
        mock_time.side_effect = [0, 1, 5, 10, 16]
        mock_run.return_value = MagicMock(stdout="", returncode=1)

        mgr = GuiManager()
        with pytest.raises(TimeoutError, match="No X11 window"):
            mgr.launch("/usr/bin/slowapp")

    def test_ensure_window_no_process(self):
        mgr = GuiManager()
        with pytest.raises(RuntimeError, match="No GUI process"):
            mgr._ensure_window()

    def test_ensure_window_process_dead(self):
        mgr = GuiManager()
        proc = MagicMock()
        proc.poll.return_value = 0  # exited
        mgr._process = proc
        mgr._window_id = "123"
        with pytest.raises(RuntimeError, match="No GUI process"):
            mgr._ensure_window()

    def test_ensure_window_no_wid(self):
        mgr = GuiManager()
        proc = MagicMock()
        proc.poll.return_value = None
        mgr._process = proc
        mgr._window_id = None
        with pytest.raises(RuntimeError, match="No window ID"):
            mgr._ensure_window()

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_click(self, mock_run):
        mgr = GuiManager()
        proc = MagicMock()
        proc.poll.return_value = None
        mgr._process = proc
        mgr._window_id = "111"

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
        mgr = GuiManager()
        proc = MagicMock()
        proc.poll.return_value = None
        mgr._process = proc
        mgr._window_id = "111"

        mgr.type_text("hello world")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "type" in args
        assert "hello world" in args

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_send_key(self, mock_run):
        mgr = GuiManager()
        proc = MagicMock()
        proc.poll.return_value = None
        mgr._process = proc
        mgr._window_id = "111"

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

        mgr = GuiManager()
        proc = MagicMock()
        proc.poll.return_value = None
        mgr._process = proc
        mgr._window_id = "222"

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
        mgr = GuiManager()
        proc = MagicMock()
        proc.poll.return_value = None
        mgr._process = proc
        mgr._window_id = "222"

        mock_run.side_effect = [
            MagicMock(returncode=0),  # windowactivate
            MagicMock(returncode=1, stdout=b"", stderr=b"import error"),  # import failed
        ]

        with patch("smolagents.bp_tools_gui.time.sleep"):
            with pytest.raises(RuntimeError, match="Screenshot capture failed"):
                mgr.capture_screenshot()

    def test_close(self):
        mgr = GuiManager()
        proc = MagicMock()
        proc.poll.return_value = None
        proc.wait.return_value = 0
        mgr._process = proc
        mgr._window_id = "333"

        mgr.close()

        proc.terminate.assert_called_once()
        assert mgr._process is None
        assert mgr._window_id is None

    def test_close_no_process(self):
        """Closing when nothing is running is a no-op."""
        mgr = GuiManager()
        mgr.close()  # should not raise

    def test_shutdown_is_close_alias(self):
        mgr = GuiManager()
        mgr.close = MagicMock()
        mgr.shutdown()
        mgr.close.assert_called_once()


# ======================================================================
# Tool forward() delegation tests
# ======================================================================

class TestGuiTools:

    def _make_ready_manager(self):
        mgr = GuiManager()
        proc = MagicMock()
        proc.pid = 100
        proc.poll.return_value = None
        mgr._process = proc
        mgr._window_id = "555"
        return mgr

    @patch("smolagents.bp_tools_gui.subprocess.Popen")
    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_launch_tool(self, mock_run, mock_popen):
        proc = MagicMock()
        proc.pid = 50
        proc.poll.return_value = None
        mock_popen.return_value = proc
        mock_run.return_value = MagicMock(stdout="999\n", returncode=0)

        mgr = GuiManager()
        tool = GuiLaunchTool(mgr)
        result = tool.forward("/tmp/myapp", args="--x --y")

        assert "PID=50" in result
        assert "999" in result

    def test_screenshot_tool_sets_flag(self):
        mgr = self._make_ready_manager()
        tool = GuiScreenshotTool(mgr)
        result = tool.forward()
        assert mgr._should_screenshot is True
        assert "requested" in result.lower()

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_click_tool(self, mock_run):
        mgr = self._make_ready_manager()
        tool = GuiClickTool(mgr)
        result = tool.forward(50, 75)
        assert "(50, 75)" in result

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_type_tool(self, mock_run):
        mgr = self._make_ready_manager()
        tool = GuiTypeTool(mgr)
        result = tool.forward("abc")
        assert "3" in result  # 3 characters

    @patch("smolagents.bp_tools_gui.subprocess.run")
    def test_key_tool(self, mock_run):
        mgr = self._make_ready_manager()
        tool = GuiKeyTool(mgr)
        result = tool.forward("Return")
        assert "Return" in result

    def test_close_tool(self):
        mgr = self._make_ready_manager()
        pid = mgr._process.pid
        mgr._process.wait.return_value = 0
        tool = GuiCloseTool(mgr)
        result = tool.forward()
        assert str(pid) in result

    def test_close_tool_no_process(self):
        mgr = GuiManager()
        tool = GuiCloseTool(mgr)
        result = tool.forward()
        assert "No GUI" in result


# ======================================================================
# create_gui_tools()
# ======================================================================

class TestCreateGuiTools:

    def test_shape(self):
        manager, tools = create_gui_tools()
        assert isinstance(manager, GuiManager)
        assert len(tools) == 6
        names = {t.name for t in tools}
        assert names == {"gui_launch", "gui_screenshot", "gui_click", "gui_type", "gui_key", "gui_close"}


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
        assert step.observations_images == [fake_img]
        assert "200x100" in step.observations

    def test_clears_old_screenshots(self):
        """Callback clears observations_images from steps older than current-2."""
        import PIL.Image
        from smolagents.memory import ActionStep

        mgr = GuiManager()
        mgr._should_screenshot = True
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
        mgr.capture_screenshot = MagicMock(side_effect=RuntimeError("X11 gone"))

        step = MagicMock()
        step.step_number = 3
        step.observations = "existing"

        agent = MagicMock()
        agent._gui_manager = mgr

        gui_screenshot_callback(step, agent=agent)

        assert mgr._should_screenshot is False
        assert "screenshot failed" in step.observations
        assert "X11 gone" in step.observations
