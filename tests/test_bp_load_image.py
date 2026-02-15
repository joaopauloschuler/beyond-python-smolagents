#!/usr/bin/env python3
"""
Unit tests for LoadImageTool and load_image_callback in bp_tools.py.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import PIL.Image
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smolagents.bp_tools import LoadImageTool, load_image_callback
from smolagents.memory import ActionStep


# ======================================================================
# LoadImageTool tests
# ======================================================================

class TestLoadImageTool:

    def test_initial_state(self):
        tool = LoadImageTool()
        assert tool._pending_images == []
        assert tool.name == "load_image"

    def test_forward_valid_file(self, tmp_path):
        img = PIL.Image.new("RGB", (10, 10), "red")
        img_path = str(tmp_path / "test.png")
        img.save(img_path)

        tool = LoadImageTool()
        result = tool.forward(img_path)

        assert img_path in tool._pending_images
        assert "queued" in result.lower()
        assert "1 image(s) pending" in result

    def test_forward_multiple_files(self, tmp_path):
        tool = LoadImageTool()
        for i in range(3):
            img = PIL.Image.new("RGB", (10, 10), "blue")
            path = str(tmp_path / f"img{i}.png")
            img.save(path)
            tool.forward(path)

        assert len(tool._pending_images) == 3

    def test_forward_file_not_found(self):
        tool = LoadImageTool()
        result = tool.forward("/nonexistent/path/image.png")

        assert "Error" in result
        assert "not found" in result.lower()
        assert len(tool._pending_images) == 0

    def test_forward_expands_tilde(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        img = PIL.Image.new("RGB", (5, 5), "green")
        img_path = tmp_path / "photo.png"
        img.save(str(img_path))

        tool = LoadImageTool()
        result = tool.forward("~/photo.png")

        assert len(tool._pending_images) == 1
        assert str(img_path) in tool._pending_images[0]


# ======================================================================
# load_image_callback tests
# ======================================================================

class TestLoadImageCallback:

    def test_noop_when_no_agent(self):
        step = MagicMock()
        step.observations_images = None
        load_image_callback(step, agent=None)
        assert step.observations_images is None

    def test_noop_when_no_tool_on_agent(self):
        step = MagicMock()
        step.observations_images = None
        agent = MagicMock(spec=[])  # no _load_image_tool attr
        load_image_callback(step, agent=agent)
        assert step.observations_images is None

    def test_noop_when_no_pending(self):
        tool = LoadImageTool()
        step = MagicMock()
        step.observations_images = None
        agent = MagicMock()
        agent._load_image_tool = tool
        load_image_callback(step, agent=agent)
        assert step.observations_images is None

    def test_loads_single_image(self, tmp_path):
        img = PIL.Image.new("RGB", (100, 50), "red")
        img_path = str(tmp_path / "test.png")
        img.save(img_path)

        tool = LoadImageTool()
        tool._pending_images.append(img_path)

        step = MagicMock()
        step.step_number = 5
        step.observations = None
        step.observations_images = None

        agent = MagicMock()
        agent._load_image_tool = tool
        agent.memory.steps = []

        load_image_callback(step, agent=agent)

        assert len(step.observations_images) == 1
        assert step.observations_images[0].size == (100, 50)
        assert "100x50" in step.observations
        assert tool._pending_images == []  # cleared

    def test_loads_multiple_images(self, tmp_path):
        tool = LoadImageTool()
        for i, color in enumerate(["red", "green", "blue"]):
            img = PIL.Image.new("RGB", (10 + i, 20 + i), color)
            path = str(tmp_path / f"img{i}.png")
            img.save(path)
            tool._pending_images.append(path)

        step = MagicMock()
        step.step_number = 3
        step.observations = "existing output"
        step.observations_images = None

        agent = MagicMock()
        agent._load_image_tool = tool
        agent.memory.steps = []

        load_image_callback(step, agent=agent)

        assert len(step.observations_images) == 3
        assert "existing output" in step.observations
        assert "Loaded images:" in step.observations

    def test_appends_to_existing_images(self, tmp_path):
        """If observations_images already has images (e.g. from gui_screenshot), append."""
        existing_img = PIL.Image.new("RGB", (5, 5), "white")

        img = PIL.Image.new("RGB", (30, 30), "black")
        path = str(tmp_path / "new.png")
        img.save(path)

        tool = LoadImageTool()
        tool._pending_images.append(path)

        step = MagicMock()
        step.step_number = 2
        step.observations = None
        step.observations_images = [existing_img]

        agent = MagicMock()
        agent._load_image_tool = tool
        agent.memory.steps = []

        load_image_callback(step, agent=agent)

        assert len(step.observations_images) == 2  # existing + new

    def test_clears_old_step_images(self, tmp_path):
        img = PIL.Image.new("RGB", (10, 10), "red")
        path = str(tmp_path / "test.png")
        img.save(path)

        tool = LoadImageTool()
        tool._pending_images.append(path)

        old_step = MagicMock(spec=ActionStep)
        old_step.step_number = 1
        old_step.observations_images = [img]

        recent_step = MagicMock(spec=ActionStep)
        recent_step.step_number = 4
        recent_step.observations_images = [img]

        current_step = MagicMock()
        current_step.step_number = 5
        current_step.observations = None
        current_step.observations_images = None

        agent = MagicMock()
        agent._load_image_tool = tool
        agent.memory.steps = [old_step, recent_step]

        load_image_callback(current_step, agent=agent)

        # Old step (5 - 1 >= 2) should be cleared
        assert old_step.observations_images is None
        # Recent step (5 - 4 < 2) should keep images
        assert recent_step.observations_images == [img]

    def test_handles_corrupt_image(self, tmp_path):
        """If a file can't be opened as an image, report error but don't crash."""
        bad_path = str(tmp_path / "bad.png")
        with open(bad_path, "w") as f:
            f.write("not an image")

        good_img = PIL.Image.new("RGB", (20, 20), "blue")
        good_path = str(tmp_path / "good.png")
        good_img.save(good_path)

        tool = LoadImageTool()
        tool._pending_images.extend([bad_path, good_path])

        step = MagicMock()
        step.step_number = 1
        step.observations = None
        step.observations_images = None

        agent = MagicMock()
        agent._load_image_tool = tool
        agent.memory.steps = []

        load_image_callback(step, agent=agent)

        # Only the good image loaded
        assert len(step.observations_images) == 1
        assert "failed" in step.observations
        assert "good.png" in step.observations
