#!/usr/bin/env python3
"""
Unit tests for image analysis and drawing tools in bp_tools_image.py.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import PIL.Image
import PIL.ImageDraw
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smolagents.bp_tools_image import (
    DiffImagesTool,
    ScreenOcrTool,
    CanvasCreateTool,
    CanvasDrawTool,
    create_image_tools,
)


# ======================================================================
# Helper
# ======================================================================

def _make_image(tmp_path, name, size=(100, 100), color="red"):
    path = str(tmp_path / name)
    PIL.Image.new("RGB", size, color).save(path)
    return path


# ======================================================================
# DiffImagesTool tests
# ======================================================================

class TestDiffImagesTool:

    def test_highlight_identical_images(self, tmp_path):
        path1 = _make_image(tmp_path, "a.png", color="blue")
        path2 = _make_image(tmp_path, "b.png", color="blue")
        out = str(tmp_path / "diff.png")

        tool = DiffImagesTool()
        result = tool.forward(path1, path2, output_path=out, mode="highlight")

        assert os.path.isfile(out)
        assert "0.0%" in result
        assert "0/" not in result or "Changed pixels: 0" in result

    def test_highlight_different_images(self, tmp_path):
        path1 = _make_image(tmp_path, "a.png", color="red")
        path2 = _make_image(tmp_path, "b.png", color="blue")
        out = str(tmp_path / "diff.png")

        tool = DiffImagesTool()
        result = tool.forward(path1, path2, output_path=out, mode="highlight")

        assert os.path.isfile(out)
        assert "100.0%" in result

        # Verify the output image has red tint (changed pixels)
        diff_img = PIL.Image.open(out)
        assert diff_img.size == (100, 100)

    def test_side_by_side_mode(self, tmp_path):
        path1 = _make_image(tmp_path, "a.png", size=(50, 50), color="white")
        path2 = _make_image(tmp_path, "b.png", size=(50, 50), color="black")
        out = str(tmp_path / "diff.png")

        tool = DiffImagesTool()
        result = tool.forward(path1, path2, output_path=out, mode="side_by_side")

        assert os.path.isfile(out)
        diff_img = PIL.Image.open(out)
        # side_by_side: 3 * width + 2 * gap
        assert diff_img.size[0] == 50 * 3 + 4 * 2
        assert diff_img.size[1] == 50

    def test_different_sizes_resized(self, tmp_path):
        path1 = _make_image(tmp_path, "a.png", size=(100, 100), color="green")
        path2 = _make_image(tmp_path, "b.png", size=(200, 200), color="green")
        out = str(tmp_path / "diff.png")

        tool = DiffImagesTool()
        result = tool.forward(path1, path2, output_path=out)

        assert os.path.isfile(out)
        # Output should match img1 size
        diff_img = PIL.Image.open(out)
        assert diff_img.size == (100, 100)

    def test_default_temp_output(self, tmp_path):
        path1 = _make_image(tmp_path, "a.png")
        path2 = _make_image(tmp_path, "b.png")

        tool = DiffImagesTool()
        result = tool.forward(path1, path2)

        assert "diff.png" in result
        assert "load_image" in result

    def test_invalid_mode(self, tmp_path):
        path1 = _make_image(tmp_path, "a.png")
        path2 = _make_image(tmp_path, "b.png")

        tool = DiffImagesTool()
        result = tool.forward(path1, path2, mode="invalid")

        assert "Error" in result
        assert "Unknown mode" in result

    def test_missing_file(self):
        tool = DiffImagesTool()
        result = tool.forward("/nonexistent/a.png", "/nonexistent/b.png")
        assert "Error" in result


# ======================================================================
# ScreenOcrTool tests
# ======================================================================

class TestScreenOcrTool:

    @patch("smolagents.bp_tools_image.shutil.which", return_value=None)
    def test_no_tesseract(self, mock_which, tmp_path):
        path = _make_image(tmp_path, "text.png")
        tool = ScreenOcrTool()
        result = tool.forward(path)
        assert "not installed" in result
        assert "sudo apt install" in result

    @patch("smolagents.bp_tools_image.shutil.which", return_value="/usr/bin/tesseract")
    @patch("smolagents.bp_tools_image.subprocess.run")
    def test_ocr_success(self, mock_run, mock_which, tmp_path):
        path = _make_image(tmp_path, "text.png")
        mock_run.return_value = MagicMock(returncode=0, stdout="Hello World\n", stderr="")

        tool = ScreenOcrTool()
        result = tool.forward(path)

        assert result == "Hello World"
        mock_run.assert_called_once()
        # Verify tesseract was called with correct language
        args = mock_run.call_args[0][0]
        assert "tesseract" in args
        assert "-l" in args
        assert "eng" in args

    @patch("smolagents.bp_tools_image.shutil.which", return_value="/usr/bin/tesseract")
    @patch("smolagents.bp_tools_image.subprocess.run")
    def test_ocr_with_region(self, mock_run, mock_which, tmp_path):
        path = _make_image(tmp_path, "text.png", size=(200, 200))
        mock_run.return_value = MagicMock(returncode=0, stdout="Cropped", stderr="")

        tool = ScreenOcrTool()
        result = tool.forward(path, region="10,20,50,30")

        assert result == "Cropped"

    @patch("smolagents.bp_tools_image.shutil.which", return_value="/usr/bin/tesseract")
    @patch("smolagents.bp_tools_image.subprocess.run")
    def test_ocr_custom_language(self, mock_run, mock_which, tmp_path):
        path = _make_image(tmp_path, "text.png")
        mock_run.return_value = MagicMock(returncode=0, stdout="Bonjour", stderr="")

        tool = ScreenOcrTool()
        result = tool.forward(path, language="fra")

        args = mock_run.call_args[0][0]
        assert "fra" in args

    @patch("smolagents.bp_tools_image.shutil.which", return_value="/usr/bin/tesseract")
    @patch("smolagents.bp_tools_image.subprocess.run")
    def test_ocr_no_text_detected(self, mock_run, mock_which, tmp_path):
        path = _make_image(tmp_path, "blank.png", color="white")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        tool = ScreenOcrTool()
        result = tool.forward(path)

        assert "No text detected" in result

    @patch("smolagents.bp_tools_image.shutil.which", return_value="/usr/bin/tesseract")
    @patch("smolagents.bp_tools_image.subprocess.run")
    def test_ocr_error(self, mock_run, mock_which, tmp_path):
        path = _make_image(tmp_path, "text.png")
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Failed")

        tool = ScreenOcrTool()
        result = tool.forward(path)

        assert "error" in result.lower()

    def test_ocr_invalid_region(self, tmp_path):
        path = _make_image(tmp_path, "text.png")
        tool = ScreenOcrTool()

        with patch("smolagents.bp_tools_image.shutil.which", return_value="/usr/bin/tesseract"):
            result = tool.forward(path, region="not,numbers,at,all")
            assert "Error" in result

    def test_ocr_wrong_region_count(self, tmp_path):
        path = _make_image(tmp_path, "text.png")
        tool = ScreenOcrTool()

        with patch("smolagents.bp_tools_image.shutil.which", return_value="/usr/bin/tesseract"):
            result = tool.forward(path, region="10,20,30")
            assert "Error" in result
            assert "4 integers" in result

    def test_ocr_missing_file(self):
        tool = ScreenOcrTool()
        with patch("smolagents.bp_tools_image.shutil.which", return_value="/usr/bin/tesseract"):
            result = tool.forward("/nonexistent/image.png")
        assert "Error" in result


# ======================================================================
# CanvasCreateTool tests
# ======================================================================

class TestCanvasCreateTool:

    def test_create_default(self, tmp_path):
        out = str(tmp_path / "canvas.png")
        tool = CanvasCreateTool()
        result = tool.forward(400, 300, out)

        assert os.path.isfile(out)
        assert "400x300" in result
        img = PIL.Image.open(out)
        assert img.size == (400, 300)
        # Default white background
        assert img.getpixel((0, 0)) == (255, 255, 255)

    def test_create_custom_color(self, tmp_path):
        out = str(tmp_path / "canvas.png")
        tool = CanvasCreateTool()
        result = tool.forward(200, 100, out, bg_color="black")

        img = PIL.Image.open(out)
        assert img.getpixel((0, 0)) == (0, 0, 0)
        assert "black" in result

    def test_create_hex_color(self, tmp_path):
        out = str(tmp_path / "canvas.png")
        tool = CanvasCreateTool()
        result = tool.forward(50, 50, out, bg_color="#FF0000")

        img = PIL.Image.open(out)
        assert img.getpixel((0, 0)) == (255, 0, 0)

    def test_create_invalid_color(self, tmp_path):
        out = str(tmp_path / "canvas.png")
        tool = CanvasCreateTool()
        result = tool.forward(50, 50, out, bg_color="notacolor")
        assert "Error" in result


# ======================================================================
# CanvasDrawTool tests
# ======================================================================

class TestCanvasDrawTool:

    def _make_canvas(self, tmp_path, name="canvas.png", size=(200, 200)):
        path = str(tmp_path / name)
        PIL.Image.new("RGB", size, "white").save(path)
        return path

    def test_draw_rect(self, tmp_path):
        path = self._make_canvas(tmp_path)
        tool = CanvasDrawTool()
        result = tool.forward(path, "rect", "10,10,50,50", color="blue")

        assert "rect" in result
        img = PIL.Image.open(path)
        # The border pixel should be blue-ish
        assert img.getpixel((10, 10)) != (255, 255, 255)

    def test_draw_rect_with_fill(self, tmp_path):
        path = self._make_canvas(tmp_path)
        tool = CanvasDrawTool()
        result = tool.forward(path, "rect", "10,10,50,50", color="blue", fill="yellow")

        img = PIL.Image.open(path)
        # Center should be filled yellow
        assert img.getpixel((30, 30)) == (255, 255, 0)

    def test_draw_circle(self, tmp_path):
        path = self._make_canvas(tmp_path)
        tool = CanvasDrawTool()
        result = tool.forward(path, "circle", "100,100,40", color="red")

        assert "circle" in result
        img = PIL.Image.open(path)
        # Center should still be white (no fill), but edge should be red
        assert img.getpixel((100, 100)) == (255, 255, 255)

    def test_draw_circle_filled(self, tmp_path):
        path = self._make_canvas(tmp_path)
        tool = CanvasDrawTool()
        result = tool.forward(path, "circle", "100,100,40", color="red", fill="red")

        img = PIL.Image.open(path)
        # Center should now be red
        assert img.getpixel((100, 100)) == (255, 0, 0)

    def test_draw_line(self, tmp_path):
        path = self._make_canvas(tmp_path)
        tool = CanvasDrawTool()
        result = tool.forward(path, "line", "0,0,199,199", color="green", line_width=3)

        assert "line" in result
        img = PIL.Image.open(path)
        # Diagonal pixel should be green-ish
        assert img.getpixel((100, 100)) != (255, 255, 255)

    def test_draw_arrow(self, tmp_path):
        path = self._make_canvas(tmp_path)
        tool = CanvasDrawTool()
        result = tool.forward(path, "arrow", "20,100,180,100", color="black")

        assert "arrow" in result
        img = PIL.Image.open(path)
        # Line midpoint should have color
        assert img.getpixel((100, 100)) != (255, 255, 255)

    def test_draw_ellipse(self, tmp_path):
        path = self._make_canvas(tmp_path)
        tool = CanvasDrawTool()
        result = tool.forward(path, "ellipse", "20,40,180,160", color="purple")

        assert "ellipse" in result

    def test_draw_text(self, tmp_path):
        path = self._make_canvas(tmp_path)
        tool = CanvasDrawTool()
        result = tool.forward(path, "text", "50,50", color="black", text="Hello!")

        assert "text" in result
        img = PIL.Image.open(path)
        # Some pixels near (50,50) should be non-white (text drawn)
        region = img.crop((50, 50, 120, 70))
        pixels = list(region.getdata())
        assert any(p != (255, 255, 255) for p in pixels)

    def test_draw_text_no_text_param(self, tmp_path):
        path = self._make_canvas(tmp_path)
        tool = CanvasDrawTool()
        result = tool.forward(path, "text", "10,10")
        assert "Error" in result
        assert "'text' parameter" in result

    def test_draw_multiple_shapes(self, tmp_path):
        """Draw several shapes on the same canvas."""
        path = self._make_canvas(tmp_path)
        tool = CanvasDrawTool()

        tool.forward(path, "rect", "10,10,60,60", color="blue", fill="lightblue")
        tool.forward(path, "circle", "150,50,30", color="red", fill="pink")
        tool.forward(path, "arrow", "60,60,120,50", color="black")
        result = tool.forward(path, "text", "70,150", color="green", text="Done!", font_size=20)

        assert "text" in result
        img = PIL.Image.open(path)
        # Just verify the image is modified (not all white)
        pixels = list(img.getdata())
        assert any(p != (255, 255, 255) for p in pixels)

    def test_draw_unknown_shape(self, tmp_path):
        path = self._make_canvas(tmp_path)
        tool = CanvasDrawTool()
        result = tool.forward(path, "hexagon", "10,10,50")
        assert "Error" in result
        assert "Unknown shape" in result

    def test_draw_bad_coords(self, tmp_path):
        path = self._make_canvas(tmp_path)
        tool = CanvasDrawTool()
        result = tool.forward(path, "rect", "not,valid,coords,here")
        assert "Error" in result

    def test_draw_wrong_coord_count(self, tmp_path):
        path = self._make_canvas(tmp_path)
        tool = CanvasDrawTool()
        result = tool.forward(path, "rect", "10,10")
        assert "Error" in result
        assert "x1,y1,x2,y2" in result

    def test_draw_missing_file(self):
        tool = CanvasDrawTool()
        result = tool.forward("/nonexistent/img.png", "rect", "10,10,50,50")
        assert "Error" in result

    def test_fill_none_string(self, tmp_path):
        """fill='none' should be treated as no fill."""
        path = self._make_canvas(tmp_path)
        tool = CanvasDrawTool()
        result = tool.forward(path, "rect", "10,10,90,90", fill="none")

        img = PIL.Image.open(path)
        # Center should still be white (no fill)
        assert img.getpixel((50, 50)) == (255, 255, 255)


# ======================================================================
# create_image_tools()
# ======================================================================

class TestCreateImageTools:

    def test_shape(self):
        tools = create_image_tools()
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"diff_images", "screen_ocr", "canvas_create", "canvas_draw"}
