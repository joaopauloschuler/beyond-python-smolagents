"""
Image analysis and drawing tools for BPSA agents.

Provides tools for visual diffing, OCR text extraction, and canvas drawing.
All tools produce image files that can be viewed via ``load_image``.

Dependencies:
- Pillow (required, already a project dependency)
- tesseract-ocr (optional, for ``screen_ocr`` only)
"""

import math
import os
import shutil
import subprocess
import tempfile

import PIL.Image
import PIL.ImageChops
import PIL.ImageDraw
import PIL.ImageFont

from .tools import Tool


# ======================================================================
# diff_images — visual image comparison
# ======================================================================


class DiffImagesTool(Tool):
    """Visually compare two images and produce a diff image."""

    name = "diff_images"
    description = (
        "Compare two images and produce a visual diff highlighting the differences. "
        "Modes: 'highlight' (red overlay on changed pixels, default), "
        "'side_by_side' (both images next to each other with diff in the middle). "
        "Returns the path to the diff image and a summary of how different they are. "
        "Use load_image on the returned path to see the result."
    )
    inputs = {
        "image1_path": {"type": "string", "description": "Path to the first image"},
        "image2_path": {"type": "string", "description": "Path to the second image"},
        "output_path": {
            "type": "string",
            "description": "Path to save the diff image (optional, defaults to a temp file)",
            "nullable": True,
        },
        "mode": {
            "type": "string",
            "description": "Diff mode: 'highlight' (default) or 'side_by_side'",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(
        self,
        image1_path: str,
        image2_path: str,
        output_path: str | None = None,
        mode: str | None = None,
    ) -> str:
        mode = (mode or "highlight").strip().lower()
        if mode not in ("highlight", "side_by_side"):
            return f"Error: Unknown mode '{mode}'. Use 'highlight' or 'side_by_side'."

        try:
            img1 = PIL.Image.open(image1_path).convert("RGB")
            img2 = PIL.Image.open(image2_path).convert("RGB")
        except Exception as e:
            return f"Error opening images: {e}"

        # Resize img2 to match img1 if dimensions differ
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, PIL.Image.LANCZOS)

        # Compute pixel difference
        diff = PIL.ImageChops.difference(img1, img2)
        diff_gray = diff.convert("L")
        total_pixels = img1.size[0] * img1.size[1]
        changed_pixels = sum(1 for p in diff_gray.getdata() if p > 10)
        pct = (changed_pixels / total_pixels * 100) if total_pixels > 0 else 0

        if mode == "highlight":
            result = self._highlight_diff(img1, img2, diff_gray)
        else:
            result = self._side_by_side(img1, img2, diff_gray)

        out = output_path or tempfile.mktemp(suffix="_diff.png", prefix="bpsa_")
        result.save(out)

        return (
            f"Diff saved to {out} ({result.size[0]}x{result.size[1]}). "
            f"Changed pixels: {changed_pixels:,}/{total_pixels:,} ({pct:.1f}%). "
            f"Use load_image(\"{out}\") to view."
        )

    def _highlight_diff(self, img1, img2, diff_gray):
        """Overlay changed pixels in red on the first image."""
        result = img1.copy()
        # Create red mask where pixels differ
        mask = diff_gray.point(lambda p: 255 if p > 10 else 0)
        red_overlay = PIL.Image.new("RGB", img1.size, (255, 0, 0))
        # Blend: 60% original + 40% red on changed pixels
        blended = PIL.Image.blend(img1, red_overlay, 0.4)
        result.paste(blended, mask=mask)
        return result

    def _side_by_side(self, img1, img2, diff_gray):
        """Place both images side by side with the diff in the middle."""
        w, h = img1.size
        gap = 4
        canvas = PIL.Image.new("RGB", (w * 3 + gap * 2, h), (40, 40, 40))
        canvas.paste(img1, (0, 0))

        # Amplified diff for visibility
        diff_vis = PIL.ImageChops.difference(img1, img2)
        from PIL import ImageEnhance
        diff_vis = ImageEnhance.Brightness(diff_vis).enhance(3.0)
        canvas.paste(diff_vis, (w + gap, 0))

        canvas.paste(img2, (w * 2 + gap * 2, 0))

        # Labels
        draw = PIL.ImageDraw.Draw(canvas)
        draw.text((5, 5), "Before", fill="white")
        draw.text((w + gap + 5, 5), "Diff", fill="white")
        draw.text((w * 2 + gap * 2 + 5, 5), "After", fill="white")
        return canvas


# ======================================================================
# screen_ocr — text extraction from images via Tesseract
# ======================================================================


class ScreenOcrTool(Tool):
    """Extract text from an image using OCR (Tesseract)."""

    name = "screen_ocr"
    description = (
        "Extract text from an image file using OCR (Tesseract). "
        "Useful for reading text from GUI screenshots, scanned documents, "
        "or any image containing text. Optionally crop to a region before OCR. "
        "Requires tesseract-ocr to be installed (sudo apt install tesseract-ocr)."
    )
    inputs = {
        "image_path": {"type": "string", "description": "Path to the image file"},
        "region": {
            "type": "string",
            "description": "Optional crop region as 'x,y,width,height' (pixels from top-left). "
                           "Leave empty to OCR the entire image.",
            "nullable": True,
        },
        "language": {
            "type": "string",
            "description": "Tesseract language code (default: 'eng'). E.g. 'deu', 'fra', 'jpn'.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(
        self,
        image_path: str,
        region: str | None = None,
        language: str | None = None,
    ) -> str:
        if not shutil.which("tesseract"):
            return (
                "Error: tesseract-ocr is not installed. "
                "Install with: sudo apt install tesseract-ocr"
            )

        try:
            img = PIL.Image.open(image_path)
        except Exception as e:
            return f"Error opening image: {e}"

        # Crop if region specified
        if region and region.strip():
            try:
                parts = [int(x.strip()) for x in region.split(",")]
                if len(parts) != 4:
                    return "Error: region must be 'x,y,width,height' (4 integers)."
                x, y, w, h = parts
                img = img.crop((x, y, x + w, y + h))
            except ValueError:
                return "Error: region must be 4 comma-separated integers: 'x,y,width,height'."

        lang = (language or "eng").strip()

        # Save to temp file for tesseract
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
            img.save(tmp_path)

        try:
            result = subprocess.run(
                ["tesseract", tmp_path, "stdout", "-l", lang],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return f"Tesseract error: {result.stderr.strip()}"
            text = result.stdout.strip()
            if not text:
                return "(No text detected in the image)"
            return text
        except subprocess.TimeoutExpired:
            return "Error: Tesseract timed out (30s limit)."
        finally:
            os.unlink(tmp_path)


# ======================================================================
# Canvas tools — create and draw on images
# ======================================================================


class CanvasCreateTool(Tool):
    """Create a blank canvas image."""

    name = "canvas_create"
    description = (
        "Create a blank canvas image of the specified size and background color. "
        "Returns the path to the saved image. Use canvas_draw to add shapes, "
        "then load_image to see the result."
    )
    inputs = {
        "width": {"type": "integer", "description": "Canvas width in pixels"},
        "height": {"type": "integer", "description": "Canvas height in pixels"},
        "output_path": {"type": "string", "description": "Path to save the canvas image"},
        "bg_color": {
            "type": "string",
            "description": "Background color (default: 'white'). Accepts names ('red', 'blue') or hex ('#FF0000').",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(
        self,
        width: int,
        height: int,
        output_path: str,
        bg_color: str | None = None,
    ) -> str:
        bg = (bg_color or "white").strip()
        try:
            img = PIL.Image.new("RGB", (width, height), bg)
        except Exception as e:
            return f"Error creating canvas: {e}"

        try:
            img.save(output_path)
        except Exception as e:
            return f"Error saving canvas: {e}"

        return f"Canvas created: {width}x{height}, bg={bg}, saved to {output_path}"


class CanvasDrawTool(Tool):
    """Draw shapes and text on an image file."""

    name = "canvas_draw"
    description = (
        "Draw a shape or text on an existing image file (canvas, screenshot, photo, etc.). "
        "The image is modified in place. Call multiple times to build up a drawing.\n\n"
        "Supported shapes and their coords format:\n"
        "- 'rect': 'x1,y1,x2,y2' — rectangle from top-left to bottom-right\n"
        "- 'circle': 'cx,cy,radius' — circle centered at (cx,cy)\n"
        "- 'ellipse': 'x1,y1,x2,y2' — ellipse within bounding box\n"
        "- 'line': 'x1,y1,x2,y2' — straight line between two points\n"
        "- 'arrow': 'x1,y1,x2,y2' — line with arrowhead at (x2,y2)\n"
        "- 'text': 'x,y' — text at position (x,y), provide text in the 'text' parameter\n\n"
        "Use load_image on the image path to see the result."
    )
    inputs = {
        "image_path": {"type": "string", "description": "Path to the image file to draw on"},
        "shape": {
            "type": "string",
            "description": "Shape type: 'rect', 'circle', 'ellipse', 'line', 'arrow', 'text'",
        },
        "coords": {
            "type": "string",
            "description": "Coordinates as comma-separated integers (format depends on shape)",
        },
        "color": {
            "type": "string",
            "description": "Color name or hex (default: 'red'). E.g. 'blue', '#00FF00'.",
            "nullable": True,
        },
        "fill": {
            "type": "string",
            "description": "Fill color for rect/circle/ellipse (default: no fill). Use 'none' for outline only.",
            "nullable": True,
        },
        "line_width": {
            "type": "integer",
            "description": "Line/outline width in pixels (default: 2)",
            "nullable": True,
        },
        "text": {
            "type": "string",
            "description": "Text string (only for shape='text')",
            "nullable": True,
        },
        "font_size": {
            "type": "integer",
            "description": "Font size for text (default: 16)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(
        self,
        image_path: str,
        shape: str,
        coords: str,
        color: str | None = None,
        fill: str | None = None,
        line_width: int | None = None,
        text: str | None = None,
        font_size: int | None = None,
    ) -> str:
        color = (color or "red").strip()
        lw = line_width or 2
        shape = shape.strip().lower()

        try:
            img = PIL.Image.open(image_path).convert("RGB")
        except Exception as e:
            return f"Error opening image: {e}"

        draw = PIL.ImageDraw.Draw(img)

        try:
            parts = [int(x.strip()) for x in coords.split(",")]
        except ValueError:
            return f"Error: coords must be comma-separated integers, got '{coords}'."

        fill_color = None
        if fill and fill.strip().lower() != "none":
            fill_color = fill.strip()

        try:
            if shape == "rect":
                if len(parts) != 4:
                    return "Error: rect requires coords 'x1,y1,x2,y2'."
                draw.rectangle(parts, outline=color, fill=fill_color, width=lw)

            elif shape == "circle":
                if len(parts) != 3:
                    return "Error: circle requires coords 'cx,cy,radius'."
                cx, cy, r = parts
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, fill=fill_color, width=lw)

            elif shape == "ellipse":
                if len(parts) != 4:
                    return "Error: ellipse requires coords 'x1,y1,x2,y2'."
                draw.ellipse(parts, outline=color, fill=fill_color, width=lw)

            elif shape == "line":
                if len(parts) != 4:
                    return "Error: line requires coords 'x1,y1,x2,y2'."
                draw.line(parts, fill=color, width=lw)

            elif shape == "arrow":
                if len(parts) != 4:
                    return "Error: arrow requires coords 'x1,y1,x2,y2'."
                x1, y1, x2, y2 = parts
                draw.line([x1, y1, x2, y2], fill=color, width=lw)
                # Draw arrowhead
                self._draw_arrowhead(draw, x1, y1, x2, y2, color, lw)

            elif shape == "text":
                if len(parts) != 2:
                    return "Error: text requires coords 'x,y'."
                if not text:
                    return "Error: 'text' parameter is required for shape='text'."
                fs = font_size or 16
                font = self._get_font(fs)
                draw.text((parts[0], parts[1]), text, fill=color, font=font)

            else:
                return f"Error: Unknown shape '{shape}'. Use rect/circle/ellipse/line/arrow/text."

        except Exception as e:
            return f"Error drawing {shape}: {e}"

        img.save(image_path)
        return f"Drew {shape} on {image_path} (color={color})"

    def _draw_arrowhead(self, draw, x1, y1, x2, y2, color, lw):
        """Draw an arrowhead at the end of a line."""
        angle = math.atan2(y2 - y1, x2 - x1)
        head_len = max(10, lw * 5)
        head_angle = math.pi / 6  # 30 degrees

        left_x = x2 - head_len * math.cos(angle - head_angle)
        left_y = y2 - head_len * math.sin(angle - head_angle)
        right_x = x2 - head_len * math.cos(angle + head_angle)
        right_y = y2 - head_len * math.sin(angle + head_angle)

        draw.polygon(
            [(x2, y2), (int(left_x), int(left_y)), (int(right_x), int(right_y))],
            fill=color,
        )

    def _get_font(self, size):
        """Try to load a TrueType font, fall back to default."""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
        for path in font_paths:
            if os.path.isfile(path):
                try:
                    return PIL.ImageFont.truetype(path, size)
                except Exception:
                    continue
        return PIL.ImageFont.load_default()


# ======================================================================
# Factory
# ======================================================================


def create_image_tools() -> list[Tool]:
    """Create and return the image analysis/drawing tools."""
    return [
        DiffImagesTool(),
        ScreenOcrTool(),
        CanvasCreateTool(),
        CanvasDrawTool(),
    ]
