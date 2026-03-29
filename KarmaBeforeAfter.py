"""
ComfyUI node for professional before/after image comparison.

This module provides a specialized node that creates publication-ready
comparison images from two inputs (before and after). Supports multiple
layout modes including side-by-side, horizontal/vertical split with
adjustable divider, and opacity blend for detailed inspection.
"""

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def tensor2pil(image):
    """Convert tensor to PIL image."""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    """Convert PIL image to tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class Karma_Before_After:
    """
    Professional before/after image comparison node.

    Creates publication-ready comparison images from two inputs using
    multiple layout modes. Ideal for showcasing post-processing effects,
    model comparisons, upscaling results, or any A/B image comparison.

    Layout Modes:
        - side_by_side: Images placed next to each other with optional gap
        - split_horizontal: Left/right split with movable vertical divider
        - split_vertical: Top/bottom split with movable horizontal divider
        - blend: Opacity crossfade between the two images

    Features:
        - Automatic image resizing when dimensions don't match
        - Configurable divider line with color and width
        - Optional text labels for before/after identification
        - Adjustable split position (0.0 to 1.0)
        - Side-by-side gap width control
        - Background color for gap/padding areas
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input parameters for the before/after comparison node.

        Returns:
            Dictionary containing required and optional input specifications
        """
        return {
            "required": {
                "before_image": ("IMAGE", {
                    "tooltip": "The 'before' image (left/top in comparisons)"
                }),
                "after_image": ("IMAGE", {
                    "tooltip": "The 'after' image (right/bottom in comparisons)"
                }),
                "mode": (["side_by_side", "split_horizontal", "split_vertical", "blend"], {
                    "default": "split_horizontal",
                    "tooltip": "Comparison layout mode"
                }),
                "split_position": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Position of the split/blend point (0.0 = all before, 1.0 = all after)"
                }),
                "divider_width": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Width of the divider line in pixels (0 = no divider)"
                }),
                "divider_color": (["white", "black", "red", "gray"], {
                    "default": "white",
                    "tooltip": "Color of the split divider line"
                }),
                "show_labels": (["none", "simple", "outlined"], {
                    "default": "none",
                    "tooltip": "Label style: none, simple text, or outlined text for visibility"
                }),
                "label_size": (["small", "medium", "large"], {
                    "default": "medium",
                    "tooltip": "Size of the before/after labels"
                }),
                "gap_width": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 100,
                    "step": 2,
                    "tooltip": "Gap between images in side-by-side mode (pixels)"
                }),
                "background_color": (["black", "white", "gray"], {
                    "default": "black",
                    "tooltip": "Background/gap color"
                }),
                "resize_mode": (["match_before", "match_after", "match_larger", "match_smaller"], {
                    "default": "match_before",
                    "tooltip": "How to handle mismatched image sizes"
                }),
            },
            "optional": {
                "before_label": ("STRING", {
                    "default": "Before",
                    "tooltip": "Custom label for the before image"
                }),
                "after_label": ("STRING", {
                    "default": "After",
                    "tooltip": "Custom label for the after image"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("comparison",)
    FUNCTION = "create_comparison"
    CATEGORY = "KarmaNodes/Post-Processing"

    # Color lookup for divider and background
    COLOR_MAP = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "red": (220, 50, 50),
        "gray": (128, 128, 128),
    }

    def create_comparison(self, before_image: torch.Tensor, after_image: torch.Tensor,
                          mode: str, split_position: float, divider_width: int,
                          divider_color: str, show_labels: str, label_size: str,
                          gap_width: int, background_color: str, resize_mode: str,
                          before_label: str = "Before",
                          after_label: str = "After") -> tuple:
        """
        Create a before/after comparison image.

        Processes both input images, resizes them to matching dimensions if
        needed, then composites them according to the selected layout mode.

        Args:
            before_image: The 'before' image tensor
            after_image: The 'after' image tensor
            mode: Layout mode (side_by_side, split_horizontal, split_vertical, blend)
            split_position: Position of split point (0.0 to 1.0)
            divider_width: Width of divider line in pixels
            divider_color: Color of the divider line
            show_labels: Label style (none, simple, outlined)
            label_size: Label size (small, medium, large)
            gap_width: Gap between images in side-by-side mode
            background_color: Background/gap fill color
            resize_mode: How to handle size mismatches
            before_label: Custom text for 'before' label
            after_label: Custom text for 'after' label

        Returns:
            Tuple containing the comparison image tensor
        """
        pil_before = tensor2pil(before_image)
        pil_after = tensor2pil(after_image)

        # Ensure both images are RGB
        pil_before = pil_before.convert("RGB")
        pil_after = pil_after.convert("RGB")

        # Resize images to match if needed
        pil_before, pil_after = self._resize_images(pil_before, pil_after, resize_mode)

        bg_color = self.COLOR_MAP.get(background_color, (0, 0, 0))
        div_color = self.COLOR_MAP.get(divider_color, (255, 255, 255))

        # Create comparison based on mode
        if mode == "side_by_side":
            result = self._side_by_side(pil_before, pil_after, gap_width, bg_color)
        elif mode == "split_horizontal":
            result = self._split_horizontal(pil_before, pil_after, split_position,
                                            divider_width, div_color)
        elif mode == "split_vertical":
            result = self._split_vertical(pil_before, pil_after, split_position,
                                          divider_width, div_color)
        elif mode == "blend":
            result = self._blend(pil_before, pil_after, split_position)
        else:
            result = pil_before

        # Add labels if requested
        if show_labels != "none":
            result = self._add_labels(result, mode, split_position, gap_width,
                                      show_labels, label_size, before_label,
                                      after_label, div_color)

        result_tensor = pil2tensor(result)
        return (result_tensor,)

    @staticmethod
    def _resize_images(before: Image.Image, after: Image.Image,
                       resize_mode: str) -> tuple:
        """
        Resize images to matching dimensions based on the selected mode.

        Args:
            before: Before PIL Image
            after: After PIL Image
            resize_mode: Strategy for matching dimensions

        Returns:
            Tuple of (resized_before, resized_after) PIL Images
        """
        bw, bh = before.size
        aw, ah = after.size

        if bw == aw and bh == ah:
            return before, after

        if resize_mode == "match_before":
            target_w, target_h = bw, bh
        elif resize_mode == "match_after":
            target_w, target_h = aw, ah
        elif resize_mode == "match_larger":
            target_w = max(bw, aw)
            target_h = max(bh, ah)
        elif resize_mode == "match_smaller":
            target_w = min(bw, aw)
            target_h = min(bh, ah)
        else:
            target_w, target_h = bw, bh

        if (bw, bh) != (target_w, target_h):
            before = before.resize((target_w, target_h), Image.LANCZOS)
        if (aw, ah) != (target_w, target_h):
            after = after.resize((target_w, target_h), Image.LANCZOS)

        return before, after

    @staticmethod
    def _side_by_side(before: Image.Image, after: Image.Image,
                      gap: int, bg_color: tuple) -> Image.Image:
        """
        Create a side-by-side comparison with optional gap.

        Places both images horizontally next to each other with a
        configurable gap between them.

        Args:
            before: Before PIL Image
            after: After PIL Image
            gap: Gap width in pixels between the two images
            bg_color: Background color for the gap area

        Returns:
            Combined side-by-side image
        """
        w, h = before.size
        total_w = w * 2 + gap

        canvas = Image.new("RGB", (total_w, h), bg_color)
        canvas.paste(before, (0, 0))
        canvas.paste(after, (w + gap, 0))

        return canvas

    @staticmethod
    def _split_horizontal(before: Image.Image, after: Image.Image,
                          position: float, divider_width: int,
                          divider_color: tuple) -> Image.Image:
        """
        Create a horizontal (left/right) split comparison.

        Shows the 'before' image on the left portion and the 'after' image
        on the right, with a vertical divider line at the split position.

        Args:
            before: Before PIL Image
            after: After PIL Image
            position: Split position (0.0 = far left, 1.0 = far right)
            divider_width: Width of the vertical divider line
            divider_color: Color of the divider line

        Returns:
            Split comparison image
        """
        w, h = before.size
        split_x = int(w * position)
        split_x = max(0, min(split_x, w))

        # Build result: left from before, right from after
        before_arr = np.array(before)
        after_arr = np.array(after)
        result_arr = np.copy(after_arr)
        if split_x > 0:
            result_arr[:, :split_x] = before_arr[:, :split_x]

        result = Image.fromarray(result_arr)

        # Draw divider line
        if divider_width > 0 and 0 < split_x < w:
            draw = ImageDraw.Draw(result)
            half_w = divider_width // 2
            x0 = max(0, split_x - half_w)
            x1 = min(w - 1, split_x + half_w)
            draw.rectangle([x0, 0, x1, h - 1], fill=divider_color)

            # Draw small triangular indicators at top and bottom
            indicator_size = max(6, divider_width * 3)
            # Top indicator (downward triangle)
            draw.polygon([
                (split_x - indicator_size, 0),
                (split_x + indicator_size, 0),
                (split_x, indicator_size)
            ], fill=divider_color)
            # Bottom indicator (upward triangle)
            draw.polygon([
                (split_x - indicator_size, h - 1),
                (split_x + indicator_size, h - 1),
                (split_x, h - 1 - indicator_size)
            ], fill=divider_color)

        return result

    @staticmethod
    def _split_vertical(before: Image.Image, after: Image.Image,
                        position: float, divider_width: int,
                        divider_color: tuple) -> Image.Image:
        """
        Create a vertical (top/bottom) split comparison.

        Shows the 'before' image on the top portion and the 'after' image
        on the bottom, with a horizontal divider line at the split position.

        Args:
            before: Before PIL Image
            after: After PIL Image
            position: Split position (0.0 = top, 1.0 = bottom)
            divider_width: Width of the horizontal divider line
            divider_color: Color of the divider line

        Returns:
            Split comparison image
        """
        w, h = before.size
        split_y = int(h * position)
        split_y = max(0, min(split_y, h))

        # Build result: top from before, bottom from after
        before_arr = np.array(before)
        after_arr = np.array(after)
        result_arr = np.copy(after_arr)
        if split_y > 0:
            result_arr[:split_y, :] = before_arr[:split_y, :]

        result = Image.fromarray(result_arr)

        # Draw divider line
        if divider_width > 0 and 0 < split_y < h:
            draw = ImageDraw.Draw(result)
            half_w = divider_width // 2
            y0 = max(0, split_y - half_w)
            y1 = min(h - 1, split_y + half_w)
            draw.rectangle([0, y0, w - 1, y1], fill=divider_color)

            # Draw small triangular indicators at left and right
            indicator_size = max(6, divider_width * 3)
            # Left indicator (rightward triangle)
            draw.polygon([
                (0, split_y - indicator_size),
                (0, split_y + indicator_size),
                (indicator_size, split_y)
            ], fill=divider_color)
            # Right indicator (leftward triangle)
            draw.polygon([
                (w - 1, split_y - indicator_size),
                (w - 1, split_y + indicator_size),
                (w - 1 - indicator_size, split_y)
            ], fill=divider_color)

        return result

    @staticmethod
    def _blend(before: Image.Image, after: Image.Image,
               opacity: float) -> Image.Image:
        """
        Create a blended comparison using opacity crossfade.

        Blends both images together using the split_position as the
        opacity/mix value. At 0.0 only the before image is shown,
        at 1.0 only the after image, and values in between show a
        transparent overlay of both.

        Args:
            before: Before PIL Image
            after: After PIL Image
            opacity: Blend amount (0.0 = all before, 1.0 = all after)

        Returns:
            Blended comparison image
        """
        before_arr = np.array(before, dtype=np.float32)
        after_arr = np.array(after, dtype=np.float32)

        blended = before_arr * (1.0 - opacity) + after_arr * opacity
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    def _add_labels(self, image: Image.Image, mode: str, split_position: float,
                    gap_width: int, label_style: str, label_size: str,
                    before_text: str, after_text: str,
                    divider_color: tuple) -> Image.Image:
        """
        Add text labels to the comparison image.

        Places 'Before' and 'After' labels at appropriate positions based
        on the layout mode. Supports simple text or outlined text for
        better visibility over varied backgrounds.

        Args:
            image: The comparison image to label
            mode: Current layout mode
            split_position: Current split position
            gap_width: Gap width (for side-by-side positioning)
            label_style: Label rendering style (simple or outlined)
            label_size: Label size (small, medium, large)
            before_text: Text for the 'before' label
            after_text: Text for the 'after' label
            divider_color: Divider color (used for label background reference)

        Returns:
            Image with labels added
        """
        result = image.copy()
        draw = ImageDraw.Draw(result)
        w, h = result.size

        # Determine font size based on image dimensions and label_size setting
        base_size = max(12, min(w, h) // 30)
        size_multipliers = {"small": 0.7, "medium": 1.0, "large": 1.5}
        font_size = int(base_size * size_multipliers.get(label_size, 1.0))

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        # Determine label positions based on mode
        margin = font_size // 2
        label_y = h - margin - font_size  # Bottom of image

        if mode == "side_by_side":
            single_w = (w - gap_width) // 2
            before_x = margin
            after_x = single_w + gap_width + margin
        elif mode == "split_horizontal":
            split_x = int((w - gap_width) / 2 * split_position) if mode == "side_by_side" else int(w * split_position)
            # Place before label in left quarter, after in right quarter
            before_x = margin
            after_x = max(split_x + margin, w * 3 // 4 - margin)
        elif mode == "split_vertical":
            before_x = margin
            after_x = margin
            split_y = int(h * split_position)
            label_y = split_y - margin - font_size  # Before label above split
        elif mode == "blend":
            before_x = margin
            after_x = w - margin  # Will be adjusted by text width below
        else:
            before_x = margin
            after_x = w // 2 + margin

        # Get text bounding boxes for positioning
        before_bbox = draw.textbbox((0, 0), before_text, font=font)
        after_bbox = draw.textbbox((0, 0), after_text, font=font)
        before_tw = before_bbox[2] - before_bbox[0]
        after_tw = after_bbox[2] - after_bbox[0]
        text_h = before_bbox[3] - before_bbox[1]

        # Adjust after_x for right-aligned modes
        if mode in ("blend",):
            after_x = w - margin - after_tw

        # For vertical split, position labels differently
        if mode == "split_vertical":
            split_y = int(h * split_position)
            before_y = max(margin, split_y - margin - text_h - 8)
            after_y = min(h - margin - text_h, split_y + margin)
        else:
            before_y = label_y
            after_y = label_y

        # Draw labels with padding background for readability
        padding = 4
        self._draw_label(draw, before_text, int(before_x), int(before_y),
                         font, label_style, padding)
        self._draw_label(draw, after_text, int(after_x), int(after_y),
                         font, label_style, padding)

        return result

    @staticmethod
    def _draw_label(draw: ImageDraw.Draw, text: str, x: int, y: int,
                    font, style: str, padding: int) -> None:
        """
        Draw a single text label with the specified style.

        For 'simple' style, draws white text on a semi-transparent dark
        background. For 'outlined' style, draws white text with a dark
        outline for maximum legibility over any background.

        Args:
            draw: PIL ImageDraw instance
            text: Label text to render
            x: X position for the label
            y: Y position for the label
            font: PIL font instance
            style: Label style ('simple' or 'outlined')
            padding: Padding around the text in pixels
        """
        bbox = draw.textbbox((x, y), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        if style == "simple":
            # Draw semi-transparent background rectangle
            bg_rect = [
                x - padding,
                y - padding,
                x + tw + padding,
                y + th + padding
            ]
            # Draw dark background (solid since PIL doesn't support alpha easily)
            draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
            draw.text((x, y), text, fill=(255, 255, 255), font=font)

        elif style == "outlined":
            # Draw text outline (dark border around white text)
            outline_range = max(1, padding // 2)
            for ox in range(-outline_range, outline_range + 1):
                for oy in range(-outline_range, outline_range + 1):
                    if ox != 0 or oy != 0:
                        draw.text((x + ox, y + oy), text,
                                  fill=(0, 0, 0), font=font)
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
