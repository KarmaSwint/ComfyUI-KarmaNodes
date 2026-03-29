"""
ComfyUI node for realistic lens effect simulation.

This module provides a specialized node that applies authentic optical
imperfections to images, simulating the characteristics of real camera lenses.
Effects include chromatic aberration, vignetting, barrel/pincushion distortion,
and halation (highlight bloom).
"""

import torch
from PIL import Image, ImageFilter
import numpy as np

def tensor2pil(image):
    """Convert tensor to PIL image."""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """Convert PIL image to tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class Karma_Lens_FX:
    """
    Professional lens effects node that simulates real-world optical imperfections.

    This node recreates the optical characteristics of physical camera lenses by
    applying chromatic aberration, vignetting, barrel/pincushion distortion, and
    halation effects. Each effect can be independently controlled for precise
    cinematic styling.

    Features:
        - Chromatic aberration with per-channel offset control
        - Smooth radial vignette with adjustable falloff
        - Barrel and pincushion lens distortion
        - Halation (bloom/glow on highlights) with threshold control
        - All effects composable and independently adjustable
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input parameters for the lens effects node.

        Returns:
            Dictionary containing required and optional input specifications
        """
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to apply lens effects to"}),
                "chromatic_aberration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Strength of color fringing at image edges (in pixels)"
                }),
                "vignette_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Intensity of edge darkening (0 = none, 1 = maximum)"
                }),
                "vignette_falloff": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Controls how gradually the vignette fades (higher = tighter center)"
                }),
                "distortion": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Lens distortion: positive = barrel, negative = pincushion"
                }),
                "halation_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Intensity of highlight bloom/glow effect"
                }),
                "halation_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Brightness threshold above which halation is applied"
                }),
                "halation_radius": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 1.0,
                    "tooltip": "Spread radius of the halation glow (in pixels)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_lens_fx"
    CATEGORY = "KarmaNodes/Post-Processing"

    def apply_lens_fx(self, image: torch.Tensor, chromatic_aberration: float,
                      vignette_strength: float, vignette_falloff: float,
                      distortion: float, halation_strength: float,
                      halation_threshold: float, halation_radius: float) -> tuple:
        """
        Apply lens effects to the input image.

        Effects are applied in optical order: distortion first (physical lens
        geometry), then chromatic aberration (light separation), halation
        (light scatter), and finally vignetting (light falloff).

        Args:
            image: Input image tensor
            chromatic_aberration: Strength of color fringing in pixels
            vignette_strength: Intensity of edge darkening (0-1)
            vignette_falloff: Vignette gradient steepness (0.5-5.0)
            distortion: Barrel (+) or pincushion (-) distortion (-1 to 1)
            halation_strength: Intensity of highlight bloom (0-1)
            halation_threshold: Brightness threshold for halation (0-1)
            halation_radius: Spread of halation glow in pixels

        Returns:
            Tuple containing the processed image tensor
        """
        pil_image = tensor2pil(image)

        # Apply effects in optical order
        if abs(distortion) > 0.001:
            pil_image = self.apply_distortion(pil_image, distortion)

        if chromatic_aberration > 0.1:
            pil_image = self.apply_chromatic_aberration(pil_image, chromatic_aberration)

        if halation_strength > 0.001:
            pil_image = self.apply_halation(pil_image, halation_strength,
                                            halation_threshold, halation_radius)

        if vignette_strength > 0.001:
            pil_image = self.apply_vignette(pil_image, vignette_strength,
                                            vignette_falloff)

        result_tensor = pil2tensor(pil_image)
        return (result_tensor,)

    @staticmethod
    def apply_chromatic_aberration(image: Image.Image, strength: float) -> Image.Image:
        """
        Apply chromatic aberration by offsetting color channels.

        Simulates the failure of a lens to focus all colors to the same point,
        creating color fringing that increases toward image edges. The red channel
        is shifted outward and the blue channel inward, mimicking real lateral
        chromatic aberration.

        Args:
            image: Input PIL Image
            strength: Offset strength in pixels

        Returns:
            Image with chromatic aberration applied
        """
        img_array = np.array(image, dtype=np.float32)
        h, w = img_array.shape[:2]
        is_color = len(img_array.shape) == 3 and img_array.shape[2] >= 3

        if not is_color:
            return image

        # Create coordinate grids for radial-weighted shifts
        cy, cx = h / 2.0, w / 2.0
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Radial distance from center (normalized to 0-1)
        max_radius = np.sqrt(cx ** 2 + cy ** 2)
        dx = (x_coords - cx) / max_radius
        dy = (y_coords - cy) / max_radius
        radius = np.sqrt(dx ** 2 + dy ** 2)

        # Scale shift by radial distance (more shift at edges)
        shift_scale = radius * strength

        # Shift red channel outward, blue channel inward
        result = img_array.copy()

        # Red channel - shift away from center
        r_x = x_coords + dx * shift_scale
        r_y = y_coords + dy * shift_scale
        r_x = np.clip(r_x, 0, w - 1).astype(np.int32)
        r_y = np.clip(r_y, 0, h - 1).astype(np.int32)
        result[:, :, 0] = img_array[r_y, r_x, 0]

        # Blue channel - shift toward center
        b_x = x_coords - dx * shift_scale
        b_y = y_coords - dy * shift_scale
        b_x = np.clip(b_x, 0, w - 1).astype(np.int32)
        b_y = np.clip(b_y, 0, h - 1).astype(np.int32)
        result[:, :, 2] = img_array[b_y, b_x, 2]

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_vignette(image: Image.Image, strength: float,
                       falloff: float) -> Image.Image:
        """
        Apply radial vignette darkening to image edges.

        Creates a smooth radial gradient that darkens the image toward its
        edges, simulating the natural light falloff of camera lenses. The
        falloff parameter controls how tight the bright center area is.

        Args:
            image: Input PIL Image
            strength: Vignette intensity (0-1)
            falloff: Gradient steepness (higher = tighter center)

        Returns:
            Image with vignette applied
        """
        img_array = np.array(image, dtype=np.float32)
        h, w = img_array.shape[:2]

        # Create radial distance map (0 at center, 1 at corners)
        cy, cx = h / 2.0, w / 2.0
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Normalize to elliptical distance so vignette follows image shape
        dx = (x_coords - cx) / cx
        dy = (y_coords - cy) / cy
        radius = np.sqrt(dx ** 2 + dy ** 2)

        # Apply falloff curve and strength
        # radius of ~1.0 at edges, ~1.41 at corners
        vignette_mask = 1.0 - strength * np.clip(radius ** falloff, 0, 1)
        vignette_mask = np.clip(vignette_mask, 0, 1)

        # Apply to all channels
        if len(img_array.shape) == 3:
            vignette_mask = vignette_mask[:, :, np.newaxis]

        result = img_array * vignette_mask
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_distortion(image: Image.Image, strength: float) -> Image.Image:
        """
        Apply barrel or pincushion lens distortion.

        Simulates the geometric distortion of real camera lenses. Barrel
        distortion (positive values) bulges the image center outward, while
        pincushion distortion (negative values) pinches it inward.

        Args:
            image: Input PIL Image
            strength: Distortion amount (positive = barrel, negative = pincushion)

        Returns:
            Image with lens distortion applied
        """
        img_array = np.array(image, dtype=np.float32)
        h, w = img_array.shape[:2]

        # Create normalized coordinate grid centered at image center
        cy, cx = h / 2.0, w / 2.0
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Normalize coordinates to -1..1 range
        nx = (x_coords - cx) / cx
        ny = (y_coords - cy) / cy

        # Radial distance from center
        r = np.sqrt(nx ** 2 + ny ** 2)

        # Apply distortion formula: r_distorted = r * (1 + k * r^2)
        k = strength * 0.5  # Scale for reasonable range
        r_distorted = r * (1.0 + k * r ** 2)

        # Avoid division by zero
        safe_r = np.where(r > 0.0001, r, 1.0)
        scale = r_distorted / safe_r
        scale = np.where(r > 0.0001, scale, 1.0)

        # Map back to pixel coordinates
        new_x = cx + nx * scale * cx
        new_y = cy + ny * scale * cy

        # Clip to valid range
        new_x = np.clip(new_x, 0, w - 1).astype(np.int32)
        new_y = np.clip(new_y, 0, h - 1).astype(np.int32)

        # Remap image
        if len(img_array.shape) == 3:
            result = img_array[new_y, new_x, :]
        else:
            result = img_array[new_y, new_x]

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_halation(image: Image.Image, strength: float,
                       threshold: float, radius: float) -> Image.Image:
        """
        Apply halation (highlight bloom) effect.

        Simulates the light-scatter phenomenon in analog film where bright
        highlights bleed into surrounding areas with a soft glow. The effect
        is isolated to pixels above the brightness threshold and blurred to
        create a natural bloom.

        Args:
            image: Input PIL Image
            strength: Intensity of the glow (0-1)
            threshold: Brightness threshold for affected pixels (0-1)
            radius: Blur radius for the glow spread

        Returns:
            Image with halation applied
        """
        img_array = np.array(image, dtype=np.float32) / 255.0
        is_color = len(img_array.shape) == 3 and img_array.shape[2] >= 3

        # Calculate luminance
        if is_color:
            luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        else:
            luminance = img_array.copy()

        # Create highlight mask (pixels above threshold)
        highlight_mask = np.clip((luminance - threshold) / (1.0 - threshold + 0.001), 0, 1)

        # Extract highlight colors and blur them
        if is_color:
            highlight_image = img_array * highlight_mask[:, :, np.newaxis]
        else:
            highlight_image = img_array * highlight_mask

        # Convert to PIL for Gaussian blur
        highlight_pil = Image.fromarray(np.clip(highlight_image * 255, 0, 255).astype(np.uint8))
        blurred_highlight = highlight_pil.filter(ImageFilter.GaussianBlur(radius=radius))
        blurred_array = np.array(blurred_highlight, dtype=np.float32) / 255.0

        # Blend: screen-like compositing for natural glow
        # Screen blend: 1 - (1 - a) * (1 - b)
        result = 1.0 - (1.0 - img_array) * (1.0 - blurred_array * strength)

        return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))
