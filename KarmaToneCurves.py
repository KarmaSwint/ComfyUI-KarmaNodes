"""
ComfyUI node for professional tone curve adjustments.

This module provides a specialized node for surgical tonal control,
including independent shadow, midtone, and highlight adjustments,
split toning, and black/white point management. Designed to complement
the Karma Kolors node by offering finer-grained control over the tonal
range of an image.
"""

import torch
from PIL import Image
import numpy as np

def tensor2pil(image):
    """Convert tensor to PIL image."""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """Convert PIL image to tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class Karma_Tone_Curves:
    """
    Professional tone curve adjustment node for fine-grained tonal control.

    This node provides independent control over shadows, midtones, and highlights,
    along with split toning capabilities and black/white point adjustment. It
    operates like a simplified version of Lightroom's tone curve panel, giving
    photographers and artists precise control over the luminance distribution
    of their images.

    Features:
        - Independent shadow, midtone, and highlight brightness control
        - Midtone contrast adjustment (S-curve)
        - Shadow and highlight split toning with hue and saturation
        - Black point and white point clipping
        - Smooth tonal transitions with no banding
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input parameters for the tone curves node.

        Returns:
            Dictionary containing required and optional input specifications
        """
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to apply tone adjustments to"}),
                "shadows": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Shadow brightness adjustment (-1 = crush, +1 = lift)"
                }),
                "midtones": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Midtone brightness adjustment (gamma correction)"
                }),
                "highlights": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Highlight brightness adjustment (-1 = pull down, +1 = push up)"
                }),
                "midtone_contrast": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Midtone contrast (S-curve): positive = more contrast, negative = flatter"
                }),
                "black_point": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.3,
                    "step": 0.005,
                    "tooltip": "Raise the black point to clip shadows (0 = pure black)"
                }),
                "white_point": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.7,
                    "max": 1.0,
                    "step": 0.005,
                    "tooltip": "Lower the white point to clip highlights (1 = pure white)"
                }),
                "shadow_tint_hue": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Hue for shadow split tone (0=red, 0.33=green, 0.6=blue, 0.83=magenta)"
                }),
                "shadow_tint_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Intensity of shadow color tinting"
                }),
                "highlight_tint_hue": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Hue for highlight split tone (0=red, 0.1=orange, 0.17=yellow)"
                }),
                "highlight_tint_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Intensity of highlight color tinting"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_tone_curves"
    CATEGORY = "KarmaNodes/Post-Processing"

    def apply_tone_curves(self, image: torch.Tensor, shadows: float, midtones: float,
                          highlights: float, midtone_contrast: float,
                          black_point: float, white_point: float,
                          shadow_tint_hue: float, shadow_tint_strength: float,
                          highlight_tint_hue: float, highlight_tint_strength: float) -> tuple:
        """
        Apply tone curve adjustments to the input image.

        Adjustments are applied in a specific order to maintain image quality:
        black/white point clipping, then shadow/midtone/highlight adjustments,
        then midtone contrast (S-curve), and finally split toning.

        Args:
            image: Input image tensor
            shadows: Shadow brightness adjustment (-1 to 1)
            midtones: Midtone brightness / gamma adjustment (-1 to 1)
            highlights: Highlight brightness adjustment (-1 to 1)
            midtone_contrast: S-curve contrast for midtones (-1 to 1)
            black_point: Black point clipping level (0-0.3)
            white_point: White point clipping level (0.7-1.0)
            shadow_tint_hue: Hue value for shadow tint (0-1)
            shadow_tint_strength: Strength of shadow tint (0-0.5)
            highlight_tint_hue: Hue value for highlight tint (0-1)
            highlight_tint_strength: Strength of highlight tint (0-0.5)

        Returns:
            Tuple containing the processed image tensor
        """
        pil_image = tensor2pil(image)
        img_array = np.array(pil_image, dtype=np.float32) / 255.0
        is_color = len(img_array.shape) == 3 and img_array.shape[2] >= 3

        # Step 1: Black/white point adjustment
        if black_point > 0.001 or white_point < 0.999:
            img_array = self.apply_point_clipping(img_array, black_point, white_point)

        # Step 2: Shadow, midtone, highlight adjustments
        if abs(shadows) > 0.001 or abs(midtones) > 0.001 or abs(highlights) > 0.001:
            img_array = self.apply_zone_adjustments(img_array, shadows, midtones, highlights)

        # Step 3: Midtone contrast (S-curve)
        if abs(midtone_contrast) > 0.001:
            img_array = self.apply_s_curve(img_array, midtone_contrast)

        # Step 4: Split toning
        if is_color:
            if shadow_tint_strength > 0.001:
                img_array = self.apply_split_tone(img_array, shadow_tint_hue,
                                                  shadow_tint_strength, zone="shadows")
            if highlight_tint_strength > 0.001:
                img_array = self.apply_split_tone(img_array, highlight_tint_hue,
                                                  highlight_tint_strength, zone="highlights")

        result = Image.fromarray(np.clip(img_array * 255, 0, 255).astype(np.uint8))
        result_tensor = pil2tensor(result)
        return (result_tensor,)

    @staticmethod
    def apply_point_clipping(img: np.ndarray, black_point: float,
                             white_point: float) -> np.ndarray:
        """
        Adjust black and white points by remapping the tonal range.

        This compresses the full tonal range into the window defined by
        the black and white points, effectively clipping the deepest
        shadows and brightest highlights.

        Args:
            img: Image array in 0-1 float range
            black_point: New minimum value (0-0.3)
            white_point: New maximum value (0.7-1.0)

        Returns:
            Remapped image array
        """
        range_width = max(white_point - black_point, 0.01)
        result = (img - black_point) / range_width
        return np.clip(result, 0, 1)

    @staticmethod
    def apply_zone_adjustments(img: np.ndarray, shadows: float,
                               midtones: float, highlights: float) -> np.ndarray:
        """
        Apply independent brightness adjustments to shadow, midtone, and highlight zones.

        Uses smooth weighting functions to isolate tonal zones and apply
        adjustments only to the relevant range. The weighting functions
        overlap smoothly to prevent visible banding or transitions.

        Args:
            img: Image array in 0-1 float range
            shadows: Shadow adjustment (-1 to 1)
            midtones: Midtone adjustment (-1 to 1)
            highlights: Highlight adjustment (-1 to 1)

        Returns:
            Adjusted image array
        """
        # Calculate luminance for zone detection
        if len(img.shape) == 3 and img.shape[2] >= 3:
            luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            luminance = img.copy() if len(img.shape) == 2 else img[:, :, 0]

        # Smooth zone weight functions using cosine-based transitions
        # Shadows: strongest at 0, fades to 0 by ~0.5
        shadow_weight = np.clip(1.0 - luminance * 2.5, 0, 1) ** 1.5

        # Highlights: 0 until ~0.5, full strength at 1.0
        highlight_weight = np.clip((luminance - 0.4) * 2.5, 0, 1) ** 1.5

        # Midtones: bell curve peaking at 0.5
        midtone_weight = 1.0 - shadow_weight - highlight_weight
        midtone_weight = np.clip(midtone_weight, 0, 1)

        # Calculate combined adjustment
        adjustment = (shadow_weight * shadows * 0.3 +
                      midtone_weight * midtones * 0.3 +
                      highlight_weight * highlights * 0.3)

        if len(img.shape) == 3:
            adjustment = adjustment[:, :, np.newaxis]

        return np.clip(img + adjustment, 0, 1)

    @staticmethod
    def apply_s_curve(img: np.ndarray, strength: float) -> np.ndarray:
        """
        Apply an S-curve contrast adjustment to the midtones.

        Positive strength increases contrast in the midtone range (steepens
        the curve around 0.5), while negative strength reduces contrast
        (flattens the curve). The curve is anchored at the black and white
        points to avoid clipping.

        Args:
            img: Image array in 0-1 float range
            strength: S-curve intensity (-1 to 1)

        Returns:
            Contrast-adjusted image array
        """
        # Use a sigmoid-based S-curve centered at 0.5
        # The strength parameter controls the steepness
        contrast_factor = 1.0 + strength * 2.0

        # Apply power-based S-curve: simple and effective
        if contrast_factor > 0:
            # Remap around 0.5 pivot point
            centered = img - 0.5
            # Apply contrast
            result = 0.5 + centered * contrast_factor
            # Smooth clipping using tanh to avoid hard edges
            result = 0.5 + 0.5 * np.tanh((result - 0.5) * 2.0) / np.tanh(1.0)
        else:
            result = img

        return np.clip(result, 0, 1)

    @staticmethod
    def apply_split_tone(img: np.ndarray, hue: float, strength: float,
                         zone: str = "shadows") -> np.ndarray:
        """
        Apply color tinting to a specific tonal zone.

        Split toning adds a color cast to either shadows or highlights
        independently. This is a classic photographic technique used to
        create mood, such as cool blue shadows with warm golden highlights.

        Args:
            img: Image array in 0-1 float range (must be RGB)
            hue: Color hue to apply (0-1, where 0=red, 0.33=green, 0.67=blue)
            strength: Tint intensity (0-0.5)
            zone: Which zone to tint: "shadows" or "highlights"

        Returns:
            Tinted image array
        """
        # Calculate luminance
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # Create zone mask
        if zone == "shadows":
            mask = np.clip(1.0 - luminance * 2.0, 0, 1)
        else:
            mask = np.clip((luminance - 0.5) * 2.0, 0, 1)

        # Convert hue to RGB color
        # Simple hue-to-RGB conversion (fully saturated colors)
        hue_360 = hue * 6.0
        x = 1.0 - abs(hue_360 % 2.0 - 1.0)

        if hue_360 < 1:
            tint_color = np.array([1.0, x, 0.0])
        elif hue_360 < 2:
            tint_color = np.array([x, 1.0, 0.0])
        elif hue_360 < 3:
            tint_color = np.array([0.0, 1.0, x])
        elif hue_360 < 4:
            tint_color = np.array([0.0, x, 1.0])
        elif hue_360 < 5:
            tint_color = np.array([x, 0.0, 1.0])
        else:
            tint_color = np.array([1.0, 0.0, x])

        # Apply tint: blend toward tint color based on mask and strength
        tint_layer = tint_color[np.newaxis, np.newaxis, :] * np.ones_like(img)
        blend_mask = mask[:, :, np.newaxis] * strength

        result = img * (1.0 - blend_mask) + tint_layer * blend_mask

        return np.clip(result, 0, 1)
