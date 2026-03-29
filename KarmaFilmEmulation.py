"""
ComfyUI node for cinematic film stock emulation.

This module provides a specialized node that emulates the color science and
tonal characteristics of iconic analog film stocks. Each preset replicates
the unique look of a specific film, including its color response, contrast
curve, grain structure, and highlight/shadow behavior.
"""

import torch
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

def tensor2pil(image):
    """Convert tensor to PIL image."""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """Convert PIL image to tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


# Film stock preset definitions
# Each preset defines the color science of a specific film stock:
#   temperature   - white balance shift (negative=cool, positive=warm)
#   tint          - green/magenta tint shift
#   contrast      - overall contrast adjustment
#   saturation    - color saturation multiplier
#   shadows_hue   - hue of shadow tinting (0-1)
#   shadows_sat   - strength of shadow tinting
#   highlights_hue - hue of highlight tinting (0-1)
#   highlights_sat - strength of highlight tinting
#   gamma         - midtone brightness (>1 = brighter, <1 = darker)
#   black_lift    - raise black point for faded look
#   grain         - film grain intensity
#   grain_size    - film grain particle size
#   halation      - highlight bloom intensity
FILM_PRESETS = {
    "Kodak Portra 400": {
        "description": "Natural skin tones, soft contrast, warm pastels. The gold standard for portrait photography.",
        "temperature": 0.04,
        "tint": 0.01,
        "contrast": 0.95,
        "saturation": 0.88,
        "shadows_hue": 0.58,
        "shadows_sat": 0.06,
        "highlights_hue": 0.10,
        "highlights_sat": 0.05,
        "gamma": 1.05,
        "black_lift": 0.02,
        "grain": 0.06,
        "grain_size": 1.2,
        "halation": 0.0,
    },
    "Kodak Ektar 100": {
        "description": "Ultra-vivid colors, fine grain, high saturation. Ideal for landscapes and travel.",
        "temperature": 0.02,
        "tint": 0.0,
        "contrast": 1.15,
        "saturation": 1.25,
        "shadows_hue": 0.60,
        "shadows_sat": 0.03,
        "highlights_hue": 0.08,
        "highlights_sat": 0.02,
        "gamma": 0.98,
        "black_lift": 0.0,
        "grain": 0.03,
        "grain_size": 0.8,
        "halation": 0.0,
    },
    "Kodak Gold 200": {
        "description": "Warm, saturated consumer film. Golden highlights, nostalgic everyday look.",
        "temperature": 0.06,
        "tint": 0.01,
        "contrast": 1.05,
        "saturation": 1.10,
        "shadows_hue": 0.08,
        "shadows_sat": 0.05,
        "highlights_hue": 0.12,
        "highlights_sat": 0.08,
        "gamma": 1.02,
        "black_lift": 0.01,
        "grain": 0.08,
        "grain_size": 1.3,
        "halation": 0.0,
    },
    "Fuji Velvia 50": {
        "description": "Extreme saturation, deep contrast, vivid greens and blues. Legendary landscape film.",
        "temperature": -0.02,
        "tint": 0.0,
        "contrast": 1.25,
        "saturation": 1.40,
        "shadows_hue": 0.55,
        "shadows_sat": 0.04,
        "highlights_hue": 0.05,
        "highlights_sat": 0.02,
        "gamma": 0.95,
        "black_lift": 0.0,
        "grain": 0.02,
        "grain_size": 0.7,
        "halation": 0.0,
    },
    "Fuji Pro 400H": {
        "description": "Soft, pastel rendering with subtle greens. Bright, airy skin tones. Wedding favorite.",
        "temperature": -0.01,
        "tint": 0.02,
        "contrast": 0.90,
        "saturation": 0.85,
        "shadows_hue": 0.42,
        "shadows_sat": 0.05,
        "highlights_hue": 0.15,
        "highlights_sat": 0.04,
        "gamma": 1.08,
        "black_lift": 0.03,
        "grain": 0.05,
        "grain_size": 1.0,
        "halation": 0.0,
    },
    "Fuji Superia 400": {
        "description": "Cool tones, strong greens and blues, punchy contrast. Classic consumer film.",
        "temperature": -0.03,
        "tint": 0.01,
        "contrast": 1.08,
        "saturation": 1.05,
        "shadows_hue": 0.55,
        "shadows_sat": 0.06,
        "highlights_hue": 0.42,
        "highlights_sat": 0.04,
        "gamma": 1.0,
        "black_lift": 0.01,
        "grain": 0.09,
        "grain_size": 1.4,
        "halation": 0.0,
    },
    "CineStill 800T": {
        "description": "Tungsten-balanced cinema film. Teal shadows, warm highlights, halation around lights.",
        "temperature": -0.08,
        "tint": -0.02,
        "contrast": 1.05,
        "saturation": 0.95,
        "shadows_hue": 0.52,
        "shadows_sat": 0.10,
        "highlights_hue": 0.08,
        "highlights_sat": 0.08,
        "gamma": 1.02,
        "black_lift": 0.02,
        "grain": 0.10,
        "grain_size": 1.5,
        "halation": 0.15,
    },
    "Kodak Tri-X 400": {
        "description": "Iconic black & white film. Rich tones, beautiful grain, deep blacks. Street photography legend.",
        "temperature": 0.0,
        "tint": 0.0,
        "contrast": 1.20,
        "saturation": 0.0,
        "shadows_hue": 0.0,
        "shadows_sat": 0.0,
        "highlights_hue": 0.0,
        "highlights_sat": 0.0,
        "gamma": 0.98,
        "black_lift": 0.01,
        "grain": 0.12,
        "grain_size": 1.4,
        "halation": 0.0,
    },
    "Ilford HP5 Plus": {
        "description": "Versatile black & white film. Smooth tones, moderate grain, excellent latitude.",
        "temperature": 0.0,
        "tint": 0.0,
        "contrast": 1.10,
        "saturation": 0.0,
        "shadows_hue": 0.0,
        "shadows_sat": 0.0,
        "highlights_hue": 0.0,
        "highlights_sat": 0.0,
        "gamma": 1.02,
        "black_lift": 0.02,
        "grain": 0.08,
        "grain_size": 1.2,
        "halation": 0.0,
    },
    "Kodak Vision3 500T": {
        "description": "Professional cinema negative film. Refined color, tungsten-balanced, modern movie look.",
        "temperature": -0.05,
        "tint": -0.01,
        "contrast": 1.0,
        "saturation": 0.92,
        "shadows_hue": 0.55,
        "shadows_sat": 0.07,
        "highlights_hue": 0.10,
        "highlights_sat": 0.05,
        "gamma": 1.03,
        "black_lift": 0.015,
        "grain": 0.05,
        "grain_size": 1.0,
        "halation": 0.05,
    },
}


class Karma_Film_Emulation:
    """
    Film stock emulation node for one-click cinematic color grading.

    This node applies the color science, tonal characteristics, and texture
    of iconic analog film stocks to digital images. Each preset is carefully
    calibrated to replicate the unique rendering of a specific film, including
    its color response curves, contrast behavior, grain structure, and special
    characteristics like CineStill's halation.

    The intensity slider allows blending between the original image and the
    full film emulation, making it easy to dial in exactly the right amount
    of analog character.

    Supported film stocks:
        Color Negative:
            - Kodak Portra 400 (portraits, natural skin tones)
            - Kodak Ektar 100 (landscapes, vivid color)
            - Kodak Gold 200 (warm, nostalgic everyday)
            - Fuji Velvia 50 (extreme saturation, landscapes)
            - Fuji Pro 400H (soft pastels, weddings)
            - Fuji Superia 400 (cool tones, consumer)
        Cinema:
            - CineStill 800T (tungsten, halation, night photography)
            - Kodak Vision3 500T (professional cinema)
        Black & White:
            - Kodak Tri-X 400 (classic, rich grain)
            - Ilford HP5 Plus (smooth, versatile)
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input parameters for the film emulation node.

        Returns:
            Dictionary containing required and optional input specifications
        """
        film_options = list(FILM_PRESETS.keys())

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to apply film emulation to"}),
                "film_stock": (film_options, {
                    "default": "Kodak Portra 400",
                    "tooltip": "Film stock to emulate"
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.5,
                    "step": 0.05,
                    "tooltip": "Blend intensity (0 = original, 1 = full emulation, >1 = exaggerated)"
                }),
                "grain_override": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Override grain amount (-1 = use film default, 0-1 = custom strength)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Random seed for reproducible grain patterns"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_film_emulation"
    CATEGORY = "KarmaNodes/Post-Processing"

    def apply_film_emulation(self, image: torch.Tensor, film_stock: str,
                             intensity: float, grain_override: float,
                             seed: int) -> tuple:
        """
        Apply film stock emulation to the input image.

        The emulation pipeline applies effects in the correct order to replicate
        how analog film actually works: color response first (how the film
        captures light), then contrast/tone (development characteristics),
        then physical artifacts (grain, halation).

        Args:
            image: Input image tensor
            film_stock: Name of the film stock preset to apply
            intensity: Blend strength (0=original, 1=full, >1=exaggerated)
            grain_override: Custom grain strength (-1 = use preset default)
            seed: Random seed for grain reproducibility

        Returns:
            Tuple containing the processed image tensor
        """
        preset = FILM_PRESETS[film_stock]

        pil_image = tensor2pil(image)
        original_array = np.array(pil_image, dtype=np.float32) / 255.0
        img_array = original_array.copy()

        # Step 1: Color temperature and tint
        if abs(preset["temperature"]) > 0.001 or abs(preset["tint"]) > 0.001:
            img_array = self.apply_temperature_tint(
                img_array, preset["temperature"] * intensity, preset["tint"] * intensity
            )

        # Step 2: Saturation (including B&W conversion for monochrome films)
        effective_sat = 1.0 + (preset["saturation"] - 1.0) * intensity
        if effective_sat < 0.01:
            # Black and white film
            img_array = self.convert_to_bw(img_array)
        elif abs(effective_sat - 1.0) > 0.01:
            img_array = self.adjust_saturation(img_array, effective_sat)

        # Step 3: Gamma / midtone brightness
        effective_gamma = 1.0 + (preset["gamma"] - 1.0) * intensity
        if abs(effective_gamma - 1.0) > 0.005:
            img_array = self.apply_gamma(img_array, effective_gamma)

        # Step 4: Contrast
        effective_contrast = 1.0 + (preset["contrast"] - 1.0) * intensity
        if abs(effective_contrast - 1.0) > 0.01:
            img_array = self.apply_contrast(img_array, effective_contrast)

        # Step 5: Black point lift (faded film look)
        effective_lift = preset["black_lift"] * intensity
        if effective_lift > 0.001:
            img_array = np.clip(img_array * (1.0 - effective_lift) + effective_lift, 0, 1)

        # Step 6: Split toning
        if preset["shadows_sat"] > 0 and intensity > 0:
            img_array = self.apply_split_tone(
                img_array, preset["shadows_hue"],
                preset["shadows_sat"] * intensity, zone="shadows"
            )
        if preset["highlights_sat"] > 0 and intensity > 0:
            img_array = self.apply_split_tone(
                img_array, preset["highlights_hue"],
                preset["highlights_sat"] * intensity, zone="highlights"
            )

        # Step 7: Halation (for CineStill and cinema stocks)
        effective_halation = preset["halation"] * intensity
        if effective_halation > 0.005:
            img_array = self.apply_halation(img_array, effective_halation)

        # Step 8: Film grain
        grain_amount = preset["grain"] if grain_override < 0 else grain_override
        grain_amount *= intensity
        if grain_amount > 0.005:
            valid_seed = int(seed) % (2**32)
            np.random.seed(valid_seed)
            img_array = self.apply_grain(
                img_array, grain_amount, preset["grain_size"]
            )

        result = Image.fromarray(np.clip(img_array * 255, 0, 255).astype(np.uint8))
        result_tensor = pil2tensor(result)
        return (result_tensor,)

    @staticmethod
    def apply_temperature_tint(img: np.ndarray, temperature: float,
                               tint: float) -> np.ndarray:
        """
        Adjust color temperature and green/magenta tint.

        Temperature shifts the blue-yellow axis (positive = warmer/yellow,
        negative = cooler/blue). Tint shifts the green-magenta axis
        (positive = more green, negative = more magenta).

        Args:
            img: Image array in 0-1 float range
            temperature: Temperature shift (-0.2 to 0.2)
            tint: Tint shift (-0.1 to 0.1)

        Returns:
            Color-adjusted image array
        """
        result = img.copy()
        if len(result.shape) == 3 and result.shape[2] >= 3:
            # Warm: boost red, reduce blue
            result[:, :, 0] = np.clip(result[:, :, 0] + temperature * 0.5, 0, 1)
            result[:, :, 2] = np.clip(result[:, :, 2] - temperature * 0.5, 0, 1)
            # Tint: adjust green channel
            result[:, :, 1] = np.clip(result[:, :, 1] + tint * 0.5, 0, 1)
        return result

    @staticmethod
    def convert_to_bw(img: np.ndarray) -> np.ndarray:
        """
        Convert image to black and white using luminance weighting.

        Uses standard BT.601 luminance coefficients for natural-looking
        monochrome conversion that matches how the human eye perceives
        brightness.

        Args:
            img: Image array in 0-1 float range

        Returns:
            Grayscale image array (still 3-channel for compatibility)
        """
        if len(img.shape) == 3 and img.shape[2] >= 3:
            luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            result = np.stack([luminance, luminance, luminance], axis=2)
            return result
        return img

    @staticmethod
    def adjust_saturation(img: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust color saturation.

        Blends between the luminance (grayscale) version and the original
        image. Factor > 1 increases saturation, < 1 decreases.

        Args:
            img: Image array in 0-1 float range
            factor: Saturation multiplier

        Returns:
            Saturation-adjusted image array
        """
        if len(img.shape) == 3 and img.shape[2] >= 3:
            luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            luminance = luminance[:, :, np.newaxis]
            result = luminance + (img - luminance) * factor
            return np.clip(result, 0, 1)
        return img

    @staticmethod
    def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction for midtone brightness adjustment.

        Gamma > 1 brightens midtones (lifts the curve), gamma < 1 darkens
        them. Black and white points are preserved.

        Args:
            img: Image array in 0-1 float range
            gamma: Gamma value (typically 0.8-1.2)

        Returns:
            Gamma-corrected image array
        """
        # Inverse gamma: gamma > 1 should brighten
        inv_gamma = 1.0 / max(gamma, 0.01)
        return np.clip(np.power(np.clip(img, 0.0001, 1.0), inv_gamma), 0, 1)

    @staticmethod
    def apply_contrast(img: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image contrast around the midpoint.

        Scales pixel values relative to 0.5 (middle gray). Factor > 1
        increases contrast, < 1 decreases it.

        Args:
            img: Image array in 0-1 float range
            factor: Contrast multiplier

        Returns:
            Contrast-adjusted image array
        """
        return np.clip(0.5 + (img - 0.5) * factor, 0, 1)

    @staticmethod
    def apply_split_tone(img: np.ndarray, hue: float, strength: float,
                         zone: str = "shadows") -> np.ndarray:
        """
        Apply color tinting to shadows or highlights.

        Args:
            img: Image array in 0-1 float range
            hue: Color hue (0-1)
            strength: Tint intensity
            zone: "shadows" or "highlights"

        Returns:
            Tinted image array
        """
        if len(img.shape) < 3 or img.shape[2] < 3:
            return img

        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        if zone == "shadows":
            mask = np.clip(1.0 - luminance * 2.0, 0, 1)
        else:
            mask = np.clip((luminance - 0.5) * 2.0, 0, 1)

        # Hue to RGB
        hue_360 = hue * 6.0
        x = 1.0 - abs(hue_360 % 2.0 - 1.0)
        if hue_360 < 1:
            color = np.array([1.0, x, 0.0])
        elif hue_360 < 2:
            color = np.array([x, 1.0, 0.0])
        elif hue_360 < 3:
            color = np.array([0.0, 1.0, x])
        elif hue_360 < 4:
            color = np.array([0.0, x, 1.0])
        elif hue_360 < 5:
            color = np.array([x, 0.0, 1.0])
        else:
            color = np.array([1.0, 0.0, x])

        tint = color[np.newaxis, np.newaxis, :] * np.ones_like(img)
        blend = mask[:, :, np.newaxis] * strength
        return np.clip(img * (1.0 - blend) + tint * blend, 0, 1)

    @staticmethod
    def apply_halation(img: np.ndarray, strength: float) -> np.ndarray:
        """
        Apply halation (highlight bloom) effect.

        Simulates the light-scatter phenomenon where bright highlights bleed
        into surrounding areas. Characteristic of CineStill and some cinema
        film stocks where the anti-halation layer is removed.

        Args:
            img: Image array in 0-1 float range
            strength: Halation intensity

        Returns:
            Image array with halation applied
        """
        if len(img.shape) == 3 and img.shape[2] >= 3:
            luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            luminance = img[:, :, 0] if len(img.shape) == 3 else img

        # Extract bright areas
        threshold = 0.75
        highlights = np.clip((luminance - threshold) / (1.0 - threshold + 0.001), 0, 1)

        if len(img.shape) == 3:
            highlight_img = img * highlights[:, :, np.newaxis]
        else:
            highlight_img = img * highlights

        # Blur highlights using PIL
        h_pil = Image.fromarray(np.clip(highlight_img * 255, 0, 255).astype(np.uint8))
        h_blurred = h_pil.filter(ImageFilter.GaussianBlur(radius=15))
        h_array = np.array(h_blurred, dtype=np.float32) / 255.0

        # Screen blend
        result = 1.0 - (1.0 - img) * (1.0 - h_array * strength)
        return np.clip(result, 0, 1)

    @staticmethod
    def apply_grain(img: np.ndarray, strength: float,
                    grain_size: float) -> np.ndarray:
        """
        Apply realistic film grain with luminance-based intensity.

        Grain is more pronounced in darker areas and less visible in bright
        highlights, mimicking real analog film behavior. Multiple noise
        layers at different frequencies create a more organic texture.

        Args:
            img: Image array in 0-1 float range
            strength: Grain intensity
            grain_size: Grain particle size

        Returns:
            Image array with grain applied
        """
        h, w = img.shape[:2]

        # Calculate luminance for intensity modulation
        if len(img.shape) == 3 and img.shape[2] >= 3:
            luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            luminance = img[:, :, 0] if len(img.shape) == 3 else img

        # Grain is stronger in shadows, weaker in highlights
        grain_mask = 1.0 - luminance * 0.5

        # Generate grain at reduced resolution for larger grain size
        grain_h = max(int(h / grain_size), 1)
        grain_w = max(int(w / grain_size), 1)

        # Multi-layer grain for organic texture
        grain = np.random.normal(0, 1, (grain_h, grain_w)).astype(np.float32)

        # Upscale grain to image size if needed
        if grain_size > 1.0:
            grain_pil = Image.fromarray(
                np.clip((grain + 3) / 6 * 255, 0, 255).astype(np.uint8)
            )
            grain_pil = grain_pil.resize((w, h), Image.BILINEAR)
            grain = (np.array(grain_pil, dtype=np.float32) / 255.0 * 6 - 3)

        # Apply grain modulated by luminance
        grain_final = grain * grain_mask * strength

        if len(img.shape) == 3:
            grain_final = grain_final[:, :, np.newaxis]

        return np.clip(img + grain_final, 0, 1)
