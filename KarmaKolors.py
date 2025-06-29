"""
ComfyUI node for color post-processing adjustments.

This module provides a specialized node for color correction and enhancement,
including white balance, brightness, contrast, and saturation adjustments.
The node is designed to provide professional-grade color grading capabilities.
"""

import torch
from PIL import Image, ImageEnhance
import numpy as np
import colorsys

def tensor2pil(image):
    """Convert tensor to PIL image."""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """Convert PIL image to tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class Karma_Kolors:
    """
    Advanced color post-processing node for professional color grading.
    
    This node provides comprehensive color adjustment capabilities including:
    - White balance correction with temperature control
    - Brightness adjustment
    - Contrast enhancement
    - Saturation control
    
    All adjustments are applied in the optimal order to maintain image quality
    and provide natural-looking results.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input parameters for the color processing node.
        
        Returns:
            Dictionary containing required and optional input specifications
        """
        # Generate temperature options from 2000K to 10000K in 100K increments
        temperature_options = ["auto"] + [f"{temp}K" for temp in range(2000, 10001, 100)]
        
        # Generate percentage options from -20% to +20% in 0.5% increments
        percentage_options = [f"{val:.1f}" for val in np.arange(-20.0, 20.5, 0.5)]
        
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to apply color adjustments to"}),
                "white_balance": (temperature_options, {
                    "default": "auto",
                    "tooltip": "White balance temperature in Kelvin or auto"
                }),
                "brightness": (percentage_options, {
                    "default": "0.0",
                    "tooltip": "Brightness adjustment in percentage (-20% to +20%)"
                }),
                "contrast": (percentage_options, {
                    "default": "0.0", 
                    "tooltip": "Contrast adjustment in percentage (-20% to +20%)"
                }),
                "saturation": (percentage_options, {
                    "default": "0.0",
                    "tooltip": "Saturation adjustment in percentage (-20% to +20%)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_color_adjustments"
    CATEGORY = "KarmaNodes/Post-Processing"

    def apply_color_adjustments(self, image: torch.Tensor, white_balance: str, 
                              brightness: str, contrast: str, saturation: str) -> tuple:
        """
        Apply color adjustments to the input image.
        
        Args:
            image: Input image tensor
            white_balance: White balance setting (temperature in K or "auto")
            brightness: Brightness adjustment percentage
            contrast: Contrast adjustment percentage  
            saturation: Saturation adjustment percentage
            
        Returns:
            Tuple containing the processed image tensor
        """
        # Convert tensor to PIL for processing
        pil_image = tensor2pil(image)
        
        # Apply adjustments in optimal order
        processed_image = pil_image
        
        # 1. White balance correction (first, as it affects overall color temperature)
        if white_balance != "auto":
            processed_image = self.apply_white_balance(processed_image, white_balance)
        else:
            processed_image = self.apply_auto_white_balance(processed_image)
        
        # 2. Brightness adjustment
        brightness_val = float(brightness)
        if brightness_val != 0.0:
            processed_image = self.apply_brightness(processed_image, brightness_val)
        
        # 3. Contrast adjustment  
        contrast_val = float(contrast)
        if contrast_val != 0.0:
            processed_image = self.apply_contrast(processed_image, contrast_val)
        
        # 4. Saturation adjustment (last, to preserve color relationships)
        saturation_val = float(saturation)
        if saturation_val != 0.0:
            processed_image = self.apply_saturation(processed_image, saturation_val)
        
        # Convert back to tensor
        result_tensor = pil2tensor(processed_image)
        
        return (result_tensor,)

    @staticmethod
    def apply_white_balance(image: Image.Image, temperature: str) -> Image.Image:
        """
        Apply white balance correction based on color temperature.
        
        Args:
            image: Input PIL image
            temperature: Color temperature string (e.g., "5500K")
            
        Returns:
            White balance corrected PIL image
        """
        # Extract temperature value
        temp_k = int(temperature.replace('K', ''))
        
        # Convert temperature to RGB multipliers
        # Based on Tanner Helland's algorithm for blackbody radiation
        temp = temp_k / 100.0
        
        if temp <= 66:
            red = 255
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
            if temp >= 19:
                blue = temp - 10
                blue = 138.5177312231 * np.log(blue) - 305.0447927307
            else:
                blue = 0
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            green = temp - 60  
            green = 288.1221695283 * (green ** -0.0755148492)
            blue = 255
        
        # Normalize to 0-1 range
        red = np.clip(red, 0, 255) / 255.0
        green = np.clip(green, 0, 255) / 255.0
        blue = np.clip(blue, 0, 255) / 255.0
        
        # Apply white balance correction
        image_array = np.array(image).astype(np.float32) / 255.0
        
        if len(image_array.shape) == 3:  # Color image
            # Apply multipliers to each channel
            image_array[:, :, 0] *= red    # Red channel
            image_array[:, :, 1] *= green  # Green channel  
            image_array[:, :, 2] *= blue   # Blue channel
            
            # Normalize to prevent clipping while maintaining ratios
            max_val = np.max(image_array)
            if max_val > 1.0:
                image_array /= max_val
        
        # Convert back to PIL image
        corrected_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(corrected_array)

    @staticmethod
    def apply_auto_white_balance(image: Image.Image) -> Image.Image:
        """
        Apply automatic white balance correction using gray world assumption.
        
        Args:
            image: Input PIL image
            
        Returns:
            Auto white balance corrected PIL image
        """
        image_array = np.array(image).astype(np.float32) / 255.0
        
        if len(image_array.shape) == 3:  # Color image
            # Calculate average values for each channel
            avg_r = np.mean(image_array[:, :, 0])
            avg_g = np.mean(image_array[:, :, 1])
            avg_b = np.mean(image_array[:, :, 2])
            
            # Calculate gray world average
            gray_avg = (avg_r + avg_g + avg_b) / 3.0
            
            # Calculate correction factors
            if avg_r > 0:
                r_factor = gray_avg / avg_r
            else:
                r_factor = 1.0
                
            if avg_g > 0:
                g_factor = gray_avg / avg_g
            else:
                g_factor = 1.0
                
            if avg_b > 0:
                b_factor = gray_avg / avg_b
            else:
                b_factor = 1.0
            
            # Apply correction factors
            image_array[:, :, 0] *= r_factor
            image_array[:, :, 1] *= g_factor
            image_array[:, :, 2] *= b_factor
            
            # Normalize to prevent clipping
            image_array = np.clip(image_array, 0, 1)
        
        # Convert back to PIL image
        corrected_array = (image_array * 255.0).astype(np.uint8)
        return Image.fromarray(corrected_array)

    @staticmethod
    def apply_brightness(image: Image.Image, brightness_percent: float) -> Image.Image:
        """
        Apply brightness adjustment.
        
        Args:
            image: Input PIL image
            brightness_percent: Brightness adjustment in percentage (-20 to +20)
            
        Returns:
            Brightness adjusted PIL image
        """
        # Convert percentage to enhancement factor
        # 0% = 1.0 (no change), +20% = 1.2, -20% = 0.8
        factor = 1.0 + (brightness_percent / 100.0)
        
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def apply_contrast(image: Image.Image, contrast_percent: float) -> Image.Image:
        """
        Apply contrast adjustment.
        
        Args:
            image: Input PIL image
            contrast_percent: Contrast adjustment in percentage (-20 to +20)
            
        Returns:
            Contrast adjusted PIL image
        """
        # Convert percentage to enhancement factor
        factor = 1.0 + (contrast_percent / 100.0)
        
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def apply_saturation(image: Image.Image, saturation_percent: float) -> Image.Image:
        """
        Apply saturation adjustment.
        
        Args:
            image: Input PIL image
            saturation_percent: Saturation adjustment in percentage (-20 to +20)
            
        Returns:
            Saturation adjusted PIL image
        """
        # Convert percentage to enhancement factor
        factor = 1.0 + (saturation_percent / 100.0)
        
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Karma_Kolors": Karma_Kolors
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Karma_Kolors": "Karma Kolors"
}