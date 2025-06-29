"""
ComfyUI node for adding realistic film grain texture to images.

This module provides a specialized node that adds authentic film grain effects
to images, simulating the texture and characteristics of analog film photography.
The grain effect is applied with luminance-based intensity for realistic results.
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

class Karma_Film_Grain:
    """
    Advanced film grain node that adds realistic film texture to images.
    
    This node simulates the grain characteristics of analog film by applying
    multiple layers of noise with luminance-based intensity. The grain effect
    is more pronounced in darker areas, mimicking real film behavior.
    
    Features:
        - Realistic multi-layer grain pattern
        - Luminance-based grain intensity
        - Configurable grain strength and size
        - Support for both color and grayscale images
        - Fallback implementation when scipy is unavailable
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input parameters for the film grain node.
        
        Returns:
            Dictionary containing required and optional input specifications
        """
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to apply film grain to"}),
                "strength": ("FLOAT", {
                    "default": 0.1, 
                    "min": 0.01, 
                    "max": 1.0, 
                    "step": 0.01, 
                    "tooltip": "Intensity of the film grain effect"
                }),
                "grain_size": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1, 
                    "max": 5.0, 
                    "step": 0.1, 
                    "tooltip": "Size/scale of grain particles"
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
    FUNCTION = "apply_film_grain"
    CATEGORY = "KarmaNodes/Post-Processing"

    def apply_film_grain(self, image: torch.Tensor, strength: float, grain_size: float, seed: int) -> tuple:
        """
        Apply realistic film grain texture to the input image.
        
        Args:
            image: Input image tensor
            strength: Intensity of the grain effect (0.01-1.0)
            grain_size: Size/scale of grain particles (0.1-5.0)
            seed: Random seed for reproducible results
            
        Returns:
            Tuple containing the processed image tensor
        """
        # Set random seed for reproducible results
        # Ensure seed is within valid range for NumPy (0 to 2^32 - 1)
        valid_seed = int(seed) % (2**32)
        np.random.seed(valid_seed)
        
        # Convert tensor to PIL for processing
        pil_image = tensor2pil(image)
        
        # Apply film grain
        grained_pil = self.add_film_grain(pil_image, strength=strength, grain_size=grain_size)
        
        # Convert back to tensor
        result_tensor = pil2tensor(grained_pil)
        
        return (result_tensor,)

    @staticmethod
    def add_film_grain(image: Image.Image, strength: float = 0.1, grain_size: float = 1.0, seed: int = 0) -> Image.Image:
        """
        Add realistic film grain texture to the image.
        
        This method creates a multi-layer grain pattern that mimics the characteristics
        of analog film. The grain intensity varies based on image luminance, with
        darker areas receiving more grain, similar to real film behavior.
        
        Args:
            image: Input PIL image
            strength: Intensity of the grain effect (0.01-1.0)
            grain_size: Size/scale of grain particles (0.1-5.0)
            seed: Random seed for reproducible grain patterns
            
        Returns:
            PIL image with film grain applied
        """
        # Set random seed for reproducible results
        # Ensure seed is within valid range for NumPy (0 to 2^32 - 1)
        valid_seed = int(seed) % (2**32)
        np.random.seed(valid_seed)
        
        # Convert PIL image to numpy array
        image_array = np.array(image).astype(np.float32) / 255.0
        height, width = image_array.shape[:2]
        
        # Create grain pattern using multiple noise layers for realism
        # Base grain layer - fine grain
        base_grain_size = max(1, int(grain_size))
        base_noise = np.random.normal(0, 1, (height // base_grain_size, width // base_grain_size))
        
        # Resize base noise to image dimensions using available interpolation
        try:
            from scipy.ndimage import zoom
            base_grain = zoom(base_noise, (base_grain_size, base_grain_size), order=1)
            
            # Ensure grain matches image dimensions exactly
            if base_grain.shape[0] != height or base_grain.shape[1] != width:
                base_grain = zoom(base_grain, (height / base_grain.shape[0], width / base_grain.shape[1]), order=1)
            
            # Add medium grain layer for texture variation
            medium_grain_size = max(1, int(grain_size * 2))
            if medium_grain_size < min(height, width) // 4:
                medium_noise = np.random.normal(0, 0.5, (height // medium_grain_size, width // medium_grain_size))
                medium_grain = zoom(medium_noise, (medium_grain_size, medium_grain_size), order=1)
                if medium_grain.shape[0] != height or medium_grain.shape[1] != width:
                    medium_grain = zoom(medium_grain, (height / medium_grain.shape[0], width / medium_grain.shape[1]), order=1)
            else:
                medium_grain = np.zeros((height, width))
                
        except ImportError:
            # Fallback to simple numpy-based upsampling if scipy is not available
            print("    Warning: scipy not available, using simple grain pattern")
            base_grain = np.repeat(np.repeat(base_noise, base_grain_size, axis=0), base_grain_size, axis=1)
            
            # Crop or pad to match exact dimensions
            if base_grain.shape[0] > height:
                base_grain = base_grain[:height, :]
            elif base_grain.shape[0] < height:
                pad_height = height - base_grain.shape[0]
                base_grain = np.pad(base_grain, ((0, pad_height), (0, 0)), mode='edge')
                
            if base_grain.shape[1] > width:
                base_grain = base_grain[:, :width]
            elif base_grain.shape[1] < width:
                pad_width = width - base_grain.shape[1]
                base_grain = np.pad(base_grain, ((0, 0), (0, pad_width)), mode='edge')
            
            # Simple medium grain layer
            medium_grain = np.random.normal(0, 0.2, (height, width))
        
        # Combine grain layers
        combined_grain = base_grain + medium_grain * 0.3
        
        # Normalize grain to prevent extreme values
        grain_std = np.std(combined_grain)
        if grain_std > 0:
            combined_grain = combined_grain / grain_std
        
        # Apply grain with luminance-based intensity (more grain in darker areas, like real film)
        if len(image_array.shape) == 3:  # Color image
            # Convert to luminance for grain intensity calculation
            luminance = 0.299 * image_array[:, :, 0] + 0.587 * image_array[:, :, 1] + 0.114 * image_array[:, :, 2]
            # More grain in darker areas (inverse luminance)
            grain_intensity = (1.0 - luminance) * 0.5 + 0.5
            
            # Apply grain to each channel
            grained_array = image_array.copy()
            for c in range(3):
                channel_grain = combined_grain * grain_intensity * strength * 0.1
                grained_array[:, :, c] = np.clip(image_array[:, :, c] + channel_grain, 0, 1)
        else:  # Grayscale image
            grain_intensity = (1.0 - image_array) * 0.5 + 0.5
            channel_grain = combined_grain * grain_intensity * strength * 0.1
            grained_array = np.clip(image_array + channel_grain, 0, 1)
        
        # Convert back to uint8 and PIL image
        grained_uint8 = (grained_array * 255.0).astype(np.uint8)
        grained_image = Image.fromarray(grained_uint8)
        return grained_image

# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Karma_Film_Grain": Karma_Film_Grain
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Karma_Film_Grain": "Karma Film Grain"
}