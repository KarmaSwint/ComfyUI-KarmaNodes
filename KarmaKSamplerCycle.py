"""
ComfyUI node for multi-cycle sampling with progressive upscaling.

This module provides a specialized KSampler node that performs multiple sampling cycles
with progressive upscaling between cycles. It supports both basic and model-based upscaling,
additive conditioning, and dynamic parameter adjustments between cycles.

The main component is the Karma_KSampler_Cycle class which handles the entire workflow
of sampling, upscaling, and re-sampling for high-quality image generation.
"""

import torch
from PIL import Image
import numpy as np
import folder_paths
import nodes
import comfy.samplers
import comfy.model_management
import comfy.utils
from skimage.filters import unsharp_mask

# Helper functions

def tensor2pil(image: torch.Tensor) -> Image.Image:
    """
    Convert a PyTorch tensor to a PIL Image.
    
    Args:
        image: Input tensor with shape [C, H, W] or [1, C, H, W]
        
    Returns:
        PIL Image converted from tensor
    """
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image to a PyTorch tensor.
    
    Args:
        image: Input PIL Image
        
    Returns:
        PyTorch tensor with shape [1, H, W, C]
    """
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class Karma_Image_Rescale:
    """
    Image rescaling utility class for high-quality image resizing operations.
    
    This class provides methods to rescale images with optional supersampling
    for improved quality. It supports both factor-based rescaling and
    resizing to specific dimensions.
    """
    
    def __init__(self):
        """Initialize the image rescaler."""
        pass
        
    def image_rescale(self, image: list[torch.Tensor], mode: str = "rescale", 
                     supersample: str = 'true', resampling: str = "lanczos", 
                     rescale_factor: float = 2, resize_width: int = 1024, 
                     resize_height: int = 1024) -> tuple[torch.Tensor]:
        """
        Rescale a batch of image tensors.
        
        Args:
            image: List of image tensors to rescale
            mode: Rescaling mode, either "rescale" (by factor) or "resize" (to dimensions)
            supersample: Whether to use supersampling ('true' or 'false')
            resampling: Resampling method ('nearest', 'bilinear', 'bicubic', or 'lanczos')
            rescale_factor: Factor to scale by when mode is "rescale"
            resize_width: Target width when mode is "resize"
            resize_height: Target height when mode is "resize"
            
        Returns:
            Tuple containing the batch of rescaled image tensors
        """
        # Process each image tensor in the batch
        resized_image_tensors = []
        
        for input_tensor in image:
            # Convert tensor to PIL for resizing
            input_pil = tensor2pil(input_tensor)
            
            # Apply resize operation
            resized_pil = self.apply_resize_image(
                input_pil, 
                mode, 
                supersample, 
                rescale_factor, 
                resize_width, 
                resize_height, 
                resampling
            )
            
            # Convert back to tensor and add to results
            resized_image_tensors.append(pil2tensor(resized_pil))
            
        # Combine all tensors into a batch
        batch_result = torch.cat(resized_image_tensors, dim=0)
        return (batch_result, )
        
    def apply_resize_image(self, image: Image.Image, mode: str = 'scale', 
                          supersample: str = 'true', factor: float = 2, 
                          width: int = 1024, height: int = 1024, 
                          resample: str = 'bicubic') -> Image.Image:
        """
        Apply resize operation to a single PIL image.
        
        Args:
            image: PIL Image to resize
            mode: 'rescale' to use factor, or 'scale'/'resize' to use dimensions
            supersample: Whether to use supersampling for higher quality
            factor: Scaling factor when mode is 'rescale'
            width: Target width when mode is not 'rescale'
            height: Target height when mode is not 'rescale'
            resample: Resampling method ('nearest', 'bilinear', 'bicubic', or 'lanczos')
            
        Returns:
            Resized PIL Image
        """
        current_width, current_height = image.size
        if mode == 'rescale':
            new_width, new_height = int(current_width * factor), int(current_height * factor)
        else:
            # Ensure dimensions are divisible by 8 for VAE compatibility
            new_width = width if width % 8 == 0 else width + (8 - width % 8)
            new_height = height if height % 8 == 0 else height + (8 - height % 8)
            
        resample_filters = {
            'nearest': 0,
            'bilinear': 2,
            'bicubic': 3,
            'lanczos': 1
        }
        
        if supersample == 'true':
            # Supersample by first upscaling to 8x the target size then downscaling
            image = image.resize((new_width * 8, new_height * 8), 
                               resample=Image.Resampling(resample_filters[resample]))
                               
        resized_image = image.resize((new_width, new_height), 
                                   resample=Image.Resampling(resample_filters[resample]))
        return resized_image

class Karma_KSampler_Cycle:
    """
    Advanced KSampler node that performs multiple sampling cycles with progressive upscaling.
    
    This node extends the standard KSampler functionality by implementing a multi-cycle
    approach where the image is progressively upscaled between sampling cycles. This allows
    for high-quality, high-resolution image generation with better detail preservation.
    
    Features:
        - Multiple sampling cycles with configurable parameters
        - Progressive upscaling between cycles (basic or model-based)
        - Primary and optional secondary model switching at specified cycle
        - Additive conditioning with strength scaling
        - Dynamic denoise strength adjustment
        - Optional sharpening between cycles
        - Automatic device management for optimal performance
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input parameters for the node.
        
        Returns:
            Dictionary containing required and optional input parameters
        """
        required = {
                # Core Models
                "primary_model": ("MODEL",),
                "vae": ("VAE",),
                "latent_image": ("LATENT", ),
                
                # Conditioning Inputs
                "positive": ("CONDITIONING", {"tooltip": "The main positive conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "The main negative conditioning."}),
                
                # Sampling Parameters
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "starting_denoise": ("FLOAT", {"default":1.0, "min":0.0, "max":1.0, "step":0.01}),
                "cycle_denoise": ("FLOAT", {"default":0.5, "min":0.0, "max":1.0, "step":0.01}),
                
                # Cycle Parameters
                "total_cycles": ("INT", {"default":2, "min":2, "max":12, "step":1}),
                "enable_denoise_scaling": ("BOOLEAN", {"default": True, "tooltip": "Enable automatic denoise strength scaling between cycles"}),
                "denoise_min_threshold": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Minimum denoise threshold to prevent going too low"}),
                "enable_steps_scaling": ("BOOLEAN", {"default": False, "tooltip": "Enable automatic steps adjustment between cycles"}),
                "steps_scaling_value": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1, "tooltip": "Amount to adjust steps by each cycle"}),
                "steps_adjustment_mode": (["decrement", "increment"], {"default": "decrement", "tooltip": "Whether to decrease or increase steps each cycle"}),
                "steps_threshold_mode": (["auto", "manual"], {"default": "auto", "tooltip": "Auto calculates threshold based on initial steps, manual uses fixed value"}),
                "steps_manual_threshold": ("INT", {"default": 5, "min": 1, "max": 200, "step": 1, "tooltip": "Manual threshold value (only used when threshold mode is manual)"}),
                "secondary_model_start_cycle": ("INT", {"default": 2, "min": 1, "max": 12, "step": 1, "tooltip": "Cycle at which to switch to secondary model (if provided)"}),
                
                # Upscaling Parameters
                "upscale_factor": ("FLOAT", {"default":2.0, "min":0.1, "max":8.0, "step":0.1}),
                "upscale_method": (["basic", "model", "latent"], {"default": "basic", "tooltip": "basic = PIL image upscale, model = upscale model, latent = direct latent space resize (fastest, no VAE decode/encode)"}),
                "scale_sampling": (["bilinear", "bicubic", "nearest", "lanczos"], {"tooltip": "Resampling method for image upscaling"}),
                "latent_upscale_method": (["bilinear", "bicubic", "nearest", "lanczos"], {"default": "lanczos", "tooltip": "Resampling method for latent space upscaling"}),
                "enable_gradual_upscaling": ("BOOLEAN", {"default": False, "tooltip": "Enable gradual upscaling with multiple intermediate steps"}),
                "gradual_upscale_steps": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1, "tooltip": "Number of gradual upscaling steps (only used when gradual upscaling is enabled)"}),
                
                # Additive Prompt Parameters
                "pos_add_strength": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01}),
                "enable_pos_strength_scaling": ("BOOLEAN", {"default": False, "tooltip": "Enable positive strength scaling between cycles"}),
                "pos_add_strength_cutoff": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "neg_add_strength": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01}),
                "enable_neg_strength_scaling": ("BOOLEAN", {"default": False, "tooltip": "Enable negative strength scaling between cycles"}),
                "neg_add_strength_cutoff": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                
                # Post-processing Parameters
                "sharpen_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "sharpen_radius": ("INT", {"default": 2, "min": 1, "max": 12, "step": 1}),
                
                # VAE Parameters
                "use_tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE processing for large images"}),
        }
        optional = {
                "secondary_model": ("MODEL", {"tooltip": "Optional secondary model to switch to at specified cycle"}),
                "upscale_model": ("UPSCALE_MODEL",),
                "pos_additive": ("CONDITIONING", {"tooltip": "Additional positive conditioning to blend in."}),
                "neg_additive": ("CONDITIONING", {"tooltip": "Additional negative conditioning to blend in."}),
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("LATENT", "VAE")
    RETURN_NAMES = ("latent(s)", "vae")
    FUNCTION = "sample"
    CATEGORY = "KarmaNodes/Sampling"

    def sample(self, primary_model, vae, positive, negative, 
                steps: int, seed: int, cfg: float, sampler_name: str, scheduler: str, 
                latent_image: dict, use_tiled_vae: bool, upscale_factor: float,
                total_cycles: int, starting_denoise: float, cycle_denoise: float, 
                enable_denoise_scaling: bool, denoise_min_threshold: float,
                scale_sampling: str, pos_add_strength: float, enable_pos_strength_scaling: bool, 
                pos_add_strength_cutoff: float, neg_add_strength: float, enable_neg_strength_scaling: bool, 
                neg_add_strength_cutoff: float, sharpen_strength: float, sharpen_radius: int, 
                enable_steps_scaling: bool, steps_scaling_value: int, steps_adjustment_mode: str,
                steps_threshold_mode: str, steps_manual_threshold: int, upscale_method: str, 
                latent_upscale_method: str, enable_gradual_upscaling: bool, gradual_upscale_steps: int,
                secondary_model_start_cycle: int, secondary_model=None, upscale_model=None, 
                pos_additive=None, neg_additive=None) -> tuple:
        """
        Execute the multi-cycle sampling process with progressive upscaling.
        
        This method performs the following steps in each cycle:
        1. Combine conditioning inputs (with optional additive conditioning)
        2. Run the sampler with current parameters
        3. Decode the latent to an image
        4. Upscale the image
        5. Re-encode to latent for the next cycle
        
        Args:
            primary_model: Primary diffusion model
            vae: VAE model for encoding/decoding
            positive: Main positive conditioning
            negative: Main negative conditioning
            steps: Number of sampling steps
            seed: Random seed for sampling
            cfg: Classifier-free guidance scale
            sampler_name: Name of the sampler to use
            scheduler: Name of the scheduler to use
            latent_image: Initial latent image
            use_tiled_vae: Whether to use tiled VAE processing
            upscale_factor: Total upscale factor to achieve
            total_cycles: Number of sampling cycles
            starting_denoise: Initial denoise strength
            cycle_denoise: Denoise strength for subsequent cycles
            enable_denoise_scaling: Whether to enable denoise scaling
            denoise_min_threshold: Minimum denoise threshold value
            scale_sampling: Resampling method for scaling
            pos_add_strength: Strength of positive additive conditioning
            enable_pos_strength_scaling: Whether to scale positive strength
            pos_add_strength_cutoff: Maximum positive additive strength
            neg_add_strength: Strength of negative additive conditioning
            enable_neg_strength_scaling: Whether to scale negative strength
            neg_add_strength_cutoff: Maximum negative additive strength
            sharpen_strength: Strength of sharpening filter
            sharpen_radius: Radius of sharpening filter
            enable_steps_scaling: Whether to adjust steps between cycles
            steps_scaling_value: Amount to adjust steps by
            steps_adjustment_mode: Direction of steps adjustment ("decrement" or "increment")
            steps_threshold_mode: How to calculate steps threshold ("auto" or "manual")
            steps_manual_threshold: Manual threshold value when steps_threshold_mode is "manual"
            upscale_method: Method for upscaling ("basic" or "model")
            latent_upscale_method: Resampling method for latent upscaling
            enable_gradual_upscaling: Whether to use gradual upscaling
            gradual_upscale_steps: Number of gradual upscaling steps
            secondary_model_start_cycle: Cycle number at which to switch to secondary model
            secondary_model: Optional secondary diffusion model to switch to
            upscale_model: Upscale model to use (required if upscale_method is "model")
            pos_additive: Additional positive conditioning to blend in (optional)
            neg_additive: Additional negative conditioning to blend in (optional)
            
        Returns:
            Tuple containing the final latent image and the VAE
        """
        # Use provided conditioning inputs directly
        
        # --- Setup --- 
        num_sampling_cycles = total_cycles if steps >= total_cycles else steps
        model_switched = False

        # Calculate steps threshold based on mode
        if steps_threshold_mode == "manual":
            steps_min_threshold = steps_manual_threshold
        else:  # auto mode
            if steps_adjustment_mode == 'decrement':
                steps_min_threshold = max(1, steps // 4)  # Minimum threshold for decrement mode
            else:  # increment mode
                steps_min_threshold = steps * 4  # Maximum threshold for increment mode

        # Validate upscale method and model
        if upscale_method == "model" and upscale_model is None:
            print("WARNING: Upscale method set to 'model' but no upscale model provided. Falling back to basic method.")
            upscale_method = "basic"
        
        # Calculate upscale factors - use total_cycles for upscaling distribution
        upscale_cycles_needed = total_cycles - 1  # Number of upscale operations (cycles - 1)
        
        # Calculate the absolute final target dimensions based on the initial latent
        initial_latent_height, initial_latent_width = latent_image['samples'].shape[2], latent_image['samples'].shape[3]
        initial_image_height, initial_image_width = initial_latent_height * 8, initial_latent_width * 8
        final_target_width = int(initial_image_width * upscale_factor)
        final_target_height = int(initial_image_height * upscale_factor)
        
        if upscale_method == "model" and upscale_model is not None:
            # For model upscaling, we'll use the model's native scale then adjust to target
            model_scale = upscale_model.scale
            if upscale_cycles_needed > 0:
                # Calculate the target scale per cycle to reach total upscale_factor
                target_scale_per_cycle = upscale_factor ** (1 / upscale_cycles_needed)
                print(f"INFO: Target scale per cycle: {target_scale_per_cycle:.2f}x, Model native scale: {model_scale}x")
                # We'll use the model's native scale and then adjust with PIL if needed
                current_upscale_factor = target_scale_per_cycle  # This is what we want to achieve
                model_native_scale = model_scale  # This is what the model provides
            else:
                current_upscale_factor = 1.0
                model_native_scale = 1.0
        else:
            # For basic upscaling, calculate factor per cycle to reach total target
            current_upscale_factor = upscale_factor ** (1 / upscale_cycles_needed) if upscale_cycles_needed > 0 else 1.0
        
        latent_image_result = None # Initialize to prevent potential UnboundLocalError if num_sampling_cycles is 0
        print(f"--- Initial Setup ---")
        print(f"  Upscale Cycles: {total_cycles}, Total Steps: {steps}")
        print(f"  Number of Sampling Cycles: {num_sampling_cycles}")
        print(f"  Target Total Upscale Factor: {upscale_factor}")
        print(f"  Initial Image Size: {initial_image_width}x{initial_image_height}")
        print(f"  Final Target Size: {final_target_width}x{final_target_height}")
        print(f"  Upscale Method: {upscale_method}")
        if upscale_method == "model" and upscale_model is not None:
            print(f"  Upscale Model Native Scale: {upscale_model.scale}x")
            if upscale_cycles_needed > 0:
                print(f"  Target Scale Per Cycle: {current_upscale_factor:.4f}x")
                actual_total = current_upscale_factor ** upscale_cycles_needed
                print(f"  Actual Total Upscale: {actual_total:.2f}x")
        else:
            print(f"  Factor Per Cycle: {current_upscale_factor:.4f}")
        print(f"  Upscale Operations: {upscale_cycles_needed} (between {total_cycles} cycles)")
        print(f"  Tiled VAE: {use_tiled_vae}")
        print(f"  Denoise Scaling Enabled: {enable_denoise_scaling}")
        if enable_denoise_scaling:
            print(f"    Denoise Min Threshold: {denoise_min_threshold}")
        print(f"  Steps Scaling Enabled: {enable_steps_scaling}")
        if enable_steps_scaling:
            print(f"    Steps Scaling Mode: {steps_adjustment_mode}")
            print(f"    Steps Scaling Value: {steps_scaling_value}")
            print(f"    Steps Threshold Mode: {steps_threshold_mode}")
            print(f"    Steps Min Threshold: {steps_min_threshold}")
        print(f"  Gradual Upscaling Enabled: {enable_gradual_upscaling}")
        if enable_gradual_upscaling:
            print(f"    Gradual Upscale Steps: {gradual_upscale_steps}")
        print(f"  Scale Sampling Method: {scale_sampling}")
        print(f"  Latent Upscale Method: {latent_upscale_method}")
        if secondary_model is not None:
            print(f"  Secondary Model: Enabled (switches at cycle {secondary_model_start_cycle})")
        else:
            print(f"  Secondary Model: Disabled")
        print(f"---------------------")

        # --- Main Sampling Loop --- 
        for i in range(num_sampling_cycles):
            # Print cycle info
            print(f"==== Main Cycle Pass {i+1}/{num_sampling_cycles} ====")
            
            # --- Model Selection Logic ---
            # Determine which model to use for this cycle
            current_cycle = i + 1  # Convert to 1-based indexing for user clarity
            if secondary_model is not None and current_cycle >= secondary_model_start_cycle:
                active_model = secondary_model
                if not model_switched:
                    print(f"  Switching to secondary model at cycle {current_cycle}")
                    model_switched = True
            else:
                active_model = primary_model
            
            # --- Get Target Device ---
            # Assume the model's device is the target device
            target_device = active_model.load_device
            print(f"  Target device for tensors: {target_device}")
            print(f"  Using {'secondary' if active_model == secondary_model else 'primary'} model")

            # Update latent_image for passes > 0
            if i > 0:
                print(f"  Updating latent_image for pass {i+1} from previous latent_image_result")
                latent_image = latent_image_result
                # Debug print for the input latent to the sampler
                if isinstance(latent_image, dict) and 'samples' in latent_image:
                    latent_samples_tensor = latent_image['samples']
                    print(f"  Input latent type for ksampler: {type(latent_image)}, shape: {latent_samples_tensor.shape}, device: {latent_samples_tensor.device}")
                    # --- Check for NaNs/Infs before sampling ---
                    if torch.isnan(latent_samples_tensor).any(): print("  FATAL WARNING: Input latent (pre-sample) contains NaNs!")
                    if torch.isinf(latent_samples_tensor).any(): print("  FATAL WARNING: Input latent (pre-sample) contains Infs!")
                    # --- Ensure latent is on the correct device ---
                    if latent_samples_tensor.device != target_device:
                        print(f"    Moving latent samples from {latent_samples_tensor.device} to {target_device}...")
                        latent_image['samples'] = latent_samples_tensor.to(target_device)
                        print(f"    Latent samples now on device: {latent_image['samples'].device}")
                else:
                    print(f"  WARNING: Input latent for ksampler is unexpected format: {type(latent_image)}")
            else:
                 # Debug print for the initial input latent
                if isinstance(latent_image, dict) and 'samples' in latent_image:
                    latent_samples_tensor = latent_image['samples']
                    print(f"  Initial input latent type: {type(latent_image)}, shape: {latent_samples_tensor.shape}, device: {latent_samples_tensor.device}")
                     # --- Ensure initial latent is on the correct device ---
                    if latent_samples_tensor.device != target_device:
                        print(f"    Moving initial latent samples from {latent_samples_tensor.device} to {target_device}...")
                        latent_image['samples'] = latent_samples_tensor.to(target_device)
                        print(f"    Initial latent samples now on device: {latent_image['samples'].device}")
                else:
                    print(f"  WARNING: Initial input latent is unexpected format: {type(latent_image)}")

            if enable_denoise_scaling:
                denoise = (
                    ( round(cycle_denoise * (2 ** (-(i-1))), 2) if i > 0 else cycle_denoise )
                    if i > 0 else round(starting_denoise, 2)
                )
            else:
                denoise = round((cycle_denoise if i > 0 else starting_denoise), 2)

            if denoise < denoise_min_threshold and enable_denoise_scaling:
                denoise = denoise_min_threshold

            # No secondary_model/secondary_start_cycle in this version

            if enable_steps_scaling and i > 0:
                original_steps = steps
                # Adjust steps based on the adjustment mode
                if steps_adjustment_mode == 'increment':
                    steps = steps + steps_scaling_value
                    # For increment mode, cap at maximum threshold
                    if steps > steps_min_threshold:
                        steps = steps_min_threshold
                        print(f"  Steps scaling: {original_steps} + {steps_scaling_value} = {original_steps + steps_scaling_value} (capped to {steps})")
                    else:
                        print(f"  Steps scaling: {original_steps} + {steps_scaling_value} = {steps}")
                else:  # decrement mode
                    steps = steps - steps_scaling_value
                    # For decrement mode, ensure we don't go below minimum threshold
                    if steps < steps_min_threshold:
                        steps = steps_min_threshold
                        print(f"  Steps scaling: {original_steps} - {steps_scaling_value} = {original_steps - steps_scaling_value} (capped to {steps})")
                    else:
                        print(f"  Steps scaling: {original_steps} - {steps_scaling_value} = {steps}")

            print("Steps:", steps)
            print("Denoise:", denoise)

            if pos_additive:
                pos_strength = 0.0 if i <= 0 else pos_add_strength
                pos_strength = (
                    ( round(pos_add_strength * (2 ** (i-1)), 2)
                    if i > 0
                    else pos_add_strength )
                    if enable_pos_strength_scaling
                    else pos_add_strength
                )
                pos_strength = (
                    pos_add_strength_cutoff
                    if pos_strength > pos_add_strength_cutoff
                    else pos_strength
                )
                comb = nodes.ConditioningAverage()
                positive = comb.addWeighted(pos_additive, positive, pos_strength)[0]
                print("Positive Additive Strength:", pos_strength)

            if neg_additive:
                neg_strength = 0.0 if i <= 0 else neg_add_strength
                neg_strength = (
                    ( round(neg_add_strength * (2 ** (i-1)), 2)
                    if i > 0
                    else neg_add_strength )
                    if enable_neg_strength_scaling
                    else neg_add_strength
                )
                neg_strength = (
                    neg_add_strength_cutoff
                    if neg_strength > neg_add_strength_cutoff
                    else neg_strength
                )
                comb = nodes.ConditioningAverage()
                negative = comb.addWeighted(neg_additive, negative, neg_strength)[0]
                print("Negative Additive Strength:", neg_strength)

            # Run KSampler
            print(f"  Running common_ksampler (Pass {i+1})...")
            samples = nodes.common_ksampler(
                active_model,
                seed + i, # Increment seed per main cycle
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image, # Input latent (already checked for device)
                denoise=denoise,
            )

            # Debug print for the sampler output
            if isinstance(samples, tuple) and len(samples) > 0 and isinstance(samples[0], dict) and 'samples' in samples[0]:
                 sampled_latent_tensor = samples[0]['samples']
                 print(f"  Sampler output type: {type(samples[0])}, shape: {sampled_latent_tensor.shape}, device: {sampled_latent_tensor.device}")
                 # Check for NaNs/Infs after sampling, just in case
                 if torch.isnan(sampled_latent_tensor).any(): print("  FATAL WARNING: Sampled latent contains NaNs!")
                 if torch.isinf(sampled_latent_tensor).any(): print("  FATAL WARNING: Sampled latent contains Infs!")
            else:
                 print(f"  WARNING: Sampled output is unexpected format: {type(samples)}")


            # Upscale between cycles (only during total_cycles, not all sampling cycles)
            if i < total_cycles - 1:
                
                # --- Latent Space Upscaling (fastest path, no VAE decode/encode) ---
                if upscale_method == "latent":
                    print(f"  Performing LATENT Upscale for next cycle (Pass {i+1}/{total_cycles})...")
                    try:
                        current_latent = samples[0]['samples'].to(target_device)
                        _, c, lh, lw = current_latent.shape
                        
                        # Calculate target pixel dimensions using same progression as image upscaling
                        upscale_progress = (i + 1) / upscale_cycles_needed
                        cycle_target_width = int(initial_image_width + (final_target_width - initial_image_width) * upscale_progress)
                        cycle_target_height = int(initial_image_height + (final_target_height - initial_image_height) * upscale_progress)
                        
                        # Convert pixel dimensions to latent dimensions (latent is 1/8 of pixel space)
                        latent_target_width = cycle_target_width // 8
                        latent_target_height = cycle_target_height // 8
                        
                        # Ensure minimum latent size
                        latent_target_width = max(latent_target_width, 1)
                        latent_target_height = max(latent_target_height, 1)
                        
                        # Map latent_upscale_method to torch interpolation mode
                        interp_mode = latent_upscale_method
                        if interp_mode == "lanczos":
                            interp_mode = "bicubic"  # Closest torch equivalent (lanczos not supported by F.interpolate)
                        
                        print(f"    Latent: {lw}x{lh} -> {latent_target_width}x{latent_target_height} (pixel target: {cycle_target_width}x{cycle_target_height})")
                        print(f"    Interpolation mode: {latent_upscale_method}" + (" (mapped to bicubic)" if latent_upscale_method == "lanczos" else ""))
                        
                        upscaled_latent = torch.nn.functional.interpolate(
                            current_latent,
                            size=(latent_target_height, latent_target_width),
                            mode=interp_mode,
                            antialias=interp_mode in ("bilinear", "bicubic")
                        )
                        
                        latent_image_result = {"samples": upscaled_latent}
                        print(f"    Upscaled latent shape: {upscaled_latent.shape}")
                        
                        # Sanity checks
                        if torch.isnan(upscaled_latent).any(): print("    WARNING: Upscaled latent contains NaNs!")
                        if torch.isinf(upscaled_latent).any(): print("    WARNING: Upscaled latent contains Infs!")
                        
                    except Exception as e:
                        print(f"\n!!! ERROR: Failed during latent upscaling: {e}")
                        print("!!! ERROR: Passing latent directly without upscaling.")
                        latent_image_result = samples[0]
                    continue  # Skip to next cycle (no decode/encode needed)
                
                # --- Pixel Space Upscaling (basic/model) ---
                print(f"  Performing Image Upscale for next cycle (Pass {i+1}/{total_cycles})...")
                
                # 1. Decode Latent to Image
                print("    1. Decoding latent to image...")
                try:
                    # Ensure sampled latent is on the correct device before decode
                    sampled_latent_tensor = samples[0]['samples'].to(target_device) 
                    if use_tiled_vae:
                        decoded_image_tensor = vae.decode_tiled(sampled_latent_tensor)
                    else:
                        decoded_image_tensor = vae.decode(sampled_latent_tensor)
                    print(f"       Decoded image tensor shape: {decoded_image_tensor.shape}, device: {decoded_image_tensor.device}")
                except Exception as e:
                    print(f"\\n!!! ERROR: Failed during VAE Decode for upscaling: {e}")
                    print("!!! ERROR: Skipping upscale, passing latent directly. This might cause issues!\\")
                    latent_image_result = samples[0] # Pass original latent if decode fails
                    continue # Skip to next iteration

                # 2. Perform Gradual Upscaling
                if enable_gradual_upscaling:
                    # Gradual upscaling with multiple intermediate steps
                    num_upscale_steps = gradual_upscale_steps
                    step_upscale_factor = current_upscale_factor ** (1 / num_upscale_steps)
                    print(f"    2. Performing gradual upscaling: {num_upscale_steps} steps, {step_upscale_factor:.4f}x per step")
                    
                    # Calculate the target for this upscale operation based on progression toward absolute final target
                    # We're in upscale operation (i+1) out of upscale_cycles_needed total operations
                    upscale_progress = (i + 1) / upscale_cycles_needed  # Progress through upscale operations (0 to 1)
                    cycle_target_width = int(initial_image_width + (final_target_width - initial_image_width) * upscale_progress)
                    cycle_target_height = int(initial_image_height + (final_target_height - initial_image_height) * upscale_progress)
                    
                    initial_pil = tensor2pil(decoded_image_tensor)
                    initial_width, initial_height = initial_pil.size
                    print(f"       Initial: {initial_width}x{initial_height}, Cycle target: {cycle_target_width}x{cycle_target_height}")
                    print(f"       Upscale progress: {upscale_progress:.2f} toward final {final_target_width}x{final_target_height}")
                    
                    current_image_tensor = decoded_image_tensor
                    for step in range(num_upscale_steps):
                        if step == num_upscale_steps - 1:
                            # For the final step, use exact cycle target dimensions
                            current_pil = tensor2pil(current_image_tensor)
                            current_width, current_height = current_pil.size
                            step_factor = cycle_target_width / current_width  # Calculate exact factor needed
                            print(f"       Upscale step {step+1}/{num_upscale_steps} (final step, exact factor: {step_factor:.4f}x)")
                            print(f"       Final step: {current_width}x{current_height} -> {cycle_target_width}x{cycle_target_height}")
                        else:
                            step_factor = step_upscale_factor
                            print(f"       Upscale step {step+1}/{num_upscale_steps} (factor: {step_factor:.4f}x)")
                        
                        current_image_tensor = self.perform_single_upscale_step(
                            current_image_tensor, step_factor, upscale_method, upscale_model, scale_sampling, 
                            is_final_step=(step == num_upscale_steps - 1),
                            final_target_width=cycle_target_width if step == num_upscale_steps - 1 else None,
                            final_target_height=cycle_target_height if step == num_upscale_steps - 1 else None
                        )
                    upscaled_image_tensor = current_image_tensor
                else:
                    # Single upscaling step - calculate target based on upscale operation progress
                    upscale_progress = (i + 1) / upscale_cycles_needed  # Progress through upscale operations (0 to 1)
                    cycle_target_width = int(initial_image_width + (final_target_width - initial_image_width) * upscale_progress)
                    cycle_target_height = int(initial_image_height + (final_target_height - initial_image_height) * upscale_progress)
                    
                    initial_pil = tensor2pil(decoded_image_tensor)
                    initial_width, initial_height = initial_pil.size
                    actual_factor = cycle_target_width / initial_width
                    print(f"    2. Performing single upscaling step (target factor: {actual_factor:.4f}x)")
                    print(f"       {initial_width}x{initial_height} -> {cycle_target_width}x{cycle_target_height}")
                    
                    upscaled_image_tensor = self.perform_single_upscale_step(
                        decoded_image_tensor, actual_factor, upscale_method, upscale_model, scale_sampling,
                        is_final_step=True, final_target_width=cycle_target_width, final_target_height=cycle_target_height
                    )

                # 3. Apply Sharpening if enabled
                if sharpen_strength > 0.0:
                   print(f"    3. Applying sharpening (Strength: {sharpen_strength}, Radius: {sharpen_radius})...")
                   # Sharpening helper works on PIL images
                   upscaled_pil_for_sharpen = tensor2pil(upscaled_image_tensor)
                   sharpened_pil = self.unsharp_filter(upscaled_pil_for_sharpen, radius=sharpen_radius, amount=sharpen_strength)
                   upscaled_image_tensor = pil2tensor(sharpened_pil) # Convert back
                   print(f"       Sharpened image tensor shape: {upscaled_image_tensor.shape}")

                # 4. Encode Upscaled Image back to Latent
                print("    4. Encoding upscaled image back to latent...")
                try:
                    # Ensure upscaled image tensor is on the target device for VAE encode
                    upscaled_image_tensor = upscaled_image_tensor.to(target_device)
                    # Crop pixels if needed before encode
                    cropped_pixels = self.vae_encode_crop_pixels(upscaled_image_tensor)
                    if use_tiled_vae:
                        latent_image_result = {"samples": vae.encode_tiled(cropped_pixels[:,:,:,:3])}
                    else:
                        latent_image_result = {"samples": vae.encode(cropped_pixels[:,:,:,:3])}
                    print(f"       Re-encoded latent shape: {latent_image_result['samples'].shape}, device: {latent_image_result['samples'].device}")
                    
                    # Final check for NaNs/Infs after the full upscale process
                    if torch.isnan(latent_image_result['samples']).any(): print("  FATAL WARNING: Re-encoded latent contains NaNs!")
                    if torch.isinf(latent_image_result['samples']).any(): print("  FATAL WARNING: Re-encoded latent contains Infs!")
                        
                except Exception as e:
                    print(f"\\n!!! ERROR: Failed during VAE Encode after upscaling: {e}")
                    print("!!! ERROR: Skipping upscale, passing latent directly. This might cause issues!\\")
                    latent_image_result = samples[0] # Pass original latent if encode fails
                    continue # Skip to next iteration
                    
            else: # Last iteration
                latent_image_result = samples[0]
                print(f"  Last iteration, latent_image_result assigned from sampler output.")

        # --- Final Device Check ---
        final_latent = latent_image_result # Get the result after main loop
        if isinstance(final_latent, dict) and 'samples' in final_latent:
            final_samples_tensor = final_latent['samples']
            # Use model device as target
            try:
                target_device = active_model.load_device
                print(f"Final check: Output latent device is {final_samples_tensor.device}, target is {target_device}")
                if final_samples_tensor.device != target_device:
                    print(f"  Moving final output latent samples from {final_samples_tensor.device} to {target_device}...")
                    final_latent['samples'] = final_samples_tensor.to(target_device)
                    print(f"  Final latent samples now on device: {final_latent['samples'].device}")
            except AttributeError:
                 print("Warning: Could not determine model device for final check.") # Handle cases where model might not have load_device
        else:
             print("WARNING: Final latent output is not in expected dict format for device check.")

        # Handle case where loop didn't run (num_sampling_cycles=0?)
        if final_latent is None:
            print("WARNING: Main loop did not run or produce a result. Returning input latent.")
            final_latent = latent_image 

        return (final_latent, vae)

    def perform_single_upscale_step(self, image_tensor: torch.Tensor, upscale_factor: float, 
                                   upscale_method: str, upscale_model, scale_sampling: str, 
                                   is_final_step: bool = False, final_target_width: int = None, 
                                   final_target_height: int = None) -> torch.Tensor:
        """
        Perform a single upscaling step on an image tensor.
        
        This method handles both model-based and basic upscaling methods.
        For model-based upscaling, it first uses the model's native scale
        and then adjusts to the exact target dimensions if needed.
        
        Args:
            image_tensor: Input image tensor to upscale
            upscale_factor: Factor to upscale by
            upscale_method: Method to use ("model" or "basic")
            upscale_model: Upscale model to use (required if method is "model")
            scale_sampling: Resampling method for basic upscaling
            is_final_step: Whether this is the final upscaling step
            final_target_width: Exact target width for final step
            final_target_height: Exact target height for final step
            
        Returns:
            Upscaled image tensor
        """
        try:
            if upscale_method == "model" and upscale_model is not None:
                # Calculate the target size
                input_pil = tensor2pil(image_tensor)
                input_width, input_height = input_pil.size
                
                if is_final_step and final_target_width is not None and final_target_height is not None:
                    # Use exact final target dimensions
                    target_width = final_target_width
                    target_height = final_target_height
                else:
                    # Calculate based on upscale factor
                    target_width = int(input_width * upscale_factor)
                    target_height = int(input_height * upscale_factor)
                
                print(f"         Input size: {input_width}x{input_height}, Target size: {target_width}x{target_height}")
                
                # First, upscale with the model at its native scale
                model_upscaled_tensor = self.upscale_with_model(upscale_model, image_tensor)
                
                # Always adjust to the exact target size using PIL
                model_upscaled_pil = tensor2pil(model_upscaled_tensor)
                model_width, model_height = model_upscaled_pil.size
                print(f"         Model output size: {model_width}x{model_height}")
                
                # Resize to exact target dimensions
                if model_width != target_width or model_height != target_height:
                    print(f"         Adjusting from {model_width}x{model_height} to {target_width}x{target_height}")
                    adjusted_pil = model_upscaled_pil.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)
                    return pil2tensor(adjusted_pil)
                else:
                    print(f"         Model output matches target size exactly")
                    return model_upscaled_tensor
            else:
                # Use Karma_Image_Rescale helper class from this file
                rescaler = Karma_Image_Rescale()
                # Need to convert tensor to PIL temporarily for the helper
                pil_image = tensor2pil(image_tensor)
                upscaled_pil = rescaler.apply_resize_image(pil_image, mode='rescale', supersample='true', factor=upscale_factor, resample=scale_sampling)
                # Convert back to tensor
                return pil2tensor(upscaled_pil)
        except Exception as e:
            print(f"ERROR in perform_single_upscale_step: {e}")
            return image_tensor  # Return original if upscaling fails

    @staticmethod
    def vae_encode_crop_pixels(pixels: torch.Tensor) -> torch.Tensor:
        """
        Crop pixel tensor to dimensions divisible by 8 for VAE compatibility.
        
        Args:
            pixels: Input pixel tensor with shape [B, H, W, C]
            
        Returns:
            Cropped pixel tensor with dimensions divisible by 8
        """
        target_height = (pixels.shape[1] // 8) * 8
        target_width = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != target_height or pixels.shape[2] != target_width:
            height_offset = (pixels.shape[1] % 8) // 2
            width_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, height_offset:target_height + height_offset, width_offset:target_width + width_offset, :]
        return pixels

    @staticmethod
    def unsharp_filter(image: Image.Image, radius: int = 2, amount: float = 1.0) -> Image.Image:
        """
        Apply unsharp mask filter to enhance image details.
        
        Args:
            image: Input PIL image
            radius: Blur radius for the unsharp mask
            amount: Strength of the sharpening effect
            
        Returns:
            Sharpened PIL image
        """
        # Convert PIL image to normalized numpy array (0-1 range)
        image_array = np.array(image)
        normalized_array = image_array / 255.0
        
        # Apply unsharp mask filter
        sharpened_array = unsharp_mask(normalized_array, radius=radius, amount=amount, channel_axis=2)
        
        # Convert back to uint8 range (0-255)
        sharpened_uint8 = (sharpened_array * 255.0).astype(np.uint8)
        
        # Convert back to PIL image
        sharpened_image = Image.fromarray(sharpened_uint8)
        return sharpened_image

    @staticmethod
    def upscale_with_model(upscale_model, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Upscale image tensor using an upscale model with tiled processing.
        
        This method handles memory management and uses tiled processing to
        avoid out-of-memory errors with large images.
        
        Args:
            upscale_model: The upscale model to use
            image_tensor: Input image tensor to upscale
            
        Returns:
            Upscaled image tensor
            
        Raises:
            OOM_EXCEPTION: If tiling cannot prevent out-of-memory error
        """
        device = comfy.model_management.get_torch_device()
        
        # Calculate memory requirements and free memory if needed
        memory_required = comfy.model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image_tensor.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += image_tensor.nelement() * image_tensor.element_size()
        comfy.model_management.free_memory(memory_required, device)

        # Move model and input to the target device
        upscale_model.to(device)
        input_tensor_rearranged = image_tensor.movedim(-1,-3).to(device)

        # Start with large tiles and reduce if OOM occurs
        tile_size = 512
        tile_overlap = 32

        out_of_memory = True
        while out_of_memory:
            try:
                total_steps = input_tensor_rearranged.shape[0] * comfy.utils.get_tiled_scale_steps(
                    input_tensor_rearranged.shape[3], 
                    input_tensor_rearranged.shape[2], 
                    tile_x=tile_size, 
                    tile_y=tile_size, 
                    overlap=tile_overlap
                )
                progress_bar = comfy.utils.ProgressBar(total_steps)
                upscaled_tensor = comfy.utils.tiled_scale(
                    input_tensor_rearranged, 
                    lambda a: upscale_model(a), 
                    tile_x=tile_size, 
                    tile_y=tile_size, 
                    overlap=tile_overlap, 
                    upscale_amount=upscale_model.scale, 
                    pbar=progress_bar
                )
                out_of_memory = False
            except comfy.model_management.OOM_EXCEPTION as e:
                # Reduce tile size and try again
                tile_size //= 2
                if tile_size < 128:
                    raise e  # If tiles are too small, propagate the error

        # Move model back to CPU to free GPU memory
        upscale_model.to("cpu")
        result_tensor = torch.clamp(upscaled_tensor.movedim(-3,-1), min=0, max=1.0)
        return result_tensor

# Register the node for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Karma-KSampler-Cycle": Karma_KSampler_Cycle,
}

# Define display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "Karma-KSampler-Cycle": "Karma KSampler Cycle",
}