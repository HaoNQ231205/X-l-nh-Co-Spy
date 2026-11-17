"""Geometric coordinate transformations and spatial mapping for CAM heatmaps.

Converts:
1. Artifact CAM grid (28x28) -> crop coordinates (224x224) -> original image space.
2. Semantic Grad-CAM grid (27x27) -> crop coordinates (384x384) -> original image space.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2


def upsample_heatmap(heatmap: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Upsample a low-resolution heatmap (e.g. 28x28 or 27x27) to target (width, height) using bilinear interpolation."""
    target_w, target_h = target_size
    return cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def map_crop_to_original(
    crop_heatmap: np.ndarray,
    original_size: Tuple[int, int],
    crop_size: int = 224,
    load_size: int = 256
) -> np.ndarray:
    """Map a crop-level heatmap back onto the original image coordinate frame.
    
    Reverse pipeline of:
    Original (W_orig, H_orig) -> Resize(load_size) -> CenterCrop(crop_size)
    """
    orig_w, orig_h = original_size

    # Reconstruct load_size image heatmap (with zeros outside center crop)
    resized_map = np.zeros((load_size, load_size), dtype=np.float32)
    start_x = (load_size - crop_size) // 2
    start_y = (load_size - crop_size) // 2
    resized_map[start_y:start_y + crop_size, start_x:start_x + crop_size] = crop_heatmap

    # Upsample to original image aspect and dimensions
    full_heatmap = cv2.resize(resized_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return full_heatmap


def compute_receptive_field_profile() -> Dict[str, Any]:
    """Receptive field analysis of the Artifact branch (Section 2 specification)."""
    return {
        "stride": 8,
        "feature_map_dimensions": [28, 28],
        "input_resolution": [224, 224],
        "effective_stride_per_pixel": 8.0,
        "receptive_field_extent": ">> 8x8 pixels",
        "description": (
            "Because input passes through SD v1.4 VAE encoder/decoder (containing residual 3x3 convolutions "
            "and self-attention in the 28x28 latent space) before entering ResNet layers, "
            "each feature cell in layer2 integrates information from a wide spatial context."
        )
    }

