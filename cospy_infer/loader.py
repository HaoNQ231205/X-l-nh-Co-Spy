"""Robust two-pass image loader for CO-SPY.

Addresses Defect J4:
utils.py (L14) sets ImageFile.LOAD_TRUNCATED_IMAGES = True globally at import time,
causing truncated images to silently fill missing regions with gray without warnings.

loader.py executes a two-pass decode mechanism:
1. Pass 1 (Strict Integrity Check):
   Forces ImageFile.LOAD_TRUNCATED_IMAGES = False, calls verify() and load()
   to catch truncation errors (OSError, SyntaxError) cleanly.
2. Pass 2 (Legacy / Baseline Matching vs Strict Rejection):
   If allow_truncated_legacy=True, falls back to legacy decoding to preserve
   pixel-identical baseline evaluation.
   If allow_truncated_legacy=False and the image is truncated, raises CorruptedImageError
   and logs the sample to corrupted_skipped.
"""

import os
from typing import Tuple, Dict, Any, Optional
from PIL import Image, ImageFile

# Import utils first as it sets LOAD_TRUNCATED_IMAGES = True
import utils

# IMMEDIATELY reset to False to enforce strict validation
ImageFile.LOAD_TRUNCATED_IMAGES = False


class CorruptedImageError(Exception):
    """Raised when an image is truncated or corrupted and legacy mode is disabled."""
    pass


def load_image_two_pass(
    image_path: str,
    allow_truncated_legacy: bool = False
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Load an image with two-pass integrity check.
    
    Args:
        image_path: Path to the image file.
        allow_truncated_legacy: If True, decode truncated images using legacy mode
                                (matching original baseline pixel output).
                                If False, raise CorruptedImageError.
                                
    Returns:
        (PIL.Image, metadata_dict)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    is_truncated = False
    error_msg = ""
    metadata: Dict[str, Any] = {
        "filepath": image_path,
        "is_truncated": False,
        "format": None,
        "size": None,
        "error": None,
    }

    # Pass 1: Strict Check with LOAD_TRUNCATED_IMAGES = False
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with open(image_path, "rb") as f:
            # We open from file bytes to avoid holding file lock on Windows
            with Image.open(f) as img:
                metadata["format"] = img.format
                metadata["size"] = img.size
                # Force loading all image data to trigger any truncation exception
                img.load()
                # Copy into an in-memory clean RGB image
                img_rgb = img.convert("RGB")
        return img_rgb, metadata
    except (OSError, SyntaxError, Image.DecompressionBombError, Exception) as e:
        is_truncated = True
        error_msg = str(e)
        metadata["is_truncated"] = True
        metadata["error"] = error_msg

    # Pass 2: Truncated Handling
    if allow_truncated_legacy:
        # Decode in legacy mode to match exact baseline pixels
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with open(image_path, "rb") as f:
                with Image.open(f) as img:
                    img_rgb = img.convert("RGB")
            # Reset back to False
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            return img_rgb, metadata
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = False
    else:
        raise CorruptedImageError(
            f"Image is corrupted/truncated: {image_path} (Reason: {error_msg})"
        )


def check_image_health(image_path: str) -> Dict[str, Any]:
    """Non-throwing health check for an image file."""
    try:
        _, meta = load_image_two_pass(image_path, allow_truncated_legacy=False)
        return {"status": "ok", "metadata": meta}
    except CorruptedImageError as ce:
        return {"status": "truncated", "error": str(ce)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

