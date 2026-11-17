"""JPEG analysis probe for CO-SPY.

Inspects JPEG compression properties, quantization tables, grid phase, and OOD status:
- estimated_jpeg_quality
- quant_table_type: "standard_ijg_table" vs "custom_table"
- 8x8 DCT grid phase detection
- Out-of-Distribution (OOD) flag (q < 70 for sd-v1_4 artifact branch; any JPEG for progan NPR)
- Generates JSON-serializable input_report
"""

import os
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image

# Standard IJG 50% Luminance Quantization Table
STD_LUMINANCE_QUANT_50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

# Standard IJG 50% Chrominance Quantization Table
STD_CHROMINANCE_QUANT_50 = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32)


def generate_ijg_table(base_table: np.ndarray, quality: int) -> np.ndarray:
    """Scale standard IJG 50 table to target quality according to standard IJG formula."""
    q = max(1, min(100, quality))
    if q < 50:
        scale = 5000.0 / q
    else:
        scale = 200.0 - 2.0 * q

    scaled = np.floor((base_table * scale + 50.0) / 100.0)
    scaled = np.clip(scaled, 1.0, 255.0)
    return scaled


def estimate_jpeg_quality_from_quant_table(
    quantization_dict: Dict[int, Any]
) -> Tuple[Optional[int], str]:
    """Estimate JPEG quality factor and table type from Pillow image quantization tables.
    
    Returns:
        (estimated_quality, quant_table_type)
        where quant_table_type is "standard_ijg_table", "custom_table", or "unknown"
    """
    if not quantization_dict or 0 not in quantization_dict:
        return None, "unknown"

    table_0 = quantization_dict[0]
    # Pillow quantization table can be a list or array of 64 elements
    tbl_arr = np.array(table_0, dtype=np.float32).reshape(8, 8)

    best_q = 50
    min_diff = float("inf")

    # Search closest matching IJG quality in 1..100
    for q in range(1, 101):
        ref_tbl = generate_ijg_table(STD_LUMINANCE_QUANT_50, q)
        diff = np.mean(np.abs(tbl_arr - ref_tbl))
        if diff < min_diff:
            min_diff = diff
            best_q = q

    # If average difference is <= 1.5, classify as standard IJG table
    if min_diff <= 1.5:
        table_type = "standard_ijg_table"
    else:
        table_type = "custom_table"

    return best_q, table_type


def detect_grid_phase(image: Image.Image) -> Tuple[int, int]:
    """Detect 8x8 DCT grid phase offset (phase_x, phase_y) in [0..7].
    
    Analyzes pixel intensity difference periodicity along block boundaries.
    """
    try:
        gray = np.array(image.convert("L"), dtype=np.float32)
        h, w = gray.shape
        if h < 64 or w < 64:
            return 0, 0

        # Horizontal block difference across modulo 8 columns
        diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
        phase_scores_x = [0.0] * 8
        for k in range(8):
            cols = diff_h[:, k::8]
            phase_scores_x[k] = float(np.mean(cols)) if cols.size > 0 else 0.0

        # Vertical block difference across modulo 8 rows
        diff_v = np.abs(gray[1:, :] - gray[:-1, :])
        phase_scores_y = [0.0] * 8
        for k in range(8):
            rows = diff_v[k::8, :]
            phase_scores_y[k] = float(np.mean(rows)) if rows.size > 0 else 0.0

        # Phase with highest boundary discontinuity
        phase_x = int(np.argmax(phase_scores_x))
        phase_y = int(np.argmax(phase_scores_y))
        return phase_x, phase_y
    except Exception:
        return 0, 0


def _inspect_pil_image(
    img: Image.Image,
    filepath: Optional[str],
    train_dataset: str = "sd-v1_4",
) -> Dict[str, Any]:
    fmt = (getattr(img, "format", None) or "UNKNOWN").upper()
    width, height = img.size
    is_jpeg = fmt in ("JPEG", "JPG")

    est_q = None
    table_type = "not_applicable"
    phase_x, phase_y = 0, 0

    if is_jpeg:
        quant = getattr(img, "quantization", None)
        if quant:
            est_q, table_type = estimate_jpeg_quality_from_quant_table(quant)
        phase_x, phase_y = detect_grid_phase(img)

    # Determine OOD conditions
    ood_reasons = []
    is_ood = False

    if train_dataset == "sd-v1_4":
        # Artifact branch saw JPEG q in [70, 95]
        if is_jpeg and est_q is not None and est_q < 70:
            is_ood = True
            ood_reasons.append(f"jpeg_quality_{est_q}_below_training_range_70_95")
    elif train_dataset == "progan":
        # NPR artifact branch was trained with jpg_prob = 0
        if is_jpeg:
            is_ood = True
            ood_reasons.append("progan_npr_artifact_never_trained_on_jpeg")

    return {
        "filepath": filepath,
        "format": fmt,
        "image_size": [width, height],
        "is_jpeg": is_jpeg,
        "estimated_jpeg_quality": est_q,
        "quant_table_type": table_type,
        "grid_phase": [phase_x, phase_y],
        "is_ood": is_ood,
        "ood_reasons": ood_reasons,
        "note": "Flagged per Sections 1 & 2 of CO-SPY audit" if is_ood else "In-distribution",
    }


def probe_jpeg(
    image_or_path: Union[str, Image.Image],
    train_dataset: str = "sd-v1_4"
) -> Dict[str, Any]:
    """Inspect an image and generate a structured input report.
    
    Args:
        image_or_path: Filepath or PIL Image.
        train_dataset: Target detector ("sd-v1_4" or "progan").
    """
    if isinstance(image_or_path, str):
        with Image.open(image_or_path) as img:
            return _inspect_pil_image(img, filepath=image_or_path, train_dataset=train_dataset)
    return _inspect_pil_image(image_or_path, filepath=None, train_dataset=train_dataset)

