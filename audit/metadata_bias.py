"""Metadata bias audit for CO-SPY benchmark and datasets.

Calculates univariate ROC-AUC between metadata attributes (format, short edge,
aspect ratio, EXIF presence, estimated JPEG quality) and ground truth Real/Fake labels.

Enforces Rule 8 (Quy tac 8):
'Do not use JPEG metadata or format artifacts as classification features.'
Flags warnings if metadata strongly correlates with the labels (Issues J5, L7).
"""

import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from PIL import Image, ExifTags
from sklearn.metrics import roc_auc_score


def extract_single_image_metadata(image_path: str) -> Dict[str, Any]:
    """Extract metadata attributes from an image file without decoding full pixels."""
    meta: Dict[str, Any] = {
        "filepath": image_path,
        "format": "UNKNOWN",
        "width": 0,
        "height": 0,
        "min_edge": 0,
        "aspect_ratio": 1.0,
        "file_size_bytes": 0,
        "has_exif": 0,
        "is_jpeg": 0,
        "is_png": 0,
        "is_webp": 0,
        "estimated_jpeg_quality": None,
    }

    if not os.path.exists(image_path):
        return meta

    meta["file_size_bytes"] = os.path.getsize(image_path)

    try:
        with Image.open(image_path) as img:
            fmt = (img.format or "UNKNOWN").upper()
            meta["format"] = fmt
            meta["width"] = img.width
            meta["height"] = img.height
            meta["min_edge"] = min(img.width, img.height)
            meta["aspect_ratio"] = round(img.width / max(1, img.height), 4)

            meta["is_jpeg"] = 1 if fmt in ("JPEG", "JPG") else 0
            meta["is_png"] = 1 if fmt == "PNG" else 0
            meta["is_webp"] = 1 if fmt == "WEBP" else 0

            # EXIF check
            try:
                exif = img.getexif()
                meta["has_exif"] = 1 if (exif is not None and len(exif) > 0) else 0
            except Exception:
                meta["has_exif"] = 0

            # Extract quantization table quality estimation if JPEG
            if meta["is_jpeg"] and hasattr(img, "quantization") and img.quantization:
                from cospy_infer.jpeg_probe import estimate_jpeg_quality_from_quant_table
                q_est, _ = estimate_jpeg_quality_from_quant_table(img.quantization)
                meta["estimated_jpeg_quality"] = q_est
    except Exception as e:
        meta["error"] = str(e)

    return meta


def compute_univariate_auc(labels: np.ndarray, feature_values: np.ndarray) -> float:
    """Calculate univariate ROC-AUC for a continuous or binary feature.
    
    If the feature is inversely correlated with the positive class,
    takes max(auc, 1 - auc) to reflect predictive signal strength regardless of sign.
    Returns 0.5 if all values are identical or invalid.
    """
    valid_mask = ~np.isnan(feature_values)
    y = labels[valid_mask]
    x = feature_values[valid_mask]

    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return 0.5

    try:
        auc = roc_auc_score(y, x)
        # Symmetrize so 0.5 is random guess, 1.0 is perfect correlation in either direction
        return float(max(auc, 1.0 - auc))
    except Exception:
        return 0.5


def audit_metadata_bias(
    image_paths: List[str],
    labels: List[int],
    bias_threshold: float = 0.65
) -> Dict[str, Any]:
    """Audit metadata bias across a dataset.
    
    Args:
        image_paths: List of image file paths.
        labels: 0 for Real, 1 for Fake.
        bias_threshold: AUC threshold above which a feature is flagged as biased.
    """
    assert len(image_paths) == len(labels), "image_paths and labels must have the same length."

    metadata_list = [extract_single_image_metadata(p) for p in image_paths]
    labels_arr = np.array(labels, dtype=np.int32)

    features = {
        "is_png": np.array([m["is_png"] for m in metadata_list], dtype=np.float32),
        "is_jpeg": np.array([m["is_jpeg"] for m in metadata_list], dtype=np.float32),
        "min_edge": np.array([m["min_edge"] for m in metadata_list], dtype=np.float32),
        "aspect_ratio": np.array([m["aspect_ratio"] for m in metadata_list], dtype=np.float32),
        "file_size_bytes": np.array([m["file_size_bytes"] for m in metadata_list], dtype=np.float32),
        "has_exif": np.array([m["has_exif"] for m in metadata_list], dtype=np.float32),
    }

    # For estimated_jpeg_quality, fill NaNs (non-JPEGs) with sentinel or compute on subset
    jpeg_mask = np.array([m["is_jpeg"] == 1 for m in metadata_list], dtype=bool)
    if np.sum(jpeg_mask) > 10 and len(np.unique(labels_arr[jpeg_mask])) >= 2:
        q_vals = np.array([
            m["estimated_jpeg_quality"] if m["estimated_jpeg_quality"] is not None else np.nan
            for m in metadata_list
        ], dtype=np.float32)
        features["estimated_jpeg_quality"] = q_vals

    auc_results: Dict[str, Dict[str, Any]] = {}
    flagged_features: List[str] = []

    for name, vals in features.items():
        auc = compute_univariate_auc(labels_arr, vals)
        real_vals = vals[labels_arr == 0]
        fake_vals = vals[labels_arr == 1]
        
        real_mean = float(np.nanmean(real_vals)) if len(real_vals) > 0 else 0.0
        fake_mean = float(np.nanmean(fake_vals)) if len(fake_vals) > 0 else 0.0

        is_biased = auc >= bias_threshold
        if is_biased:
            flagged_features.append(name)

        auc_results[name] = {
            "univariate_auc": round(auc, 4),
            "real_mean": round(real_mean, 2),
            "fake_mean": round(fake_mean, 2),
            "biased_warning": is_biased,
        }

    return {
        "total_samples": len(image_paths),
        "real_count": int(np.sum(labels_arr == 0)),
        "fake_count": int(np.sum(labels_arr == 1)),
        "bias_threshold": bias_threshold,
        "flagged_features": flagged_features,
        "auc_results": auc_results,
        "verdict": "FLAGGED_METADATA_BIAS" if len(flagged_features) > 0 else "CLEAN_NO_METADATA_BIAS",
        "rule_8_compliance": len(flagged_features) == 0,
    }

