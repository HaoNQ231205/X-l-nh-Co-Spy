"""Rendering and visualization tools for CO-SPY explanations.

Includes:
- Signed heatmap visualization (positive = AI evidence, negative = Real evidence)
- Fusion contribution table: logit_fusion = b + w_sem * l_sem + w_art * l_art
- Difference CAM (Delta CAM = m_orig - m_degraded)
- Heatmap overlay onto original image
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import cv2


def render_signed_heatmap(
    signed_map: np.ndarray,
    vmax: Optional[float] = None
) -> np.ndarray:
    """Render a signed evidence map into an RGB image using a diverging colormap.
    
    Positive values (AI evidence) -> Red / Warm
    Negative values (Real evidence) -> Blue / Cool
    Zero (Neutral) -> White
    """
    if vmax is None or vmax <= 0:
        vmax = max(float(np.max(np.abs(signed_map))), 1e-5)

    # Normalize map to [-1.0, 1.0]
    norm_map = np.clip(signed_map / vmax, -1.0, 1.0)

    h, w = signed_map.shape
    rgb_img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Positive (Red): [255, 255 * (1 - v), 255 * (1 - v)]
    pos_mask = norm_map > 0
    pos_vals = norm_map[pos_mask]
    rgb_img[pos_mask, 1] = np.uint8(255 * (1.0 - pos_vals))
    rgb_img[pos_mask, 2] = np.uint8(255 * (1.0 - pos_vals))

    # Negative (Blue): [255 * (1 - |v|), 255 * (1 - |v|), 255]
    neg_mask = norm_map < 0
    neg_vals = np.abs(norm_map[neg_mask])
    rgb_img[neg_mask, 0] = np.uint8(255 * (1.0 - neg_vals))
    rgb_img[neg_mask, 1] = np.uint8(255 * (1.0 - neg_vals))

    return rgb_img


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    signed: bool = True
) -> Image.Image:
    """Overlay a 2D heatmap on a PIL Image."""
    w, h = image.size
    # Upsample heatmap to match image size
    resized_hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    if signed:
        hm_rgb = render_signed_heatmap(resized_hm)
    else:
        # Standard positive heatmap (JET or VIRIDIS)
        norm_hm = np.uint8(255 * np.clip(resized_hm, 0.0, 1.0))
        hm_rgb = cv2.applyColorMap(norm_hm, cv2.COLORMAP_JET)
        hm_rgb = cv2.cvtColor(hm_rgb, cv2.COLOR_BGR2RGB)

    img_np = np.array(image.convert("RGB"))
    blended = np.uint8((1.0 - alpha) * img_np + alpha * hm_rgb)
    return Image.fromarray(blended)


def compute_delta_cam(m_original: np.ndarray, m_degraded: np.ndarray) -> np.ndarray:
    """Compute difference map Delta CAM = m_original - m_degraded.
    
    Reveals regions where degradation erased or inverted artifact evidence.
    """
    assert m_original.shape == m_degraded.shape, "CAM shapes must match for delta calculation."
    return m_original - m_degraded


def format_fusion_decomposition_table(
    b_fusion: float,
    w_sem: float,
    logit_sem: float,
    w_art: float,
    logit_art: float
) -> Dict[str, Any]:
    """Calculate and format linear contribution decomposition:
    
    logit_fusion = b + w_sem * logit_sem + w_art * logit_art
    """
    contrib_sem = w_sem * logit_sem
    contrib_art = w_art * logit_art
    logit_fusion = b_fusion + contrib_sem + contrib_art
    prob_fusion = 1.0 / (1.0 + np.exp(-logit_fusion))

    table_str = (
        f"----------------------------------------------------------\n"
        f"CO-SPY Fusion Decomposition (Threshold = 0.5)\n"
        f"----------------------------------------------------------\n"
        f"  Bias (b)                 : {b_fusion:+.4f}\n"
        f"  Semantic Contribution    : {contrib_sem:+.4f}  (w_s={w_sem:+.4f} * l_s={logit_sem:+.4f})\n"
        f"  Artifact Contribution    : {contrib_art:+.4f}  (w_a={w_art:+.4f} * l_a={logit_art:+.4f})\n"
        f"  --------------------------------------------------------\n"
        f"  Total logit_fusion       : {logit_fusion:+.4f}\n"
        f"  Final Probability        : {prob_fusion * 100:.2f}%\n"
        f"  Predicted Class          : {'AI-Generated' if prob_fusion > 0.5 else 'Real'}\n"
        f"----------------------------------------------------------"
    )

    return {
        "bias": round(b_fusion, 4),
        "w_sem": round(w_sem, 4),
        "logit_sem": round(logit_sem, 4),
        "contrib_sem": round(contrib_sem, 4),
        "w_art": round(w_art, 4),
        "logit_art": round(logit_art, 4),
        "contrib_art": round(contrib_art, 4),
        "logit_fusion": round(logit_fusion, 4),
        "prob_fusion": round(float(prob_fusion), 4),
        "prediction": "AI-Generated" if prob_fusion > 0.5 else "Real",
        "formatted_table": table_str
    }

