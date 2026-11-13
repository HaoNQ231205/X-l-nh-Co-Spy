"""Sanity checks and faithfulness validation for CO-SPY explanations.

Includes:
1. Additive Precision Check: Verifies abs(logit_art - (b + mean(m))) < 1e-3.
2. Gradient Existence Check: Verifies gradient norm > 0 for Semantic Grad-CAM.
3. Model Parameter Randomization Test (Adebayo et al., NeurIPS 2018):
   Randomizes the classifier fc layer in memory, verifies that the resulting
   activation map changes dramatically (Pearson correlation drops).
   Restores original weights faithfully after test.
"""

from typing import Dict, Any, Tuple
import torch
import numpy as np
from scipy.stats import pearsonr

from explain.cam_artifact import ArtifactCAM
from explain.gradcam_semantic import SemanticGradCAM


def run_additive_sanity_check(artifact_cam: ArtifactCAM, x_art: torch.Tensor) -> Dict[str, Any]:
    """Test mathematical additivity of the artifact CAM."""
    res = artifact_cam.compute_cam(x_art)
    err = res["additive_error"]
    passed = err < 1e-3
    return {
        "test_name": "exact_additivity_check",
        "passed": bool(passed),
        "additive_error": float(err),
        "tolerance": 1e-3,
        "actual_logit": res["logit_art"],
        "reconstructed_logit": res["reconstructed_logit"],
    }


def run_gradient_sanity_check(semantic_gradcam: SemanticGradCAM, x_sem: torch.Tensor) -> Dict[str, Any]:
    """Test that Semantic Grad-CAM receives valid, non-zero gradients."""
    res = semantic_gradcam.compute_gradcam(x_sem)
    norm = res["gradient_norm"]
    passed = norm > 1e-7
    return {
        "test_name": "gradient_existence_check",
        "passed": bool(passed),
        "gradient_norm": float(norm),
        "threshold": 1e-7,
    }


def run_adebayo_randomization_check(
    artifact_model: torch.nn.Module,
    x_art: torch.Tensor,
    seed: int = 42
) -> Dict[str, Any]:
    """Execute model parameter randomization test (Adebayo et al., 2018).
    
    Temporarily replaces artifact_model.fc weights with random Gaussian noise,
    verifies that CAM map degrades/changes, and restores original parameters.
    """
    cam_tool = ArtifactCAM(artifact_model)
    try:
        # 1. Compute original CAM
        orig_res = cam_tool.compute_cam(x_art)
        orig_m = orig_res["cam_map_28x28"].flatten()

        # 2. Backup original weights
        with torch.no_grad():
            orig_weight = artifact_model.fc.weight.clone()
            orig_bias = artifact_model.fc.bias.clone()

            # Randomize fc layer in-memory
            torch.manual_seed(seed)
            torch.nn.init.normal_(artifact_model.fc.weight, mean=0.0, std=1.0)
            torch.nn.init.normal_(artifact_model.fc.bias, mean=0.0, std=1.0)

            # Compute randomized CAM
            rand_res = cam_tool.compute_cam(x_art)
            rand_m = rand_res["cam_map_28x28"].flatten()

            # Restore original weights
            artifact_model.fc.weight.copy_(orig_weight)
            artifact_model.fc.bias.copy_(orig_bias)

        # 3. Measure Pearson correlation
        if np.std(orig_m) > 1e-8 and np.std(rand_m) > 1e-8:
            r_val, p_val = pearsonr(orig_m, rand_m)
        else:
            r_val = 0.0

        # Pass if correlation is low (|r| < 0.5), meaning CAM is genuinely dependent on weights
        passed = abs(r_val) < 0.5

        return {
            "test_name": "adebayo_model_randomization_test",
            "passed": bool(passed),
            "pearson_correlation": round(float(r_val), 4),
            "correlation_threshold": 0.5,
            "verdict": (
                "PASSED: CAM is sensitive to learned weights (not an edge detector artifact)."
                if passed else
                "FAILED: CAM failed parameter randomization test."
            )
        }
    finally:
        cam_tool.remove_hook()

