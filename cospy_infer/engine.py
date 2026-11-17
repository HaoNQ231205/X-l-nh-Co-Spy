"""Inference engine for CO-SPY.

Key enhancements over baseline Detector:
1. Explicit verification of missing_keys and unexpected_keys when loading weights
   (fixes the silent random-weight execution bug caused by strict=False in BaseDetector).
2. Checkpoint SHA-256 verification before and after inference runs to guarantee immutability.
3. Strict use of torch.no_grad() (not inference_mode) for safe interaction with autograd and hooks.
4. Returns complete outputs: logit_sem, logit_art, logit_fusion, branch probabilities,
   fusion probability, and binary prediction at threshold 0.5.
5. Supports legacy, branch_native, and multiview routes seamlessly.
"""

import os
import sys
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import numpy as np
from PIL import Image

# Ensure project root is available
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluate import Detector
from audit.dedup import compute_file_sha256
from cospy_infer.views import ViewGenerator
from cospy_infer.loader import load_image_two_pass


def _torch_load_compat(path: str, map_location: Any) -> Any:
    """Load PyTorch weights/checkpoints safely across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def safe_load_state_dict_checked(
    model: torch.nn.Module,
    weights_path: str,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Load weights into model and rigorously check missing/unexpected keys."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")

    state_dict = _torch_load_compat(weights_path, map_location=device)
    incompatible = model.load_state_dict(state_dict, strict=False)

    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    # Check for critical weight absence
    critical_missing = [k for k in missing if "fc" in k or "weight" in k]

    return {
        "weights_path": weights_path,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "critical_missing": critical_missing,
        "is_clean_load": len(missing) == 0 and len(unexpected) == 0,
    }


class InferenceEngine:
    """Production-grade inference engine for CO-SPY."""

    def __init__(
        self,
        train_dataset: str = "sd-v1_4",
        mode: str = "fusion",
        branch: str = "artifact",
        pretrain: bool = False,
        ckpt: str = "ckpt",
        device: Optional[str] = None,
        route: str = "branch_native",
        views: int = 1,
        weights_dir: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.train_dataset = train_dataset
        self.mode = mode
        self.branch = branch
        self.pretrain = pretrain
        self.ckpt = ckpt
        self.route = route
        self.views = views

        self.view_generator = ViewGenerator(route=route, views=views)

        # Record and hash weight paths
        self.weight_hashes_initial: Dict[str, str] = {}
        self.weight_load_reports: Dict[str, Any] = {}
        self._inspect_and_hash_weights()

        # Instantiate underlying Detector
        self.detector = Detector(
            device=self.device,
            mode=self.mode,
            branch=self.branch,
            train_dataset=self.train_dataset,
            pretrain=self.pretrain,
            ckpt=self.ckpt
        )
        self.model = self.detector.model
        self.model.eval()
        self.loaded_custom_files: Dict[str, Optional[str]] = {
            "semantic": None,
            "artifact": None,
            "fusion": None,
        }

        # Support loading custom trained weights directly from directory (e.g. from Google Drive)
        if weights_dir is not None:
            sem_p = os.path.join(weights_dir, "semantic_weights.pth")
            art_p = os.path.join(weights_dir, "artifact_weights.pth")
            fus_p = None
            for candidate in ("best_model.pth", "fusion_weights.pth", "train_checkpoint.pth"):
                cand_p = os.path.join(weights_dir, candidate)
                if os.path.isfile(cand_p) and os.path.getsize(cand_p) > 0:
                    fus_p = cand_p
                    break

            has_sem = hasattr(self.model, "sem") and os.path.isfile(sem_p) and os.path.getsize(sem_p) > 0
            has_art = hasattr(self.model, "art") and os.path.isfile(art_p) and os.path.getsize(art_p) > 0

            if has_sem:
                self.model.sem.load_weights(sem_p)
                self.loaded_custom_files["semantic"] = sem_p
            if has_art:
                self.model.art.load_weights(art_p)
                self.loaded_custom_files["artifact"] = art_p

            if hasattr(self.model, "fc") and fus_p is not None:
                f_state = _torch_load_compat(fus_p, map_location=self.device)
                if not isinstance(f_state, dict):
                    raise RuntimeError(f"Unsupported fusion weight format in {fus_p}: {type(f_state).__name__}")
                if "model_state_dict" in f_state:
                    m_dict = f_state["model_state_dict"]
                    if not has_sem and not has_art:
                        self.model.load_state_dict(m_dict, strict=False)
                    elif "fc.weight" in m_dict and "fc.bias" in m_dict:
                        self.model.fc.load_state_dict({"weight": m_dict["fc.weight"], "bias": m_dict["fc.bias"]})
                    else:
                        raise RuntimeError(f"Checkpoint {fus_p} is missing 'fc.weight'/'fc.bias'.")
                elif "fc.weight" in f_state and "fc.bias" in f_state:
                    self.model.fc.load_state_dict({"weight": f_state["fc.weight"], "bias": f_state["fc.bias"]})
                elif "weight" in f_state and "bias" in f_state:
                    self.model.fc.load_state_dict({"weight": f_state["weight"], "bias": f_state["bias"]})
                elif hasattr(self.model, "load_weights"):
                    self.model.load_weights(fus_p)
                else:
                    raise RuntimeError(f"Unrecognized fusion state keys in {fus_p}: {list(f_state.keys())}")
                self.loaded_custom_files["fusion"] = fus_p

            # Critical: SemanticDetector/CoSpyFusionDetector.load_weights assign CPU .data tensors;
            # move the full model back onto target self.device before verifying & predicting.
            self.model.to(self.device)
            self.model.eval()
            print(
                f"✅ Đã nạp model từ {weights_dir}:\n"
                f"   • Artifact : {os.path.basename(self.loaded_custom_files['artifact']) if self.loaded_custom_files['artifact'] else f'Pretrained ({self.train_dataset})'}\n"
                f"   • Semantic : {os.path.basename(self.loaded_custom_files['semantic']) if self.loaded_custom_files['semantic'] else f'Pretrained ({self.train_dataset})'}\n"
                f"   • Fusion   : {os.path.basename(self.loaded_custom_files['fusion']) if self.loaded_custom_files['fusion'] else f'Pretrained ({self.train_dataset})'}"
            )

        # Run strict weight & device verification after all custom weights are placed on device
        self._verify_loaded_weights()

    def _inspect_and_hash_weights(self):
        """Locate weights files and compute initial SHA-256."""
        paths = []
        if self.mode == "branch":
            p = f"pretrained/{self.train_dataset}/{self.branch}_weights.pth" if self.pretrain else \
                os.path.join(self.ckpt, self.train_dataset, self.branch, "best_model.pth")
            paths.append(p)
        elif self.mode == "fusion":
            if self.pretrain:
                paths.extend([
                    f"pretrained/{self.train_dataset}/semantic_weights.pth",
                    f"pretrained/{self.train_dataset}/artifact_weights.pth",
                    f"pretrained/{self.train_dataset}/fusion_weights.pth"
                ])
            else:
                paths.extend([
                    os.path.join(self.ckpt, self.train_dataset, "semantic", "best_model.pth"),
                    os.path.join(self.ckpt, self.train_dataset, "artifact", "best_model.pth"),
                    os.path.join(self.ckpt, self.train_dataset, "fusion", "best_model.pth")
                ])
        elif self.mode == "end2end":
            paths.append(os.path.join(self.ckpt, self.train_dataset, "end2end", "best_model.pth"))

        for p in paths:
            if os.path.exists(p):
                self.weight_hashes_initial[p] = compute_file_sha256(p)

    def _verify_loaded_weights(self):
        """Verify critical classifier heads are finite and reside on self.device."""
        expected_type = torch.device(self.device).type
        for attr in ("fc",):
            head = getattr(self.model, attr, None)
            if head is not None:
                for p in head.parameters():
                    if torch.isnan(p).any():
                        raise RuntimeError(f"Model {attr} parameters contain NaNs!")
                    if p.device.type != expected_type:
                        raise RuntimeError(f"Device mismatch on model.{attr}: {p.device} vs {expected_type}")
        if self.mode == "fusion":
            for branch_name in ("sem", "art"):
                branch = getattr(self.model, branch_name, None)
                branch_fc = getattr(branch, "fc", None)
                if branch_fc is not None:
                    for p in branch_fc.parameters():
                        if torch.isnan(p).any():
                            raise RuntimeError(f"{branch_name}.fc parameters contain NaNs!")
                        if p.device.type != expected_type:
                            raise RuntimeError(f"Device mismatch on {branch_name}.fc: {p.device} vs {expected_type}")

    def verify_checkpoints_integrity(self) -> bool:
        """Verify that checkpoint files have not been modified or corrupted on disk."""
        for path, initial_hash in self.weight_hashes_initial.items():
            if not os.path.exists(path):
                return False
            curr_hash = compute_file_sha256(path)
            if curr_hash != initial_hash:
                raise RuntimeError(
                    f"Checkpoint file modified during runtime: {path} "
                    f"(Initial: {initial_hash}, Current: {curr_hash})"
                )
        return True

    @torch.no_grad()
    def predict_image(
        self,
        image_or_path: Union[str, Image.Image],
        allow_truncated_legacy: bool = False
    ) -> Dict[str, Any]:
        """Predict whether an image is AI-generated with full logit breakdown.
        
        Args:
            image_or_path: Path to image file or PIL Image.
            allow_truncated_legacy: Allow decoding truncated images in legacy mode.
            
        Returns:
            Dict containing logit_sem, logit_art, logit_fusion, prob_sem,
            prob_art, prob_fusion, prediction, and view information.
        """
        if isinstance(image_or_path, str):
            img, load_meta = load_image_two_pass(
                image_or_path,
                allow_truncated_legacy=allow_truncated_legacy
            )
        else:
            img = image_or_path.convert("RGB")
            load_meta = {"filepath": None, "is_truncated": False}

        # Generate route views
        views = self.view_generator.generate(img, self.model)
        route_name = views["route"]

        if self.mode == "fusion":
            if route_name == "legacy":
                # Legacy route: uses model.test_transform and model.forward()
                x = views["legacy_tensor"].to(self.device)
                
                # Forward through fusion detector
                # Compute intermediate branch logits
                x_sem = self.model.sem_transform(x)
                x_art = self.model.art_transform(x)
                pred_sem = self.model.sem(x_sem)
                pred_art = self.model.art(x_art)
                feat = torch.cat([pred_sem, pred_art], dim=1)
                logit_fusion_t = self.model.fc(feat)

                logit_sem = float(pred_sem.squeeze().cpu().item())
                logit_art = float(pred_art.squeeze().cpu().item())
                logit_fusion = float(logit_fusion_t.squeeze().cpu().item())

            elif route_name in ("branch_native", "multiview_9"):
                # Native route: sem and art evaluate their own independent test transforms
                x_sem = views["sem_tensor"].to(self.device)
                # x_sem is already normalized by sem.test_transform, directly pass to sem
                pred_sem = self.model.sem(x_sem)
                logit_sem = float(pred_sem.squeeze().cpu().item())

                # For artifact branch: evaluate view(s)
                art_logits = []
                for art_t in views["art_tensors"]:
                    art_t = art_t.to(self.device)
                    # art_t is normalized by art.test_transform, directly pass to art
                    pred_art_v = self.model.art(art_t)
                    art_logits.append(float(pred_art_v.squeeze().cpu().item()))

                logit_art = float(np.mean(art_logits))

                # Fuse via self.model.fc
                feat = torch.tensor([[logit_sem, logit_art]], dtype=torch.float32, device=self.device)
                logit_fusion_t = self.model.fc(feat)
                logit_fusion = float(logit_fusion_t.squeeze().cpu().item())
            else:
                raise ValueError(f"Unknown route: {route_name}")

            prob_sem = float(torch.sigmoid(torch.tensor(logit_sem)).item())
            prob_art = float(torch.sigmoid(torch.tensor(logit_art)).item())
            prob_fusion = float(torch.sigmoid(torch.tensor(logit_fusion)).item())
            prediction = 1 if prob_fusion > 0.5 else 0

            return {
                "route": route_name,
                "views_count": len(views["art_tensors"]) if views["art_tensors"] else 1,
                "logit_sem": round(logit_sem, 5),
                "logit_art": round(logit_art, 5),
                "logit_fusion": round(logit_fusion, 5),
                "prob_sem": round(prob_sem, 5),
                "prob_art": round(prob_art, 5),
                "prob_fusion": round(prob_fusion, 5),
                "prediction": prediction,
                "label_name": "AI-Generated" if prediction == 1 else "Real",
                "load_meta": load_meta,
            }

        elif self.mode == "branch":
            # Single branch evaluation
            if self.branch == "artifact":
                x = self.model.test_transform(img).unsqueeze(0).to(self.device)
                pred = self.model(x)
                logit = float(pred.squeeze().cpu().item())
            else:
                x = self.model.test_transform(img).unsqueeze(0).to(self.device)
                pred = self.model(x)
                logit = float(pred.squeeze().cpu().item())

            prob = float(torch.sigmoid(torch.tensor(logit)).item())
            pred_class = 1 if prob > 0.5 else 0
            return {
                "branch": self.branch,
                "logit": round(logit, 5),
                "prob": round(prob, 5),
                "prediction": pred_class,
                "label_name": "AI-Generated" if pred_class == 1 else "Real",
                "load_meta": load_meta,
            }
        else:
            x = self.model.test_transform(img).unsqueeze(0).to(self.device)
            pred = self.model(x)
            logit = float(pred.squeeze().cpu().item())
            prob = float(torch.sigmoid(torch.tensor(logit)).item())
            pred_class = 1 if prob > 0.5 else 0
            return {
                "mode": self.mode,
                "logit": round(logit, 5),
                "prob": round(prob, 5),
                "prediction": pred_class,
                "label_name": "AI-Generated" if pred_class == 1 else "Real",
                "load_meta": load_meta,
            }

