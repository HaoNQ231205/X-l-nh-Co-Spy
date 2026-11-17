"""Grad-CAM implementation for CO-SPY Semantic branch.

Autograd and Tensor Isolation Rules (Section 2 & 13.7 of specification):
- Grad-CAM MUST create a separate input tensor with .detach().clone().requires_grad_(True).
- MUST run inside `with torch.enable_grad():` outside inference_mode to avoid
  RuntimeError: 'Inference tensors cannot be saved for backward'.
- Target hook location for SigLIP (ViT-SO400M-14-SigLIP-384):
  SigLIP uses attn_pool on 27x27 = 729 patch tokens (not a single CLS token).
  Hook is registered at the output of the final transformer block:
  `model.sem.clip.visual.trunk.blocks[-1]` (prior to attn_pool).
- Target hook for CLIP ViT-L/14:
  `model.sem.clip.visual.transformer.resblocks[-1]` (excluding CLS token).
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np


class SemanticGradCAM:
    """Computes Grad-CAM for Semantic detector branch."""

    def __init__(self, semantic_model: torch.nn.Module):
        self.model = semantic_model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hook_handles = []
        self._locate_and_hook_target_layer()

    def _locate_and_hook_target_layer(self):
        """Locate final transformer block in SigLIP or standard CLIP."""
        target_layer = None
        clip_model = getattr(self.model, "clip", None)
        if clip_model is None:
            raise AttributeError("Semantic model does not have 'clip' attribute.")

        visual = getattr(clip_model, "visual", None)
        if visual is None:
            raise AttributeError("CLIP model does not have 'visual' attribute.")

        # Check for SigLIP structure: visual.trunk.blocks[-1]
        if hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
            target_layer = visual.trunk.blocks[-1]
            self.model_type = "siglip"
        # Check for standard open_clip: visual.transformer.resblocks[-1]
        elif hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
            target_layer = visual.transformer.resblocks[-1]
            self.model_type = "clip_vit"
        else:
            # Fallback search
            for name, module in visual.named_modules():
                if "block" in name.lower() or "resblock" in name.lower():
                    target_layer = module
            self.model_type = "generic_vit"

        if target_layer is None:
            raise AttributeError("Could not locate final transformer block in visual trunk.")

        def forward_hook(module, input, output):
            # SigLIP returns output tensor or tuple
            if isinstance(output, tuple):
                self.activations = output[0]
            else:
                self.activations = output

        def backward_hook(module, grad_in, grad_out):
            if isinstance(grad_out, tuple):
                self.gradients = grad_out[0]
            else:
                self.gradients = grad_out

        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def compute_gradcam(
        self,
        x_sem: torch.Tensor
    ) -> Dict[str, Any]:
        """Compute Grad-CAM for Semantic branch.
        
        Args:
            x_sem: Semantic input tensor [1, 3, 384, 384] (or 224 for progan).
            
        Returns:
            Dict containing gradcam heatmap [H, W], logit_sem, and gradient check.
        """
        self.activations = None
        self.gradients = None
        device = next(self.model.parameters()).device

        # CRITICAL: Create isolated input tensor with requires_grad=True
        input_tensor = x_sem.detach().clone().to(device).requires_grad_(True)

        # Run with enable_grad to build computational graph for backward
        with torch.enable_grad():
            logit = self.model(input_tensor)
            self.model.zero_grad()
            logit.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")

        # Tensor shapes:
        # For SigLIP ViT-SO400M-14-384: activations is [1, 729, D] where 729 = 27x27, D = 1152
        act = self.activations.detach()
        grad = self.gradients.detach()

        # Handle token sequence [B, N, D] vs feature map [B, D, H, W]
        if act.dim() == 3:
            B, N, D = act.shape
            # For standard CLIP with CLS token: N = 1 + 256 or 1 + 196
            # For SigLIP without CLS token (attn_pool): N = 729 (27x27)
            grid_size = int(round(np.sqrt(N)))
            if grid_size * grid_size == N:
                # No CLS token (SigLIP)
                act = act.permute(0, 2, 1).view(B, D, grid_size, grid_size)
                grad = grad.permute(0, 2, 1).view(B, D, grid_size, grid_size)
            elif (grid_size - 1) * (grid_size - 1) == (N - 1):
                # Has CLS token at index 0
                act = act[:, 1:, :].permute(0, 2, 1).view(B, D, grid_size - 1, grid_size - 1)
                grad = grad[:, 1:, :].permute(0, 2, 1).view(B, D, grid_size - 1, grid_size - 1)
            else:
                grid_size = int(np.floor(np.sqrt(N)))
                act = act[:, :grid_size*grid_size, :].permute(0, 2, 1).view(B, D, grid_size, grid_size)
                grad = grad[:, :grid_size*grid_size, :].permute(0, 2, 1).view(B, D, grid_size, grid_size)

        # Compute importance weights alpha_k = spatial average of gradients
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [B, D, 1, 1]
        cam = torch.sum(weights * act, dim=1, keepdim=True)    # [B, 1, H, W]
        cam = F.relu(cam)  # ReLU to focus on positive contributions
        
        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        cam_np = cam.squeeze().cpu().numpy()
        grad_norm = float(torch.norm(grad).item())

        return {
            "gradcam_map": cam_np,
            "grid_shape": list(cam_np.shape),
            "logit_sem": float(logit.detach().squeeze().item()),
            "gradient_norm": grad_norm,
            "has_nonzero_gradient": bool(grad_norm > 1e-7),
        }

