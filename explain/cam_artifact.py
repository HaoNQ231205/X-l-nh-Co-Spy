"""Exact additive Class Activation Mapping (CAM) for CO-SPY Artifact branch.

Algebraic Decomposition (Section 2 & 13.2 of specification):
Luồng xử lý: 224 -> conv1 (stride 2) -> maxpool (stride 2) -> layer1 (stride 1) -> layer2 (stride 2)
tạo ra feature map layer2 kích thước 28x28 (stride = 8, 784 vị trí),
tiếp tục qua avgpool = AdaptiveAvgPool2d((1, 1)) rồi fc = Linear(512, 1).

Đẳng thức phân rã cộng tính tuyệt đối:
    logit_art = b + (1 / 784) * sum_{i,j} m(i, j)
trong đó:
    m(i, j) = sum_{c=1}^{512} w_c * F_{c, i, j}

Lưu ý Receptive Field (Vùng cảm thụ):
Trước khi vào ResNet, ảnh đã đi qua VAE Encoder và Decoder (chứa tích chập 3x3 và MidBlock self-attention
trên không gian 28x28), sau đó qua 3 Bottleneck layer1 và 4 Bottleneck layer2.
Do vùng cảm thụ chồng lấn rất rộng, mỗi ô (i, j) trên bản đồ 28x28 KHÔNG đại diện độc lập
cho một ô vuông 8x8 pixel tách biệt trên ảnh gốc.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import numpy as np


class ArtifactCAM:
    """Extracts exact additive CAM from Artifact branch at layer2."""

    def __init__(self, artifact_model: torch.nn.Module):
        self.model = artifact_model
        self.feature_map: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self):
        """Register forward hook at artifact_encoder.layer2."""
        # Find layer2 in artifact_encoder
        target_layer = None
        if hasattr(self.model, "artifact_encoder") and hasattr(self.model.artifact_encoder, "layer2"):
            target_layer = self.model.artifact_encoder.layer2
        else:
            # Search recursively
            for name, module in self.model.named_modules():
                if name.endswith("layer2"):
                    target_layer = module
                    break

        if target_layer is None:
            raise AttributeError("Could not find layer2 in artifact model.")

        def hook_fn(module, input, output):
            self.feature_map = output

        self._hook_handle = target_layer.register_forward_hook(hook_fn)

    def remove_hook(self):
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def compute_cam(
        self,
        x_art: torch.Tensor
    ) -> Dict[str, Any]:
        """Compute exact additive CAM.
        
        Args:
            x_art: Input tensor for artifact branch [1, 3, 224, 224].
            
        Returns:
            Dict containing:
            - 'cam_map_28x28': numpy array [28, 28] of signed evidence m(i, j)
            - 'logit_art': model output logit
            - 'reconstructed_logit': b + mean(m)
            - 'additive_error': abs(logit_art - reconstructed_logit)
            - 'is_additive_exact': True if error < 1e-3
        """
        self.feature_map = None
        device = next(self.model.parameters()).device
        x_art = x_art.to(device)

        # Forward pass (CAM only needs activations, runs safely in no_grad)
        with torch.no_grad():
            logit_out = self.model(x_art)

        if self.feature_map is None:
            raise RuntimeError("Forward hook failed to capture layer2 feature map.")

        # Feature map F: [1, 512, 28, 28]
        F = self.feature_map.squeeze(0)  # [512, 28, 28]
        C, H, W = F.shape
        assert H == 28 and W == 28, f"Expected 28x28 feature map, got {H}x{W}"

        # Weights and bias from fc layer
        fc_weight = self.model.fc.weight.squeeze(0)  # [512]
        fc_bias = float(self.model.fc.bias.item())    # scalar

        # Exact contribution map: m(i, j) = sum_c w_c * F_{c, i, j}
        # Using tensordot or einsum
        m = torch.einsum("c,chw->hw", fc_weight, F)  # [28, 28]

        # Reconstructed logit: b + (1 / 784) * sum_{i,j} m(i, j)
        mean_m = float(m.mean().item())
        reconstructed_logit = fc_bias + mean_m
        actual_logit = float(logit_out.squeeze().item())
        additive_error = abs(actual_logit - reconstructed_logit)

        return {
            "cam_map_28x28": m.cpu().numpy(),
            "fc_bias": fc_bias,
            "mean_m": mean_m,
            "logit_art": actual_logit,
            "reconstructed_logit": reconstructed_logit,
            "additive_error": additive_error,
            "is_additive_exact": bool(additive_error < 1e-3),
            "receptive_field_note": (
                "Feature map has stride 8, but receptive field of VAE + ResNet spans far wider "
                "than isolated 8x8 blocks due to VAE self-attention and conv bottleneck cascades."
            )
        }

