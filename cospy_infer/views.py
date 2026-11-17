"""Input tensor view generators for CO-SPY inference.

Provides 3 route strategies:
1. legacy:
   Uses the common fusion_model.test_transform as in the original Detector.
   (Known defect: causes double-resizing and aliasing for Artifact branch in sd-v1_4).
2. branch_native:
   Calls model.sem.test_transform(img) and model.art.test_transform(img) independently,
   preserving native geometric and frequency characteristics of each branch.
3. multiview(K):
   K in {1, 9}. Generates K deterministic crops on PIL Resize(256) with offsets
   that are exact multiples of 8 (0, 16, 32) to align with the 8x downsampling stride
   of the VAE on the resized image.
"""

from typing import Dict, Any, List, Tuple, Union
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image


class ViewGenerator:
    """Generates branch input tensors according to specified route."""

    def __init__(self, route: str = "branch_native", views: int = 1):
        assert route in ("legacy", "branch_native", "native", "multiview"), f"Unknown route: {route}"
        if route == "native":
            route = "branch_native"
        self.route = route
        self.views = views
        if self.route == "multiview":
            assert views in (1, 9), "multiview route only supports K in {1, 9}."

        # Offsets for 9 deterministic crops on 256x256 image with 224x224 crop
        # Margin is 256 - 224 = 32 pixels. Offsets: 0, 16, 32 (all multiples of 8)
        self.crop_offsets = [
            (0, 0), (0, 16), (0, 32),
            (16, 0), (16, 16), (16, 32),
            (32, 0), (32, 16), (32, 32)
        ]

    def generate(
        self,
        img: Image.Image,
        model: Any
    ) -> Dict[str, Any]:
        """Generate input tensors for model evaluation.
        
        Returns:
            Dict containing:
            - 'route': name of the route
            - 'legacy_tensor': tensor if route == 'legacy', else None
            - 'sem_tensor': tensor [1, 3, 384, 384] for semantic branch
            - 'art_tensors': list of tensors [1, 3, 224, 224] for artifact branch
        """
        if self.route == "legacy":
            # Original common test transform
            tensor = model.test_transform(img).unsqueeze(0)
            return {
                "route": "legacy",
                "legacy_tensor": tensor,
                "sem_tensor": None,
                "art_tensors": None,
            }

        elif self.route == "branch_native" or (self.route == "multiview" and self.views == 1):
            # Branch-native: call each branch's native test_transform independently
            sem_tensor = model.sem.test_transform(img).unsqueeze(0)
            art_tensor = model.art.test_transform(img).unsqueeze(0)
            return {
                "route": "branch_native",
                "legacy_tensor": None,
                "sem_tensor": sem_tensor,
                "art_tensors": [art_tensor],
            }

        elif self.route == "multiview" and self.views == 9:
            # Semantic branch still uses its native 384 center crop
            sem_tensor = model.sem.test_transform(img).unsqueeze(0)

            # Artifact branch: resize to 256, then extract 9 deterministic 224x224 crops
            # with offsets (0, 16, 32)
            load_size = getattr(model.art, "loadSize", 256)
            crop_size = getattr(model.art, "cropSize", 224)
            mean = getattr(model.art, "mean", [0.0, 0.0, 0.0])
            std = getattr(model.art, "std", [1.0, 1.0, 1.0])

            img_resized = img.resize((load_size, load_size), Image.Resampling.BICUBIC)
            normalize = transforms.Normalize(mean=mean, std=std)

            art_tensors = []
            for (ox, oy) in self.crop_offsets:
                crop = img_resized.crop((ox, oy, ox + crop_size, oy + crop_size))
                crop_t = TF.to_tensor(crop)
                crop_t = normalize(crop_t).unsqueeze(0)
                art_tensors.append(crop_t)

            return {
                "route": "multiview_9",
                "legacy_tensor": None,
                "sem_tensor": sem_tensor,
                "art_tensors": art_tensors,
            }
        else:
            raise ValueError(f"Unsupported configuration: route={self.route}, views={self.views}")

