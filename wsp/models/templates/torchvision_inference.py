"""
WSP classifier trained with training/train_species_classifier.py.

Copied into a WSP model folder by `wsp/tools/wsp_library.py add-model`,
next to the checkpoint (saved as model.pt). The checkpoint is the dict the
trainer writes: state_dict, class_names, arch, input_size, normalization.
Runs in the app's "pytorch" environment (torch + torchvision only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def build_model(arch: str, num_classes: int):
    """Same architectures and heads as training/wildlife_classifier.py."""
    import torch.nn as nn
    from torchvision import models

    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture '{arch}'")


class ModelInference:
    def __init__(self, model_dir: Path, model_path: Path):
        self.model_dir = Path(model_dir)
        self.model_path = Path(model_path)
        self.model = None
        self.device = "cpu"
        self.names: list[str] = []
        self.input_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def check_gpu(self) -> bool:
        import torch

        return torch.cuda.is_available() or torch.backends.mps.is_available()

    def load_model(self) -> None:
        import torch

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state_dict = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in checkpoint["state_dict"].items()
        }
        names = checkpoint["class_names"]
        self.names = (
            [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else list(names)
        )
        self.input_size = int(checkpoint.get("input_size", self.input_size))
        norm = checkpoint.get("normalization") or {}
        self.mean = list(norm.get("mean", self.mean))
        self.std = list(norm.get("std", self.std))
        model = build_model(str(checkpoint.get("arch", "mobilenet_v3_small")), len(self.names))
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device).eval()
        self.model = model

    def get_crop(
        self, image: Image.Image, bbox: tuple[float, float, float, float]
    ) -> Image.Image:
        x, y, w, h = bbox
        left, top = int(x * image.width), int(y * image.height)
        right = left + max(int(w * image.width), 1)
        bottom = top + max(int(h * image.height), 1)
        return image.convert("RGB").crop((left, top, right, bottom))

    def get_tensor(self, crop: Image.Image) -> np.ndarray:
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])
        return transform(crop.convert("RGB")).numpy()

    def classify_batch(self, batch: np.ndarray) -> list[list[list]]:
        import torch

        with torch.no_grad():
            logits = self.model(torch.from_numpy(batch).to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return [[[self.names[j], float(p[j])] for j in range(len(self.names))] for p in probs]

    def get_classification(self, crop: Image.Image) -> list[list]:
        return self.classify_batch(np.stack([self.get_tensor(crop)]))[0]

    def get_class_names(self) -> dict[str, str]:
        if not self.names:
            import torch

            names = torch.load(self.model_path, map_location="cpu", weights_only=False)["class_names"]
            self.names = (
                [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else list(names)
            )
        return {str(i + 1): name for i, name in enumerate(self.names)}
