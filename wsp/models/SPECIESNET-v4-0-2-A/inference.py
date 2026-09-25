"""
SpeciesNet 4.0.2a (always_crop) for WSP CameraTrap.

Put this file in the SpeciesNet model folder of the WSP model library,
next to the files of Google's SpeciesNet PyTorch release v4.0.2a:

    always_crop_99710272_22x8_v12_epoch_00148.pt
    always_crop_99710272_22x8_v12_epoch_00148.labels.txt
    geofence_release.*.json
    taxonomy.csv            (written by wsp/tools/wsp_library.py add-speciesnet)
    inference.py            (this file)

It runs in the app's "wsp-base" environment and needs only torch,
torchvision and onnx2torch (the classifier is a torch.fx GraphModule that
pickled onnx2torch operators). Preprocessing follows the speciesnet
package's SpeciesNetClassifier exactly: crop the MegaDetector box, resize
to 480x480 without antialiasing, round-trip through uint8, HWC in [0, 1].
Class names follow the same de-duplication rule as the app's geofence and
taxonomy code, so labels, geofence and taxonomy tree all agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 480


def class_names_from_labels(labels_path: Path) -> list[str]:
    """
    Display names in label-file order. Empty or duplicate common names
    fall back to the most specific taxonomy rank, then a UUID prefix.
    Keep in step with app/ml/geofence.py::_parse_labels_cached.
    """
    names: list[str] = []
    seen: set[str] = set()
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 7:
                # Keep indices aligned with the model output even for a
                # malformed line.
                names.append(line)
                seen.add(line)
                continue
            name = parts[6]
            if not name or name in seen:
                taxonomy = [p for p in parts[1:6] if p]
                if taxonomy:
                    name = taxonomy[-1]
            if name in seen:
                name = f"{name} ({parts[0][:8]})"
            seen.add(name)
            names.append(name)
    return names


class ModelInference:
    def __init__(self, model_dir: Path, model_path: Path):
        self.model_dir = Path(model_dir)
        self.model_path = Path(model_path)
        self.model = None
        self.device = "cpu"
        labels = sorted(self.model_dir.glob("*.labels*.txt"))
        if not labels:
            raise FileNotFoundError(f"No SpeciesNet labels file in {self.model_dir}")
        self.names = class_names_from_labels(labels[-1])

    def check_gpu(self) -> bool:
        import torch

        return torch.cuda.is_available() or torch.backends.mps.is_available()

    def load_model(self) -> None:
        import onnx2torch  # noqa: F401  (needed to unpickle the GraphModule)
        import torch

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        model = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model

    def get_crop(
        self, image: Image.Image, bbox: tuple[float, float, float, float]
    ) -> Image.Image:
        x, y, w, h = bbox
        left = int(x * image.width)
        top = int(y * image.height)
        width = max(int(w * image.width), 1)
        height = max(int(h * image.height), 1)
        return image.convert("RGB").crop((left, top, left + width, top + height))

    def get_tensor(self, crop: Image.Image) -> np.ndarray:
        import torch
        import torchvision.transforms.functional as F

        tensor = F.pil_to_tensor(crop.convert("RGB"))
        tensor = F.convert_image_dtype(tensor, torch.float32)
        tensor = F.resize(tensor, [IMG_SIZE, IMG_SIZE], antialias=False)
        tensor = F.convert_image_dtype(tensor, torch.uint8)
        return (tensor.permute(1, 2, 0).numpy() / 255).astype(np.float32)

    def classify_batch(self, batch: np.ndarray) -> list[list[list]]:
        import torch

        with torch.no_grad():
            logits = self.model(torch.from_numpy(batch).to(self.device)).cpu()
            probs = torch.softmax(logits, dim=-1).numpy()
        return [
            [[self.names[j], float(p[j])] for j in range(len(self.names))]
            for p in probs
        ]

    def get_classification(self, crop: Image.Image) -> list[list]:
        return self.classify_batch(np.stack([self.get_tensor(crop)]))[0]

    def get_class_names(self) -> dict[str, str]:
        return {str(i + 1): name for i, name in enumerate(self.names)}
