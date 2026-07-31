"""Utilities for loading and running wildlife species classification models."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

from species_postprocessing import format_species_detection_label

DEFAULT_INPUT_SIZE = 224
DEFAULT_NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
__all__ = [
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_NORMALIZATION",
    "build_classifier_model",
    "WildlifeClassificationAdapter",
    "format_species_detection_label",
]


def build_classifier_model(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Build a supported backbone configured for a class count."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture '{arch}'. Supported: mobilenet_v3_small")


def _as_class_name_map(class_names: object) -> Dict[int, str]:
    if isinstance(class_names, Mapping):
        parsed: Dict[int, str] = {}
        for key, value in class_names.items():
            parsed[int(key)] = str(value)
        return parsed
    if isinstance(class_names, Sequence) and not isinstance(class_names, (str, bytes)):
        return {idx: str(name) for idx, name in enumerate(class_names)}
    return {}


def _infer_num_classes_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> int:
    for key in ("classifier.3.weight", "fc.weight", "head.weight"):
        weights = state_dict.get(key)
        if isinstance(weights, torch.Tensor) and weights.ndim >= 2:
            return int(weights.shape[0])
    # fallback: choose first 2D tensor output channels
    for weights in state_dict.values():
        if isinstance(weights, torch.Tensor) and weights.ndim >= 2:
            return int(weights.shape[0])
    return 0


def _normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    if keys and all(key.startswith("module.") for key in keys):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return dict(state_dict)


class WildlifeClassificationAdapter:
    """Adapter that loads legacy or checkpoint-based models for inference."""

    def __init__(self, weights_path: str, device: str | torch.device):
        self.device = torch.device(device)
        self.arch = None
        self.metrics = {}
        self.checkpoint = {}
        self.supports_batch = False  # Single-image only for now.

        loaded = self._torch_load_compatible(weights_path)
        self.model, metadata = self._load_model_and_metadata(loaded)
        self.checkpoint = metadata
        self.arch = metadata.get("arch")
        self.metrics = metadata.get("metrics", {})

        normalization = metadata.get("normalization", DEFAULT_NORMALIZATION)
        mean = normalization.get("mean", DEFAULT_NORMALIZATION["mean"])
        std = normalization.get("std", DEFAULT_NORMALIZATION["std"])
        input_size = metadata.get("input_size", DEFAULT_INPUT_SIZE)
        if isinstance(input_size, Sequence) and not isinstance(input_size, (str, bytes)):
            input_size = int(input_size[0])
        else:
            input_size = int(input_size)

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.CLASS_NAMES = self._resolve_class_names(metadata)

    def _torch_load_compatible(self, weights_path: str):
        try:
            return torch.load(weights_path, map_location=self.device)
        except Exception as exc:
            message = str(exc)
            # PyTorch >=2.6 defaults to weights_only=True. Retry for trusted legacy full-model checkpoints.
            if "weights_only" in message.lower() or "unsupported global" in message.lower():
                return torch.load(weights_path, map_location=self.device, weights_only=False)
            raise

    def _load_model_and_metadata(self, loaded: object) -> tuple[nn.Module, Dict[str, object]]:
        if isinstance(loaded, Mapping) and "state_dict" in loaded:
            checkpoint = dict(loaded)
            state_dict = _normalize_state_dict_keys(checkpoint["state_dict"])
            class_names = _as_class_name_map(checkpoint.get("class_names"))
            num_classes = len(class_names) or _infer_num_classes_from_state_dict(state_dict)
            if num_classes <= 0:
                raise ValueError("Unable to infer class count from checkpoint state_dict")

            arch = str(checkpoint.get("arch", "mobilenet_v3_small"))
            model = build_classifier_model(arch=arch, num_classes=num_classes)
            model.load_state_dict(state_dict, strict=True)
            model = model.to(self.device)
            model.eval()
            checkpoint["class_names"] = class_names or {idx: f"Species {idx}" for idx in range(num_classes)}
            return model, checkpoint

        if hasattr(loaded, "to"):
            loaded = loaded.to(self.device)
        if hasattr(loaded, "eval"):
            loaded.eval()
        return loaded, {}

    def _resolve_class_names(self, metadata: Mapping[str, object]) -> Dict[int, str]:
        metadata_names = _as_class_name_map(metadata.get("class_names"))
        if metadata_names:
            return metadata_names

        model_names = _as_class_name_map(getattr(self.model, "CLASS_NAMES", None))
        if model_names:
            return model_names

        num_classes = self._infer_num_classes()
        if num_classes:
            return {idx: f"Species {idx}" for idx in range(num_classes)}
        return {}

    def _infer_num_classes(self) -> int:
        try:
            with torch.no_grad():
                sample = torch.zeros(1, 3, DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE, device=self.device)
                output = self.model(sample)
            if output.ndim >= 2:
                return int(output.shape[-1])
        except Exception:
            return 0
        return 0

    def _prepare_tensor(self, image: Image.Image | np.ndarray) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL Image or numpy array for classification")
        return self.transform(image).unsqueeze(0).to(self.device)

    def _format_label(self, class_idx: int) -> str:
        if self.CLASS_NAMES:
            return self.CLASS_NAMES.get(class_idx, f"Species {class_idx}")
        return f"Species {class_idx}"

    def single_image_classification(self, image: Image.Image | np.ndarray) -> Dict[str, object]:
        tensor = self._prepare_tensor(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, class_idx = torch.max(probs, dim=1)
        idx = int(class_idx.item())
        return {
            "prediction": self._format_label(idx),
            "confidence": float(confidence.item()),
            "class_id": idx,
        }
