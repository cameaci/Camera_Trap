"""Adapter for running SpeciesNet as a crop classification model."""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

DEFAULT_SPECIESNET_MODEL = "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"
DEFAULT_SPECIESNET_CACHE_PARTS = (
    "models",
    "google",
    "speciesnet",
    "pyTorch",
    "v4.0.2a",
    "1",
)


def _is_speciesnet_model_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "info.json").is_file()
        and any(path.glob("*.pt"))
        and any(path.glob("*.labels.*.txt"))
    )


def find_cached_speciesnet_model() -> Path | None:
    """Find the default SpeciesNet model in KaggleHub's local cache."""
    cache_roots = []
    kagglehub_cache = os.environ.get("KAGGLEHUB_CACHE")
    if kagglehub_cache:
        cache_roots.append(Path(kagglehub_cache))
    cache_roots.append(Path.home() / ".cache" / "kagglehub")

    for cache_root in cache_roots:
        candidate = cache_root.joinpath(*DEFAULT_SPECIESNET_CACHE_PARTS)
        if _is_speciesnet_model_dir(candidate):
            return candidate
    return None


def resolve_speciesnet_model_name(model_name: str | None) -> str:
    """Prefer the local cached default model to avoid unnecessary downloads."""
    if model_name and model_name != DEFAULT_SPECIESNET_MODEL:
        return model_name

    cached_model = find_cached_speciesnet_model()
    if cached_model is not None:
        return str(cached_model)
    return model_name or DEFAULT_SPECIESNET_MODEL


def format_speciesnet_label(raw_label: object) -> str:
    """Convert SpeciesNet taxonomy strings into compact display labels."""
    label = str(raw_label or "").strip()
    if not label:
        return "Unknown"

    parts = [part.strip() for part in label.split(";")]
    if len(parts) == 1:
        return label

    ignored = {"", "no cv result"}
    for part in reversed(parts[1:]):
        if part.lower() not in ignored:
            return part
    return "Unknown"


class SpeciesNetClassificationAdapter:
    """Small adapter matching the app's single-image classifier contract."""

    supports_batch = False

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        classifier_cls: type | None = None,
    ) -> None:
        if classifier_cls is None:
            try:
                speciesnet = import_module("speciesnet")
            except ImportError as exc:
                raise ImportError(
                    "SpeciesNet is not installed. Install it with `pip install speciesnet` "
                    "to use the SpeciesNet classifier."
                ) from exc

            classifier_cls = getattr(speciesnet, "SpeciesNetClassifier")
            default_model = getattr(speciesnet, "DEFAULT_MODEL", DEFAULT_SPECIESNET_MODEL)
        else:
            default_model = DEFAULT_SPECIESNET_MODEL

        requested_model_name = model_name or default_model
        self.model_name = resolve_speciesnet_model_name(requested_model_name)
        self.device = device
        try:
            self.classifier = classifier_cls(model_name=self.model_name, device=device)
        except Exception as exc:
            message = str(exc)
            if "CERTIFICATE_VERIFY_FAILED" in message or "SSLCertVerificationError" in message:
                raise RuntimeError(
                    "SpeciesNet model could not be downloaded because SSL certificate "
                    "verification failed. Use a local SpeciesNet model folder in the "
                    "Weights / Model Path field, or fix the Kaggle/HuggingFace SSL "
                    "certificate chain in this Python environment."
                ) from exc
            raise
        self.CLASS_NAMES = {
            idx: format_speciesnet_label(label)
            for idx, label in getattr(self.classifier, "labels", {}).items()
        }
        self._label_to_idx = {
            label: idx for idx, label in getattr(self.classifier, "labels", {}).items()
        }

    def _prepare_image(self, image: Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL Image or numpy array for classification")
        return image.convert("RGB")

    def single_image_classification(self, image: Image.Image | np.ndarray) -> Dict[str, Any]:
        pil_image = self._prepare_image(image)
        preprocessed = self.classifier.preprocess(pil_image, bboxes=None)
        result = self.classifier.predict("<memory>", preprocessed)
        classifications = result.get("classifications", {}) if isinstance(result, dict) else {}
        classes = classifications.get("classes") or []
        scores = classifications.get("scores") or []

        if not classes or not scores:
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "class_id": -1,
                "raw_prediction": None,
            }

        raw_prediction = classes[0]
        return {
            "prediction": format_speciesnet_label(raw_prediction),
            "confidence": float(scores[0]),
            "class_id": int(self._label_to_idx.get(raw_prediction, -1)),
            "raw_prediction": raw_prediction,
        }
