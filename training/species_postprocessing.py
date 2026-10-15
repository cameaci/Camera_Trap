"""Post-processing helpers for species classification output."""

from __future__ import annotations

from typing import Iterable, Mapping

DEFAULT_OTHER_ALIASES = ("other", "unknown", "background", "blank", "empty")


def normalize_species_label(label: str | None) -> str:
    """Normalize a label for robust comparisons."""
    if label is None:
        return ""
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


def _normalize_other_aliases(other_aliases: Iterable[str] | None) -> set[str]:
    aliases = other_aliases or DEFAULT_OTHER_ALIASES
    return {normalize_species_label(alias) for alias in aliases}


def is_unknown_prediction(
    prediction: str | None,
    confidence: float,
    conf_threshold: float,
    other_aliases: Iterable[str] | None = None,
) -> bool:
    """Decide whether a classification should be suppressed as unknown."""
    if confidence < conf_threshold:
        return True
    return normalize_species_label(prediction) in _normalize_other_aliases(other_aliases)


def format_species_detection_label(
    result: Mapping[str, object] | None,
    conf_threshold: float,
    other_aliases: Iterable[str] | None = None,
) -> str:
    """Format a model prediction for detection overlays."""
    if not isinstance(result, Mapping):
        return "Unknown"

    prediction = str(result.get("prediction", "Unknown"))
    confidence = float(result.get("confidence", 0.0))
    if is_unknown_prediction(prediction, confidence, conf_threshold, other_aliases=other_aliases):
        return "Unknown"
    return f"{prediction} {confidence:.2f}"
