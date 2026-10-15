"""
Model manifest schema for ML models.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Clear documentation

"""

from typing import Literal

from pydantic import BaseModel

# Region the cls model is trained for. Drives how the classification
# dropdown groups its options. None for detection / embedding models
# (region-agnostic) and as a fallback for legacy cls manifests.
ModelRegion = Literal[
    "global", "africa", "americas", "asia", "europe", "oceania"
]

class ModelManifest(BaseModel):
    """
    Model manifest defining all metadata and configuration for an ML model.

    This schema is used to define both detection and classification models.
    Manifests are stored in JSON format and loaded at runtime.
    """

    # Identity
    model_id: str
    friendly_name: str
    # Optional decorative/regional icon. Classification models carry a
    # regional flag; detection / embedding models omit it.
    emoji: str | None = None
    type: str | None = (
        None  # Unused legacy field, kept for backward compatibility with existing manifests
    )
    model_category: str | None = (
        None  # "detection"/"classification"/"embedding" - set during loading
    )

    # Environment & Model Files
    env: str
    model_fname: str
    # Public URL of a single-file model (weights only). Used when the WSP
    # model library does not have the model.
    download_url: str | None = None
    # A local manifest.json holds nothing beyond its catalog entry. Whether
    # an install still matches the library is answered by comparing the
    # files themselves (model_library.find_stale_files), so there is no
    # recorded state here to fall out of date.

    # Metadata
    description: str
    description_short: str | None = None
    developer: str
    owner: str | None = None
    citation: str | None = None
    license: str | None = None
    info_url: str
    min_app_version: str

    # Classification-specific
    species_list: list[str] | None = None
    # Region the model is trained for. Used to group cls models in the
    # UI dropdown. None for detection / embedding (region-agnostic).
    region: ModelRegion | None = None
    # Full-image classifier flag. When True, the model labels the whole
    # frame and the worker skips MegaDetector entirely; a synthetic
    # detection covering the full image is fed straight into the
    # classification phase. See app.ml.full_image_detection.
    full_image_cls: bool = False
    # Picture of what the model expects to see, shown in the model info
    # sheet. A URL only, the image never lives in the repo or the app.
    # Meant for models with a specific setup (a drift-fence bucket, a
    # baited tray) so a user can compare it with their own photos.
    example_image_url: str | None = None

    # Embedding-specific
    embedding_dim: int | None = None  # 384, 768, or 1024
    input_size: int | None = None  # e.g., 224
    torch_hub_model: str | None = None  # e.g., "dinov2_vits14" (for architecture loading)

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "model_id": "MD5A-0-0",
                "friendly_name": "MegaDetector 5a",
                "emoji": "🔍",
                "env": "megadetector",
                "model_fname": "md_v5a.0.0.pt",
                "description": "MegaDetector v5a for animal detection in camera trap images",
                "developer": "Dan Morris",
                "license": "MIT",
                "info_url": "https://github.com/agentmorris/MegaDetector",
                "min_app_version": "0.1.0",
            }
        }
