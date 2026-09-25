"""
Model storage: where installed models live and how they get there.

Models live in ``<user data>/models/{det,cls,emb}/<model_id>/``. They are
installed from the WSP model library (see app/ml/model_library.py), or,
for a public single-file model, from its ``download_url``. Nothing is ever
fetched from a model hub.
"""

from collections.abc import Callable
from pathlib import Path

from app.core.config import get_settings
from app.core.job_cancellation import JobCancelledError
from app.core.logging_config import get_logger
from app.ml import model_library
from app.ml.schemas.model_manifest import ModelManifest
from app.utils.fs_remove import safe_rmtree

logger = get_logger(__name__)

_TYPE_DIRS = {"detection": "det", "classification": "cls", "embedding": "emb"}


def _clear_downloaded_files(model_dir: Path) -> None:
    """
    Remove everything an install put in `model_dir`, keeping manifest.json.

    manifest.json is written from the catalog by the catalog updater and
    is never part of an install, so an install can delete it but never put
    it back. Losing it drops the model out of the catalog until the next
    launch's sync, while the weights sit on disk. Per-entry `safe_rmtree`
    so one locked file cannot abort the rest of the cleanup.
    """
    if not model_dir.exists():
        return
    for entry in model_dir.iterdir():
        if entry.name == "manifest.json":
            continue
        safe_rmtree(entry)


class ModelStorage:
    """Installs models into, and locates them in, the local models folder."""

    def __init__(self, models_dir: Path | None = None):
        """
        Initialize model storage manager.

        Args:
            models_dir: Directory to store model weights (default: settings.models_dir)
        """
        self.models_dir = models_dir or get_settings().models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def check_weights_ready(self, manifest: ModelManifest) -> bool:
        """
        Check if model files are downloaded and ready for inference.

        For embedding models that load their architecture via
        torch.hub.load(..., source="local"), the model folder also ships the
        dinov2/ source and a hubconf.py next to the .pth. An install missing
        those is "not ready", so the normal Prepare-model flow fills them in
        (files already present with the right size are skipped).

        Args:
            manifest: Model manifest

        Returns:
            True if all required files are present, False if download needed
        """
        # Model is in models/det/{model_id}/ or models/cls/{model_id}/
        # Use model_category (set by ManifestManager based on directory) to determine path
        model_type = _TYPE_DIRS[manifest.model_category]
        model_path = self.models_dir / model_type / manifest.model_id
        model_file = model_path / manifest.model_fname

        if not model_file.exists():
            return False

        # Architecture source check: only applies to models that load via
        # torch.hub.load(source="local"). Other models load their
        # architecture from PyPI packages in the analysis env and don't
        # ship source alongside the weights.
        if manifest.torch_hub_model and not (model_path / "hubconf.py").is_file():
            return False

        return True

    def download_weights(
        self,
        manifest: ModelManifest,
        progress_callback: Callable[[str, float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Path:
        """
        Install a model if it is not installed yet.

        Sources, in order: the WSP model library, then the model's
        download_url. A failure keeps whatever was copied, so a retry only
        fetches what is missing (every file lands via a .tmp sibling, so a
        file at its final path is whole). A cancel removes the partial
        install; manifest.json survives.

        Raises:
            ModelSourceMissingError: no library or URL can provide it.
            RuntimeError: the install failed.
            JobCancelledError: cancelled via should_cancel.
        """
        model_type = _TYPE_DIRS[manifest.model_category]
        model_path = self.models_dir / model_type / manifest.model_id

        if self.check_weights_ready(manifest):
            logger.info(f"Model {manifest.model_id} already installed at {model_path}")
            if progress_callback:
                progress_callback("Model already installed", 1.0)
            return model_path

        src = model_library.library_model_dir(model_type, manifest.model_id)
        try:
            if src is not None:
                logger.info(f"Copying {manifest.model_id} from the library at {src}")
                model_library.copy_model(src, model_path, progress_callback, should_cancel)
            elif manifest.download_url:
                logger.info(f"Downloading {manifest.model_id} from {manifest.download_url}")
                if progress_callback:
                    progress_callback(f"Downloading {manifest.friendly_name}...", 0.0)
                model_library.download_url(
                    manifest.download_url,
                    model_path / manifest.model_fname,
                    progress_callback,
                    should_cancel,
                )
            else:
                raise model_library.ModelSourceMissingError(
                    manifest.model_id, manifest.friendly_name, model_path
                )
        except JobCancelledError:
            logger.info(f"Cleaning up cancelled install at {model_path}")
            _clear_downloaded_files(model_path)
            raise
        except model_library.ModelSourceMissingError:
            raise
        except Exception as e:
            source = src or manifest.download_url
            raise RuntimeError(
                f"Failed to install {manifest.model_id} from {source}: {e}"
            ) from e

        if not self.check_weights_ready(manifest):
            raise RuntimeError(
                f"{manifest.model_id} was installed but {manifest.model_fname} "
                f"is missing from {model_path}. Check the model folder in the library."
            )
        logger.info(f"Installed {manifest.model_id} at {model_path}")
        if progress_callback:
            progress_callback("Install complete", 1.0)
        return model_path

    def update_stale_files(self, manifest: ModelManifest) -> list[str]:
        """
        Copy again only the files whose local copy differs from the
        library. Nothing is deleted.

        Returns:
            The sorted relative paths that were refreshed ([] when the
            install already matches the library).

        Raises:
            FileNotFoundError: the model is not installed on this machine.
            ConnectionError: the library cannot be reached to decide.
        """
        model_dir = self.get_model_file(manifest).parent
        model_type = model_dir.parent.name
        src = model_library.library_model_dir(model_type, manifest.model_id)
        if src is None:
            raise ConnectionError(
                f"The WSP model library has no folder for {manifest.model_id}"
            )
        stale = model_library.find_stale_files(model_dir, src)
        if stale is None:
            raise ConnectionError(f"Could not read {src} to check for updates")
        if stale:
            model_library.copy_model(src, model_dir, include=stale, overwrite=True)
            logger.info(
                f"Updated {len(stale)} file(s) for {manifest.model_id}: {', '.join(stale)}"
            )
        return stale

    def get_model_path(self, manifest: ModelManifest) -> Path:
        """
        Get path to model directory.

        Args:
            manifest: Model manifest

        Returns:
            Path to model directory (models/det/{model_id}/ or models/cls/{model_id}/)

        Raises:
            FileNotFoundError: If model not downloaded
        """
        # Use model_category (set by ManifestManager based on directory) to determine path
        model_type = _TYPE_DIRS[manifest.model_category]
        model_path = self.models_dir / model_type / manifest.model_id
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model {manifest.model_id} not found at {model_path}. "
                f"Please download it first."
            )

        return model_path

    def get_model_file(self, manifest: ModelManifest) -> Path:
        """
        Get path to model weight file.

        Args:
            manifest: Model manifest

        Returns:
            Path to model file (e.g., .pt, .pth)

        Raises:
            FileNotFoundError: If model file not found
        """
        model_path = self.get_model_path(manifest)
        model_file = model_path / manifest.model_fname

        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {manifest.model_fname}\n" f"Expected at: {model_file}"
            )

        return model_file

    def get_weights_size(self, manifest: ModelManifest) -> float | None:
        """
        Get size of downloaded weights in MB.

        Args:
            manifest: Model manifest

        Returns:
            Size in MB or None if not downloaded
        """
        # Use model_category (set by ManifestManager based on directory) to determine path
        model_type = _TYPE_DIRS[manifest.model_category]
        model_path = self.models_dir / model_type / manifest.model_id
        if not model_path.exists():
            return None

        # Calculate directory size
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())

        return total_size / (1024 * 1024)  # Convert to MB
