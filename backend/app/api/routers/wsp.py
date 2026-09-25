"""
WSP: the model library folder setting.

The library is where WSP CameraTrap installs models from (see
app/ml/model_library.py). The user picks it once in Settings; it is saved
in <user data>/wsp-config.json. Saving it re-runs the catalog sync, so
models published in the library show up without restarting the app.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from app.api.routers import ml_models
from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.ml import model_library
from app.ml.catalog_updater import ModelCatalogUpdater

logger = get_logger(__name__)
router = APIRouter(prefix="/api/wsp", tags=["WSP"])


class LibraryStatus(BaseModel):
    # The folder in use, or None when no library is reachable.
    library_dir: str | None
    # Where it came from: "env", "settings", "autodetect" or None.
    source: str | None
    # The folder saved in Settings, even when it cannot be reached right now.
    configured_dir: str | None
    models_dir: str


class LibraryUpdate(BaseModel):
    # None or "" forgets the saved folder and goes back to autodetection.
    library_dir: str | None


def _status() -> LibraryStatus:
    settings = get_settings()
    configured = model_library.read_config().get("model_library_dir")
    library = model_library.get_library_dir()
    if library is None:
        source = None
    elif settings.model_library_dir:
        source = "env"
    elif configured:
        source = "settings"
    else:
        source = "autodetect"
    return LibraryStatus(
        library_dir=str(library) if library else None,
        source=source,
        configured_dir=configured,
        models_dir=str(settings.models_dir),
    )


@router.get("/library", response_model=LibraryStatus)
def get_library() -> LibraryStatus:
    return _status()


@router.post("/library", response_model=LibraryStatus)
async def set_library(update: LibraryUpdate, request: Request) -> LibraryStatus:
    raw = (update.library_dir or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not model_library.is_library_dir(path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"{path} is not a WSP model library. Pick the folder that "
                    f"holds models.json (usually 'WSP CameraTrap/models')."
                ),
            )
        model_library.write_library_dir(path)
    else:
        model_library.write_library_dir(None)

    try:
        request.app.state.model_updates = await ModelCatalogUpdater().sync()
        if ml_models.manifest_manager is not None:
            ml_models.manifest_manager.load_manifests(force_refresh=True)
    except Exception as e:
        logger.error(f"Catalog sync after a library change failed: {e}", exc_info=True)

    return _status()
