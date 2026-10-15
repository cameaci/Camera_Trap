"""
The WSP model library setting.

Models are installed from the local models folder or the WSP model
library (see app/ml/model_library.py). From File > WSP model library the
user either installs a model library .zip (unpacked straight into the
local models folder) or points the app at a library folder (a synced
OneDrive folder or a network share); the folder is saved in
<user data>/wsp-config.json. Either way the catalog is re-read afterwards
so the models appear at once.
"""

import asyncio
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
    # The library folder in use, or None when no library folder is reachable.
    library_dir: str | None
    # Where it came from: "env", "settings", "autodetect" or None.
    source: str | None
    # The folder saved in the app, even when it cannot be reached right now.
    configured_dir: str | None
    models_dir: str


class LibraryUpdate(BaseModel):
    # A folder, or None/"" to stop using a saved folder.
    library_dir: str | None = None


class ZipImport(BaseModel):
    # The model library .zip on this computer.
    zip_path: str


class ZipImportResult(BaseModel):
    # The installed model folders, as "<type>/<id>".
    imported: list[str]
    status: LibraryStatus


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


async def _resync(request: Request) -> None:
    try:
        request.app.state.model_updates = await ModelCatalogUpdater().sync()
        if ml_models.manifest_manager is not None:
            ml_models.manifest_manager.load_manifests(force_refresh=True)
    except Exception as e:
        logger.error(f"Catalog sync after a library change failed: {e}", exc_info=True)


@router.get("/library", response_model=LibraryStatus)
async def get_library() -> LibraryStatus:
    # A thread: an unreachable network share must not block the server.
    return await asyncio.to_thread(_status)


@router.post("/library", response_model=LibraryStatus)
async def set_library(update: LibraryUpdate, request: Request) -> LibraryStatus:
    """Use a library folder, or forget the saved one when none is given."""
    folder = (update.library_dir or "").strip()
    if folder:
        path = Path(folder).expanduser()
        if not await asyncio.to_thread(model_library.is_library_dir, path):
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
    await _resync(request)
    return await asyncio.to_thread(_status)


@router.post("/library/import", response_model=ZipImportResult)
async def import_library_zip(body: ZipImport, request: Request) -> ZipImportResult:
    """Unpack a model library .zip into the local models folder."""
    path = Path(body.zip_path.strip()).expanduser()
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"{path} is not a file."
        )
    try:
        imported = await asyncio.to_thread(model_library.import_models_zip, path)
    except model_library.ModelZipError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Could not write the models ({e}). If an analysis is running, "
                f"let it finish, then try again."
            ),
        ) from e
    await _resync(request)
    return ZipImportResult(imported=imported, status=await asyncio.to_thread(_status))
