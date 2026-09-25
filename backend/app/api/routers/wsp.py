"""
The WSP model library setting.

The library is where WSP CameraTrap installs models from (see
app/ml/model_library.py): a OneDrive/SharePoint share link to the library
.zip, or a folder (a synced OneDrive folder or a network share). The user
connects it once from File > WSP model library; the choice is saved in
<user data>/wsp-config.json. A new link is downloaded in the background,
and the catalog is re-read afterwards so its models appear at once.
"""

import asyncio
import threading
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


class _DownloadState:
    def __init__(self) -> None:
        self.in_progress = False
        self.progress = 0.0
        self.message = ""
        self.error: str | None = None
        # Set when a download is asked for while one is running (a new
        # link was saved mid-download), so the running task goes again.
        self._again = False
        self._lock = threading.Lock()

    def start(self) -> bool:
        """True when the caller should start the task; else it reruns."""
        with self._lock:
            if self.in_progress:
                self._again = True
                return False
            self.in_progress, self.progress, self.message, self.error = True, 0.0, "", None
            self._again = False
            return True

    def update(self, message: str, progress: float) -> None:
        self.message, self.progress = message, progress

    def finish(self, error: str | None = None) -> bool:
        """End the download, or return True when another pass was asked for."""
        with self._lock:
            if self._again:
                self._again = False
                self.progress, self.message = 0.0, ""
                return True
            self.in_progress, self.error = False, error
            return False


_download = _DownloadState()


class LibraryStatus(BaseModel):
    # The library folder in use, or None when no library is reachable.
    library_dir: str | None
    # Where it came from: "env", "settings", "link", "autodetect" or None.
    source: str | None
    # The folder saved in the app, even when it cannot be reached right now.
    configured_dir: str | None
    # The share link in use (saved in the app or shipped with it), or None.
    library_url: str | None
    models_dir: str
    download_in_progress: bool
    download_progress: float
    download_message: str
    download_error: str | None


class LibraryUpdate(BaseModel):
    # A folder, or None/"" to stop using a folder.
    library_dir: str | None = None
    # A share link, or None/"" to stop using a link saved in the app.
    library_url: str | None = None


def _status() -> LibraryStatus:
    settings = get_settings()
    configured = model_library.read_config().get("model_library_dir")
    url = model_library.get_library_url()
    library = model_library.get_library_dir()
    if library is None:
        source = None
    elif settings.model_library_dir:
        source = "env"
    elif configured:
        source = "settings"
    elif url and library == model_library.library_cache_dir():
        source = "link"
    else:
        source = "autodetect"
    return LibraryStatus(
        library_dir=str(library) if library else None,
        source=source,
        configured_dir=configured,
        library_url=url,
        models_dir=str(settings.models_dir),
        download_in_progress=_download.in_progress,
        download_progress=_download.progress,
        download_message=_download.message,
        download_error=_download.error,
    )


async def _resync(request: Request) -> None:
    try:
        request.app.state.model_updates = await ModelCatalogUpdater().sync()
        if ml_models.manifest_manager is not None:
            ml_models.manifest_manager.load_manifests(force_refresh=True)
    except Exception as e:
        logger.error(f"Catalog sync after a library change failed: {e}", exc_info=True)


# Strong references to the running download tasks: the event loop keeps
# only a weak one, so an unreferenced task can be collected mid-run.
_tasks: set[asyncio.Task] = set()


def _start_download(request: Request) -> None:
    """
    Download the linked library in the background, then re-read the
    catalog. The status reports the download as running until the catalog
    is re-read, so the dialog refreshes the model lists only once the new
    models are there. A request while one runs makes it go once more.
    """
    if not model_library.get_library_url() or not _download.start():
        return

    async def run() -> None:
        again = True
        while again:
            error = None
            try:
                await asyncio.to_thread(model_library.sync_library_url, _download.update)
            except Exception as e:
                logger.error(f"WSP model library download failed: {e}")
                error = str(e)
            try:
                await _resync(request)
            finally:
                again = _download.finish(error)

    task = asyncio.create_task(run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


@router.get("/library", response_model=LibraryStatus)
def get_library() -> LibraryStatus:
    return _status()


@router.post("/library", response_model=LibraryStatus)
async def set_library(update: LibraryUpdate, request: Request) -> LibraryStatus:
    folder = (update.library_dir or "").strip()
    link = (update.library_url or "").strip()

    if folder:
        path = Path(folder).expanduser()
        if not model_library.is_library_dir(path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"{path} is not a WSP model library. Pick the folder that "
                    f"holds models.json (usually 'WSP CameraTrap/models')."
                ),
            )
        model_library.write_library_dir(path)
        await _resync(request)
        return _status()

    if link:
        if not link.lower().startswith(("https://", "http://")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Paste the full share link, starting with https://",
            )
        # A link replaces a saved folder, which would otherwise win.
        model_library.write_library_dir(None)
        model_library.write_library_url(link)
    else:
        # Neither: forget both and go back to the shipped link / OneDrive.
        model_library.write_library_dir(None)
        model_library.write_library_url(None)

    if model_library.get_library_url():
        _start_download(request)
    else:
        await _resync(request)
    return _status()


@router.post("/library/refresh", response_model=LibraryStatus)
async def refresh_library(request: Request) -> LibraryStatus:
    """Check the library link again (downloads only when the file changed)."""
    _start_download(request)
    return _status()
