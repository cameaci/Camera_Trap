"""
The WSP model library: where WSP CameraTrap installs its models from.

Models come from places the WSP network allows:

1. The local models folder (``~/WSP-CameraTrap/models/<type>/<id>/``).
   A model folder copied there by hand is used as is.
2. The WSP model library, laid out like the local models folder and
   holding the catalog next to it::

       <library>/models.json
       <library>/det/MD5A-0-0/md_v5a.0.0.pt
       <library>/cls/SPECIESNET-v4-0-2-A/...

   The library is either a folder (a SharePoint library synced by
   OneDrive, or a network share) or a .zip of that folder behind a
   OneDrive/SharePoint share link. A linked library is downloaded into
   ``<user data>/library-cache/`` whenever the file behind the link
   changes, and then used exactly like a folder. "Installing" a model
   copies its folder from the library.
3. A ``download_url`` in the catalog entry, for a public single-file
   weight such as MegaDetector in this repository's releases.

The copy follows these rules:
every file lands in a ``.tmp`` sibling and is renamed into place once it
is complete, a file already present with the same size is skipped, a
failure leaves what was copied so a retry only fetches what is missing,
and a cancel raises ``JobCancelledError`` for the caller to clean up.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import urllib.parse
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

import requests

from app.core.config import get_settings
from app.core.job_cancellation import JobCancelledError
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# The folder name the app looks for when no library path is configured:
# "<OneDrive or SharePoint root>/WSP CameraTrap/models".
LIBRARY_FOLDER_NAME = "WSP CameraTrap"
LIBRARY_MODELS_SUBDIR = "models"
LIBRARY_CATALOG_NAME = "models.json"

# Files that are never copied from, or compared against, the library.
# manifest.json is written from the catalog by the catalog updater, so a
# copy of it in the library would fight that writer.
IGNORED_FILES = frozenset(
    {"manifest.json", ".DS_Store", "Thumbs.db", "desktop.ini"}
)

# Files up to this size are compared by content when checking for
# updates; bigger ones (the weights) by size only. Weights are versioned
# by model_id, as upstream does: publish a new id rather than replacing
# a weights file in place.
CONTENT_COMPARE_MAX_BYTES = 20 * 1024 * 1024

_CHUNK = 4 * 1024 * 1024

ProgressCallback = Callable[[str, float], None]
CancelCheck = Callable[[], bool]


class ModelSourceMissingError(RuntimeError):
    """No folder, library or URL can provide this model. Worded for the user."""

    def __init__(self, model_id: str, friendly_name: str, local_dir: Path) -> None:
        self.model_id = model_id
        library = get_library_dir()
        where = (
            f"The WSP model library at {library} does not contain it."
            if library
            else "No WSP model library is connected on this computer."
        )
        super().__init__(
            f"{friendly_name} ({model_id}) is not installed. {where} "
            f"Connect the library under File > WSP model library (OneDrive link "
            f"or synced folder), or copy the model folder to {local_dir}."
        )


# ---------------------------------------------------------------------------
# Locating the library
# ---------------------------------------------------------------------------


def _config_path() -> Path:
    return get_settings().user_data_dir / "wsp-config.json"


def read_config() -> dict:
    """The WSP settings file, or {} when it is missing or unreadable."""
    path = _config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning(f"Ignoring unreadable {path}: {e}")
        return {}
    return data if isinstance(data, dict) else {}


def write_library_dir(path: Path | None) -> None:
    """Remember the library folder the user picked (None forgets it)."""
    config = read_config()
    if path is None:
        config.pop("model_library_dir", None)
    else:
        config["model_library_dir"] = str(path)
    _write_config(config)


def _write_config(config: dict) -> None:
    target = _config_path()
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
    tmp.replace(target)


def bundled_config() -> dict:
    """wsp/config.json shipped with the app (deployment defaults), or {}."""
    candidates: list[Path] = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "wsp" / "config.json")
    candidates.append(Path(__file__).resolve().parents[3] / "wsp" / "config.json")
    for path in candidates:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                logger.error(f"Could not read {path}: {e}")
                return {}
            return data if isinstance(data, dict) else {}
    return {}


def get_library_url() -> str | None:
    """
    The share link of the WSP model library bundle, or None.

    Order: WSP_MODEL_LIBRARY_URL, then the link saved in the app, then the
    link shipped in wsp/config.json.
    """
    settings = get_settings()
    if settings.model_library_url is not None:
        return settings.model_library_url.strip() or None
    for source in (read_config(), bundled_config()):
        value = str(source.get("model_library_url") or "").strip()
        if value:
            return value
    return None


def write_library_url(url: str | None) -> None:
    """Remember the library link the user entered (None forgets it)."""
    config = read_config()
    if url:
        config["model_library_url"] = url
    else:
        config.pop("model_library_url", None)
    _write_config(config)


def is_library_dir(path: Path) -> bool:
    """A library holds a catalog or at least one model type folder."""
    if not path.is_dir():
        return False
    if (path / LIBRARY_CATALOG_NAME).is_file():
        return True
    return any((path / t).is_dir() for t in ("det", "cls", "emb"))


def _search_roots() -> list[Path]:
    """Folders OneDrive syncs into on Windows and macOS."""
    roots: list[Path] = []
    for var in ("OneDriveCommercial", "OneDrive"):
        value = os.environ.get(var, "").strip()
        if value:
            roots.append(Path(value))
    home = Path.home()
    try:
        for child in home.iterdir():
            name = child.name.lower()
            if child.is_dir() and (name.startswith("onedrive") or name.startswith("wsp")):
                roots.append(child)
    except OSError:
        pass
    cloud = home / "Library" / "CloudStorage"  # macOS OneDrive location
    if cloud.is_dir():
        try:
            roots.extend(p for p in cloud.iterdir() if p.is_dir())
        except OSError:
            pass
    # Keep order, drop duplicates.
    seen: set[Path] = set()
    unique = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique.append(root)
    return unique


def autodetect_library_dir() -> Path | None:
    """
    Find "<root>/WSP CameraTrap/models" or one level deeper, which is where
    a synced SharePoint library ("<root>/<Site - Library>/WSP CameraTrap")
    or an "Add shortcut to My files" folder lands.
    """
    for root in _search_roots():
        candidates = [root / LIBRARY_FOLDER_NAME / LIBRARY_MODELS_SUBDIR]
        try:
            candidates += [
                child / LIBRARY_FOLDER_NAME / LIBRARY_MODELS_SUBDIR
                for child in root.iterdir()
                if child.is_dir()
            ]
        except OSError:
            continue
        for candidate in candidates:
            if is_library_dir(candidate):
                return candidate
    return None


def _configured_dir() -> Path | None:
    """The folder set by WSP_MODEL_LIBRARY_DIR or saved in the app, or None."""
    configured = get_settings().model_library_dir or read_config().get("model_library_dir")
    return Path(configured).expanduser() if configured else None


def get_library_dir() -> Path | None:
    """
    The WSP model library folder, or None.

    Order: WSP_MODEL_LIBRARY_DIR, then the folder saved in the app, then
    the downloaded copy of the linked library (see sync_library_url), then
    the usual OneDrive locations. A configured folder that does not exist
    (OneDrive not synced yet, share offline) is reported as missing rather
    than silently replaced by a guess.
    """
    settings = get_settings()
    configured = _configured_dir()
    if configured is not None:
        return configured if is_library_dir(configured) else None
    if get_library_url():
        cached = library_cache_dir()
        if is_library_dir(cached):
            return cached
    if not settings.model_library_autodetect:
        return None
    return autodetect_library_dir()


def library_model_dir(model_type: str, model_id: str) -> Path | None:
    """The library's folder for this model, or None."""
    library = get_library_dir()
    if library is None:
        return None
    candidate = library / model_type / model_id
    return candidate if candidate.is_dir() else None


def read_library_catalog() -> dict | None:
    """The library's models.json, or None when there is none or it is unreadable."""
    library = get_library_dir()
    if library is None:
        return None
    path = library / LIBRARY_CATALOG_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.error(f"Could not read the library catalog {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# The linked library (a .zip behind a OneDrive/SharePoint share link)
# ---------------------------------------------------------------------------


class LibraryDownloadError(RuntimeError):
    """The linked library could not be downloaded. Worded for the user."""


_SYNC_LOCK = threading.Lock()
_SOURCE_FILE = ".source.json"


def library_cache_dir() -> Path:
    """Where the linked library is unpacked: <user data>/library-cache/models."""
    return get_settings().user_data_dir / "library-cache" / "models"


def normalize_share_url(url: str) -> str:
    """
    Turn a OneDrive/SharePoint share link into a direct download link.

    A share link opens a preview page; adding download=1 makes OneDrive and
    SharePoint answer with the file itself. Other URLs are left alone.
    """
    parts = urllib.parse.urlsplit(url.strip())
    host = parts.netloc.lower()
    if not (
        host.endswith("sharepoint.com")
        or host.endswith("1drv.ms")
        or host.endswith("onedrive.live.com")
    ):
        return url.strip()
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    if not any(k == "download" for k, _ in query):
        query.append(("download", "1"))
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def _read_source(cache_root: Path) -> dict:
    try:
        return json.loads((cache_root / _SOURCE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _fingerprint(url: str, headers) -> dict:
    return {
        "url": url,
        "etag": headers.get("ETag"),
        "last_modified": headers.get("Last-Modified"),
        "content_length": headers.get("Content-Length"),
    }


def _find_catalog_root(extracted: Path) -> Path | None:
    """The folder inside an unpacked bundle that holds models.json."""
    if (extracted / LIBRARY_CATALOG_NAME).is_file():
        return extracted
    for candidate in sorted(extracted.rglob(LIBRARY_CATALOG_NAME)):
        if len(candidate.relative_to(extracted).parts) <= 3:
            return candidate.parent
    return None


def sync_library_url(
    progress_callback: ProgressCallback | None = None,
    timeout: float = 60.0,
) -> Path | None:
    """
    Download the linked library if the file behind the link changed.

    Returns the unpacked library folder, or None when no link is set or
    a configured folder wins over it (see get_library_dir), in which case
    the download would never be used.
    Skips the download when the link, ETag, Last-Modified and size all
    match the last download. Keeps the previous copy until a new one is
    complete, so a failed download never loses a working library.

    Raises:
        LibraryDownloadError: the link does not serve a library bundle.
    """
    url = get_library_url()
    if not url or _configured_dir() is not None:
        return None
    direct = normalize_share_url(url)
    target = library_cache_dir()
    cache_root = target.parent
    with _SYNC_LOCK:
        cache_root.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(direct, stream=True, timeout=timeout)
        except requests.RequestException as e:
            raise LibraryDownloadError(
                f"Could not reach the WSP model library link: {e}"
            ) from e
        with response:
            if response.status_code in (401, 403):
                raise LibraryDownloadError(
                    "The WSP model library link asks for a sign-in. Share the "
                    "library .zip with 'Anyone with the link can view', or sync "
                    "the OneDrive folder instead."
                )
            if response.status_code >= 400:
                raise LibraryDownloadError(
                    f"The WSP model library link answered HTTP {response.status_code}."
                )
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise LibraryDownloadError(
                    "The WSP model library link opens a web page instead of the "
                    "library .zip (usually a sign-in page). Share the .zip with "
                    "'Anyone with the link can view', or sync the OneDrive folder."
                )
            fingerprint = _fingerprint(url, response.headers)
            previous = _read_source(cache_root)
            unchanged = (
                previous == fingerprint
                and is_library_dir(target)
                and any(fingerprint[k] for k in ("etag", "last_modified", "content_length"))
            )
            if unchanged:
                logger.info("WSP model library link unchanged; using the cached copy")
                return target

            total = int(response.headers.get("Content-Length") or 0)
            zip_path = cache_root / "library.zip.tmp"
            done = 0
            with open(zip_path, "wb") as fout:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    fout.write(chunk)
                    done += len(chunk)
                    if progress_callback and total:
                        progress_callback(
                            f"Downloading the WSP model library "
                            f"({done / 1e6:.0f} / {total / 1e6:.0f} MB)",
                            min(done / total, 1.0),
                        )
        staging = cache_root / "models.new"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        try:
            with zipfile.ZipFile(zip_path) as bundle:
                for member in bundle.namelist():
                    resolved = (staging / member).resolve()
                    if not str(resolved).startswith(str(staging.resolve())):
                        raise LibraryDownloadError(f"Unsafe path in library bundle: {member}")
                bundle.extractall(staging)
        except zipfile.BadZipFile as e:
            raise LibraryDownloadError(
                "The file behind the WSP model library link is not a .zip."
            ) from e
        finally:
            zip_path.unlink(missing_ok=True)
        root = _find_catalog_root(staging)
        if root is None:
            shutil.rmtree(staging, ignore_errors=True)
            raise LibraryDownloadError(
                "The WSP model library .zip has no models.json. Build it with "
                "wsp/tools/wsp_library.py bundle."
            )
        old = cache_root / "models.old"
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        if target.exists():
            target.rename(old)
        root.rename(target)
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)
        (cache_root / _SOURCE_FILE).write_text(json.dumps(fingerprint), encoding="utf-8")
        logger.info(f"Downloaded the WSP model library into {target}")
        return target


# ---------------------------------------------------------------------------
# Copying
# ---------------------------------------------------------------------------


def _library_files(src: Path) -> list[Path]:
    """Every file under src that belongs to the model, as relative paths."""
    files = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        if path.name in IGNORED_FILES or path.name.endswith(".tmp"):
            continue
        files.append(path.relative_to(src))
    return files


def _copy_one(
    src: Path,
    dst: Path,
    on_bytes: Callable[[int], None],
    should_cancel: CancelCheck | None,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            while True:
                if should_cancel is not None and should_cancel():
                    raise JobCancelledError("Model copy cancelled")
                chunk = fin.read(_CHUNK)
                if not chunk:
                    break
                fout.write(chunk)
                on_bytes(len(chunk))
        if tmp.stat().st_size != src.stat().st_size:
            raise OSError(f"Copied size of {src.name} does not match the source")
        shutil.copystat(src, tmp)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def copy_model(
    src: Path,
    dst: Path,
    progress_callback: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
    include: Iterable[str] | None = None,
    overwrite: bool = False,
) -> list[str]:
    """
    Copy a model folder from the library into the local models folder.

    Args:
        src: The model's folder in the library.
        dst: The model's local folder.
        include: Only these relative paths (posix style). None copies all.
        overwrite: Copy even when a local file of the same size exists.

    Returns:
        The relative paths that were copied.
    """
    wanted = set(include) if include is not None else None
    files = [
        rel for rel in _library_files(src)
        if wanted is None or rel.as_posix() in wanted
    ]
    todo = [
        rel for rel in files
        if overwrite
        or not (dst / rel).is_file()
        or (dst / rel).stat().st_size != (src / rel).stat().st_size
    ]
    total = sum((src / rel).stat().st_size for rel in todo) or 1
    done = 0

    def on_bytes(n: int) -> None:
        nonlocal done
        done += n
        if progress_callback:
            mb_done, mb_total = done / 1e6, total / 1e6
            progress_callback(
                f"Copying from the WSP model library ({mb_done:.0f} / {mb_total:.0f} MB)",
                min(done / total, 1.0),
            )

    for rel in todo:
        _copy_one(src / rel, dst / rel, on_bytes, should_cancel)
    if progress_callback:
        progress_callback("Copy complete", 1.0)
    return [rel.as_posix() for rel in todo]


def download_url(
    url: str,
    dst: Path,
    progress_callback: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
    timeout: float = 60.0,
) -> None:
    """Stream one public file (e.g. a GitHub release asset) to dst via a .tmp."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            done = 0
            with open(tmp, "wb") as fout:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if should_cancel is not None and should_cancel():
                        raise JobCancelledError("Model download cancelled")
                    fout.write(chunk)
                    done += len(chunk)
                    if progress_callback and total:
                        progress_callback(
                            f"Downloading {dst.name} ({done / 1e6:.0f} / {total / 1e6:.0f} MB)",
                            min(done / total, 1.0),
                        )
        if total and tmp.stat().st_size != total:
            raise OSError(f"Download of {dst.name} was incomplete")
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Updates
# ---------------------------------------------------------------------------


def _same_content(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    if a.stat().st_size > CONTENT_COMPARE_MAX_BYTES:
        return True
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            ca, cb = fa.read(_CHUNK), fb.read(_CHUNK)
            if ca != cb:
                return False
            if not ca:
                return True


def find_stale_files(model_dir: Path, src: Path) -> list[str] | None:
    """
    Relative paths whose local copy differs from the library, [] when the
    install matches, None when the library folder cannot be read. Local
    files the library does not have are left alone. Never raises.
    """
    try:
        stale = []
        for rel in _library_files(src):
            local = model_dir / rel
            if not local.is_file() or not _same_content(local, src / rel):
                stale.append(rel.as_posix())
        return stale
    except OSError as e:
        logger.warning(f"Could not compare {model_dir} with {src}: {e}")
        return None
