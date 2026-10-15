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

   The library is a folder (a SharePoint library synced by OneDrive, or a
   network share). "Installing" a model copies its folder from it. The
   same layout also travels as one .zip (wsp/tools/wsp_library.py
   bundle), which File > WSP model library unpacks straight into the
   local models folder (import_models_zip); its models.json then sits in
   the local models folder and is read like the library's.
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
import threading
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
            f"Install the model library .zip under File > WSP model library, "
            f"or copy the model folder to {local_dir}."
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
    the usual OneDrive locations. A configured folder that does not exist
    (OneDrive not synced yet, share offline) is reported as missing rather
    than silently replaced by a guess.
    """
    settings = get_settings()
    configured = _configured_dir()
    if configured is not None:
        return configured if is_library_dir(configured) else None
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
# Installing a library .zip into the local models folder
# ---------------------------------------------------------------------------


class ModelZipError(RuntimeError):
    """The file is not a model library .zip. Worded for the user."""


_IMPORT_LOCK = threading.Lock()


def read_local_catalog() -> dict | None:
    """models.json in the local models folder (left by an imported .zip), or None."""
    path = get_settings().models_dir / LIBRARY_CATALOG_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.error(f"Could not read the model catalog {path}: {e}")
        return None


def _merge_catalogs(old: dict | None, new: dict) -> dict:
    """new's entries over old's, by model_id, per model type."""
    merged: dict = {"models": {}}
    old_models = (old or {}).get("models") or {}
    new_models = new.get("models") or {}
    for model_type in sorted(set(old_models) | set(new_models)):
        entries = {
            e["model_id"]: e
            for e in [*(old_models.get(model_type) or []), *(new_models.get(model_type) or [])]
            if isinstance(e, dict) and isinstance(e.get("model_id"), str)
        }
        merged["models"][model_type] = list(entries.values())
    return merged


def import_models_zip(
    zip_path: Path, progress_callback: ProgressCallback | None = None
) -> list[str]:
    """
    Unpack a model library .zip (wsp_library.py bundle) into the local
    models folder, next to the models already there.

    Files in the .zip replace local files of the same name; nothing else
    is removed. The .zip's models.json is merged into the local one, so a
    .zip holding only a new model adds it without hiding the others.
    Returns the imported model folders as "<type>/<id>".

    Raises:
        ModelZipError: the file is not a .zip, or has no models.json.
    """
    models_dir = get_settings().models_dir
    root = models_dir.resolve()
    with _IMPORT_LOCK:
        try:
            bundle = zipfile.ZipFile(zip_path)
        except (OSError, zipfile.BadZipFile) as e:
            raise ModelZipError(f"{Path(zip_path).name} is not a .zip file: {e}") from e
        with bundle:
            infos = [i for i in bundle.infolist() if not i.is_dir()]
            catalogs = sorted(
                (i.filename for i in infos
                 if i.filename.rsplit("/", 1)[-1] == LIBRARY_CATALOG_NAME
                 and i.filename.count("/") <= 2),
                key=lambda n: n.count("/"),
            )
            if not catalogs:
                raise ModelZipError(
                    f"{Path(zip_path).name} is not a WSP model library .zip "
                    f"(it has no models.json). Build it with "
                    f"wsp/tools/wsp_library.py bundle."
                )
            prefix = catalogs[0][: -len(LIBRARY_CATALOG_NAME)]
            members = []
            for info in infos:
                if not info.filename.startswith(prefix):
                    continue
                rel = info.filename[len(prefix):]
                parts = rel.split("/")
                if (
                    rel == LIBRARY_CATALOG_NAME
                    or parts[-1] in IGNORED_FILES
                    or "__pycache__" in parts
                    or rel.endswith(".pyc")
                ):
                    continue
                dest = (models_dir / rel).resolve()
                if root not in dest.parents:
                    raise ModelZipError(f"Unsafe path in {Path(zip_path).name}: {info.filename}")
                members.append((info, dest))

            total = sum(i.file_size for i, _ in members) or 1
            done = 0
            imported: set[str] = set()
            for info, dest in members:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".tmp")
                try:
                    with bundle.open(info) as fin, open(tmp, "wb") as fout:
                        while chunk := fin.read(_CHUNK):
                            fout.write(chunk)
                            done += len(chunk)
                            if progress_callback:
                                progress_callback(
                                    f"Installing models ({done / 1e6:.0f} / {total / 1e6:.0f} MB)",
                                    min(done / total, 1.0),
                                )
                    tmp.replace(dest)
                finally:
                    if tmp.exists():
                        tmp.unlink()
                rel_parts = dest.relative_to(root).parts
                if len(rel_parts) >= 3:
                    imported.add(f"{rel_parts[0]}/{rel_parts[1]}")

            try:
                new_catalog = json.loads(bundle.read(catalogs[0]).decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                raise ModelZipError(
                    f"The models.json in {Path(zip_path).name} is invalid: {e}"
                ) from e
        catalog_path = models_dir / LIBRARY_CATALOG_NAME
        merged = _merge_catalogs(read_local_catalog(), new_catalog)
        tmp = catalog_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(catalog_path)
    logger.info(f"Installed {len(imported)} model(s) from {zip_path}: {sorted(imported)}")
    return sorted(imported)


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
        # No copystat: a library synced read-only from SharePoint would pass
        # its read-only attribute on, and Windows then refuses to replace
        # the file when the model is updated.
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
    last: str | None = None,
) -> list[str]:
    """
    Copy a model folder from the library into the local models folder.

    Args:
        src: The model's folder in the library.
        dst: The model's local folder.
        include: Only these relative paths (posix style). None copies all.
        overwrite: Copy even when a local file of the same size exists.
        last: Copy this relative path (posix style) after all the others.
            An install passes the weights file, which is what marks the
            model as installed, so a copy that fails part way never leaves
            weights without the files next to them.

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
    todo.sort(key=lambda rel: rel.as_posix() == last)
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
