"""
Model catalog updater - reads the WSP catalog and creates stubs for its models.

Following DEVELOPERS.md principles:
- Fail silently if offline (non-critical operation)
- Never overwrite existing model directories
- Log all operations for debugging
"""

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.ml import model_library

logger = get_logger(__name__)


def _bundled_catalog_path() -> Path | None:
    """
    The catalog shipped inside the app (wsp/models.json), or None.

    Two locations: the PyInstaller bundle when frozen, the repo when
    running from source.
    """
    candidates: list[Path] = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "wsp" / "models.json")
    candidates.append(Path(__file__).resolve().parents[3] / "wsp" / "models.json")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _validate_catalog(catalog: Any) -> dict[str, Any] | None:
    """The catalog, or None when it is not shaped like one."""
    if not isinstance(catalog, dict) or "models" not in catalog:
        logger.error("Invalid catalog structure: missing 'models'")
        return None
    if "det" not in catalog["models"] or "cls" not in catalog["models"]:
        logger.error("Invalid catalog structure: missing 'det' or 'cls' in models")
        return None
    # One malformed entry (a hand-edited library models.json) must not take
    # the whole sync down, the shipped models included: drop it and go on.
    models: dict[str, list[dict[str, Any]]] = {}
    for model_type, entries in catalog["models"].items():
        if not isinstance(entries, list):
            logger.error(f"Invalid catalog structure: '{model_type}' is not a list")
            return None
        models[model_type] = []
        for entry in entries:
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("model_id"), str)
                and isinstance(entry.get("model_fname"), str)
            ):
                models[model_type].append(entry)
            else:
                logger.error(f"Skipping a catalog entry without model_id/model_fname: {entry!r}")
    return {**catalog, "models": models}

# Names of envs whose drift we surface in the toast. Kept here rather
# than in EnvironmentManager because env_manager treats env_name as an
# opaque parameter; this list tracks which ones the app actually ships.
_DRIFT_CHECKED_ENVS: tuple[str, ...] = (
    "wsp-base",
    "pytorch",
    "pywildlife",
    "tensorflow-v1",
    "tensorflow-v2",
)


def find_drifted_envs() -> list[dict[str, str]]:
    """
    Which shipped envs no longer match the YAML this app version carries.

    Reads a 64-byte sentinel and hashes a small YAML per env, all local,
    so this is cheap enough to call per request. `GET /api/ml/updates`
    does exactly that rather than serving the startup snapshot: rebuilding
    an env fixes the sentinel, and a user who then reloads the window must
    not be told again to rebuild what they just rebuilt.

    An env that is not installed, or predates the sentinel, reports
    nothing (`check_yaml_drift` returns None) and is skipped.
    """
    from app.ml.environment_manager import EnvironmentManager

    env_manager = EnvironmentManager()
    drifted: list[dict[str, str]] = []
    for env_name in _DRIFT_CHECKED_ENVS:
        try:
            has_drifted = env_manager.check_yaml_drift(env_name)
        except Exception as e:
            logger.warning(f"Env drift check for {env_name} raised: {e}")
            continue
        if has_drifted:
            drifted.append({"env_name": env_name})
    return drifted


class ModelCatalogUpdater:
    """
    Fetches central model catalog and creates local directory stubs for new models.

    Only creates manifest.json files - does not download weights.
    """

    def __init__(self, models_dir: Path | None = None):
        """
        Args:
            models_dir: Directory where models are stored (default: settings.models_dir)
        """
        self.models_dir = models_dir or get_settings().models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def fetch_catalog(self) -> dict[str, Any] | None:
        """
        The WSP model library's catalog merged over the shipped one.

        The shipped catalog is what keeps an unreachable library from
        emptying the app: manifest.json is written from this catalog and
        nothing else writes it, and ManifestManager skips any model
        directory without one.

        Never raises; returns None when neither source yields a catalog.
        """
        return self._library_catalog()

    def _library_catalog(self) -> dict[str, Any] | None:
        """
        WSP: the WSP model library's models.json, falling back to the
        catalog shipped with the app. Entries in the library override
        shipped entries with the same model_id, and the library can add
        models the app has never heard of (a new WSP model), which is how
        a model is published without a new app release.
        """
        bundled = self._bundled_catalog()
        library = _validate_catalog(model_library.read_library_catalog())
        if library is None:
            return bundled
        if bundled is None:
            return library
        merged: dict[str, Any] = {"models": {}}
        for model_type in ("det", "cls", "emb"):
            entries = {
                m["model_id"]: m for m in bundled["models"].get(model_type, [])
            }
            for m in library["models"].get(model_type, []):
                entries[m["model_id"]] = m
            merged["models"][model_type] = list(entries.values())
        logger.info(
            "Using the WSP model library catalog: "
            + ", ".join(f"{len(v)} {k}" for k, v in merged["models"].items())
        )
        return merged

    def _bundled_catalog(self) -> dict[str, Any] | None:
        """The catalog shipped with the app, or None if it cannot be read."""
        path = _bundled_catalog_path()
        if path is None:
            logger.error("No bundled models.json to fall back on")
            return None
        try:
            catalog = _validate_catalog(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.error(f"Failed to read bundled catalog {path}: {e}", exc_info=True)
            return None
        if catalog is not None:
            logger.warning(f"Using the model catalog shipped with the app: {path}")
        return catalog

    def get_local_models(self) -> dict[str, set[str]]:
        """
        Scan local models directory and return existing model IDs.

        Returns:
            Dict with 'det' and 'cls' keys, values are sets of model_ids
        """
        local_models: dict[str, set[str]] = {"det": set(), "cls": set(), "emb": set()}

        for model_type in ["det", "cls", "emb"]:
            type_dir = self.models_dir / model_type
            if not type_dir.exists():
                continue

            for model_dir in type_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "manifest.json").exists():
                    local_models[model_type].add(model_dir.name)

        logger.debug(
            f"Found {len(local_models['det'])} local det models, "
            f"{len(local_models['cls'])} local cls models, "
            f"{len(local_models['emb'])} local emb models"
        )
        return local_models

    def download_taxonomy(self, model_id: str, model_dir: Path) -> None:
        """
        Copy taxonomy.csv for a model from the WSP model library, if it has
        one. Never raises.
        """
        src = model_library.library_model_dir(model_dir.parent.name, model_id)
        if src is None or not (src / "taxonomy.csv").is_file():
            logger.debug(f"No taxonomy.csv in the library for {model_id}")
            return
        try:
            model_library.copy_model(src, model_dir, include={"taxonomy.csv"})
            logger.info(f"Copied taxonomy.csv for {model_id} from the library")
        except Exception as e:
            logger.warning(f"Failed to copy taxonomy.csv for {model_id}: {e}")

    def write_manifest(
        self, model_type: str, manifest_data: dict[str, Any]
    ) -> str:
        """
        Idempotently sync the local manifest.json for a model with the
        central catalog. Creates the model directory, refreshes the
        manifest in place when the catalog has newer content (citation,
        URL, license, friendly_name etc.), no-ops when content is
        identical, and fetches taxonomy.csv whenever it is missing.

        Returns one of "created" / "updated" / "unchanged" so the caller
        can decide what to surface in the UI. Never raises; logs and
        returns "unchanged" on unexpected I/O errors so a single bad
        entry can't take the whole sync down.
        """
        model_id = manifest_data["model_id"]
        model_dir = self.models_dir / model_type / model_id
        manifest_path = model_dir / "manifest.json"
        is_new_dir = not model_dir.exists()

        try:
            # Compare existing content. Identical bytes-or-equivalent
            # JSON means the catalog hasn't moved and the file is left
            # alone, but we still fall through to the taxonomy check.
            unchanged = False
            if manifest_path.exists():
                try:
                    with open(manifest_path, encoding="utf-8") as f:
                        existing = json.load(f)
                    unchanged = existing == manifest_data
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        f"Existing manifest at {manifest_path} unreadable, "
                        f"will overwrite: {e}"
                    )

            if not unchanged:
                model_dir.mkdir(parents=True, exist_ok=True)
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest_data, f, indent=2)

            # Taxonomy ships in the model folder, not the catalog, so fetch it
            # whenever it is missing rather than only on first creation.
            # This check must sit outside the `unchanged` branch: a stub
            # whose taxonomy never landed (model published before its
            # taxonomy.csv existed, or first synced while offline) has a
            # perfectly unchanged manifest, so an early return would leave
            # it broken forever, on a flat label list with no rollup.
            # Present file means no request, so once a model's taxonomy is
            # on disk this costs nothing; a repo that genuinely has no
            # taxonomy.csv pays one cheap 404 per launch.
            if model_type == "cls" and not (model_dir / "taxonomy.csv").exists():
                self.download_taxonomy(model_id, model_dir)

            if unchanged:
                return "unchanged"

            if is_new_dir:
                logger.info(f"Created manifest stub for {model_type}/{model_id}")
                return "created"

            logger.info(f"Refreshed manifest for {model_type}/{model_id}")
            return "updated"

        except Exception as e:
            logger.error(
                f"Failed to sync manifest for {model_type}/{model_id}: {e}",
                exc_info=True,
            )
            return "unchanged"

    async def _find_stale_files(
        self, model_type: str, manifest_data: dict[str, Any]
    ) -> list[str] | None:
        """
        Relative paths of this model's local files that no longer match the
        WSP model library, or None when the model is not installed or the
        library does not have it.

        Only installed models can be stale, so a catalog stub (a manifest
        with no weights next to it) is skipped.
        """
        model_dir = self.models_dir / model_type / manifest_data["model_id"]
        if not (model_dir / manifest_data["model_fname"]).is_file():
            return None

        def compare() -> list[str] | None:
            src = model_library.library_model_dir(model_type, manifest_data["model_id"])
            if src is None:
                return None
            return model_library.find_stale_files(model_dir, src)

        # Off the event loop, locating the library included: a slow or
        # unreachable network share must not stall the server.
        return await asyncio.to_thread(compare)

    async def sync(self, refresh_library: bool = True) -> dict[str, Any]:
        """
        Fetch the central catalog, then for every entry write the local
        manifest.json: create on first appearance, refresh in place when
        the catalog moved (citation, URL, license, friendly_name, etc.),
        no-op when identical. Idempotent: safe to run on every startup.

        `refresh_library=False` skips checking the library link, for a
        caller that has just downloaded it.

        Everything that reads the library runs in a worker thread: it may
        be a network share, and an unreachable one must not block the
        event loop and with it every other request.

        Returns:
            {
                "new_models":       [{"model_id", "friendly_name", "emoji"}, ...],
                "refreshed_models": [{"model_id", "friendly_name"}, ...],
                "drifted_models":   [{"model_id", "friendly_name", "emoji"}, ...],
                    installed models with at least one file that no longer
                    matches the library. The file names go to the log rather
                    than over the wire: nothing renders them, and this
                    snapshot goes stale the moment the library changes, so the
                    update endpoint recomputes the list itself.
                "drifted_envs":     [{"env_name"}, ...],
                "checked_at":       "<UTC ISO timestamp>",
                "error":            "<message>" (only if fetch failed),
            }

        Note: async so the lifespan startup task doesn't block boot.
        """
        result: dict[str, Any] = {
            "new_models": [],
            "refreshed_models": [],
            "drifted_models": [],
            "drifted_envs": [],
            "checked_at": datetime.now(UTC).isoformat(),
        }

        try:
            # A linked library (OneDrive share link) is refreshed first, so a
            # model published since the last launch is in the catalog below.
            try:
                if refresh_library:
                    await asyncio.to_thread(model_library.sync_library_url)
            except model_library.LibraryDownloadError as e:
                logger.warning(f"WSP model library link not refreshed: {e}")
                result["library_error"] = str(e)
            except Exception as e:
                logger.error(f"WSP model library link sync failed: {e}", exc_info=True)
                result["library_error"] = str(e)

            catalog = await asyncio.to_thread(self.fetch_catalog)
            if catalog is None:
                result["error"] = "Failed to fetch catalog"
                return result

            local_models = self.get_local_models()
            total_local = sum(len(s) for s in local_models.values())
            is_fresh_install = total_local == 0

            for model_type in ["det", "cls", "emb"]:
                for manifest_data in catalog["models"].get(model_type, []):
                    # A thread: a missing taxonomy.csv is copied from the library.
                    state = await asyncio.to_thread(
                        self.write_manifest, model_type, manifest_data
                    )

                    if state == "created" and not is_fresh_install:
                        # Surface as "new model" toast on existing
                        # installs only. Fresh installs just want the
                        # catalog to populate silently.
                        result["new_models"].append(
                            {
                                "model_id": manifest_data["model_id"],
                                "friendly_name": manifest_data.get(
                                    "friendly_name", manifest_data["model_id"]
                                ),
                                "emoji": manifest_data.get("emoji", "🤖"),
                            }
                        )
                    elif state == "updated":
                        result["refreshed_models"].append(
                            {
                                "model_id": manifest_data["model_id"],
                                "friendly_name": manifest_data.get(
                                    "friendly_name", manifest_data["model_id"]
                                ),
                            }
                        )

                    # Compare the installed files against the library.
                    # Skipped on fresh installs: nothing is on disk yet.
                    if not is_fresh_install:
                        stale = await self._find_stale_files(model_type, manifest_data)
                        if stale:
                            logger.info(
                                f"{model_type}/{manifest_data['model_id']} has "
                                f"{len(stale)} file(s) to update: {', '.join(stale)}"
                            )
                            result["drifted_models"].append(
                                {
                                    "model_id": manifest_data["model_id"],
                                    "friendly_name": manifest_data.get(
                                        "friendly_name", manifest_data["model_id"]
                                    ),
                                    "emoji": manifest_data.get("emoji", "🤖"),
                                }
                            )

            # Env drift: hash each shipped env's bundled YAML and
            # compare to the sentinel written when the env was built.
            # Done outside the catalog loop because envs are shipped
            # by the app, not by the central models.json. Recorded here
            # only so the count reaches the log; the endpoint recomputes
            # it per request so a rebuild takes effect immediately.
            if not is_fresh_install:
                result["drifted_envs"] = find_drifted_envs()

            if is_fresh_install:
                # On first launch every entry is "created"; no point
                # listing them; the setup wizard handles weight downloads.
                total_entries = sum(
                    len(catalog["models"].get(t, []))
                    for t in ("det", "cls", "emb")
                )
                logger.info(
                    f"Model catalog sync complete: catalog initialized "
                    f"({total_entries} entries)"
                )
            else:
                logger.info(
                    f"Model catalog sync complete: "
                    f"{len(result['new_models'])} new, "
                    f"{len(result['refreshed_models'])} refreshed, "
                    f"{len(result['drifted_models'])} model(s) with files to update, "
                    f"{len(result['drifted_envs'])} env(s) drifted"
                )

            return result

        except Exception as e:
            logger.error(f"Model catalog sync failed: {e}", exc_info=True)
            result["error"] = str(e)
            return result
