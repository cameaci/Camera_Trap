"""
Prebuilt analysis environments ("env packs").

Building an environment with micromamba downloads Python packages from
conda-forge, PyPI and pytorch.org on the user's machine. Instead, CI builds
it once on Windows with this app's own EnvironmentManager, zips it, and
publishes it to this repository's ``runtime`` release. The app then only
downloads from this repository:

    <base>/<env>-<platform>-<yamlsha>.json        manifest
    <base>/<env>-<platform>-<yamlsha>.zip.001..   the zip, in parts

``yamlsha`` is the first 12 hex digits of the bundled YAML's hash, so a
changed YAML asks for a new pack, and an app never installs a pack built
from a different YAML. When no pack exists for the YAML (404), the caller
falls back to building with micromamba.

Build (CI):  python -m app.ml.env_pack build --env wsp-base --out dist-envs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import requests

from app.core.config import RUNTIME_RELEASE_URL, get_settings
from app.core.job_cancellation import JobCancelledError
from app.core.logging_config import get_logger

logger = get_logger(__name__)

ProgressCallback = Callable[[str, float], None]

# GitHub release assets are limited to 2 GiB each; stay well below.
PART_SIZE = 1_900_000_000
# Text files up to this size are scanned for the build machine's prefix.
_PREFIX_SCAN_MAX_BYTES = 8 * 1024 * 1024
_CHUNK = 4 * 1024 * 1024

# Imported in the freshly built env before it is packed: MegaDetector, the
# classifier stack (SpeciesNet needs onnx2torch) and the image libraries.
_SMOKE_TEST = (
    "import torch, torchvision, onnx2torch, cv2, numpy, PIL, piexif, exifread; "
    "import megadetector.detection.run_detector_batch; "
    "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
)


class EnvPackError(RuntimeError):
    """A pack exists but could not be installed."""


def platform_tag() -> str | None:
    """The platform packs are built for, or None where there are none."""
    machine = platform.machine().lower()
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        return "win-64"
    if sys.platform.startswith("linux") and machine in ("x86_64", "amd64"):
        return "linux-64"
    return None


def pack_stem(env_name: str, yaml_sha: str, tag: str) -> str:
    """`env-wsp-base` + sha + tag -> `env-wsp-base-win-64-<sha12>`."""
    return f"{env_name}-{tag}-{yaml_sha[:12]}"


def pack_base_url() -> str:
    return (get_settings().env_pack_url or RUNTIME_RELEASE_URL).rstrip("/")


# ---------------------------------------------------------------------------
# Installing (the app)
# ---------------------------------------------------------------------------


def install_env_pack(
    env_name: str,
    env_path: Path,
    yaml_sha: str,
    progress_callback: ProgressCallback | None = None,
    timeout: float = 60.0,
    should_cancel: Callable[[], bool] | None = None,
) -> bool:
    """
    Install a prebuilt environment at `env_path`.

    Returns False when there is no pack for this platform and YAML (the
    caller then builds the environment itself). Raises EnvPackError when a
    pack exists but cannot be installed, and JobCancelledError when
    `should_cancel` says so during the download; either way nothing is
    left at `env_path`.
    """
    tag = platform_tag()
    if tag is None:
        return False
    stem = pack_stem(env_name, yaml_sha, tag)
    base = pack_base_url()
    try:
        response = requests.get(f"{base}/{stem}.json", timeout=timeout)
    except requests.RequestException as e:
        logger.warning(f"Could not reach the environment pack at {base}: {e}")
        return False
    if response.status_code == 404:
        logger.info(f"No prebuilt environment {stem} published; building locally")
        return False
    if response.status_code >= 400:
        logger.warning(f"Environment pack manifest answered HTTP {response.status_code}")
        return False
    try:
        manifest = response.json()
        parts: list[str] = manifest["parts"]
        expected_sha = manifest["sha256"]
        total = int(manifest["size"])
        old_prefix = manifest.get("prefix", "")
    except (ValueError, KeyError, TypeError) as e:
        raise EnvPackError(f"The environment pack manifest {stem}.json is invalid: {e}") from e

    work = env_path.parent / f".{env_path.name}.pack"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    zip_path = work / "pack.zip"
    try:
        digest = hashlib.sha256()
        done = 0
        with open(zip_path, "wb") as out:
            for part in parts:
                with requests.get(f"{base}/{part}", stream=True, timeout=timeout) as r:
                    if r.status_code >= 400:
                        raise EnvPackError(f"Downloading {part} failed: HTTP {r.status_code}")
                    for chunk in r.iter_content(chunk_size=_CHUNK):
                        if should_cancel is not None and should_cancel():
                            raise JobCancelledError("Environment download cancelled")
                        out.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        if progress_callback and total:
                            progress_callback(
                                f"Downloading the analysis environment "
                                f"({done / 1e6:.0f} / {total / 1e6:.0f} MB)",
                                0.85 * min(done / total, 1.0),
                            )
        if done != total or digest.hexdigest() != expected_sha:
            raise EnvPackError(
                "The downloaded analysis environment is incomplete or damaged; try again."
            )

        if progress_callback:
            progress_callback("Unpacking the analysis environment...", 0.88)
        staging = work / env_path.name
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(staging)
        zip_path.unlink()
        if progress_callback:
            progress_callback("Finishing the analysis environment...", 0.96)
        if old_prefix:
            # Rewrite for the final location, then move the folder there.
            n = _rewrite_prefix(staging, old_prefix, str(env_path))
            logger.info(f"Relocated {n} file(s) in the environment pack")
        if env_path.exists():
            shutil.rmtree(env_path)
        staging.rename(env_path)
    except EnvPackError:
        raise
    except (OSError, zipfile.BadZipFile, requests.RequestException) as e:
        raise EnvPackError(f"Installing the analysis environment failed: {e}") from e
    finally:
        shutil.rmtree(work, ignore_errors=True)
    logger.info(f"Installed prebuilt environment {stem} at {env_path}")
    return True


def _rewrite_prefix(root: Path, old_prefix: str, new_prefix: str) -> int:
    """
    Replace the build machine's prefix with `new_prefix` in the text files
    under `root`, the way conda-unpack does. Python itself is relocatable;
    what embeds the prefix are conda's text files (activation scripts,
    .pth files, configs). Returns the number of files rewritten.
    """
    pairs: list[tuple[bytes, bytes]] = []
    for old, new in (
        (old_prefix, new_prefix),
        (old_prefix.replace("\\", "/"), new_prefix.replace("\\", "/")),
        (old_prefix.replace("\\", "\\\\"), new_prefix.replace("\\", "\\\\")),
    ):
        pair = (old.encode(), new.encode())
        if old and pair not in pairs:
            pairs.append(pair)
    count = 0
    for path in root.rglob("*"):
        try:
            if not path.is_file() or path.stat().st_size > _PREFIX_SCAN_MAX_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:4096]:
            continue  # binary
        new_data = data
        for old, new in pairs:
            new_data = new_data.replace(old, new)
        if new_data != data:
            path.write_bytes(new_data)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Building (CI)
# ---------------------------------------------------------------------------


def write_pack(env_dir: Path, env_name: str, yaml_sha: str, tag: str,
               prefix: str, out_dir: Path, part_size: int = PART_SIZE) -> Path:
    """Zip `env_dir`, split it into parts and write the manifest. Returns it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pack_stem(env_name, yaml_sha, tag)
    zip_path = out_dir / f"{stem}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as z:
        for path in sorted(env_dir.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(env_dir).as_posix())
    digest = hashlib.sha256()
    parts: list[str] = []
    size = zip_path.stat().st_size
    with open(zip_path, "rb") as f:
        index = 1
        while True:
            chunk = f.read(part_size)
            if not chunk:
                break
            digest.update(chunk)
            name = f"{stem}.zip.{index:03d}"
            (out_dir / name).write_bytes(chunk)
            parts.append(name)
            index += 1
    zip_path.unlink()
    manifest = {
        "env": env_name,
        "platform": tag,
        "yaml_sha256": yaml_sha,
        "prefix": prefix,
        "size": size,
        "sha256": digest.hexdigest(),
        "parts": parts,
    }
    manifest_path = out_dir / f"{stem}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _build(env: str, out: Path) -> Path:
    """Build `env` with the app's own EnvironmentManager and pack it."""
    from app.ml.environment_manager import (
        ENV_YAML_SHA_FILENAME,
        EnvironmentManager,
        hash_yaml_file,
    )
    from app.ml.schemas.model_manifest import ModelManifest

    tag = platform_tag()
    if tag is None:
        raise SystemExit("Env packs are built on win-64 or linux-64 only")
    # Build from the YAML, never by downloading an already published pack.
    os.environ["WSP_ENV_PACK_URL"] = "http://127.0.0.1:9/building"
    envs_root = Path(tempfile.mkdtemp(prefix="wspct-envs-"))
    manager = EnvironmentManager(envs_dir=envs_root / "envs")
    manifest = ModelManifest(
        model_id=f"pack-{env}", friendly_name=env, env=env, model_fname="-",
        description="env pack build", developer="WSP", info_url="-",
        min_app_version="0.1.0",
    )
    env_path = manager.get_or_create_env(
        manifest, lambda m, p: print(f"[{p:5.1%}] {m}", flush=True)
    )
    yaml_sha = hash_yaml_file(manager.get_env_yaml_path(env))
    assert (env_path / ENV_YAML_SHA_FILENAME).read_text().strip() == yaml_sha
    # Everything the workers import must load from the built env.
    python = manager.get_python(env_path.name)
    subprocess.run(
        [str(python), "-c", _SMOKE_TEST],
        check=True,
        timeout=600,
    )
    # micromamba built it in a temp folder next to env_path and renamed it;
    # that temp path is the prefix baked into its files.
    prefix = str(env_path.parent / f".{env_path.name}.tmp")
    return write_pack(env_path, env_path.name, yaml_sha, tag, prefix, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a prebuilt environment pack")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--env", default="wsp-base")
    b.add_argument("--out", default="dist-envs")
    args = parser.parse_args(argv)
    manifest = _build(args.env, Path(args.out))
    print(manifest.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
