"""The WSP model library: folders, the OneDrive link, installs and updates."""

import json
import stat
import zipfile
from pathlib import Path

import pytest

from app.core.job_cancellation import JobCancelledError
from app.ml import model_library
from app.ml.catalog_updater import ModelCatalogUpdater
from app.ml.model_storage import ModelStorage
from app.ml.schemas.model_manifest import ModelManifest


@pytest.fixture
def library(tmp_path, monkeypatch) -> Path:
    """An empty library folder, named by the env var."""
    lib = tmp_path / "library"
    lib.mkdir()
    monkeypatch.setenv("WSP_MODEL_LIBRARY_DIR", str(lib))
    return lib


def _manifest(model_id="WSP-UK-v1", fname="model.pt", category="classification", **kw):
    m = ModelManifest(
        model_id=model_id,
        friendly_name=model_id,
        env="wsp-base",
        model_fname=fname,
        description="test",
        developer="WSP",
        info_url="https://example.invalid",
        min_app_version="0.1.0",
        **kw,
    )
    m.model_category = category
    return m


def _entry(model_id="WSP-UK-v1", fname="model.pt"):
    return {
        "model_id": model_id,
        "friendly_name": model_id,
        "env": "wsp-base",
        "model_fname": fname,
        "description": "d",
        "developer": "WSP",
        "info_url": "https://example.invalid",
        "min_app_version": "0.1.0",
    }


def _put(root: Path, rel: str, data: bytes = b"x") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# --- locating the library -------------------------------------------------


def test_env_var_names_the_library(library):
    assert model_library.get_library_dir() is None  # empty folder is not a library
    _put(library, "models.json", b"{}")
    assert model_library.get_library_dir() == library


def test_saved_folder_is_used_when_no_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("WSP_MODEL_LIBRARY_DIR", raising=False)
    lib = tmp_path / "saved"
    _put(lib, "cls/X/model.pt")
    model_library.write_library_dir(lib)
    try:
        assert model_library.get_library_dir() == lib
    finally:
        model_library.write_library_dir(None)
    assert model_library.get_library_dir() is None


def test_a_configured_folder_that_is_missing_is_not_guessed(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_MODEL_LIBRARY_DIR", str(tmp_path / "offline-share"))
    monkeypatch.setenv("WSP_MODEL_LIBRARY_AUTODETECT", "true")
    onedrive = tmp_path / "OneDrive - WSP"
    _put(onedrive / "WSP CameraTrap" / "models", "models.json", b"{}")
    monkeypatch.setenv("OneDriveCommercial", str(onedrive))
    assert model_library.get_library_dir() is None


def test_autodetect_finds_the_onedrive_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("WSP_MODEL_LIBRARY_DIR", raising=False)
    monkeypatch.setenv("WSP_MODEL_LIBRARY_AUTODETECT", "true")
    onedrive = tmp_path / "OneDrive - WSP"
    target = onedrive / "Ecology - Shared" / "WSP CameraTrap" / "models"
    _put(target, "models.json", b"{}")
    monkeypatch.setenv("OneDriveCommercial", str(onedrive))
    assert model_library.autodetect_library_dir() == target


def test_copy_model_copies_files_but_not_manifest(library, tmp_path):
    src = library / "cls" / "WSP-UK-v1"
    _put(src, "model.pt", b"weights")
    _put(src, "sub/inference.py", b"code")
    _put(src, "manifest.json", b"{}")
    dst = tmp_path / "local"
    progress = []

    copied = model_library.copy_model(src, dst, lambda m, p: progress.append(p))

    assert sorted(copied) == ["model.pt", "sub/inference.py"]
    assert (dst / "model.pt").read_bytes() == b"weights"
    assert not (dst / "manifest.json").exists()
    assert progress[-1] == 1.0
    assert not list(dst.rglob("*.tmp"))


def test_copy_model_skips_a_file_already_there(library, tmp_path):
    src = library / "cls" / "M"
    _put(src, "model.pt", b"abc")
    dst = tmp_path / "local"
    _put(dst, "model.pt", b"xyz")  # same size: treated as complete
    assert model_library.copy_model(src, dst) == []
    assert model_library.copy_model(src, dst, overwrite=True) == ["model.pt"]
    assert (dst / "model.pt").read_bytes() == b"abc"


def test_a_read_only_library_file_is_installed_writable(library, tmp_path):
    """OneDrive syncs a view-only SharePoint library read-only. The local
    copy must not inherit that, or Windows refuses to replace it when the
    model is updated."""
    src = library / "cls" / "M"
    _put(src, "inference.py", b"v1").chmod(0o444)
    dst = tmp_path / "local"

    model_library.copy_model(src, dst)

    assert (dst / "inference.py").stat().st_mode & stat.S_IWUSR
    (src / "inference.py").chmod(0o644)
    (src / "inference.py").write_bytes(b"v2")
    model_library.copy_model(src, dst, overwrite=True)
    assert (dst / "inference.py").read_bytes() == b"v2"


def test_cancel_raises_and_leaves_no_partial_file(library, tmp_path):
    src = library / "cls" / "M"
    _put(src, "model.pt", b"x" * 10)
    dst = tmp_path / "local"
    with pytest.raises(JobCancelledError):
        model_library.copy_model(src, dst, should_cancel=lambda: True)
    assert not (dst / "model.pt").exists()
    assert not list(dst.rglob("*.tmp"))


def test_find_stale_files_compares_content(library, tmp_path):
    src = library / "cls" / "M"
    _put(src, "model.pt", b"weights")
    _put(src, "inference.py", b"v2")
    _put(src, "taxonomy.csv", b"a,b")
    dst = tmp_path / "local"
    _put(dst, "model.pt", b"weights")
    _put(dst, "inference.py", b"v1")
    _put(dst, "extra.txt", b"local only")
    assert model_library.find_stale_files(dst, src) == ["inference.py", "taxonomy.csv"]


# --- installing -------------------------------------------------------------


def test_download_weights_copies_from_the_library(library, tmp_path):
    _put(library, "cls/WSP-UK-v1/model.pt", b"weights")
    _put(library, "cls/WSP-UK-v1/inference.py", b"code")
    storage = ModelStorage(tmp_path / "models")

    path = storage.download_weights(_manifest())

    assert path == tmp_path / "models" / "cls" / "WSP-UK-v1"
    assert (path / "inference.py").read_bytes() == b"code"


def test_download_weights_falls_back_to_the_url(library, tmp_path, monkeypatch):
    calls = []

    def fake_download(url, dst, progress_callback=None, should_cancel=None):
        calls.append(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"md")

    monkeypatch.setattr(model_library, "download_url", fake_download)
    storage = ModelStorage(tmp_path / "models")
    manifest = _manifest(
        "MD5A-0-0", "md_v5a.0.0.pt", "detection",
        download_url="https://example.invalid/md_v5a.0.0.pt",
    )

    storage.download_weights(manifest)

    assert calls == ["https://example.invalid/md_v5a.0.0.pt"]
    assert (tmp_path / "models" / "det" / "MD5A-0-0" / "md_v5a.0.0.pt").is_file()


def test_download_weights_names_the_fix_when_no_source(library, tmp_path):
    storage = ModelStorage(tmp_path / "models")
    with pytest.raises(model_library.ModelSourceMissingError) as err:
        storage.download_weights(_manifest())
    assert "WSP model library" in str(err.value)
    assert "WSP-UK-v1" in str(err.value)


def test_an_installed_model_is_not_copied_again(library, tmp_path, monkeypatch):
    local = tmp_path / "models" / "cls" / "WSP-UK-v1"
    _put(local, "model.pt", b"weights")
    monkeypatch.setattr(model_library, "copy_model", lambda *a, **k: pytest.fail("copied"))
    assert ModelStorage(tmp_path / "models").download_weights(_manifest()) == local


def test_cancelled_install_clears_files_but_keeps_the_manifest(library, tmp_path, monkeypatch):
    local = tmp_path / "models" / "cls" / "WSP-UK-v1"
    _put(local, "manifest.json", b"{}")
    _put(library, "cls/WSP-UK-v1/model.pt", b"weights")

    def cancel_midway(src, dst, progress_callback=None, should_cancel=None, **kw):
        _put(dst, "a.txt", b"a")
        raise JobCancelledError("stop")

    monkeypatch.setattr(model_library, "copy_model", cancel_midway)
    with pytest.raises(JobCancelledError):
        ModelStorage(tmp_path / "models").download_weights(_manifest())
    assert (local / "manifest.json").is_file()
    assert not (local / "a.txt").exists()


def test_failed_install_keeps_what_was_copied(library, tmp_path, monkeypatch):
    local = tmp_path / "models" / "cls" / "WSP-UK-v1"
    _put(library, "cls/WSP-UK-v1/model.pt", b"weights")

    def fail_midway(src, dst, progress_callback=None, should_cancel=None, **kw):
        _put(dst, "inference.py", b"code")
        raise OSError("share went away")

    monkeypatch.setattr(model_library, "copy_model", fail_midway)
    with pytest.raises(RuntimeError, match="share went away"):
        ModelStorage(tmp_path / "models").download_weights(_manifest())
    assert (local / "inference.py").is_file()


def test_a_failed_copy_never_leaves_the_weights_without_the_rest(
    library, tmp_path, monkeypatch
):
    """The weights mark a model as installed, so they are copied last: a
    copy that fails part way is retried, not taken for a finished one."""
    # "a_weights.pt" sorts before "inference.py".
    _put(library, "cls/WSP-UK-v1/a_weights.pt", b"weights")
    _put(library, "cls/WSP-UK-v1/inference.py", b"code")
    real_copy_one = model_library._copy_one

    def share_drops_on_code(src, dst, on_bytes, should_cancel):
        if src.name == "inference.py":
            raise OSError("share went away")
        real_copy_one(src, dst, on_bytes, should_cancel)

    monkeypatch.setattr(model_library, "_copy_one", share_drops_on_code)
    storage = ModelStorage(tmp_path / "models")
    manifest = _manifest(fname="a_weights.pt")
    with pytest.raises(RuntimeError, match="share went away"):
        storage.download_weights(manifest)
    assert not storage.check_weights_ready(manifest)

    monkeypatch.setattr(model_library, "_copy_one", real_copy_one)
    storage.download_weights(manifest)
    local = tmp_path / "models" / "cls" / "WSP-UK-v1"
    assert (local / "inference.py").read_bytes() == b"code"


def test_update_stale_files_refreshes_from_the_library(library, tmp_path):
    _put(library, "cls/WSP-UK-v1/model.pt", b"weights")
    _put(library, "cls/WSP-UK-v1/inference.py", b"fixed")
    local = tmp_path / "models" / "cls" / "WSP-UK-v1"
    _put(local, "model.pt", b"weights")
    _put(local, "inference.py", b"buggy")

    updated = ModelStorage(tmp_path / "models").update_stale_files(_manifest())

    assert updated == ["inference.py"]
    assert (local / "inference.py").read_bytes() == b"fixed"


def test_update_raises_when_the_library_is_missing(library, tmp_path):
    local = tmp_path / "models" / "cls" / "WSP-UK-v1"
    _put(local, "model.pt", b"weights")
    with pytest.raises(ConnectionError):
        ModelStorage(tmp_path / "models").update_stale_files(_manifest())


# --- catalog ----------------------------------------------------------------


def test_library_catalog_adds_and_overrides_shipped_entries(library, tmp_path):
    override = dict(_entry("MD5A-0-0", "md_v5a.0.0.pt"), friendly_name="MD from library")
    catalog = {"models": {"det": [override], "cls": [_entry()], "emb": []}}
    _put(library, "models.json", json.dumps(catalog).encode())

    merged = ModelCatalogUpdater(tmp_path / "models").fetch_catalog()

    cls_ids = [m["model_id"] for m in merged["models"]["cls"]]
    assert "WSP-UK-v1" in cls_ids
    assert "SPECIESNET-v4-0-2-A" in cls_ids  # from the shipped catalog
    det = {m["model_id"]: m for m in merged["models"]["det"]}
    assert det["MD5A-0-0"]["friendly_name"] == "MD from library"


def test_a_malformed_library_entry_is_skipped(library, tmp_path):
    """One bad hand-edited entry must not drop the rest of the catalog."""
    broken = {k: v for k, v in _entry("BROKEN").items() if k != "model_id"}
    catalog = {"models": {"det": [], "cls": [broken, _entry()]}}
    _put(library, "models.json", json.dumps(catalog).encode())

    merged = ModelCatalogUpdater(tmp_path / "models").fetch_catalog()

    cls_ids = [m["model_id"] for m in merged["models"]["cls"]]
    assert "WSP-UK-v1" in cls_ids
    assert "SPECIESNET-v4-0-2-A" in cls_ids
    assert [m["model_id"] for m in merged["models"]["det"]] == ["MD5A-0-0"]


def test_shipped_catalog_is_used_without_a_library(tmp_path, monkeypatch):
    monkeypatch.delenv("WSP_MODEL_LIBRARY_DIR", raising=False)
    catalog = ModelCatalogUpdater(tmp_path / "models").fetch_catalog()
    ids = [m["model_id"] for t in ("det", "cls") for m in catalog["models"][t]]
    assert ids == ["MD5A-0-0", "SPECIESNET-v4-0-2-A"]


async def test_sync_reports_installed_models_with_stale_files(library, tmp_path):
    models = tmp_path / "models"
    _put(library, "models.json", json.dumps({"models": {"det": [], "cls": [_entry()]}}).encode())
    _put(library, "cls/WSP-UK-v1/model.pt", b"weights")
    _put(library, "cls/WSP-UK-v1/inference.py", b"v2")
    # Something else is already installed, so this is not a fresh install.
    _put(models, "det/OTHER/manifest.json", b"{}")
    _put(models, "cls/WSP-UK-v1/model.pt", b"weights")
    _put(models, "cls/WSP-UK-v1/inference.py", b"v1")

    result = await ModelCatalogUpdater(models).sync()

    assert [m["model_id"] for m in result["drifted_models"]] == ["WSP-UK-v1"]


async def test_sync_never_checks_a_stub_without_weights(library, tmp_path):
    models = tmp_path / "models"
    _put(library, "models.json", json.dumps({"models": {"det": [], "cls": [_entry()]}}).encode())
    _put(library, "cls/WSP-UK-v1/model.pt", b"weights")
    _put(models, "det/OTHER/manifest.json", b"{}")

    result = await ModelCatalogUpdater(models).sync()

    assert result["drifted_models"] == []
    assert (models / "cls" / "WSP-UK-v1" / "manifest.json").is_file()


def test_taxonomy_is_copied_for_a_cls_model(library, tmp_path):
    _put(library, "cls/WSP-UK-v1/taxonomy.csv", b"model_class\nbadger\n")
    updater = ModelCatalogUpdater(tmp_path / "models")

    assert updater.write_manifest("cls", _entry()) == "created"

    taxonomy = tmp_path / "models" / "cls" / "WSP-UK-v1" / "taxonomy.csv"
    assert taxonomy.read_text().startswith("model_class")
    # A later launch with an unchanged manifest leaves it alone.
    taxonomy.write_text("edited")
    assert updater.write_manifest("cls", _entry()) == "unchanged"
    assert taxonomy.read_text() == "edited"


def test_missing_taxonomy_is_not_fatal(library, tmp_path):
    updater = ModelCatalogUpdater(tmp_path / "models")
    assert updater.write_manifest("cls", _entry()) == "created"
    assert not (tmp_path / "models" / "cls" / "WSP-UK-v1" / "taxonomy.csv").exists()


# --- settings endpoint ------------------------------------------------------


def test_library_endpoint_saves_and_reports_the_folder(client, tmp_path, monkeypatch):
    monkeypatch.delenv("WSP_MODEL_LIBRARY_DIR", raising=False)
    lib = tmp_path / "WSP CameraTrap" / "models"
    _put(lib, "models.json", json.dumps({"models": {"det": [], "cls": []}}).encode())

    body = client.post("/api/wsp/library", json={"library_dir": str(lib)}).json()
    assert body["library_dir"] == str(lib)
    assert body["source"] == "settings"
    assert client.get("/api/wsp/library").json()["library_dir"] == str(lib)

    client.post("/api/wsp/library", json={})
    assert client.get("/api/wsp/library").json()["library_dir"] is None


def test_library_endpoint_rejects_a_folder_that_is_not_a_library(client, tmp_path):
    response = client.post("/api/wsp/library", json={"library_dir": str(tmp_path)})
    assert response.status_code == 400
    assert "models.json" in response.json()["detail"]


def test_first_setup_installs_the_library_models(library, tmp_path, monkeypatch):
    from app.api.routers import setup as setup_router

    monkeypatch.setenv("WSP_USER_DATA_DIR", str(tmp_path / "user"))
    monkeypatch.setattr(setup_router, "_env_present", lambda: True)
    monkeypatch.setattr(setup_router, "_get_env_manager", lambda: None)
    monkeypatch.setattr(setup_router, "_refresh_catalog", lambda: None)
    sn = "always_crop_99710272_22x8_v12_epoch_00148.pt"
    _put(library, "models.json", b'{"models": {"det": [], "cls": []}}')
    _put(library, "det/MD5A-0-0/md_v5a.0.0.pt", b"md")
    _put(library, f"cls/SPECIESNET-v4-0-2-A/{sn}", b"sn")

    setup_router.run_setup(lambda m, p: None)

    models = tmp_path / "user" / "models"
    assert (models / "det" / "MD5A-0-0" / "md_v5a.0.0.pt").read_bytes() == b"md"
    assert (models / "cls" / "SPECIESNET-v4-0-2-A" / sn).read_bytes() == b"sn"
    assert not (models / "emb" / "DINOV2-VITS14").exists()  # not in the library: skipped


# --- installing a library .zip ------------------------------------------------


def _zip(path: Path, files: dict[str, bytes], prefix: str = "models/") -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for rel, data in files.items():
            z.writestr(prefix + rel, data)
    return path


@pytest.fixture
def user_models(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("WSP_USER_DATA_DIR", str(tmp_path / "user"))
    monkeypatch.delenv("WSP_MODEL_LIBRARY_DIR", raising=False)
    return tmp_path / "user" / "models"


def test_a_library_zip_installs_into_the_models_folder(user_models, tmp_path):
    catalog = {"models": {"det": [], "cls": [_entry()], "emb": []}}
    bundle = _zip(tmp_path / "lib.zip", {
        "models.json": json.dumps(catalog).encode(),
        "cls/WSP-UK-v1/model.pt": b"weights",
        "cls/WSP-UK-v1/inference.py": b"# code",
        "cls/WSP-UK-v1/__pycache__/inference.cpython-311.pyc": b"c",
        "cls/WSP-UK-v1/manifest.json": b"{}",
        "det/MD5A-0-0/md_v5a.0.0.pt": b"md",
    })

    imported = model_library.import_models_zip(bundle)

    assert imported == ["cls/WSP-UK-v1", "det/MD5A-0-0"]
    assert (user_models / "cls" / "WSP-UK-v1" / "model.pt").read_bytes() == b"weights"
    assert not (user_models / "cls" / "WSP-UK-v1" / "__pycache__").exists()
    assert not (user_models / "cls" / "WSP-UK-v1" / "manifest.json").exists()
    assert not list(user_models.rglob("*.tmp"))
    # Its catalog makes a model the shipped catalog does not know visible.
    merged = ModelCatalogUpdater(user_models).fetch_catalog()
    assert "WSP-UK-v1" in [m["model_id"] for m in merged["models"]["cls"]]


def test_a_second_zip_adds_to_the_catalog(user_models, tmp_path):
    first = {"models": {"det": [], "cls": [_entry("WSP-UK-v1")]}}
    second = {"models": {"det": [], "cls": [_entry("WSP-UK-v2")]}}
    model_library.import_models_zip(
        _zip(tmp_path / "a.zip", {"models.json": json.dumps(first).encode()})
    )
    model_library.import_models_zip(
        _zip(tmp_path / "b.zip", {"models.json": json.dumps(second).encode()})
    )
    local = model_library.read_local_catalog()
    assert [m["model_id"] for m in local["models"]["cls"]] == ["WSP-UK-v1", "WSP-UK-v2"]


def test_a_zip_without_catalog_or_not_a_zip_is_refused(user_models, tmp_path):
    with pytest.raises(model_library.ModelZipError, match="models.json"):
        model_library.import_models_zip(_zip(tmp_path / "x.zip", {"cls/A/model.pt": b"w"}))
    not_zip = tmp_path / "notes.zip"
    not_zip.write_text("hello")
    with pytest.raises(model_library.ModelZipError, match="not a .zip"):
        model_library.import_models_zip(not_zip)


def test_a_zip_cannot_write_outside_the_models_folder(user_models, tmp_path):
    bundle = _zip(tmp_path / "evil.zip", {
        "models.json": b'{"models": {}}',
        "../../evil.txt": b"x",
    })
    with pytest.raises(model_library.ModelZipError, match="Unsafe"):
        model_library.import_models_zip(bundle)
    assert not (tmp_path / "evil.txt").exists()


def test_import_endpoint_installs_and_lists_the_models(client, user_models, tmp_path):
    catalog = {"models": {"det": [], "cls": [_entry()], "emb": []}}
    bundle = _zip(tmp_path / "lib.zip", {
        "models.json": json.dumps(catalog).encode(),
        "cls/WSP-UK-v1/model.pt": b"weights",
    })
    response = client.post("/api/wsp/library/import", json={"zip_path": str(bundle)})
    assert response.status_code == 200, response.text
    assert response.json()["imported"] == ["cls/WSP-UK-v1"]
    assert (user_models / "cls" / "WSP-UK-v1" / "manifest.json").is_file()

    response = client.post("/api/wsp/library/import", json={"zip_path": str(tmp_path / "no.zip")})
    assert response.status_code == 400
