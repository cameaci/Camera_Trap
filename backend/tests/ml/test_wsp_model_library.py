"""WSP: models installed from the model library folder or a public URL."""

import json
from pathlib import Path

import pytest

from app.core.job_cancellation import JobCancelledError
from app.ml import model_library
from app.ml.catalog_updater import ModelCatalogUpdater
from app.ml.model_storage import ModelStorage
from app.ml.schemas.model_manifest import ModelManifest


@pytest.fixture
def library(tmp_path, monkeypatch) -> Path:
    """An empty library, switched on as the model source."""
    lib = tmp_path / "library"
    lib.mkdir()
    monkeypatch.setenv("ADDAXAI_MODEL_SOURCE", "library")
    monkeypatch.setenv("ADDAXAI_MODEL_LIBRARY_DIR", str(lib))
    return lib


def _manifest(model_id="WSP-UK-v1", fname="model.pt", category="classification", **kw):
    m = ModelManifest(
        model_id=model_id,
        friendly_name=model_id,
        env="pytorch",
        model_fname=fname,
        description="test",
        developer="WSP",
        info_url="https://example.invalid",
        min_app_version="0.1.0",
        **kw,
    )
    m.model_category = category
    return m


def _put(lib: Path, rel: str, data: bytes = b"x") -> Path:
    path = lib / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# --- locating the library -------------------------------------------------


def test_env_var_names_the_library(library):
    assert model_library.get_library_dir() is None  # empty folder is not a library
    _put(library, "models.json", b"{}")
    assert model_library.get_library_dir() == library


def test_saved_folder_is_used_when_no_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("ADDAXAI_MODEL_LIBRARY_DIR", raising=False)
    lib = tmp_path / "saved"
    _put(lib, "cls/X/model.pt")
    model_library.write_library_dir(lib)
    assert model_library.get_library_dir() == lib
    model_library.write_library_dir(None)
    assert model_library.get_library_dir() is None


def test_a_configured_folder_that_is_missing_is_not_guessed(tmp_path, monkeypatch):
    monkeypatch.setenv("ADDAXAI_MODEL_LIBRARY_DIR", str(tmp_path / "offline-share"))
    monkeypatch.setenv("ADDAXAI_MODEL_LIBRARY_AUTODETECT", "true")
    onedrive = tmp_path / "OneDrive - WSP"
    _put(onedrive / "WSP CameraTrap" / "models", "models.json", b"{}")
    monkeypatch.setenv("OneDriveCommercial", str(onedrive))
    assert model_library.get_library_dir() is None


def test_autodetect_finds_the_onedrive_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("ADDAXAI_MODEL_LIBRARY_DIR", raising=False)
    monkeypatch.setenv("ADDAXAI_MODEL_LIBRARY_AUTODETECT", "true")
    onedrive = tmp_path / "OneDrive - WSP"
    target = onedrive / "Ecology - Shared" / "WSP CameraTrap" / "models"
    _put(target, "models.json", b"{}")
    monkeypatch.setenv("OneDriveCommercial", str(onedrive))
    assert model_library.autodetect_library_dir() == target


# --- copying --------------------------------------------------------------


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


# --- ModelStorage in library mode ------------------------------------------


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
        download_url="https://github.com/example/md_v5a.0.0.pt",
    )

    storage.download_weights(manifest)

    assert calls == ["https://github.com/example/md_v5a.0.0.pt"]
    assert (tmp_path / "models" / "det" / "MD5A-0-0" / "md_v5a.0.0.pt").is_file()


def test_download_weights_names_the_fix_when_no_source(library, tmp_path):
    storage = ModelStorage(tmp_path / "models")
    with pytest.raises(model_library.ModelSourceMissingError) as err:
        storage.download_weights(_manifest())
    assert "WSP CameraTrap" in str(err.value)
    assert "WSP-UK-v1" in str(err.value)


def test_download_weights_never_contacts_huggingface(library, tmp_path, monkeypatch):
    import app.ml.model_storage as storage_module

    def forbidden(*args, **kwargs):
        raise AssertionError("HuggingFace must not be contacted in library mode")

    monkeypatch.setattr(storage_module, "_download_repo_with_relay", forbidden)
    _put(library, "cls/WSP-UK-v1/model.pt", b"weights")
    ModelStorage(tmp_path / "models").download_weights(_manifest())


def test_update_stale_files_refreshes_from_the_library(library, tmp_path):
    _put(library, "cls/WSP-UK-v1/model.pt", b"weights")
    _put(library, "cls/WSP-UK-v1/inference.py", b"fixed")
    local = tmp_path / "models" / "cls" / "WSP-UK-v1"
    _put(local, "model.pt", b"weights")
    _put(local, "inference.py", b"buggy")

    updated = ModelStorage(tmp_path / "models").update_stale_files(_manifest())

    assert updated == ["inference.py"]
    assert (local / "inference.py").read_bytes() == b"fixed"


# --- catalog in library mode ------------------------------------------------


def test_library_catalog_adds_and_overrides_bundled_entries(library, tmp_path):
    entry = {
        "model_id": "WSP-UK-v1",
        "friendly_name": "WSP UK v1",
        "env": "pytorch",
        "model_fname": "model.pt",
        "description": "d",
        "developer": "WSP",
        "info_url": "https://example.invalid",
        "min_app_version": "0.1.0",
    }
    override = dict(entry, model_id="MD5A-0-0", friendly_name="MD from library")
    catalog = {"models": {"det": [override], "cls": [entry], "emb": []}}
    _put(library, "models.json", json.dumps(catalog).encode())

    merged = ModelCatalogUpdater(tmp_path / "models").fetch_catalog()

    cls_ids = [m["model_id"] for m in merged["models"]["cls"]]
    assert "WSP-UK-v1" in cls_ids
    assert "SPECIESNET-v4-0-2-A" in cls_ids  # from the bundled WSP catalog
    det = {m["model_id"]: m for m in merged["models"]["det"]}
    assert det["MD5A-0-0"]["friendly_name"] == "MD from library"


def test_bundled_wsp_catalog_is_used_without_a_library(tmp_path, monkeypatch):
    monkeypatch.setenv("ADDAXAI_MODEL_SOURCE", "library")
    monkeypatch.delenv("ADDAXAI_MODEL_LIBRARY_DIR", raising=False)
    catalog = ModelCatalogUpdater(tmp_path / "models").fetch_catalog()
    ids = [m["model_id"] for t in ("det", "cls") for m in catalog["models"][t]]
    assert ids == ["MD5A-0-0", "SPECIESNET-v4-0-2-A"]


# --- settings endpoint ------------------------------------------------------


def test_library_endpoint_saves_and_reports_the_folder(client, tmp_path, monkeypatch):
    monkeypatch.setenv("ADDAXAI_MODEL_SOURCE", "library")
    monkeypatch.delenv("ADDAXAI_MODEL_LIBRARY_DIR", raising=False)
    lib = tmp_path / "WSP CameraTrap" / "models"
    _put(lib, "models.json", json.dumps({"models": {"det": [], "cls": []}}).encode())

    body = client.post("/api/wsp/library", json={"library_dir": str(lib)}).json()
    assert body["library_dir"] == str(lib)
    assert body["source"] == "settings"
    assert client.get("/api/wsp/library").json()["library_dir"] == str(lib)

    client.post("/api/wsp/library", json={"library_dir": None})
    assert client.get("/api/wsp/library").json()["library_dir"] is None


def test_library_endpoint_rejects_a_folder_that_is_not_a_library(client, tmp_path):
    response = client.post("/api/wsp/library", json={"library_dir": str(tmp_path)})
    assert response.status_code == 400
    assert "models.json" in response.json()["detail"]
