"""The WSP model library: folders, the OneDrive link, installs and updates."""

import http.server
import io
import json
import stat
import threading
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


def test_library_url_order(monkeypatch):
    monkeypatch.setenv("WSP_MODEL_LIBRARY_URL", "")
    assert model_library.get_library_url() is None
    monkeypatch.delenv("WSP_MODEL_LIBRARY_URL")
    model_library.write_library_url("https://example.invalid/saved.zip")
    try:
        assert model_library.get_library_url() == "https://example.invalid/saved.zip"
        monkeypatch.setenv("WSP_MODEL_LIBRARY_URL", "https://example.invalid/env.zip")
        assert model_library.get_library_url() == "https://example.invalid/env.zip"
    finally:
        model_library.write_library_url(None)


def test_the_shipped_config_has_a_library_url_key():
    assert "model_library_url" in model_library.bundled_config()


@pytest.mark.parametrize(
    "url, expected",
    [
        (
            "https://wsponline-my.sharepoint.com/:u:/g/personal/me/EAbc?e=xyz",
            "https://wsponline-my.sharepoint.com/:u:/g/personal/me/EAbc?e=xyz&download=1",
        ),
        ("https://1drv.ms/u/s!Abc", "https://1drv.ms/u/s!Abc?download=1"),
        (
            "https://wsponline.sharepoint.com/x.zip?download=1",
            "https://wsponline.sharepoint.com/x.zip?download=1",
        ),
        ("https://example.com/library.zip", "https://example.com/library.zip"),
    ],
)
def test_normalize_share_url(url, expected):
    assert model_library.normalize_share_url(url) == expected


# --- the linked library ---------------------------------------------------


def _bundle(files: dict[str, bytes], prefix: str = "models/") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for rel, data in files.items():
            z.writestr(prefix + rel, data)
    return buf.getvalue()


class _Server:
    """A tiny local HTTP server with a swappable response."""

    def __init__(self) -> None:
        self.body = b""
        self.content_type = "application/zip"
        self.status = 200
        self.etag = '"1"'
        # Bytes announced but never sent: a connection that drops.
        self.extra_length = 0
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(outer.status)
                self.send_header("Content-Type", outer.content_type)
                self.send_header("Content-Length", str(len(outer.body) + outer.extra_length))
                self.send_header("ETag", outer.etag)
                self.end_headers()
                self.wfile.write(outer.body)

            def log_message(self, *args):
                pass

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/library.zip"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def server(monkeypatch):
    srv = _Server()
    monkeypatch.delenv("WSP_MODEL_LIBRARY_DIR", raising=False)
    monkeypatch.setenv("WSP_MODEL_LIBRARY_URL", srv.url)
    yield srv
    srv.close()


def test_linked_library_is_downloaded_and_used(server):
    catalog = {"models": {"det": [], "cls": [_entry()]}}
    server.body = _bundle({
        "models.json": json.dumps(catalog).encode(),
        "cls/WSP-UK-v1/model.pt": b"weights",
    })
    progress = []

    lib = model_library.sync_library_url(lambda m, p: progress.append(p))

    assert lib == model_library.library_cache_dir()
    assert model_library.get_library_dir() == lib
    assert (lib / "cls" / "WSP-UK-v1" / "model.pt").read_bytes() == b"weights"
    assert progress and progress[-1] == 1.0


def test_linked_library_is_not_downloaded_again_when_unchanged(server, monkeypatch):
    server.body = _bundle({"models.json": b'{"models": {"det": [], "cls": []}}'})
    model_library.sync_library_url()
    lib = model_library.library_cache_dir()
    assert (lib / "models.json").is_file()

    extracted = []
    real_zipfile = zipfile.ZipFile
    monkeypatch.setattr(
        model_library.zipfile,
        "ZipFile",
        lambda *a, **k: extracted.append(1) or real_zipfile(*a, **k),
    )
    model_library.sync_library_url()
    assert extracted == []

    # A new file behind the same link is picked up.
    server.etag = '"2"'
    server.body = _bundle({
        "models.json": b'{"models": {"det": [], "cls": []}}',
        "cls/NEW/model.pt": b"n",
    })
    model_library.sync_library_url()
    assert (lib / "cls" / "NEW" / "model.pt").is_file()


def test_a_configured_folder_wins_over_the_link(server, tmp_path, monkeypatch):
    """The link is not downloaded when a folder takes priority over it."""
    folder = tmp_path / "synced"
    _put(folder, "models.json", b'{"models": {"det": [], "cls": []}}')
    monkeypatch.setenv("WSP_MODEL_LIBRARY_DIR", str(folder))

    def no_request(*args, **kwargs):
        raise AssertionError("the link must not be requested")

    monkeypatch.setattr(model_library.requests, "get", no_request)

    assert model_library.sync_library_url() is None
    assert model_library.get_library_dir() == folder


def test_a_sign_in_page_is_reported_and_keeps_the_old_copy(server):
    server.body = _bundle({"models.json": b'{"models": {"det": [], "cls": []}}'})
    model_library.sync_library_url()

    server.content_type = "text/html; charset=utf-8"
    server.body = b"<html>Sign in</html>"
    server.etag = '"3"'
    with pytest.raises(model_library.LibraryDownloadError, match="web page"):
        model_library.sync_library_url()
    assert (model_library.library_cache_dir() / "models.json").is_file()


def test_a_bundle_without_catalog_is_rejected(server):
    server.etag = '"4"'
    server.body = _bundle({"cls/X/model.pt": b"x"})
    with pytest.raises(model_library.LibraryDownloadError, match="models.json"):
        model_library.sync_library_url()


def test_a_broken_download_is_reported_and_leaves_nothing_behind(server):
    """A connection that drops mid-download must not leave a partial .zip."""
    server.body = _bundle({"models.json": b'{"models": {"det": [], "cls": []}}'})
    server.extra_length = 1000  # promise more bytes than are sent
    with pytest.raises(model_library.LibraryDownloadError):
        model_library.sync_library_url()
    assert not (model_library.library_cache_dir().parent / "library.zip.tmp").exists()


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


async def test_a_sync_right_after_a_download_does_not_check_the_link_again(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(model_library, "sync_library_url", lambda *a, **k: calls.append(1))

    await ModelCatalogUpdater(tmp_path / "models").sync(refresh_library=False)
    assert calls == []
    await ModelCatalogUpdater(tmp_path / "models").sync()
    assert calls == [1]


async def test_a_failing_library_link_does_not_fail_the_sync(tmp_path, monkeypatch):
    monkeypatch.delenv("WSP_MODEL_LIBRARY_DIR", raising=False)
    monkeypatch.setenv("WSP_MODEL_LIBRARY_URL", "http://127.0.0.1:9/unreachable.zip")

    result = await ModelCatalogUpdater(tmp_path / "models").sync()

    assert "library_error" in result
    assert (tmp_path / "models" / "det" / "MD5A-0-0" / "manifest.json").is_file()


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


def test_library_endpoint_rejects_a_link_that_is_not_a_url(client):
    response = client.post("/api/wsp/library", json={"library_url": "OneDrive folder"})
    assert response.status_code == 400


def test_library_endpoint_refuses_a_plain_http_link(client):
    """The library carries inference.py files the app runs."""
    response = client.post(
        "/api/wsp/library", json={"library_url": "http://example.invalid/library.zip"}
    )
    assert response.status_code == 400
    assert "https://" in response.json()["detail"]


def test_a_link_saved_mid_download_runs_the_download_again():
    """Saving a new link while one downloads must not be dropped."""
    from app.api.routers.wsp import _DownloadState

    state = _DownloadState()
    assert state.start() is True
    assert state.start() is False  # the running task will go again
    assert state.finish() is True  # ... so it does
    assert state.in_progress
    assert state.finish("boom") is False
    assert not state.in_progress and state.error == "boom"


def test_first_setup_installs_speciesnet_from_a_linked_library(server, tmp_path, monkeypatch):
    """The linked library is only downloaded during setup, so SpeciesNet
    must not be ruled out before that download has run."""
    from app.api.routers import setup as setup_router

    monkeypatch.setenv("WSP_USER_DATA_DIR", str(tmp_path / "user"))
    monkeypatch.setattr(setup_router, "_env_present", lambda: True)
    monkeypatch.setattr(setup_router, "_get_env_manager", lambda: None)
    monkeypatch.setattr(setup_router, "_refresh_catalog", lambda: None)
    sn = "always_crop_99710272_22x8_v12_epoch_00148.pt"
    server.body = _bundle({
        "models.json": json.dumps({"models": {"det": [], "cls": []}}).encode(),
        "det/MD5A-0-0/md_v5a.0.0.pt": b"md",
        f"cls/SPECIESNET-v4-0-2-A/{sn}": b"sn",
    })

    setup_router.run_setup(lambda m, p: None)

    models = tmp_path / "user" / "models"
    assert (models / "det" / "MD5A-0-0" / "md_v5a.0.0.pt").read_bytes() == b"md"
    assert (models / "cls" / "SPECIESNET-v4-0-2-A" / sn).read_bytes() == b"sn"
