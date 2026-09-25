"""Prebuilt environment packs: building, publishing layout, installing."""

import hashlib
import http.server
import json
import threading
from pathlib import Path

import pytest

from app.ml import env_pack


class _Files(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Serve a folder over HTTP and point the app's pack URL at it."""
    root = tmp_path / "release"
    root.mkdir()

    def handler(*args, **kwargs):
        return _Files(*args, directory=str(root), **kwargs)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("WSP_ENV_PACK_URL", f"http://127.0.0.1:{httpd.server_port}")
    monkeypatch.setattr(env_pack, "platform_tag", lambda: "win-64")
    yield root
    httpd.shutdown()


def _fake_env(root: Path, prefix: str) -> Path:
    env = root / "env-wsp-base"
    (env / "Scripts").mkdir(parents=True)
    (env / "python.exe").write_bytes(b"MZ\0\0binary " + prefix.encode())
    (env / "Scripts" / "activate.bat").write_text(f"set PREFIX={prefix}\n")
    (env / "Lib" / "site-packages").mkdir(parents=True)
    (env / "Lib" / "site-packages" / "x.pth").write_text(prefix.replace("\\", "/") + "/lib\n")
    (env / ".wsp-cameratrap-yaml-sha256").write_text("abc123def4567890")
    return env


def test_pack_round_trip_relocates_text_files(served, tmp_path):
    prefix = "D:\\a\\_temp\\envs\\.env-wsp-base.tmp"
    built = _fake_env(tmp_path / "build", prefix)
    manifest = env_pack.write_pack(
        built, "env-wsp-base", "abc123def4567890", "win-64", prefix, served, part_size=64
    )
    data = json.loads(manifest.read_text())
    assert manifest.name == "env-wsp-base-win-64-abc123def456.json"
    assert len(data["parts"]) > 1  # split into parts
    assert all((served / part).is_file() for part in data["parts"])

    target = tmp_path / "user" / "envs" / "env-wsp-base"
    target.parent.mkdir(parents=True)
    progress = []
    assert env_pack.install_env_pack(
        "env-wsp-base", target, "abc123def4567890", lambda m, p: progress.append(p)
    )

    assert (target / "Scripts" / "activate.bat").read_text() == f"set PREFIX={target}\n"
    pth = (target / "Lib" / "site-packages" / "x.pth").read_text()
    assert pth == str(target).replace("\\", "/") + "/lib\n"
    # Binary files are left alone.
    assert prefix.encode() in (target / "python.exe").read_bytes()
    assert (target / ".wsp-cameratrap-yaml-sha256").read_text() == "abc123def4567890"
    assert progress and max(progress) <= 1.0
    assert not list(target.parent.glob(".*.pack"))


def test_no_published_pack_means_build_locally(served, tmp_path):
    assert env_pack.install_env_pack("env-wsp-base", tmp_path / "env", "0" * 16) is False


def test_unreachable_release_means_build_locally(tmp_path, monkeypatch):
    monkeypatch.setattr(env_pack, "platform_tag", lambda: "win-64")
    monkeypatch.setenv("WSP_ENV_PACK_URL", "http://127.0.0.1:9/nothing")
    assert env_pack.install_env_pack("env-wsp-base", tmp_path / "env", "0" * 16) is False


def test_a_damaged_download_is_refused(served, tmp_path):
    prefix = "C:\\build\\.env-wsp-base.tmp"
    built = _fake_env(tmp_path / "build", prefix)
    manifest = env_pack.write_pack(built, "env-wsp-base", "f" * 16, "win-64", prefix, served)
    data = json.loads(manifest.read_text())
    part = served / data["parts"][0]
    part.write_bytes(part.read_bytes()[:-1] + b"X")

    target = tmp_path / "user" / "env-wsp-base"
    target.parent.mkdir(parents=True)
    with pytest.raises(env_pack.EnvPackError, match="damaged"):
        env_pack.install_env_pack("env-wsp-base", target, "f" * 16)
    assert not target.exists()


def test_a_cancelled_download_leaves_nothing(served, tmp_path):
    from app.core.job_cancellation import JobCancelledError

    prefix = "C:\\build\\.env-wsp-base.tmp"
    built = _fake_env(tmp_path / "build", prefix)
    env_pack.write_pack(built, "env-wsp-base", "c" * 16, "win-64", prefix, served)

    target = tmp_path / "user" / "env-wsp-base"
    target.parent.mkdir(parents=True)
    with pytest.raises(JobCancelledError):
        env_pack.install_env_pack("env-wsp-base", target, "c" * 16, should_cancel=lambda: True)
    assert not target.exists()
    assert not list(target.parent.glob(".*.pack"))


def test_platforms_without_packs_skip_the_download(monkeypatch, tmp_path):
    monkeypatch.setattr(env_pack, "platform_tag", lambda: None)
    assert env_pack.install_env_pack("env-wsp-base", tmp_path / "env", "0" * 16) is False


def test_pack_stem_uses_the_yaml_hash():
    sha = hashlib.sha256(b"yaml").hexdigest()
    assert env_pack.pack_stem("env-wsp-base", sha, "win-64") == f"env-wsp-base-win-64-{sha[:12]}"
