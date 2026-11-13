import hashlib
import socket
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import MappingProxyType

import pytest
from typer.testing import CliRunner

from copypaw import models
from copypaw.cli import app

_CONTENT = b"local model weights"


@pytest.fixture
def model_server() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(_CONTENT)))
            self.end_headers()
            payload = _CONTENT[:4] if self.path == "/interrupted" else _CONTENT
            self.wfile.write(payload)
            self.wfile.flush()
            if self.path == "/interrupted":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()

        def log_message(self, *_args: object, **_kwargs: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def local_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_server: str
) -> models.Model:
    model = models.Model(
        "weights.onnx",
        model_server + "/weights",
        len(_CONTENT),
        hashlib.sha256(_CONTENT).hexdigest(),
    )
    monkeypatch.setenv("COPYPAW_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(models, "REGISTRY", MappingProxyType({"local": model}))
    monkeypatch.setattr(models, "MODELS", MappingProxyType(dict.fromkeys(models.Mode, ("local",))))
    return model


def test_registry_pins_all_models() -> None:
    assert len(models.REGISTRY) == 5
    for model in models.REGISTRY.values():
        assert len(bytes.fromhex(model.sha256)) == 32
        assert model.size_bytes > 0
        assert model.url.startswith("https://")
        assert model.url.endswith(model.filename)
    assert models.MODELS[models.Mode.LIGHTWEIGHT] == ("yolox_tiny", "vitpose_s_apt36k")
    assert models.MODELS[models.Mode.BALANCED] == ("yolox_m", "vitpose_b_apt36k")
    assert models.MODELS[models.Mode.ACCURATE] == ("yolox_m", "vitpose_l_apt36k")


def test_fetch_installs_only_after_verification(
    tmp_path: Path, local_registry: models.Model
) -> None:
    updates: list[int] = []
    destination = tmp_path / "cache" / local_registry.filename

    def progress(filename: str, downloaded: int, total: int) -> None:
        assert filename == local_registry.filename
        assert total == len(_CONTENT)
        assert not destination.exists()
        updates.append(downloaded)

    assert models.ensure_models("lightweight", progress) == {"local": destination}
    assert destination.read_bytes() == _CONTENT
    assert updates[0] == 0
    assert updates[-1] == len(_CONTENT)
    assert list(destination.parent.iterdir()) == [destination]


def test_verified_cache_needs_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_registry: models.Model
) -> None:
    destination = models.ensure_models("balanced")["local"]
    monkeypatch.setattr(
        models,
        "REGISTRY",
        MappingProxyType({"local": replace(local_registry, url="http://127.0.0.1:1/offline")}),
    )
    assert models.ensure_models("balanced") == {"local": destination}
    assert destination.parent == tmp_path / "cache"


def test_bad_download_names_hashes_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_registry: models.Model
) -> None:
    expected = "0" * 64
    monkeypatch.setattr(
        models, "REGISTRY", MappingProxyType({"local": replace(local_registry, sha256=expected)})
    )
    with pytest.raises(models.ModelChecksumError) as caught:
        models.ensure_models("lightweight")
    assert expected in str(caught.value)
    assert local_registry.sha256 in str(caught.value)
    assert local_registry.filename in str(caught.value)
    assert list((tmp_path / "cache").iterdir()) == []


def test_interrupted_download_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_registry: models.Model
) -> None:
    interrupted = replace(
        local_registry, url=local_registry.url.replace("/weights", "/interrupted")
    )
    monkeypatch.setattr(models, "REGISTRY", MappingProxyType({"local": interrupted}))
    with pytest.raises(models.ModelChecksumError):
        models.ensure_models("balanced")
    assert list((tmp_path / "cache").iterdir()) == []


def test_corrupt_cache_is_never_accepted(tmp_path: Path, local_registry: models.Model) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / local_registry.filename).write_bytes(b"x" * len(_CONTENT))
    with pytest.raises(models.ModelChecksumError):
        models.ensure_models("accurate")


def test_wrong_size_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_registry: models.Model
) -> None:
    monkeypatch.setattr(
        models,
        "REGISTRY",
        MappingProxyType({"local": replace(local_registry, size_bytes=len(_CONTENT) + 1)}),
    )
    with pytest.raises(models.ModelChecksumError, match=r"has \d+ bytes; expected \d+ bytes"):
        models.ensure_models("balanced")
    assert list((tmp_path / "cache").iterdir()) == []


def test_invalid_mode_does_not_create_cache(tmp_path: Path, local_registry: models.Model) -> None:
    assert local_registry.filename
    with pytest.raises(ValueError, match="not a valid Mode"):
        models.ensure_models("unknown")
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("mode", list(models.Mode))
def test_cli_fetch(mode: models.Mode, local_registry: models.Model) -> None:
    result = CliRunner().invoke(app, ["models", "fetch", "--mode", mode.value])
    assert result.exit_code == 0, result.output
    assert "Verified local:" in result.output
    assert f"{local_registry.filename}: 0/{len(_CONTENT)} bytes" in result.output


def test_cli_reports_checksum_failure(tmp_path: Path, local_registry: models.Model) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / local_registry.filename).write_bytes(b"corrupt")
    result = CliRunner().invoke(app, ["models", "fetch"])
    assert result.exit_code == 2
    assert result.stderr.startswith(
        f"error: The model file {local_registry.filename} does not match"
    )
    assert "expected SHA-256" in result.stderr
    assert "next:  Delete" in result.stderr


def test_cli_rejects_invalid_mode(local_registry: models.Model) -> None:
    assert local_registry.filename
    result = CliRunner().invoke(app, ["models", "fetch", "--mode", "unknown"])
    assert result.exit_code == 2


def test_unreachable_server_is_a_download_error(
    monkeypatch: pytest.MonkeyPatch, local_registry: models.Model
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    unreachable = replace(local_registry, url=f"http://127.0.0.1:{port}/weights")
    monkeypatch.setattr(models, "REGISTRY", MappingProxyType({"local": unreachable}))
    with pytest.raises(models.ModelFetchError) as caught:
        models.ensure_models("balanced")
    assert caught.value.code == "model_download"
    assert caught.value.what.startswith(f"Copypaw could not download {local_registry.filename}:")


def test_unwritable_cache_is_a_cache_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("file", encoding="utf-8")
    monkeypatch.setenv("COPYPAW_CACHE", str(blocker))
    with pytest.raises(models.ModelFetchError) as caught:
        models.ensure_models("balanced")
    assert caught.value.code == "model_cache"
    assert f"cannot write to the model cache {blocker}" in caught.value.what
