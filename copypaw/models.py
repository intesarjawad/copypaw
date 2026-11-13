"""Pinned model weights with verified, atomic cache installation."""

import hashlib
import os
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from http.client import HTTPException
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

from copypaw.errors import CopypawError

_CHUNK_BYTES = 1024 * 1024
_DOWNLOAD_TIMEOUT_S = 60
_VITPOSE_URL = (
    "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/"
    "e83805274e89428969355ec4afffcbc413e79188/onnx/apt36k/"
)
_YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"


class Mode(StrEnum):
    """Available detector and pose model pairs."""

    LIGHTWEIGHT = "lightweight"
    BALANCED = "balanced"
    ACCURATE = "accurate"


@dataclass(frozen=True)
class Model:
    """A pinned download and its expected on-disk identity."""

    filename: str
    url: str
    size_bytes: int
    sha256: str


REGISTRY = MappingProxyType(
    {
        "vitpose_b_apt36k": Model(
            "vitpose-b-apt36k.onnx",
            _VITPOSE_URL + "vitpose-b-apt36k.onnx",
            360123609,
            "7f7240941aef7505fd6f57c87ea2305cc8778a7ac5c741120a2e782d66c7aa91",
        ),
        "vitpose_s_apt36k": Model(
            "vitpose-s-apt36k.onnx",
            _VITPOSE_URL + "vitpose-s-apt36k.onnx",
            97307814,
            "2a40f76421d87f1b03f1204949d67014dbb876b7572c2bbc60ce2be45f6b4172",
        ),
        "vitpose_l_apt36k": Model(
            "vitpose-l-apt36k.onnx",
            _VITPOSE_URL + "vitpose-l-apt36k.onnx",
            1234459913,
            "595e2a10cf20a744446b924f0cea7b9a4c7b35e4f730a148510cac863a3d8737",
        ),
        "yolox_m": Model(
            "yolox_m.onnx",
            _YOLOX_URL + "yolox_m.onnx",
            101259744,
            "21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f",
        ),
        "yolox_tiny": Model(
            "yolox_tiny.onnx",
            _YOLOX_URL + "yolox_tiny.onnx",
            20219662,
            "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
        ),
    }
)
MODELS = MappingProxyType(
    {
        Mode.LIGHTWEIGHT: ("yolox_tiny", "vitpose_s_apt36k"),
        Mode.BALANCED: ("yolox_m", "vitpose_b_apt36k"),
        Mode.ACCURATE: ("yolox_m", "vitpose_l_apt36k"),
    }
)
Progress = Callable[[str, int, int], None]


class ModelChecksumError(CopypawError):
    """Model bytes do not match the pinned SHA-256 or byte count."""

    codes = ("model_checksum", "model_size")


class ModelFetchError(CopypawError):
    """A model could not be downloaded or its cache could not be written."""

    codes = ("model_download", "model_cache")


def _verify(path: Path, model: Model) -> None:
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != model.sha256:
        raise ModelChecksumError(
            "model_checksum", filename=model.filename, expected=model.sha256, actual=actual
        )
    actual_bytes = path.stat().st_size
    if actual_bytes != model.size_bytes:
        raise ModelChecksumError(
            "model_size", filename=model.filename, expected=model.size_bytes, actual=actual_bytes
        )


def _fetch(model: Model, destination: Path, progress: Progress | None) -> None:
    if urlsplit(model.url).scheme not in {"https", "http"}:
        raise ValueError(f"Unsupported model URL scheme: {model.url}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".part", delete=False
        ) as out:
            temporary = Path(out.name)
            downloaded = 0
            if progress is not None:
                progress(model.filename, downloaded, model.size_bytes)
            with urllib.request.build_opener().open(
                model.url,
                timeout=_DOWNLOAD_TIMEOUT_S,
            ) as response:
                while chunk := response.read(_CHUNK_BYTES):
                    out.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(model.filename, downloaded, model.size_bytes)
        _verify(temporary, model)
        temporary.replace(destination)
    except (OSError, HTTPException) as error:
        raise ModelFetchError("model_download", filename=model.filename, reason=error) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_models(mode: str | Mode, progress: Progress | None = None) -> dict[str, Path]:
    """Return verified model paths, downloading missing weights into COPYPAW_CACHE.

    Progress receives the filename, downloaded bytes, and expected bytes.
    Existing files are verified on every call; invalid files raise an error.
    """
    selected = MODELS[Mode(mode)]
    cache = Path(os.environ.get("COPYPAW_CACHE", "~/.cache/copypaw/models")).expanduser()
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ModelFetchError("model_cache", path=cache, reason=error) from error
    paths: dict[str, Path] = {}
    for key in selected:
        model = REGISTRY[key]
        destination = cache / model.filename
        if destination.exists():
            _verify(destination, model)
        else:
            _fetch(model, destination, progress)
        paths[key] = destination
    return paths
