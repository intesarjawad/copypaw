"""Inspect the local environment needed to run Copypaw."""

import importlib
import multiprocessing
import os
import platform
import sys
from collections.abc import Callable
from pathlib import Path

import glfw
import onnxruntime

# Model filenames and byte sizes of the pinned releases; a wrong size means a partial download.
MODELS = {
    "lightweight": (("yolox_tiny.onnx", 20219662), ("vitpose-s-apt36k.onnx", 97307814)),
    "balanced": (("yolox_m.onnx", 101259744), ("vitpose-b-apt36k.onnx", 360123609)),
    "accurate": (("yolox_m.onnx", 101259744), ("vitpose-l-apt36k.onnx", 1234459913)),
}
PROBE_TIMEOUT_SECONDS = 10


def _probe_glfw() -> None:
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(1, 1, "probe", None, None)
    if not window:
        raise RuntimeError("GLFW window creation failed")
    glfw.destroy_window(window)
    glfw.terminate()


def _probe_osmesa() -> None:
    importlib.import_module("OpenGL.osmesa")


def render_backend() -> str:
    """Probe rendering in a separate process to isolate native failures."""
    backend = "osmesa" if os.environ.get("PYOPENGL_PLATFORM") == "osmesa" else "glfw"
    probe = _probe_osmesa if backend == "osmesa" else _probe_glfw
    process = multiprocessing.get_context("spawn").Process(target=probe)
    process.start()
    process.join(PROBE_TIMEOUT_SECONDS)
    if process.is_alive():
        process.terminate()
        process.join()
    return backend if process.exitcode == 0 else "none"


def report(backend_probe: Callable[[], str] = render_backend) -> str:
    """Return a diagnostic report without downloading models."""
    system = platform.system()
    machine = platform.machine()
    supported = (
        platform.python_implementation() == "CPython"
        and (3, 12) <= sys.version_info[:2] <= (3, 14)
        and (system, machine) in {("Darwin", "arm64"), ("Linux", "x86_64")}
    )
    backend = backend_probe()
    render = f"render: {backend}"
    if backend == "none":
        render += (
            " (on Linux, install libosmesa6 and libosmesa6-dev and set "
            "PYOPENGL_PLATFORM=osmesa; on macOS, run in a graphical session)"
        )
    cache = Path(os.environ.get("COPYPAW_CACHE", "~/.cache/copypaw/models")).expanduser()
    lines = [
        f"python: {platform.python_implementation()} {platform.python_version()}",
        f"platform: {platform.platform()} ({machine})",
        f"onnxruntime providers: {', '.join(onnxruntime.get_available_providers())}",
        render,
    ]
    for mode, models in MODELS.items():
        present = sum(
            1
            for filename, size in models
            if (cache / filename).is_file() and (cache / filename).stat().st_size == size
        )
        total_bytes = sum(size for _, size in models)
        lines.append(
            f"models {mode}: {present}/{len(models)} cached ({total_bytes} bytes required)"
        )
    lines.append(f"support: {'supported' if supported else f'untested ({system} {machine})'}")
    return "\n".join(lines)
