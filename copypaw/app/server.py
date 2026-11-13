"""Serve the web page and its JSON API on loopback for one local session."""

import hmac
import json
import logging
import mimetypes
import os
import secrets
import shutil
import tempfile
import threading
import zipfile
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, Literal, Protocol, cast
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np

from copypaw import __version__
from copypaw.errors import CopypawError, InternalError
from copypaw.models import Mode, Progress, ensure_models
from copypaw.motion.bvh_writer import write_bvh
from copypaw.motion.motion_file import MotionFile
from copypaw.motion.skeleton import BONES
from copypaw.rig.char_cfg import write_character
from copypaw.rig.edits import RigResult, load_drawing, load_joints, rig_drawing, write_edits

__all__ = [
    "GALLERY_DIR",
    "MAX_BODY_BYTES",
    "SESSION_PREFIX",
    "STATIC_DIR",
    "TOKEN_HEADER",
    "AppError",
    "LocalApp",
    "Pipeline",
    "remove_stale_sessions",
]

STATIC_DIR = Path(__file__).parent / "static"
GALLERY_DIR = Path(__file__).parent / "gallery"
# Raw bodies keep a multipart parser out of the server; a phone clip of a few seconds fits.
MAX_BODY_BYTES = 60_000_000
_BYTES_PER_MB = 1_000_000
SESSION_PREFIX = "copypaw-session-"
TOKEN_HEADER = "X-Copypaw-Token"  # noqa: S105
_OWNER_FILE = "owner.pid"
_LOOPBACK = "127.0.0.1"
_TOKEN_BYTES = 32
_ID_BYTES = 8
_SKELETON_DECIMALS = 1
_GALLERY_SUFFIX = ".motion.npz"
_log = logging.getLogger("copypaw.app")

type JobState = Literal["queued", "running", "done", "failed"]
type JobStage = Literal["queued", "models", "capture", "render", "done"]
type FetchModels = Callable[[str, Progress | None], Mapping[str, Path]]


class AppError(CopypawError):
    """The local app cannot start, or the page sent a request it cannot use."""

    codes = ("app_request", "app_static_missing", "app_port", "app_body_too_large")


class CaptureStep(Protocol):
    """Write motion.npz for a clip into an existing folder."""

    def __call__(self, clip: Path, *, out: Path) -> None: ...


class AnimateStep(Protocol):
    """Write the side-by-side, animation and fidelity files into a folder."""

    def __call__(
        self, char_dir: Path, motion: Path, *, clip: Path, out: Path, credit: bool
    ) -> None: ...


@dataclass(frozen=True)
class Pipeline:
    """The capture and animation steps the jobs run, and the model download."""

    capture: CaptureStep
    animate: AnimateStep
    fetch_models: FetchModels = ensure_models


@dataclass(frozen=True)
class _GalleryMotion:
    motion: Path
    clip: Path
    thumbnail: str


@dataclass
class _Job:
    state: JobState = "queued"
    stage: JobStage = "queued"
    results: dict[str, str] = field(default_factory=dict[str, str])
    timing: str | None = None
    error: CopypawError | None = None
    motion: Path | None = None
    clip: Path | None = None

    def source(self) -> tuple[Path, Path]:
        if self.motion is None or self.clip is None:
            raise self.error or AppError(problem="the clip has not been captured")
        return self.motion, self.clip

    def as_json(self) -> dict[str, object]:
        return {
            "state": self.state,
            "stage": self.stage,
            "results": self.results,
            "timing": self.timing,
            "error": None if self.error is None else _error_json(self.error),
        }


class _ModelDownload:
    """Fetch the balanced models once, so a clip waits only for what is still missing."""

    def __init__(self, fetch: FetchModels) -> None:
        self._fetch = fetch
        self._lock = threading.Lock()
        self._bytes: dict[str, tuple[int, int]] = {}
        self._done = threading.Event()
        self._error: CopypawError | None = None
        self._thread = threading.Thread(target=self._run, name="copypaw-models", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wait(self) -> None:
        self._done.wait()
        if self._error is not None:
            raise self._error

    def status(self) -> dict[str, object]:
        with self._lock:
            downloaded = sum(done for done, _ in self._bytes.values())
            total = sum(expected for _, expected in self._bytes.values())
        state = "failed" if self._error else "ready" if self._done.is_set() else "downloading"
        return {"state": state, "downloaded": downloaded, "total": total}

    def _progress(self, filename: str, downloaded: int, total: int) -> None:
        with self._lock:
            self._bytes[filename] = (downloaded, total)

    def _run(self) -> None:
        try:
            self._fetch(Mode.BALANCED, self._progress)
        except CopypawError as error:
            self._error = error
        except Exception:
            _log.exception("Model download failed")
            self._error = InternalError()
        finally:
            self._done.set()


def _error_json(error: CopypawError) -> dict[str, str]:
    return {"code": error.code, "what": error.what, "next": error.next}


def _owner_alive(folder: Path) -> bool:
    try:
        pid = int((folder / _OWNER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def remove_stale_sessions(root: Path) -> None:
    """Delete session folders left by an app that ended without cleaning up."""
    for folder in root.glob(f"{SESSION_PREFIX}*"):
        if folder.is_dir() and not _owner_alive(folder):
            shutil.rmtree(folder, ignore_errors=True)


def _gallery(source: Path, thumbnails: Path) -> dict[str, _GalleryMotion]:
    found: dict[str, _GalleryMotion] = {}
    for motion in sorted(source.glob(f"*{_GALLERY_SUFFIX}")):
        name = motion.name.removesuffix(_GALLERY_SUFFIX)
        clip = motion.with_name(f"{name}.mp4")
        thumbnails.mkdir(exist_ok=True)
        video = cv2.VideoCapture(str(clip))
        read, frame = video.read()
        video.release()
        if read:
            cv2.imwrite(str(thumbnails / f"{name}.jpg"), frame)
        found[name] = _GalleryMotion(motion, clip, f"gallery/{name}.jpg")
    return found


def _outline(mask: np.ndarray[tuple[int, int], np.dtype[np.uint8]]) -> list[list[int]]:
    binary = np.asarray(mask > 0, dtype=np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    return [[int(x), int(y)] for x, y in np.asarray(largest).reshape(-1, 2)]


def _zip(folder: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for path in sorted(folder.iterdir()):
            out.write(path, path.name)


def _write_skeleton(motion: MotionFile, path: Path) -> None:
    record = {
        "fps": motion.fps,
        "facing": motion.facing,
        "edges": [[bone.prox_kp, bone.dist_kp] for bone in BONES],
        "frames": np.round(motion.kp, _SKELETON_DECIMALS).tolist(),
    }
    path.write_text(json.dumps(record), encoding="utf-8")


def _timing(fidelity: Path) -> str | None:
    if not fidelity.is_file():
        return None
    data: object = json.loads(fidelity.read_text(encoding="utf-8"))
    value = cast("dict[str, object]", data).get("timing") if isinstance(data, dict) else None
    return value if isinstance(value, str) else None


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise AppError(problem=f"{key} must be text")
    return value


def _file_url(relative: str) -> str:
    return f"/api/files/{relative}"


class LocalApp:
    """One launch of the local app: its server, token, session folder and jobs."""

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        static_dir: Path = STATIC_DIR,
        gallery_dir: Path = GALLERY_DIR,
        port: int = 0,
    ) -> None:
        """Check the built page, clear stale sessions and bind 127.0.0.1 without serving yet."""
        if not (static_dir / "index.html").is_file():
            raise AppError("app_static_missing", path=static_dir)
        self.static_dir = static_dir.resolve()
        root = Path(tempfile.gettempdir())
        remove_stale_sessions(root)
        self.session = Path(tempfile.mkdtemp(prefix=SESSION_PREFIX, dir=root)).resolve()
        (self.session / _OWNER_FILE).write_text(str(os.getpid()), encoding="utf-8")
        self.token = secrets.token_urlsafe(_TOKEN_BYTES)
        try:
            self._server = ThreadingHTTPServer((_LOOPBACK, port), _handler_for(self))
        except OSError as error:
            shutil.rmtree(self.session, ignore_errors=True)
            raise AppError("app_port", port=port, reason=error) from error
        self.port = self._server.server_port
        self.hosts = frozenset({f"{_LOOPBACK}:{self.port}", f"localhost:{self.port}"})
        self._pipeline = pipeline
        self._lock = threading.Lock()
        self._rigs: dict[str, RigResult] = {}
        self._jobs: dict[str, _Job] = {}
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="copypaw-job")
        self._download = _ModelDownload(pipeline.fetch_models)
        self.gallery = _gallery(gallery_dir, self.session / "gallery")
        self._serving: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        """The bound interface and port."""
        return _LOOPBACK, self.port

    @property
    def url(self) -> str:
        """The page address; the fragment carries the token and never reaches the server."""
        return f"http://{_LOOPBACK}:{self.port}/#t={self.token}"

    def start(self) -> None:
        """Start the model download and serve requests on a background thread."""
        self._download.start()
        self._serving = threading.Thread(
            target=self._server.serve_forever, name="copypaw-server", daemon=True
        )
        self._serving.start()

    def wait(self) -> None:
        """Block until the server stops."""
        if self._serving is not None:
            self._serving.join()

    def close(self) -> None:
        """Stop serving, cancel queued jobs and delete the session folder."""
        if self._serving is not None:
            self._server.shutdown()
        self._server.server_close()
        self._worker.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(self.session, ignore_errors=True)

    def health(self) -> dict[str, object]:
        """Version, model download progress and the gallery motions."""
        return {
            "version": __version__,
            "models": self._download.status(),
            "gallery": [
                {"id": name, "thumbnail": _file_url(entry.thumbnail)}
                for name, entry in self.gallery.items()
            ],
        }

    def rig(self, body: bytes) -> dict[str, object]:
        """Rig an uploaded drawing and return its outline and snapped joints."""
        rig_id = secrets.token_hex(_ID_BYTES)
        folder = self.session / "rigs" / rig_id
        folder.mkdir(parents=True)
        upload = folder / "drawing"
        upload.write_bytes(body)
        try:
            image = load_drawing(upload)
        finally:
            upload.unlink()
        rigged = rig_drawing(image)
        with self._lock:
            self._rigs[rig_id] = rigged
        height, width = image.shape[:2]
        return {
            "rig": rig_id,
            "width": width,
            "height": height,
            "box_height": rigged.box_height,
            "outline": _outline(rigged.mask),
            "joints": {name: [joint.x, joint.y] for name, joint in rigged.snapped.items()},
        }

    def joints(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Write the character with the page's joints and record the edits beside it."""
        rig_id = _text(payload, "rig")
        with self._lock:
            rigged = self._rigs.get(rig_id)
        if rigged is None:
            raise AppError(problem="the drawing is not in this session")
        folder = self.session / "rigs" / rig_id
        positions = folder / "joints.json"
        positions.write_text(json.dumps(payload.get("joints")), encoding="utf-8")
        final = load_joints(positions, rigged.snapped)
        character = folder / "character"
        write_character(rigged.image, final, character)
        write_edits(character / "rig_edits.json", rigged.snapped, final, rigged.box_height)
        _zip(character, folder / "character.zip")
        return {"character": _file_url(f"rigs/{rig_id}/character.zip")}

    def capture(self, body: bytes) -> dict[str, object]:
        """Queue motion capture for an uploaded clip."""
        job_id, job = self._new_job()
        folder = self.session / "clips" / job_id
        folder.mkdir(parents=True)
        clip = folder / "clip.mp4"
        clip.write_bytes(body)

        def run() -> None:
            job.stage = "models"
            self._download.wait()
            job.stage = "capture"
            self._pipeline.capture(clip, out=folder)
            job.motion, job.clip = folder / "motion.npz", clip

        self._submit(job, run)
        return {"job": job_id}

    def animate(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Queue an animation of a rigged drawing with a gallery or captured motion."""
        rig_id = _text(payload, "rig")
        character = self.session / "rigs" / rig_id / "character"
        if rig_id not in self._rigs or not character.is_dir():
            raise AppError(problem="the drawing has no joints yet")
        credit = payload.get("credit", True)
        if not isinstance(credit, bool):
            raise AppError(problem="credit must be true or false")
        source = self._motion_source(payload)
        job_id, job = self._new_job()
        folder = self.session / "jobs" / job_id

        def run() -> None:
            job.stage = "render"
            motion_path, source_clip = source()
            folder.mkdir(parents=True)
            clip = folder / "clip.mp4"
            shutil.copyfile(source_clip, clip)
            self._pipeline.animate(character, motion_path, clip=clip, out=folder, credit=credit)
            motion = MotionFile.load(motion_path)
            write_bvh(motion.kp, motion.fps, motion.travel_root_x, out=folder / "motion.bvh")
            _write_skeleton(motion, folder / "skeleton.json")
            shutil.copyfile(character.parent / "character.zip", folder / "character.zip")
            job.timing = _timing(folder / "fidelity.json")
            files = {
                "video": "side_by_side.mp4",
                "gif": "side_by_side.gif",
                "mp4": "side_by_side.mp4",
                "bvh": "motion.bvh",
                "character": "character.zip",
                "clip": "clip.mp4",
                "skeleton": "skeleton.json",
            }
            if (folder / "footfall.json").is_file():
                files["footfall"] = "footfall.json"
            job.results = {key: _file_url(f"jobs/{job_id}/{name}") for key, name in files.items()}
            job.stage = "done"

        self._submit(job, run)
        return {"job": job_id}

    def job(self, job_id: str) -> dict[str, object]:
        """Report a job's state, stage, result URLs and any error."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise AppError(problem="the job is not in this session")
        return job.as_json()

    def _motion_source(self, payload: Mapping[str, object]) -> Callable[[], tuple[Path, Path]]:
        if "gallery" in payload:
            entry = self.gallery.get(_text(payload, "gallery"))
            if entry is None:
                raise AppError(problem="the gallery has no such motion")
            return lambda: (entry.motion, entry.clip)
        with self._lock:
            capture = self._jobs.get(_text(payload, "capture"))
        if capture is None:
            raise AppError(problem="the clip is not in this session")
        return capture.source

    def _new_job(self) -> tuple[str, _Job]:
        job_id, job = secrets.token_hex(_ID_BYTES), _Job()
        with self._lock:
            self._jobs[job_id] = job
        return job_id, job

    def _submit(self, job: _Job, work: Callable[[], None]) -> None:
        def run() -> None:
            job.state = "running"
            try:
                work()
            except CopypawError as error:
                job.error = error
                job.state = "failed"
            except Exception:
                _log.exception("Job failed")
                job.error = InternalError()
                job.state = "failed"
            else:
                job.state = "done"

        self._worker.submit(run)


def _route(path: str) -> str:
    """Keep two path segments so file names and ids never reach the log."""
    return "/".join(urlsplit(path).path.split("/")[:3]) or "/"


def _payload(body: bytes) -> Mapping[str, object]:
    try:
        data: object = json.loads(body)
    except ValueError as error:
        raise AppError(problem="the body is not JSON") from error
    if not isinstance(data, dict):
        raise AppError(problem="the body is not a JSON object")
    return cast("dict[str, object]", data)


class _Handler(BaseHTTPRequestHandler):
    local: ClassVar[LocalApp]
    server_version = "Copypaw"
    sys_version = ""

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        del size
        _log.info("%s %s %s", self.command, _route(getattr(self, "path", "")), code)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Drop the standard lines: they carry the whole request path."""
        del format, args

    def _handle(self, method: str) -> None:
        if self.headers.get("Host") not in self.local.hosts:
            self._send_status(HTTPStatus.FORBIDDEN)
            return
        path = urlsplit(self.path).path
        if not path.startswith("/api/"):
            if method == "GET":
                self._static(path)
            else:
                self._send_status(HTTPStatus.METHOD_NOT_ALLOWED)
            return
        token = self.headers.get(TOKEN_HEADER, "")
        if not hmac.compare_digest(token.encode(), self.local.token.encode()):
            error = AppError(problem="the session token is missing or wrong")
            self._send_json(HTTPStatus.FORBIDDEN, _error_json(error))
            return
        try:
            self._api(method, path.split("/")[2:])
        except CopypawError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, _error_json(error))
        except Exception:
            _log.exception("Request failed")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, _error_json(InternalError()))

    def _api(self, method: str, parts: list[str]) -> None:
        match method, parts:
            case "GET", ["health"]:
                self._send_json(HTTPStatus.OK, self.local.health())
            case "GET", ["jobs", job_id]:
                self._send_json(HTTPStatus.OK, self.local.job(job_id))
            case "GET", ["files", *rest]:
                self._file(self.local.session, unquote("/".join(rest)))
            case "POST", ["rig" | "capture" | "joints" | "animate" as action]:
                body = self._body()
                if body is None:
                    return
                result = {
                    "rig": lambda: self.local.rig(body),
                    "capture": lambda: self.local.capture(body),
                    "joints": lambda: self.local.joints(_payload(body)),
                    "animate": lambda: self.local.animate(_payload(body)),
                }[action]()
                self._send_json(HTTPStatus.OK, result)
            case _:
                self._send_status(HTTPStatus.NOT_FOUND)

    def _body(self) -> bytes | None:
        length = self.headers.get("Content-Length", "")
        if not length.isdecimal():
            self._send_status(HTTPStatus.LENGTH_REQUIRED)
            return None
        if int(length) > MAX_BODY_BYTES:
            error = AppError("app_body_too_large", limit_mb=MAX_BODY_BYTES // _BYTES_PER_MB)
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, _error_json(error))
            return None
        return self.rfile.read(int(length))

    def _static(self, path: str) -> None:
        relative = unquote(path).lstrip("/") or "index.html"
        self._file(self.local.static_dir, relative)

    def _file(self, root: Path, relative: str) -> None:
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            self._send_status(HTTPStatus.NOT_FOUND)
            return
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, target.read_bytes(), kind)

    def _send_json(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _send_status(self, status: HTTPStatus) -> None:
        self._send(status, status.phrase.encode(), "text/plain; charset=utf-8")

    def _send(self, status: HTTPStatus, data: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)


def _handler_for(local: LocalApp) -> type[_Handler]:
    class Handler(_Handler):
        pass

    Handler.local = local
    return Handler
