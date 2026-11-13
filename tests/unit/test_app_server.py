import http.client
import io
import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

import copypaw.cli as cli_module
from copypaw.app.server import SESSION_PREFIX, AppError, LocalApp, Pipeline
from copypaw.cli import app
from copypaw.models import Progress
from copypaw.motion.motion_file import MotionFile

PAGE_BGR = 245
INK_BGR = 20
JOINT_COUNT = 16
OVER_LIMIT_BYTES = 61_000_000
JOB_TIMEOUT_S = 30
ROOT = Path(__file__).resolve().parents[2]
GOLDEN_MOTION = ROOT / "tests/golden/dog_walk.motion.npz"
SAMPLE_CLIP = ROOT / "samples/pets/dog_walk.mp4"
INDEX = b"<!doctype html><title>Copypaw</title>"


class FakeSteps:
    def __init__(self) -> None:
        self.fetched: list[str] = []
        self.credits: list[bool] = []

    def fetch(self, mode: str, progress: Progress | None) -> Mapping[str, Path]:
        self.fetched.append(str(mode))
        if progress is not None:
            progress("detector.onnx", 5, 10)
        return {}

    def capture(self, clip: Path, *, out: Path) -> None:
        assert clip.read_bytes() == b"clip bytes"
        shutil.copyfile(GOLDEN_MOTION, out / "motion.npz")

    def animate(self, char_dir: Path, motion: Path, *, clip: Path, out: Path, credit: bool) -> None:
        assert (char_dir / "char_cfg.yaml").is_file()
        assert motion.is_file()
        assert clip.is_file()
        self.credits.append(credit)
        for name in ("side_by_side.mp4", "side_by_side.gif"):
            (out / name).write_bytes(b"video")
        (out / "fidelity.json").write_text('{"timing": "Steps match."}', encoding="utf-8")


@pytest.fixture
def temp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "temp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


@pytest.fixture
def static(tmp_path: Path) -> Path:
    folder = tmp_path / "static"
    folder.mkdir()
    (folder / "index.html").write_bytes(INDEX)
    return folder


@pytest.fixture
def gallery(tmp_path: Path) -> Path:
    folder = tmp_path / "gallery"
    folder.mkdir()
    shutil.copyfile(GOLDEN_MOTION, folder / "walk.motion.npz")
    shutil.copyfile(SAMPLE_CLIP, folder / "walk.mp4")
    return folder


@pytest.fixture
def steps() -> FakeSteps:
    return FakeSteps()


@pytest.fixture
def local(temp_root: Path, static: Path, gallery: Path, steps: FakeSteps) -> Iterator[LocalApp]:
    del temp_root
    pipeline = Pipeline(capture=steps.capture, animate=steps.animate, fetch_models=steps.fetch)
    server = LocalApp(pipeline, static_dir=static, gallery_dir=gallery)
    server.start()
    yield server
    server.close()


@pytest.fixture
def drawing_png() -> bytes:
    page = np.full((200, 300, 3), PAGE_BGR, dtype=np.uint8)
    cv2.rectangle(page, (40, 30), (260, 170), (INK_BGR,) * 3, 3)
    encoded, data = cv2.imencode(".png", page)
    assert encoded
    return data.tobytes()


def send_with_headers(
    local: LocalApp,
    method: str,
    path: str,
    body: bytes = b"",
    headers: Mapping[str, str] | None = None,
    *,
    send_body: bool = True,
) -> tuple[int, list[tuple[str, str]], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", local.port, timeout=10)
    try:
        connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        merged = {
            "Host": f"127.0.0.1:{local.port}",
            "X-Copypaw-Token": local.token,
            "Content-Length": str(len(body)),
            **(headers or {}),
        }
        for name, value in merged.items():
            connection.putheader(name, value)
        connection.endheaders(body if send_body else None)
        response = connection.getresponse()
        received = [(name.lower(), value) for name, value in response.getheaders()]
        return response.status, received, response.read()
    finally:
        connection.close()


def send(
    local: LocalApp,
    method: str,
    path: str,
    body: bytes = b"",
    headers: Mapping[str, str] | None = None,
    *,
    send_body: bool = True,
) -> tuple[int, bytes]:
    status, _, data = send_with_headers(local, method, path, body, headers, send_body=send_body)
    return status, data


def send_json(local: LocalApp, method: str, path: str, payload: object = None) -> dict[str, Any]:
    body = b"" if payload is None else json.dumps(payload).encode()
    status, data = send(local, method, path, body)
    assert status == 200, data
    return json.loads(data)


def wait_for(local: LocalApp, job_id: object) -> dict[str, Any]:
    deadline = time.monotonic() + JOB_TIMEOUT_S
    while time.monotonic() < deadline:
        job = send_json(local, "GET", f"/api/jobs/{job_id}")
        if job["state"] in {"done", "failed"}:
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_binds_loopback_only(local: LocalApp) -> None:
    assert local.address == ("127.0.0.1", local.port)
    assert local.port > 0


def test_token_travels_in_the_fragment(local: LocalApp) -> None:
    assert local.url == f"http://127.0.0.1:{local.port}/#t={local.token}"
    assert len(local.token) >= 32


def test_page_loads_without_the_token(local: LocalApp) -> None:
    assert send(local, "GET", "/", headers={"X-Copypaw-Token": ""}) == (200, INDEX)


@pytest.mark.parametrize("token", ["", "wrong"])
def test_api_without_the_token_is_forbidden(local: LocalApp, token: str) -> None:
    status, _ = send(local, "GET", "/api/health", headers={"X-Copypaw-Token": token})
    assert status == 403


@pytest.mark.parametrize("token", ["", "wrong"])
def test_a_forbidden_request_names_the_next_step(local: LocalApp, token: str) -> None:
    status, data = send(local, "GET", "/api/health", headers={"X-Copypaw-Token": token})

    error = json.loads(data)
    assert status == 403
    assert error["code"] == "app_request"
    assert "session token" in error["what"]
    assert error["next"].startswith("Reload the address `copypaw app` printed")


def test_files_need_the_token_too(local: LocalApp) -> None:
    status, _ = send(local, "GET", "/api/files/gallery/walk.jpg", headers={"X-Copypaw-Token": ""})
    assert status == 403


@pytest.mark.parametrize("path", ["/", "/api/health"])
@pytest.mark.parametrize("host", ["evil.example:80", "127.0.0.1", "0.0.0.0:{port}", "localhost"])
def test_foreign_host_is_forbidden(local: LocalApp, path: str, host: str) -> None:
    status, _ = send(local, "GET", path, headers={"Host": host.format(port=local.port)})
    assert status == 403


def test_localhost_host_is_accepted(local: LocalApp) -> None:
    status, _ = send(local, "GET", "/api/health", headers={"Host": f"localhost:{local.port}"})
    assert status == 200


def test_health_reports_version_downloads_and_gallery(local: LocalApp, steps: FakeSteps) -> None:
    health = send_json(local, "GET", "/api/health")
    assert health["version"] == "0.1.0"
    assert health["gallery"] == [{"id": "walk", "thumbnail": "/api/files/gallery/walk.jpg"}]
    models = health["models"]
    assert models["state"] in {"downloading", "ready"}
    assert (models["downloaded"], models["total"]) == (5, 10)
    assert steps.fetched == ["balanced"]


def test_body_over_sixty_megabytes_is_refused(local: LocalApp) -> None:
    status, data = send(
        local,
        "POST",
        "/api/rig",
        headers={"Content-Length": str(OVER_LIMIT_BYTES)},
        send_body=False,
    )
    assert status == 413
    assert json.loads(data)["what"] == "The file is larger than the 60 MB Copypaw accepts."


def test_body_without_a_length_is_refused(local: LocalApp) -> None:
    status, _ = send(local, "POST", "/api/rig", headers={"Content-Length": "x"})
    assert status == 411


def test_files_outside_the_session_are_not_served(local: LocalApp, static: Path) -> None:
    assert send(local, "GET", "/api/files/../static/index.html")[0] == 404
    assert send(local, "GET", "/api/files/%2E%2E/static/index.html")[0] == 404
    assert send(local, "GET", f"/{'%2E%2E/' * 3}{static.name}/index.html")[0] == 404


def test_unknown_route_is_not_found(local: LocalApp) -> None:
    assert send(local, "GET", "/api/nothing")[0] == 404
    assert send(local, "POST", "/")[0] == 405


def test_unreadable_drawing_returns_the_message(local: LocalApp) -> None:
    status, data = send(local, "POST", "/api/rig", b"not an image")
    error = json.loads(data)
    assert status == 400
    assert error["code"] == "drawing_unreadable"
    assert error["next"].startswith("Save the drawing as a PNG or JPEG")


def test_bad_json_returns_the_request_message(local: LocalApp) -> None:
    status, data = send(local, "POST", "/api/joints", b"[1")
    assert status == 400
    assert json.loads(data)["what"] == (
        "The page sent a request Copypaw cannot use: the body is not JSON."
    )


def test_rig_edit_and_animate_with_a_gallery_motion(
    local: LocalApp, steps: FakeSteps, drawing_png: bytes
) -> None:
    status, data = send(local, "POST", "/api/rig", drawing_png)
    assert status == 200, data
    rig = json.loads(data)
    assert (rig["width"], rig["height"]) == (300, 200)
    assert len(rig["joints"]) == JOINT_COUNT
    xs = [x for x, _ in rig["outline"]]
    ys = [y for _, y in rig["outline"]]
    assert 30 <= min(xs) <= 45
    assert 255 <= max(xs) <= 270
    assert 20 <= min(ys) <= 35
    assert 165 <= max(ys) <= 180

    joints = send_json(local, "POST", "/api/joints", {"rig": rig["rig"], "joints": rig["joints"]})
    status, archive = send(local, "GET", str(joints["character"]))
    assert status == 200
    with zipfile.ZipFile(io.BytesIO(archive)) as character:
        assert {"char_cfg.yaml", "texture.png", "mask.png", "rig_edits.json"} <= set(
            character.namelist()
        )
        edits = json.loads(character.read("rig_edits.json"))
    assert edits["joints_moved"] == 0
    assert edits["final"] == rig["joints"]

    queued = send_json(
        local, "POST", "/api/animate", {"rig": rig["rig"], "gallery": "walk", "credit": False}
    )
    job = wait_for(local, queued["job"])

    assert job["state"] == "done", job
    assert job["timing"] == "Steps match."
    assert steps.credits == [False]
    results = job["results"]
    assert {"video", "gif", "mp4", "bvh", "character", "clip", "skeleton"} <= set(results)
    status, bvh = send(local, "GET", str(results["bvh"]))
    assert status == 200
    assert bvh.startswith(b"HIERARCHY")
    status, skeleton = send(local, "GET", str(results["skeleton"]))
    assert len(json.loads(skeleton)["frames"]) == len(MotionFile.load(GOLDEN_MOTION).kp)
    status, thumbnail = send(local, "GET", "/api/files/gallery/walk.jpg")
    assert status == 200
    assert thumbnail.startswith(b"\xff\xd8")


def test_capture_waits_for_models_then_feeds_animate(local: LocalApp, drawing_png: bytes) -> None:
    rig = json.loads(send(local, "POST", "/api/rig", drawing_png)[1])
    send_json(local, "POST", "/api/joints", {"rig": rig["rig"], "joints": rig["joints"]})

    status, data = send(local, "POST", "/api/capture", b"clip bytes")
    assert status == 200, data
    capture_job = json.loads(data)["job"]
    queued = send_json(local, "POST", "/api/animate", {"rig": rig["rig"], "capture": capture_job})

    assert wait_for(local, capture_job)["state"] == "done"
    assert wait_for(local, queued["job"])["state"] == "done"


def test_animate_before_joints_is_refused(local: LocalApp, drawing_png: bytes) -> None:
    rig = json.loads(send(local, "POST", "/api/rig", drawing_png)[1])
    status, data = send(
        local, "POST", "/api/animate", json.dumps({"rig": rig["rig"], "gallery": "walk"}).encode()
    )
    assert status == 400
    assert "the drawing has no joints yet" in json.loads(data)["what"]


def test_bad_joints_return_the_joints_message(local: LocalApp, drawing_png: bytes) -> None:
    rig = json.loads(send(local, "POST", "/api/rig", drawing_png)[1])
    status, data = send(
        local, "POST", "/api/joints", json.dumps({"rig": rig["rig"], "joints": {}}).encode()
    )
    assert status == 400
    assert json.loads(data)["code"] == "joints_invalid"


def test_log_has_method_route_and_status_only(
    local: LocalApp, drawing_png: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="copypaw.app"):
        send(local, "GET", "/api/files/jobs/abc/my_cat_photo.png")
        send(local, "POST", "/api/rig?name=my_cat_photo.png", drawing_png)
        send(local, "GET", "/?t=secret")
    lines = [record.getMessage() for record in caplog.records]
    assert lines == ["GET /api/files 404", "POST /api/rig 200", "GET / 200"]
    assert local.token not in caplog.text


@pytest.mark.parametrize(
    ("path", "headers", "status"),
    [
        ("/", {}, 200),
        ("/api/health", {}, 200),
        ("/api/health", {"X-Copypaw-Token": "wrong"}, 403),
        ("/", {"Host": "evil.example"}, 403),
        ("/api/nothing", {}, 404),
    ],
)
def test_every_response_forbids_caching_sniffing_and_referrers(
    local: LocalApp, path: str, headers: dict[str, str], status: int
) -> None:
    code, received, _ = send_with_headers(local, "GET", path, headers=headers)

    assert code == status
    assert dict(received)["cache-control"] == "no-store"
    assert dict(received)["x-content-type-options"] == "nosniff"
    assert dict(received)["referrer-policy"] == "no-referrer"


def test_responses_set_no_cookie_and_name_no_interpreter(local: LocalApp) -> None:
    _, received, _ = send_with_headers(local, "GET", "/api/health")
    names = [name for name, _ in received]

    assert "set-cookie" not in names
    assert dict(received)["server"].strip() == "Copypaw"


def test_each_launch_has_its_own_token_and_session(
    temp_root: Path, static: Path, gallery: Path, steps: FakeSteps
) -> None:
    pipeline = Pipeline(capture=steps.capture, animate=steps.animate, fetch_models=steps.fetch)
    first = LocalApp(pipeline, static_dir=static, gallery_dir=gallery)
    second = LocalApp(pipeline, static_dir=static, gallery_dir=gallery)
    try:
        assert first.token != second.token
        assert first.session != second.session
        assert first.session.parent == second.session.parent == temp_root.resolve()
    finally:
        first.close()
        second.close()


def test_the_uploaded_drawing_is_not_kept_after_rigging(
    local: LocalApp, drawing_png: bytes
) -> None:
    status, _ = send(local, "POST", "/api/rig", drawing_png)

    assert status == 200
    kept = [path for path in local.session.rglob("*") if path.is_file()]
    assert all(path.read_bytes() != drawing_png for path in kept)
    assert not [path for path in kept if path.name == "drawing"]


def test_session_folder_is_removed_on_close(
    temp_root: Path, static: Path, gallery: Path, steps: FakeSteps
) -> None:
    pipeline = Pipeline(capture=steps.capture, animate=steps.animate, fetch_models=steps.fetch)
    server = LocalApp(pipeline, static_dir=static, gallery_dir=gallery)
    server.start()
    assert server.session.parent == temp_root.resolve()
    assert server.session.name.startswith(SESSION_PREFIX)
    assert server.session.is_dir()

    server.close()

    assert not server.session.exists()


def test_stale_sessions_are_removed_on_launch(
    temp_root: Path, static: Path, gallery: Path, steps: FakeSteps
) -> None:
    stale = temp_root / f"{SESSION_PREFIX}old"
    stale.mkdir()
    (stale / "clip.mp4").write_bytes(b"left over")
    dead = temp_root / f"{SESSION_PREFIX}dead"
    dead.mkdir()
    (dead / "owner.pid").write_text("999999999", encoding="utf-8")
    running = temp_root / f"{SESSION_PREFIX}running"
    running.mkdir()
    (running / "owner.pid").write_text(str(os.getpid()), encoding="utf-8")
    unrelated = temp_root / "other-folder"
    unrelated.mkdir()

    pipeline = Pipeline(capture=steps.capture, animate=steps.animate, fetch_models=steps.fetch)
    server = LocalApp(pipeline, static_dir=static, gallery_dir=gallery)
    server.close()

    assert not stale.exists()
    assert not dead.exists()
    assert running.is_dir()
    assert unrelated.is_dir()


def test_missing_page_names_the_build_command(tmp_path: Path, steps: FakeSteps) -> None:
    pipeline = Pipeline(capture=steps.capture, animate=steps.animate, fetch_models=steps.fetch)

    with pytest.raises(AppError) as raised:
        LocalApp(pipeline, static_dir=tmp_path / "static")

    assert raised.value.code == "app_static_missing"
    assert "`cd web && npm ci && npm run build`" in raised.value.next


def test_app_command_offers_no_way_to_leave_loopback() -> None:
    result = CliRunner().invoke(app, ["app", "--help"], env={"NO_COLOR": "1", "TERM": "dumb"})

    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--host" not in result.output
    assert "--share" not in result.output


def test_app_command_reports_a_missing_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "STATIC_DIR", tmp_path / "static")

    result = CliRunner().invoke(app, ["app", "--no-browser"])

    assert result.exit_code == 2
    assert f"error: The local app has no built web page in {tmp_path / 'static'}." in (
        result.stderr
    )
    assert "next:  Build it with `cd web && npm ci && npm run build`" in result.stderr
