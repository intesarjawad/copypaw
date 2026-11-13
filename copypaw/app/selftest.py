"""Run the packaged sample pair through the app's own API and check the cold-start budgets."""

import http.client
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import copypaw
from copypaw.app.server import GALLERY_DIR, STATIC_DIR, LocalApp, Pipeline

__all__ = [
    "CLIP_COMPUTE_BUDGET_S",
    "CLIP_TOTAL_BUDGET_S",
    "GALLERY_BUDGET_S",
    "MODEL_BYTES",
    "PAGE_OPEN_BUDGET_S",
    "REFERENCE_BYTES_PER_S",
    "SELFTEST_DIR",
    "SelfTestReport",
    "Step",
    "credit_line",
    "format_report",
    "format_table",
    "misses",
    "overlap_s",
    "parse_labels",
    "run_self_test",
    "throughput",
]

SELFTEST_DIR = Path(copypaw.__file__).parent / "selftest"
DRAWING = SELFTEST_DIR / "dragon.png"
CLIP = SELFTEST_DIR / "dog_walk.mp4"
ATTRIBUTION = SELFTEST_DIR / "ATTRIBUTION.txt"

# Seconds a first run may take. Transfer is taken out of the page-open and clip rows first.
PAGE_OPEN_BUDGET_S = 30.0
GALLERY_BUDGET_S = 30.0
CLIP_TOTAL_BUDGET_S = 90.0
MODEL_BYTES = 70_000_000
# 50 Mbps: the connection the clip budget assumes for the model download.
REFERENCE_BYTES_PER_S = 6_250_000
CLIP_COMPUTE_BUDGET_S = CLIP_TOTAL_BUDGET_S - MODEL_BYTES / REFERENCE_BYTES_PER_S

PAGE_OPEN = "Page open"
DRAWING_RIGGED = "Drawing rigged"
GALLERY_RESULT = "Gallery result"
CLIP_CAPTURE = "Clip capture"
CLIP_RESULT = "Clip result"

_JOB_TIMEOUT_S = 900.0
_POLL_S = 0.2
_HTTP_TIMEOUT_S = 120
_HTTP_OK = 200
_TOKEN_HEADER = "X-Copypaw-Token"  # noqa: S105
_HEADER = ("Step", "Elapsed (s)", "Transfer (s)", "Compute (s)", "Budget (s)", "Result")
_BYTES_PER_MB = 1_000_000


@dataclass(frozen=True)
class Step:
    """One timed step; elapsed is None when the step did not run."""

    label: str
    elapsed_s: float | None = None
    transfer_s: float = 0.0
    budget_s: float | None = None
    note: str = ""
    failed: bool = False

    @property
    def compute_s(self) -> float | None:
        """Elapsed time without the model or wheel transfer."""
        return None if self.elapsed_s is None else max(0.0, self.elapsed_s - self.transfer_s)

    @property
    def verdict(self) -> str:
        """`ok`, `-` (no budget), `skipped`, `FAILED` or `MISS`."""
        compute = self.compute_s
        if self.failed:
            return "FAILED"
        if compute is None:
            return "skipped"
        if self.budget_s is None:
            return "-"
        return "ok" if compute <= self.budget_s else "MISS"


@dataclass(frozen=True)
class SelfTestReport:
    """The steps, and what the model download cost."""

    steps: tuple[Step, ...]
    model_bytes: int
    model_ready_s: float

    @property
    def passed(self) -> bool:
        """True when no step failed or missed its budget."""
        return not misses(self.steps)


def misses(steps: tuple[Step, ...] | list[Step]) -> list[Step]:
    """Steps that failed or took longer than their budget."""
    return [step for step in steps if step.verdict in {"FAILED", "MISS"}]


def throughput(size_bytes: int, seconds: float) -> float | None:
    """Bytes per second, or None when nothing was measured."""
    return size_bytes / seconds if size_bytes > 0 and seconds > 0 else None


def overlap_s(start_s: float, end_s: float, window_start_s: float, window_end_s: float) -> float:
    """Seconds the span `start_s`..`end_s` shares with the window."""
    return max(0.0, min(end_s, window_end_s) - max(start_s, window_start_s))


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def format_table(steps: tuple[Step, ...] | list[Step]) -> str:
    """Render the steps as a Markdown table; a note follows the verdict."""
    lines = [f"| {' | '.join(_HEADER)} |", f"| {' | '.join('---' for _ in _HEADER)} |"]
    for step in steps:
        ran = step.elapsed_s is not None
        cells = (
            step.label,
            _seconds(step.elapsed_s),
            _seconds(step.transfer_s) if ran else "-",
            _seconds(step.compute_s),
            _seconds(step.budget_s),
            f"{step.verdict}: {step.note}" if step.note else step.verdict,
        )
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines)


def parse_labels(text: str) -> list[str]:
    """The first cell of every data row in a printed table."""
    labels: list[str] = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0] not in {_HEADER[0], "---"}:
            labels.append(cells[0])
    return labels


def credit_line() -> str:
    """The first line of the attribution that ships with the sample clip."""
    return ATTRIBUTION.read_text(encoding="utf-8").splitlines()[0]


def format_report(report: SelfTestReport) -> str:
    """The table, the model download line and the clip credit."""
    rate = throughput(report.model_bytes, report.model_ready_s)
    if report.model_bytes == 0 or rate is None:
        models = f"Models: ready after {report.model_ready_s:.1f} s, nothing downloaded"
    else:
        megabytes = report.model_bytes / _BYTES_PER_MB
        models = (
            f"Models: {megabytes:.1f} MB in {report.model_ready_s:.1f} s "
            f"({rate / _BYTES_PER_MB:.1f} MB/s)"
        )
    verdict = "Within budget" if report.passed else "Budget missed"
    return "\n".join([format_table(report.steps), "", models, verdict, f"Credit: {credit_line()}"])


class _StepError(Exception):
    """A step could not finish; the message is the line printed for it."""


class _Api:
    """The requests the page makes, sent with the session token."""

    def __init__(self, local: LocalApp) -> None:
        self._local = local

    def _request(self, method: str, path: str, body: bytes = b"", *, token: bool = True) -> bytes:
        connection = http.client.HTTPConnection(*self._local.address, timeout=_HTTP_TIMEOUT_S)
        headers = {_TOKEN_HEADER: self._local.token} if token else {}
        try:
            connection.request(
                method, path, body=body if method == "POST" else None, headers=headers
            )
            response = connection.getresponse()
            data = response.read()
        except OSError as error:
            raise _StepError(f"{method} {path} failed: {error}") from error
        finally:
            connection.close()
        if response.status != _HTTP_OK:
            raise _StepError(_failure(data) or f"{method} {path} returned {response.status}")
        return data

    def page(self, path: str) -> bytes:
        return self._request("GET", path, token=False)

    def get(self, path: str) -> dict[str, object]:
        return cast("dict[str, object]", json.loads(self._request("GET", path)))

    def post(self, path: str, body: bytes) -> dict[str, object]:
        return cast("dict[str, object]", json.loads(self._request("POST", path, body)))

    def file(self, url: str) -> bytes:
        return self._request("GET", url)


def _failure(data: bytes) -> str:
    try:
        error = json.loads(data).get("what")
    except (ValueError, AttributeError):
        return ""
    return error if isinstance(error, str) else ""


class _ModelWatch:
    """Record when the model download finished and how many bytes it fetched."""

    def __init__(self, api: _Api, clock: Callable[[], float]) -> None:
        self._api, self._clock = api, clock
        self.ready_at: float | None = None
        self.size_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="copypaw-selftest", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                models = cast("Mapping[str, object]", self._api.get("/api/health")["models"])
            except _StepError:
                return
            self.size_bytes = max(self.size_bytes, cast("int", models["downloaded"]))
            if models["state"] != "downloading":
                self.ready_at = self._clock()
                return
            self._stop.wait(_POLL_S)


def _wait(api: _Api, job_id: str, clock: Callable[[], float]) -> dict[str, object]:
    deadline = clock() + _JOB_TIMEOUT_S
    while clock() < deadline:
        job = api.get(f"/api/jobs/{job_id}")
        if job["state"] == "done":
            return job
        if job["state"] == "failed":
            error = cast("Mapping[str, str]", job["error"])
            raise _StepError(error["what"])
        time.sleep(_POLL_S)
    raise _StepError("the job did not finish in time")


def _json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload).encode()


def _result_video(api: _Api, job: Mapping[str, object]) -> None:
    results = cast("Mapping[str, str]", job["results"])
    if not api.file(results["video"]):
        raise _StepError("the result video is empty")


class _Flow:
    """The steps of one run, in order; a failure skips everything after it."""

    def __init__(
        self,
        api: _Api,
        local: LocalApp,
        watch: _ModelWatch,
        started: float,
        clock: Callable[[], float],
    ) -> None:
        self.api, self.local, self.watch = api, local, watch
        self.started, self.clock = started, clock
        self.steps: list[Step] = []
        self.failed = False

    def skip(self, label: str, note: str) -> None:
        self.steps.append(Step(label, note=note))

    def step(
        self,
        label: str,
        work: Callable[[], None],
        *,
        budget_s: float | None = None,
        since: float | None = None,
        net_of_models: bool = True,
    ) -> None:
        """Time `work` from `since`; waiting on the model download counts as transfer."""
        if self.failed:
            self.skip(label, "an earlier step failed")
            return
        begun = self.clock() if since is None else since
        try:
            work()
        except _StepError as error:
            self.failed = True
            self.steps.append(Step(label, self.clock() - begun, note=str(error), failed=True))
            return
        ended = self.clock()
        ready = self.watch.ready_at if self.watch.ready_at is not None else ended
        transfer = overlap_s(self.started, ready, begun, ended) if net_of_models else 0.0
        self.steps.append(Step(label, ended - begun, transfer, budget_s))


def _run_steps(
    api: _Api,
    local: LocalApp,
    watch: _ModelWatch,
    started: float,
    *,
    render: bool,
    clock: Callable[[], float],
) -> list[Step]:
    flow = _Flow(api, local, watch, started, clock)
    rig: dict[str, object] = {}
    capture: dict[str, object] = {}

    def open_page() -> None:
        if not api.page("/index.html"):
            raise _StepError("the page is empty")
        api.get("/api/health")

    def rig_drawing() -> None:
        rig.update(api.post("/api/rig", DRAWING.read_bytes()))
        api.post("/api/joints", _json({"rig": rig["rig"], "joints": rig["joints"]}))

    def animate(motion: Mapping[str, object]) -> Callable[[], None]:
        def work() -> None:
            job = api.post("/api/animate", _json({"rig": rig["rig"], **motion}))
            _result_video(api, _wait(api, cast("str", job["job"]), clock))

        return work

    def track_clip() -> None:
        capture.update(api.post("/api/capture", CLIP.read_bytes()))
        _wait(api, cast("str", capture["job"]), clock)

    flow.step(PAGE_OPEN, open_page, budget_s=PAGE_OPEN_BUDGET_S, since=started, net_of_models=False)
    flow.step(DRAWING_RIGGED, rig_drawing, net_of_models=False)
    gallery = next(iter(local.gallery), None)
    if not render:
        flow.skip(GALLERY_RESULT, "render skipped")
    elif gallery is None:
        flow.skip(GALLERY_RESULT, "no gallery motion is packaged")
    else:
        # The animate job never waits on the download, so its whole span is compute.
        flow.step(
            GALLERY_RESULT,
            animate({"gallery": gallery}),
            budget_s=GALLERY_BUDGET_S,
            net_of_models=False,
        )
    clip_started = clock()
    flow.step(CLIP_CAPTURE, track_clip, since=clip_started)

    def render_clip() -> None:
        animate({"capture": capture["job"]})()

    if render:
        # The clip row spans capture and render, as the page's Animate wait does.
        flow.step(CLIP_RESULT, render_clip, budget_s=CLIP_COMPUTE_BUDGET_S, since=clip_started)
    else:
        flow.skip(CLIP_RESULT, "render skipped")
    return flow.steps


def run_self_test(
    pipeline: Pipeline,
    *,
    static_dir: Path = STATIC_DIR,
    gallery_dir: Path = GALLERY_DIR,
    render: bool = True,
    clock: Callable[[], float] = time.monotonic,
) -> SelfTestReport:
    """Serve the page, then rig the sample drawing and track the sample clip like the UI does."""
    started = clock()
    local = LocalApp(pipeline, static_dir=static_dir, gallery_dir=gallery_dir)
    try:
        local.start()
        api = _Api(local)
        watch = _ModelWatch(api, clock)
        watch.start()
        try:
            steps = _run_steps(api, local, watch, started, render=render, clock=clock)
        finally:
            watch.stop()
    finally:
        local.close()
    ready_s = (clock() if watch.ready_at is None else watch.ready_at) - started
    return SelfTestReport(tuple(steps), watch.size_bytes, ready_s)
