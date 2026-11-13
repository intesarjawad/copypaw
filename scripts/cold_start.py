"""Time the demo moment from nothing, print the table and exit 1 on any budget miss.

--browser URL drives a page in Chromium, WebKit and Firefox, each with a fresh profile.
--local installs the app with uvx into empty caches and drives its page.
"""

import argparse
import http.client
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from playwright.sync_api import BrowserType, Page, Playwright, sync_playwright
from playwright.sync_api import Error as PlaywrightError

from copypaw import __version__
from copypaw.app.selftest import (
    CLIP_CAPTURE,
    CLIP_COMPUTE_BUDGET_S,
    CLIP_RESULT,
    CLIP_TOTAL_BUDGET_S,
    DRAWING_RIGGED,
    GALLERY_BUDGET_S,
    GALLERY_RESULT,
    PAGE_OPEN,
    PAGE_OPEN_BUDGET_S,
    REFERENCE_BYTES_PER_S,
    SELFTEST_DIR,
    Step,
    format_table,
    misses,
    overlap_s,
    parse_labels,
    throughput,
)

ROOT = Path(__file__).resolve().parents[1]
DRAWING = SELFTEST_DIR / "dragon.png"
CLIP = SELFTEST_DIR / "dog_walk.mp4"
# Browser tiers that passed their parity and performance checks; --tiers overrides it.
SHIPPED_TIERS: frozenset[int] = frozenset({1})
BROWSERS = ("chromium", "webkit", "firefox")
MODEL_SUFFIX = ".onnx"
DEFAULT_GALLERY_MOTION = "dog_walk"
JOINT_COUNT = 16
TIER_1, TIER_2 = 1, 2
MS_PER_S = 1000.0
BYTES_PER_MB = 1_000_000
LATENCY_MS = 20
UNLIMITED = -1
STEP_TIMEOUT_MS = 900_000
SERVER_START_TIMEOUT_S = 300.0
HEALTH_TIMEOUT_S = 120.0
STOP_TIMEOUT_S = 30
POLL_S = 0.25
HTTP_OK = 200
HTTP_TIMEOUT_S = 60
FETCH_CHUNK_BYTES = 1024 * 1024
RUNNING_LINE = re.compile(r"is running at (http://\S+)")
RESULT_CHANGED = """(previous) => {
    const canvas = document.querySelector('.result canvas');
    const video = document.querySelector('#result-video');
    return document.body.dataset.state === 'result' && (
        (canvas !== null && canvas !== previous.canvas) ||
        (video !== null && video.src.startsWith('blob:') && video.src !== previous.video)
    );
}"""
MODEL_REQUESTS = f"""() => performance.getEntriesByType('resource')
    .filter((entry) => new URL(entry.name).pathname.endsWith('{MODEL_SUFFIX}'))
    .map((entry) => [entry.startTime, entry.responseEnd, entry.encodedBodySize])"""

# Software rasterizers: a hosted runner has no GPU, so its times say nothing about a user's laptop.
SOFTWARE_RENDERERS = ("swiftshader", "llvmpipe", "softpipe")
WEBGL_RENDERER = """() => {
    const gl = document.createElement('canvas').getContext('webgl2');
    if (gl === null) return '';
    const info = gl.getExtension('WEBGL_debug_renderer_info');
    return String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
}"""

type Interval = tuple[float, float]


@dataclass(frozen=True)
class Plan:
    """What one pass over the page does."""

    drawing: Path
    clip: Path
    gallery: str | None
    tier1: bool
    tier2: bool
    render: bool


@dataclass(frozen=True)
class Marks:
    """When each step of a pass began and ended, in seconds on one clock."""

    opened: float
    rig_start: float
    rigged: float
    gallery: Interval | None
    clip_start: float
    captured: float | None
    clip_end: float | None


@dataclass(frozen=True)
class Install:
    """What getting the app running cost: uv's install and the model download window."""

    seconds: float
    wheel_bytes: int
    bytes_per_s: float | None
    started: float
    models_ready: float | None


def union_s(intervals: Iterable[Interval]) -> float:
    """Total seconds covered by the intervals, counting overlaps once."""
    total, reach = 0.0, float("-inf")
    for start, end in sorted(intervals):
        total += max(0.0, end - max(start, reach))
        reach = max(reach, end)
    return total


def merged(intervals: Iterable[Interval]) -> list[Interval]:
    """The intervals with overlapping and touching ones joined."""
    joined: list[Interval] = []
    for start, end in sorted(intervals):
        if joined and start <= joined[-1][1]:
            joined[-1] = (joined[-1][0], max(joined[-1][1], end))
        else:
            joined.append((start, end))
    return joined


def transfer_in_window_s(intervals: Iterable[Interval], start_s: float, end_s: float) -> float:
    """Seconds of model transfer, each counted once, that fall inside the window."""
    return sum(overlap_s(start, end, start_s, end_s) for start, end in merged(intervals))


def package_spec(*, from_pypi: bool, wheel: Path | None) -> str:
    """What uvx installs: the released version, a given wheel, or a fresh build of this tree."""
    if from_pypi:
        return f"copypaw=={__version__}"
    if wheel is None:
        raise ValueError("a wheel is needed when the released version is not used")
    return str(wheel)


def uvx(spec: str, *arguments: str) -> list[str]:
    """The uvx command line for a copypaw invocation."""
    return ["uvx", "--from", spec, "copypaw", *arguments]


def directory_bytes(folder: Path) -> int:
    """Bytes of every file under the folder."""
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def net_of_wheel_s(wheel_bytes: int, bytes_per_s: float | None) -> float:
    """Transfer seconds to take out of an install, or 0 when no throughput was measured."""
    return 0.0 if bytes_per_s is None else wheel_bytes / bytes_per_s


def labels_match(rows: Sequence[Step], self_test_output: str) -> bool:
    """True when the packaged self-test printed the same steps in the same order."""
    return parse_labels(self_test_output) == [row.label for row in rows]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _fetch_models(urls: Sequence[str]) -> tuple[int, float]:
    """Download each model once with no cache; return the bytes and seconds taken."""
    total, started = 0, time.monotonic()
    for url in urls:
        request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})  # noqa: S310
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:  # noqa: S310
            while chunk := response.read(FETCH_CHUNK_BYTES):
                total += len(chunk)
    return total, time.monotonic() - started


def _drive(
    page: Page,
    url: str,
    plan: Plan,
    clock: Callable[[], float],
    *,
    from_navigation: bool,
    track_capture: Callable[[str], None] | None = None,
) -> Marks:
    """Open the page, rig the drawing, then animate with the gallery motion and the clip."""
    page.set_default_timeout(STEP_TIMEOUT_MS)
    page.goto(url)
    page.locator("#drawing-input").wait_for(state="attached", timeout=STEP_TIMEOUT_MS)
    opened = clock()
    rig_start = clock()
    page.locator("#drawing-input").set_input_files(plan.drawing)
    page.locator("svg .dot").nth(JOINT_COUNT - 1).wait_for(timeout=STEP_TIMEOUT_MS)
    rigged = clock()
    animate = page.get_by_role("button", name="Animate", exact=True)

    def animate_and_wait() -> None:
        previous = page.evaluate_handle("""() => ({
            canvas: document.querySelector('.result canvas'),
            video: document.querySelector('#result-video')?.src ?? ''
        })""")
        try:
            animate.click()
            page.wait_for_function(RESULT_CHANGED, arg=previous, timeout=STEP_TIMEOUT_MS)
        finally:
            previous.dispose()

    gallery: Interval | None = None
    if plan.tier1 and plan.render:
        started = 0.0 if from_navigation else clock()
        motion = (
            page.locator(f"input[name=motion][value='{plan.gallery}']")
            if plan.gallery
            else page.locator("input[name=motion]").first
        )
        if motion.count() > 0 and motion.is_visible():
            motion.check()
        # A lone gallery motion with no clip to track is chosen for the user, with no radio.
        if animate.is_enabled():
            animate_and_wait()
            gallery = (started, clock())
    clip_start = clock()
    captured: float | None = None
    clip_end: float | None = None
    if plan.tier2:
        if track_capture is None:
            page.locator("#clip-input").set_input_files(plan.clip)
        else:
            with page.expect_response(lambda r: r.url.endswith("/api/capture")) as sent:
                page.locator("#clip-input").set_input_files(plan.clip)
            track_capture(cast("str", sent.value.json()["job"]))
            captured = clock()
        if plan.render:
            animate_and_wait()
            clip_end = clock()
    return Marks(
        opened=opened,
        rig_start=rig_start,
        rigged=rigged,
        gallery=gallery,
        clip_start=clip_start,
        captured=captured,
        clip_end=clip_end,
    )


def _page_clock(page: Page) -> Callable[[], float]:
    """Seconds since the page's navigation began, the timeline Resource Timing uses."""
    return lambda: cast("float", page.evaluate("performance.now()")) / MS_PER_S


def _model_requests(page: Page) -> tuple[list[Interval], int]:
    entries = cast("list[list[float]]", page.evaluate(MODEL_REQUESTS))
    spans = [(start / MS_PER_S, end / MS_PER_S) for start, end, _ in entries]
    return spans, int(sum(size for _, _, size in entries))


def is_software_renderer(renderer: str) -> bool:
    """Whether a WebGL renderer string names a CPU rasterizer."""
    return any(name in renderer.lower() for name in SOFTWARE_RENDERERS)


def _browser_steps(
    page: Page,
    name: str,
    url: str,
    plan: Plan,
) -> tuple[list[Step], list[str]]:
    clock = _page_clock(page)
    marks = _drive(page, url, plan, clock, from_navigation=True)
    spans, size = _model_requests(page)
    renderer = cast("str", page.evaluate(WEBGL_RENDERER))
    # The budgets describe a machine with a GPU; the flow itself is still asserted.
    gated = not is_software_renderer(renderer)
    note = "" if gated else f"not gated: software renderer ({renderer})"
    steps: list[Step] = []
    notes = [
        f"{name}: model transfer {union_s(spans):.1f} s, {size / BYTES_PER_MB:.1f} MB "
        "(reported apart from the budgets)",
        f"{name}: WebGL renderer {renderer or 'none'}",
    ]
    if marks.gallery is not None:
        steps.append(
            Step(
                f"{name}: Tier 1 gallery result",
                marks.gallery[1],
                0.0,
                GALLERY_BUDGET_S if gated else None,
                note,
            )
        )
    elif plan.tier1 and plan.render:
        steps.append(Step(f"{name}: Tier 1 gallery result", note="no gallery result", failed=True))
    if marks.clip_end is not None:
        transfer = transfer_in_window_s(spans, marks.clip_start, marks.clip_end)
        steps.append(
            Step(
                f"{name}: Tier 2 clip result",
                marks.clip_end - marks.clip_start,
                transfer,
                CLIP_COMPUTE_BUDGET_S if gated else None,
                note,
            )
        )
    return steps, notes


def _throttled_step(kind: BrowserType, user_data_dir: Path, url: str, plan: Plan) -> Step:
    """Tier 2 from nothing under a fixed 50 Mbps link, against the full budget."""
    only_clip = Plan(plan.drawing, plan.clip, None, tier1=False, tier2=True, render=True)
    context = kind.launch_persistent_context(user_data_dir, headless=True)
    try:
        page = context.new_page()
        session = context.new_cdp_session(page)
        send = cast("Any", session).send
        send("Network.enable")
        send(
            "Network.emulateNetworkConditions",
            {
                "offline": False,
                "latency": LATENCY_MS,
                "downloadThroughput": REFERENCE_BYTES_PER_S,
                "uploadThroughput": UNLIMITED,
            },
        )
        marks = _drive(page, url, only_clip, _page_clock(page), from_navigation=True)
    finally:
        context.close()
    label = "chromium: Tier 2 total at 50 Mbps"
    if marks.clip_end is None:
        return Step(label, note="the clip did not finish", failed=True)
    return Step(label, marks.clip_end, 0.0, CLIP_TOTAL_BUDGET_S)


def run_browsers(
    playwright: Playwright, names: Sequence[str], url: str, plan: Plan, model_urls: Sequence[str]
) -> tuple[list[Step], list[str]]:
    """One fresh-profile pass per browser, plus the throttled Chromium pass."""
    steps: list[Step] = []
    notes: list[str] = []
    for name in names:
        kind = cast("BrowserType", getattr(playwright, name))
        runs = [False, True] if name == "chromium" and plan.tier2 and plan.render else [False]
        for throttled in runs:
            if model_urls:
                size, seconds = _fetch_models(model_urls)
                rate = throughput(size, seconds)
                if rate is not None:
                    notes.append(f"{name}: network {rate / BYTES_PER_MB:.1f} MB/s")
            with tempfile.TemporaryDirectory(prefix=f"cold-start-{name}-") as profile:
                try:
                    if throttled:
                        steps.append(_throttled_step(kind, Path(profile), url, plan))
                        continue
                    # Headless Firefox has no WebGL 2 on a hosted runner; CI runs it under xvfb.
                    headless = not (name == "firefox" and "FIREFOX_HEADED" in os.environ)
                    context = kind.launch_persistent_context(profile, headless=headless)
                    try:
                        found, more = _browser_steps(context.new_page(), name, url, plan)
                    finally:
                        context.close()
                except PlaywrightError as error:
                    steps.append(
                        Step(f"{name}: page flow", note=str(error).splitlines()[0], failed=True)
                    )
                    continue
                steps.extend(found)
                notes.extend(more)
    return steps, notes


class _Server:
    """The app started by uvx, its page address and the lines it printed."""

    def __init__(self, command: list[str], env: dict[str, str]) -> None:
        self.started = time.monotonic()
        self.process = subprocess.Popen(  # noqa: S603
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        if self.process.stdout is not None:
            for line in self.process.stdout:
                self.lines.put(line)

    def url(self) -> str:
        """Wait for the line that gives the address, token included."""
        deadline = time.monotonic() + SERVER_START_TIMEOUT_S
        seen: list[str] = []
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=POLL_S)
            except queue.Empty:
                if self.process.poll() is not None:
                    break
                continue
            seen.append(line)
            if found := RUNNING_LINE.search(line):
                return found.group(1)
        raise RuntimeError("the app did not print its address: " + " ".join(seen[-5:]).strip())

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


class _Api:
    """Health and job requests to the app, with its session token."""

    def __init__(self, url: str) -> None:
        parts = urlsplit(url)
        self.host, self.port = parts.hostname or "127.0.0.1", parts.port or 80
        self.token = parts.fragment.removeprefix("t=")

    def get(self, path: str) -> dict[str, Any]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=HTTP_TIMEOUT_S)
        try:
            connection.request("GET", path, headers={"X-Copypaw-Token": self.token})
            response = connection.getresponse()
            data = response.read()
        finally:
            connection.close()
        if response.status != HTTP_OK:
            raise RuntimeError(f"GET {path} returned {response.status}")
        return cast("dict[str, Any]", json.loads(data))

    def wait_healthy(self) -> None:
        deadline = time.monotonic() + HEALTH_TIMEOUT_S
        while True:
            try:
                self.get("/api/health")
            except (OSError, RuntimeError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(POLL_S)
            else:
                return

    def wait_job(self, job_id: str) -> None:
        while True:
            job = self.get(f"/api/jobs/{job_id}")
            if job["state"] == "failed":
                raise RuntimeError(job["error"]["what"])
            if job["state"] == "done":
                return
            time.sleep(POLL_S)


class _ModelClock:
    """Note when the app's model download ended and how many bytes it fetched."""

    def __init__(self, api: _Api) -> None:
        self.api = api
        self.ready_at: float | None = None
        self.size_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                models = self.api.get("/api/health")["models"]
            except (OSError, RuntimeError):
                return
            self.size_bytes = max(self.size_bytes, int(models["downloaded"]))
            if models["state"] != "downloading":
                self.ready_at = time.monotonic()
                return
            self._stop.wait(POLL_S)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def _local_steps(
    spec: str, plan: Plan, env: dict[str, str], cache: Path, notes: list[str]
) -> list[Step]:
    """Install into the empty caches, start the app, and drive its page."""
    began = time.monotonic()
    subprocess.run(uvx(spec, "--help"), env=env, check=True, capture_output=True)  # noqa: S603
    install_s = time.monotonic() - began
    wheel_bytes = directory_bytes(Path(env["UV_CACHE_DIR"]))
    server = _Server(uvx(spec, "app", "--no-browser", "--port", str(_free_port())), env)
    try:
        url = server.url()
        api = _Api(url)
        api.wait_healthy()
        models = _ModelClock(api)
        try:
            with sync_playwright() as playwright:
                context = playwright.chromium.launch_persistent_context(
                    cache / "profile", headless=True
                )
                try:
                    page = context.new_page()
                    marks = _drive(
                        page,
                        url,
                        plan,
                        time.monotonic,
                        from_navigation=False,
                        track_capture=api.wait_job,
                    )
                finally:
                    context.close()
        finally:
            models.stop()
    finally:
        server.stop()
    ready_s = (time.monotonic() if models.ready_at is None else models.ready_at) - server.started
    rate = throughput(models.size_bytes, ready_s)
    notes.append(
        f"uv fetched {wheel_bytes / BYTES_PER_MB:.1f} MB in {install_s:.1f} s; "
        f"models fetched {models.size_bytes / BYTES_PER_MB:.1f} MB in {ready_s:.1f} s"
        + ("" if rate is None else f" ({rate / BYTES_PER_MB:.1f} MB/s)")
    )
    install = Install(install_s, wheel_bytes, rate, server.started, models.ready_at)
    return local_rows(plan, marks, install)


def local_rows(plan: Plan, marks: Marks, install: Install) -> list[Step]:
    window_end = install.models_ready if install.models_ready is not None else marks.rigged
    model_span = [(install.started, window_end)]

    def skipped(label: str, note: str) -> Step:
        return Step(label, note=note)

    opened_s = install.seconds + (marks.opened - install.started)
    wheel_s = net_of_wheel_s(install.wheel_bytes, install.bytes_per_s)
    rows = [
        Step(PAGE_OPEN, opened_s, wheel_s, PAGE_OPEN_BUDGET_S),
        Step(DRAWING_RIGGED, marks.rigged - marks.rig_start),
    ]
    if not plan.render:
        rows.append(skipped(GALLERY_RESULT, "render skipped"))
    elif marks.gallery is None:
        rows.append(skipped(GALLERY_RESULT, "no gallery motion is packaged"))
    else:
        rows.append(
            Step(GALLERY_RESULT, marks.gallery[1] - marks.gallery[0], 0.0, GALLERY_BUDGET_S)
        )
    if marks.captured is None:
        rows.append(skipped(CLIP_CAPTURE, "the capture was not observed"))
    else:
        span = marks.captured - marks.clip_start
        transfer = transfer_in_window_s(model_span, marks.clip_start, marks.captured)
        rows.append(Step(CLIP_CAPTURE, span, transfer))
    if marks.clip_end is None:
        rows.append(skipped(CLIP_RESULT, "render skipped"))
    else:
        transfer = transfer_in_window_s(model_span, marks.clip_start, marks.clip_end)
        rows.append(
            Step(CLIP_RESULT, marks.clip_end - marks.clip_start, transfer, CLIP_COMPUTE_BUDGET_S)
        )
    return rows


def run_local(
    *, from_pypi: bool, wheel: Path | None, plan: Plan, check_budget: bool
) -> tuple[list[Step], list[str], list[str]]:
    """The local flow, then the packaged self-test on the same caches for comparison."""
    notes: list[str] = []
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="cold-start-") as work:
        root = Path(work)
        env = {
            **os.environ,
            "UV_CACHE_DIR": str(root / "uv-cache"),
            "UV_TOOL_DIR": str(root / "uv-tools"),
            "COPYPAW_CACHE": str(root / "models"),
        }
        for key in ("UV_CACHE_DIR", "UV_TOOL_DIR", "COPYPAW_CACHE"):
            Path(env[key]).mkdir()
        if from_pypi:
            spec = package_spec(from_pypi=True, wheel=None)
        else:
            built = wheel or _build_wheel(root / "wheel")
            spec = package_spec(from_pypi=False, wheel=built)
        rows = _local_steps(spec, plan, env, root, notes)
        command = uvx(spec, "app", "--self-test", *([] if plan.render else ["--no-render"]))
        result = subprocess.run(  # noqa: S603
            command, env=env, capture_output=True, text=True, check=False
        )
        notes.append("self-test output:\n" + result.stdout.rstrip())
        if not labels_match(rows, result.stdout):
            problems.append("the packaged self-test printed different steps")
        failed = "FAILED" in result.stdout if not check_budget else result.returncode != 0
        if failed:
            problems.append(f"the packaged self-test exited {result.returncode}")
    return rows, notes, problems


def _build_wheel(out: Path) -> Path:
    subprocess.run(  # noqa: S603
        ["uv", "build", "--wheel", "--out-dir", str(out)],  # noqa: S607
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return next(out.glob("copypaw-*.whl"))


def _tiers(text: str) -> frozenset[int]:
    chosen = frozenset(int(part) for part in text.split(",") if part)
    if not chosen <= {TIER_1, TIER_2}:
        raise argparse.ArgumentTypeError("tiers are 1 and 2")
    return chosen


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--browser", metavar="URL", help="the published root page to open in each browser"
    )
    parser.add_argument("--local", action="store_true", help="install with uvx and drive the app")
    parser.add_argument("--from-pypi", action="store_true", help="install the released version")
    parser.add_argument("--wheel", type=Path, help="install this wheel instead of building one")
    parser.add_argument("--no-render", action="store_true", help="skip the steps that render")
    parser.add_argument(
        "--no-budget", action="store_true", help="print the times but assert no budget"
    )
    parser.add_argument(
        "--tiers",
        type=_tiers,
        default=SHIPPED_TIERS,
        help="browser tiers to run, as 1,2 (default: the tiers that shipped)",
    )
    parser.add_argument(
        "--browsers", default=",".join(BROWSERS), help="comma-separated browser engines"
    )
    parser.add_argument(
        "--model-url",
        action="append",
        default=[],
        help="a tier 2 model file to fetch for the throughput note; repeat for each file",
    )
    parser.add_argument("--drawing", type=Path, default=DRAWING)
    parser.add_argument("--clip", type=Path, default=CLIP)
    parser.add_argument(
        "--gallery-motion", default=DEFAULT_GALLERY_MOTION, help="gallery motion for --browser"
    )
    return parser


def _check(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    """Reject option combinations that cannot run; return the browser engines to use."""
    if not args.browser and not args.local:
        parser.error("pick --browser URL, --local or both")
    if args.from_pypi and not args.local:
        parser.error("--from-pypi needs --local")
    if TIER_2 in args.tiers and TIER_1 not in args.tiers:
        parser.error("tier 2 needs tier 1")
    names = [name for name in args.browsers.split(",") if name]
    if unknown := [name for name in names if name not in BROWSERS]:
        parser.error(f"unknown browser: {', '.join(unknown)}")
    return names


def _browser_pass(args: argparse.Namespace, names: list[str]) -> tuple[list[Step], list[str]]:
    if not args.tiers:
        return [], ["No browser tier is shipped, so no browser run was made."]
    plan = Plan(
        args.drawing,
        args.clip,
        args.gallery_motion,
        tier1=TIER_1 in args.tiers,
        tier2=TIER_2 in args.tiers,
        render=not args.no_render,
    )
    with sync_playwright() as playwright:
        return run_browsers(playwright, names, args.browser, plan, args.model_url)


def _local_pass(args: argparse.Namespace) -> tuple[list[Step], list[str], list[str]]:
    plan = Plan(args.drawing, args.clip, None, tier1=True, tier2=True, render=not args.no_render)
    try:
        return run_local(
            from_pypi=args.from_pypi,
            wheel=args.wheel,
            plan=plan,
            check_budget=not args.no_budget,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError, PlaywrightError) as error:
        return [Step("Local flow", note=str(error), failed=True)], [], []


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested measurements and return the exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    names = _check(parser, args)
    rows: list[Step] = []
    notes: list[str] = []
    problems: list[str] = []
    if args.browser:
        found, more = _browser_pass(args, names)
        rows.extend(found)
        notes.extend(more)
    if args.local:
        found, more, problems = _local_pass(args)
        rows.extend(found)
        notes.extend(more)
    if rows:
        print(format_table(rows))
    for note in notes:
        print(note)
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    missed = [row for row in rows if row.failed] if args.no_budget else misses(rows)
    for row in missed:
        print(f"{'failed' if row.failed else 'over budget'}: {row.label}", file=sys.stderr)
    return 1 if missed or problems else 0


if __name__ == "__main__":
    sys.exit(main())
