import math
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import urlparse

import pytest
from typer.testing import CliRunner

import copypaw.cli as cli_module
from copypaw.app import selftest
from copypaw.app.selftest import SelfTestReport, Step, format_report, format_table
from copypaw.app.server import Pipeline
from copypaw.errors import CopypawError
from copypaw.models import Progress

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_MOTION = ROOT / "tests/golden/dog_walk.motion.npz"
SAMPLE_CLIP = ROOT / "samples/pets/dog_walk.mp4"
LABELS = ["Page open", "Drawing rigged", "Gallery result", "Clip capture", "Clip result"]


class RefusedError(CopypawError):
    codes = ("app_request",)


class FakeSteps:
    def __init__(self, *, capture_fails: bool = False) -> None:
        self.capture_fails = capture_fails
        self.animated = 0

    def fetch(self, mode: str, progress: Progress | None) -> Mapping[str, Path]:
        del mode
        if progress is not None:
            progress("detector.onnx", 7, 10)
        return {}

    def capture(self, clip: Path, *, out: Path) -> None:
        assert clip.read_bytes() == selftest.CLIP.read_bytes()
        if self.capture_fails:
            raise RefusedError(problem="no animal")
        shutil.copyfile(GOLDEN_MOTION, out / "motion.npz")

    def animate(self, char_dir: Path, motion: Path, *, clip: Path, out: Path, credit: bool) -> None:
        del char_dir, motion, clip, credit
        self.animated += 1
        for name in ("side_by_side.mp4", "side_by_side.gif"):
            (out / name).write_bytes(b"video")

    def pipeline(self) -> Pipeline:
        return Pipeline(capture=self.capture, animate=self.animate, fetch_models=self.fetch)


@pytest.fixture(autouse=True)
def temp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "temp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))


@pytest.fixture
def static(tmp_path: Path) -> Path:
    folder = tmp_path / "static"
    folder.mkdir()
    (folder / "index.html").write_bytes(b"<!doctype html><title>Copypaw</title>")
    return folder


@pytest.fixture
def gallery(tmp_path: Path) -> Path:
    folder = tmp_path / "gallery"
    folder.mkdir()
    shutil.copyfile(GOLDEN_MOTION, folder / "walk.motion.npz")
    shutil.copyfile(SAMPLE_CLIP, folder / "walk.mp4")
    return folder


def test_compute_is_elapsed_without_transfer() -> None:
    step = Step("Clip result", 100.0, 11.2, 78.8)

    assert step.compute_s == pytest.approx(88.8)
    assert step.verdict == "MISS"
    assert Step("Clip result", 90.0, 11.2, 78.8).verdict == "ok"


def test_budget_is_ninety_seconds_less_seventy_megabytes_at_fifty_mbps() -> None:
    assert math.isclose(selftest.CLIP_COMPUTE_BUDGET_S, 78.8)


def test_verdicts_for_skipped_unbudgeted_and_failed_steps() -> None:
    assert Step("a").verdict == "skipped"
    assert Step("a", 5.0).verdict == "-"
    assert Step("a", 5.0, note="boom", failed=True).verdict == "FAILED"
    assert [step.label for step in selftest.misses([Step("a", 9.0, budget_s=1.0), Step("b")])] == [
        "a"
    ]


def test_table_layout() -> None:
    table = format_table(
        [
            Step("Page open", 4.0, 0.0, 30.0),
            Step("Gallery result", note="render skipped"),
            Step("Clip result", 12.0, 2.0, 5.0),
        ]
    )

    assert table.splitlines() == [
        "| Step | Elapsed (s) | Transfer (s) | Compute (s) | Budget (s) | Result |",
        "| --- | --- | --- | --- | --- | --- |",
        "| Page open | 4.0 | 0.0 | 4.0 | 30.0 | ok |",
        "| Gallery result | - | - | - | - | skipped: render skipped |",
        "| Clip result | 12.0 | 2.0 | 10.0 | 5.0 | MISS |",
    ]


def test_labels_are_read_back_from_a_printed_table() -> None:
    steps = [Step("Page open", 1.0), Step("Clip result")]

    assert selftest.parse_labels("intro\n" + format_table(steps) + "\n\nModels: x") == [
        "Page open",
        "Clip result",
    ]


def test_overlap_is_the_shared_span() -> None:
    assert selftest.overlap_s(0.0, 10.0, 4.0, 20.0) == 6.0
    assert selftest.overlap_s(0.0, 3.0, 4.0, 20.0) == 0.0


def test_throughput_needs_bytes_and_time() -> None:
    assert selftest.throughput(10_000_000, 2.0) == 5_000_000
    assert selftest.throughput(0, 2.0) is None
    assert selftest.throughput(10, 0.0) is None


def test_report_names_the_download_and_the_credit() -> None:
    report = SelfTestReport((Step("Page open", 1.0, 0.0, 30.0),), 20_000_000, 4.0)
    credit = (ROOT / "copypaw/selftest/ATTRIBUTION.txt").read_text(encoding="utf-8").splitlines()[0]

    lines = format_report(report).splitlines()

    assert "Models: 20.0 MB in 4.0 s (5.0 MB/s)" in lines
    assert "Within budget" in lines
    assert lines[-1] == f"Credit: {credit}"
    assert "CC BY 2.5" in credit
    assert any(
        urlparse(word.strip(".,:")).hostname == "commons.wikimedia.org" for word in credit.split()
    )


def test_cached_models_report_nothing_downloaded() -> None:
    report = SelfTestReport((), 0, 0.5)

    assert "Models: ready after 0.5 s, nothing downloaded" in format_report(report)


def test_sample_pair_runs_every_step(static: Path, gallery: Path) -> None:
    steps = FakeSteps()

    report = selftest.run_self_test(steps.pipeline(), static_dir=static, gallery_dir=gallery)

    assert [step.label for step in report.steps] == LABELS
    assert [step.verdict for step in report.steps] == ["ok", "-", "ok", "-", "ok"]
    assert steps.animated == 2
    assert report.model_bytes == 7
    assert report.passed


def test_gallery_result_keeps_its_compute_time(
    static: Path, gallery: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def overlapping(_start: float, _end: float, _from: float, _to: float) -> float:
        return 1.0

    monkeypatch.setattr(selftest, "overlap_s", overlapping)

    report = selftest.run_self_test(FakeSteps().pipeline(), static_dir=static, gallery_dir=gallery)

    gallery_step, clip_step = report.steps[2], report.steps[4]
    assert (gallery_step.transfer_s, gallery_step.compute_s) == (0.0, gallery_step.elapsed_s)
    assert clip_step.transfer_s == 1.0


def test_without_a_gallery_motion_that_row_is_skipped(static: Path, tmp_path: Path) -> None:
    report = selftest.run_self_test(
        FakeSteps().pipeline(), static_dir=static, gallery_dir=tmp_path / "none"
    )

    assert report.steps[2].verdict == "skipped"
    assert report.passed


def test_no_render_skips_both_render_rows(static: Path, gallery: Path) -> None:
    steps = FakeSteps()

    report = selftest.run_self_test(
        steps.pipeline(), static_dir=static, gallery_dir=gallery, render=False
    )

    assert [step.label for step in report.steps] == LABELS
    assert [step.verdict for step in report.steps] == ["ok", "-", "skipped", "-", "skipped"]
    assert steps.animated == 0
    assert report.passed


def test_a_budget_miss_fails_the_report(
    static: Path, gallery: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(selftest, "CLIP_COMPUTE_BUDGET_S", -1.0)

    report = selftest.run_self_test(FakeSteps().pipeline(), static_dir=static, gallery_dir=gallery)

    assert report.steps[-1].verdict == "MISS"
    assert not report.passed
    assert "Budget missed" in format_report(report)


def test_a_failed_step_is_named_and_later_steps_are_skipped(static: Path, gallery: Path) -> None:
    report = selftest.run_self_test(
        FakeSteps(capture_fails=True).pipeline(), static_dir=static, gallery_dir=gallery
    )

    assert [step.verdict for step in report.steps] == ["ok", "-", "ok", "FAILED", "skipped"]
    assert not report.passed
    assert report.steps[3].note


def _returning(report: SelfTestReport) -> Callable[..., SelfTestReport]:
    def run(pipeline: Pipeline, *, render: bool) -> SelfTestReport:
        del pipeline, render
        return report

    return run


def test_app_command_exits_with_the_report(monkeypatch: pytest.MonkeyPatch) -> None:
    report = SelfTestReport((Step("Page open", 1.0, 0.0, 30.0),), 0, 0.1)
    monkeypatch.setattr(cli_module, "run_self_test", _returning(report))

    result = CliRunner().invoke(cli_module.app, ["app", "--self-test"])

    assert result.exit_code == 0
    assert "| Page open | 1.0 | 0.0 | 1.0 | 30.0 | ok |" in result.output
    assert "Within budget" in result.output


def test_app_command_exits_one_on_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    report = SelfTestReport((Step("Page open", 99.0, 0.0, 30.0),), 0, 0.1)
    monkeypatch.setattr(cli_module, "run_self_test", _returning(report))

    result = CliRunner().invoke(cli_module.app, ["app", "--self-test"])

    assert result.exit_code == 1
    assert "Budget missed" in result.output


def test_no_render_alone_is_refused() -> None:
    result = CliRunner().invoke(cli_module.app, ["app", "--no-render"])

    assert result.exit_code == 2


def test_packaged_pair_is_small_and_credited() -> None:
    package = ROOT / "copypaw/selftest"

    sizes = [(package / name).stat().st_size for name in ("dragon.png", "dog_walk.mp4")]

    assert all(size > 0 for size in sizes)
    assert sum(sizes) < 2_000_000
    assert "Shin" in (package / "ATTRIBUTION.txt").read_text(encoding="utf-8")
