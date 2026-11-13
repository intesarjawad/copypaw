import argparse
import math
from collections.abc import Callable
from pathlib import Path

import pytest

from copypaw import __version__
from copypaw.app.selftest import Step, format_table
from scripts import cold_start
from scripts.cold_start import Install, Marks, Plan

PLAN = Plan(Path("drawing.png"), Path("clip.mp4"), None, tier1=True, tier2=True, render=True)


def test_the_default_gallery_motion_is_a_folder_name() -> None:
    # Motion ids are folder names, so a hyphen would never match the radio.
    assert cold_start.DEFAULT_GALLERY_MOTION == "dog_walk"


def test_union_counts_overlapping_requests_once() -> None:
    assert cold_start.union_s([(0.0, 4.0), (2.0, 6.0), (10.0, 11.0)]) == 7.0
    assert cold_start.union_s([]) == 0.0


def test_union_is_not_changed_by_a_request_inside_another() -> None:
    assert cold_start.union_s([(1.0, 9.0), (2.0, 3.0)]) == 8.0


def test_transfer_in_window_counts_each_second_once() -> None:
    spans = [(0.0, 4.0), (2.0, 6.0), (10.0, 11.0)]

    assert cold_start.transfer_in_window_s(spans, 3.0, 10.5) == 3.5
    assert cold_start.transfer_in_window_s(spans, 7.0, 9.0) == 0.0


def test_spec_is_the_released_version_or_the_wheel() -> None:
    assert cold_start.package_spec(from_pypi=True, wheel=None) == f"copypaw=={__version__}"
    assert (
        cold_start.package_spec(from_pypi=False, wheel=Path("/w/copypaw.whl")) == "/w/copypaw.whl"
    )
    with pytest.raises(ValueError, match="wheel"):
        cold_start.package_spec(from_pypi=False, wheel=None)


def test_uvx_names_the_copypaw_command() -> None:
    assert cold_start.uvx("copypaw==1", "app", "--self-test") == [
        "uvx",
        "--from",
        "copypaw==1",
        "copypaw",
        "app",
        "--self-test",
    ]


def test_directory_bytes_sums_nested_files(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one").write_bytes(b"x" * 10)
    (tmp_path / "two").write_bytes(b"y" * 5)

    assert cold_start.directory_bytes(tmp_path) == 15


def test_wheel_transfer_is_bytes_over_the_measured_throughput() -> None:
    assert cold_start.net_of_wheel_s(50_000_000, 5_000_000.0) == 10.0
    assert cold_start.net_of_wheel_s(50_000_000, None) == 0.0


def test_labels_match_only_in_the_same_order() -> None:
    rows = [Step("Page open", 1.0), Step("Clip result")]
    printed = format_table(rows)

    assert cold_start.labels_match(rows, "text\n" + printed)
    assert not cold_start.labels_match(rows[::-1], printed)
    assert not cold_start.labels_match(rows[:1], printed)


def testlocal_rows_take_the_wheel_and_the_models_out_of_the_budgeted_rows() -> None:
    marks = Marks(
        opened=5.0,
        rig_start=5.0,
        rigged=6.0,
        gallery=None,
        clip_start=6.0,
        captured=20.0,
        clip_end=40.0,
    )
    install = Install(10.0, 50_000_000, 5_000_000.0, 0.0, 12.0)

    page, drawing, gallery, capture, clip = cold_start.local_rows(PLAN, marks, install)

    assert (page.elapsed_s, page.transfer_s, page.budget_s) == (15.0, 10.0, 30.0)
    assert drawing.elapsed_s == 1.0
    assert gallery.note == "no gallery motion is packaged"
    assert (capture.elapsed_s, capture.transfer_s) == (14.0, 6.0)
    assert (clip.elapsed_s, clip.transfer_s) == (34.0, 6.0)
    assert clip.compute_s == 28.0
    assert clip.budget_s is not None
    assert math.isclose(clip.budget_s, 78.8)


def testlocal_rows_without_render_skip_the_render_rows() -> None:
    marks = Marks(5.0, 5.0, 6.0, None, 6.0, 20.0, None)
    install = Install(10.0, 0, None, 0.0, 12.0)
    plan = Plan(PLAN.drawing, PLAN.clip, None, tier1=True, tier2=True, render=False)

    rows = cold_start.local_rows(plan, marks, install)

    assert [row.verdict for row in rows] == ["ok", "-", "skipped", "-", "skipped"]
    assert rows[0].transfer_s == 0.0


def test_a_browser_run_needs_a_shipped_tier(capsys: pytest.CaptureFixture[str]) -> None:
    code = cold_start.main(["--browser", "http://localhost:4173/try/", "--tiers", ""])

    assert code == 0
    assert "No browser tier is shipped" in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--from-pypi", "--browser", "http://localhost/try/"],
        ["--browser", "http://localhost/try/", "--tiers", "2"],
        ["--browser", "http://localhost/try/", "--tiers", "3"],
        ["--browser", "http://localhost/try/", "--browsers", "edge"],
    ],
)
def test_unusable_options_are_refused(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        cold_start.main(arguments)

    assert stopped.value.code == 2


def _stub_local(
    monkeypatch: pytest.MonkeyPatch, rows: list[Step], problems: list[str] | None = None
) -> None:
    def local_pass(args: argparse.Namespace) -> tuple[list[Step], list[str], list[str]]:
        del args
        return rows, [], problems or []

    monkeypatch.setattr(cold_start, "_local_pass", local_pass)


def test_a_budget_miss_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_local(monkeypatch, [Step("Page open", 40.0, 0.0, 30.0)])

    assert cold_start.main(["--local"]) == 1
    captured = capsys.readouterr()
    assert "| Page open | 40.0 | 0.0 | 40.0 | 30.0 | MISS |" in captured.out
    assert "over budget: Page open" in captured.err


def test_within_budget_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_local(monkeypatch, [Step("Page open", 4.0, 0.0, 30.0)])

    assert cold_start.main(["--local"]) == 0


def test_no_budget_still_fails_a_step_that_did_not_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_local(
        monkeypatch, [Step("Page open", 99.0, 0.0, 30.0), Step("Clip result", 1.0, failed=True)]
    )

    assert cold_start.main(["--local", "--no-budget"]) == 1


def test_no_budget_ignores_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_local(monkeypatch, [Step("Page open", 99.0, 0.0, 30.0)])

    assert cold_start.main(["--local", "--no-budget"]) == 0


def test_a_self_test_mismatch_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_local(monkeypatch, [Step("Page open", 4.0, 0.0, 30.0)], ["different steps"])

    assert cold_start.main(["--local"]) == 1


def test_default_browser_run_enables_the_shipped_gallery_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[frozenset[int]] = []

    def browser_pass(args: argparse.Namespace, names: list[str]) -> tuple[list[Step], list[str]]:
        assert names
        seen.append(args.tiers)
        return [Step("Gallery result", 1.0)], []

    monkeypatch.setattr(cold_start, "_browser_pass", browser_pass)
    assert cold_start.main(["--browser", "http://localhost/"]) == 0
    assert seen == [frozenset({1})]


@pytest.mark.parametrize(
    ("renderer", "gated"),
    [
        ("ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device), SwiftShader driver)", False),
        ("llvmpipe (LLVM 15.0.7, 256 bits)", False),
        ("ANGLE (Apple, ANGLE Metal Renderer: Apple M5, Unspecified Version)", True),
    ],
)
def test_a_software_renderer_is_reported_but_not_gated(
    monkeypatch: pytest.MonkeyPatch, renderer: str, gated: bool
) -> None:
    class StubPage:
        def evaluate(self, _script: str) -> str:
            return renderer

    marks = Marks(1.0, 1.0, 2.0, (0.0, 99.0), 2.0, None, None)

    def no_clock(_page: object) -> Callable[[], float]:
        return lambda: 0.0

    def drive(*_args: object, **_kwargs: object) -> Marks:
        return marks

    def no_models(_page: object) -> tuple[list[cold_start.Interval], int]:
        return [], 0

    monkeypatch.setattr(cold_start, "_page_clock", no_clock)
    monkeypatch.setattr(cold_start, "_drive", drive)
    monkeypatch.setattr(cold_start, "_model_requests", no_models)

    steps, notes = cold_start._browser_steps(StubPage(), "chromium", "http://x/", PLAN)  # type: ignore[arg-type]

    assert [step.verdict for step in steps] == ["MISS" if gated else "-"]
    assert (renderer in steps[0].note) is (not gated)
    assert f"chromium: WebGL renderer {renderer}" in notes
