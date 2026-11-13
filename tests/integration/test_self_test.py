import os
from pathlib import Path

import pytest

from copypaw.app.selftest import format_report, run_self_test
from copypaw.app.server import Pipeline
from copypaw.cli import animate, capture
from copypaw.models import MODELS, REGISTRY, Mode

pytestmark = [pytest.mark.slow, pytest.mark.render]


def test_packaged_pair_runs_through_the_api(tmp_path: Path) -> None:
    cache = Path(os.environ.get("COPYPAW_CACHE", "~/.cache/copypaw/models")).expanduser()
    for key in MODELS[Mode.BALANCED]:
        if not (cache / REGISTRY[key].filename).is_file():
            pytest.skip(f"Cached model is absent: {key}; run copypaw models fetch")
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>Copypaw</title>", encoding="utf-8")

    report = run_self_test(Pipeline(capture=capture, animate=animate), static_dir=static)

    printed = format_report(report)
    assert [step.verdict for step in report.steps if step.verdict == "FAILED"] == [], printed
    clip = report.steps[-1]
    assert clip.label == "Clip result"
    assert clip.elapsed_s is not None
    assert "Credit: dog_walk.mp4" in printed
