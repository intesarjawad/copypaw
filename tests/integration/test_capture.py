import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from copypaw.cli import app
from copypaw.models import MODELS, REGISTRY, Mode
from copypaw.motion.motion_file import MotionFile

pytestmark = pytest.mark.slow
_SAMPLE = Path(__file__).resolve().parents[2] / "samples/pets/dog_walk.mp4"


def test_capture_dog_walk_writes_loadable_outputs(tmp_path: Path) -> None:
    if not _SAMPLE.is_file():
        pytest.skip("Sample clip is absent: samples/pets/dog_walk.mp4")
    cache = Path(os.environ.get("COPYPAW_CACHE", "~/.cache/copypaw/models")).expanduser()
    for key in MODELS[Mode.BALANCED]:
        if not (cache / REGISTRY[key].filename).is_file():
            pytest.skip(f"Cached model is absent: {key}; run copypaw models fetch")

    result = CliRunner().invoke(app, ["capture", str(_SAMPLE), "-o", str(tmp_path)])

    assert result.exit_code == 0, result.output
    motion = MotionFile.load(tmp_path / "motion.npz")
    assert motion.source == "dog_walk.mp4"
    assert motion.kp.shape[1:] == (17, 2)
    assert len(motion.kp) == len(motion.contacts) == len(motion.travel_root_x)
    bvh = (tmp_path / "motion.bvh").read_text(encoding="utf-8")
    assert f"Frames: {len(motion.kp)}" in bvh
    quality = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))
    assert quality["errors"] == []
