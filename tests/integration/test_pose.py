import os
from pathlib import Path

import numpy as np
import pytest

from copypaw.capture.detect import YOLOX, track_animal
from copypaw.capture.pose import estimate_keypoints
from copypaw.capture.video import read_frames
from copypaw.models import MODELS, REGISTRY, Mode

pytestmark = pytest.mark.slow
_SAMPLE = Path(__file__).resolve().parents[2] / "samples/pets/dog_walk.mp4"
_PAW_INDICES = (7, 10, 13, 16)


@pytest.mark.parametrize("flip_test", [False, True])
def test_dog_walk_detection_and_paw_confidence(flip_test: bool) -> None:
    if not _SAMPLE.is_file():
        pytest.skip("Sample clip is absent: samples/pets/dog_walk.mp4")
    cache = Path(os.environ.get("COPYPAW_CACHE", "~/.cache/copypaw/models")).expanduser()
    models = {key: cache / REGISTRY[key].filename for key in MODELS[Mode.BALANCED]}
    for path in models.values():
        if not path.is_file():
            pytest.skip(
                f"Cached model is absent: {path.name}; run copypaw models fetch --mode balanced"
            )

    frames = read_frames(_SAMPLE)
    track = track_animal(frames, YOLOX(str(models["yolox_m"])))
    assert np.mean(track.present) >= 0.9

    result = estimate_keypoints(frames, track, Mode.BALANCED, flip_test, models)
    assert result.kp.shape == (len(frames.rgb), 17, 2)
    assert result.scores.shape == (len(frames.rgb), 17)
    available = np.isfinite(track.boxes).all(axis=1)
    assert np.isfinite(result.kp[available]).all()
    assert np.isfinite(result.scores).all()
    assert np.mean(result.scores[:, _PAW_INDICES]) > 0.3
