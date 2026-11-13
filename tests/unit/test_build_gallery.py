import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import cv2
import numpy as np
import pytest
from PIL import Image

from copypaw.animate.render import RenderResult
from copypaw.motion.motion_file import MotionFile
from scripts import build_gallery
from scripts.build_gallery import Budget, Entry

PET = (200, 0, 0)
INK = (0, 0, 200)


def _rgb(count: int, height: int, width: int, colour: tuple[int, int, int]) -> np.ndarray[Any, Any]:
    return np.full((count, height, width, 3), colour, dtype=np.uint8)


def _rgba(count: int, size: int, colour: tuple[int, int, int]) -> np.ndarray[Any, Any]:
    frames = np.full((count, size, size, 4), 255, dtype=np.uint8)
    frames[..., :3] = colour
    return frames


def _compose(**overrides: Any) -> np.ndarray[Any, Any]:
    arguments: dict[str, Any] = {
        "clip_frames": _rgb(3, 8, 8, PET),
        "drawing_frames_rgba": _rgba(3, 16, INK),
        "facing_clip": "right",
        "facing_drawing": "right",
        "contacts": np.zeros((3, 4), dtype=np.bool_),
        "drawing_stance": None,
        "count": 3,
    }
    arguments.update(overrides)
    return build_gallery.compose_clip_frames(**arguments)


def test_letterbox_centres_a_wide_frame_on_the_ground_colour() -> None:
    boxed = build_gallery.letterbox(_rgb(1, 4, 8, PET), 16)

    assert boxed.shape == (1, 16, 16, 3)
    assert boxed[0, 3, 8].tolist() == [255, 255, 255]
    assert boxed[0, 4, 8].tolist() == list(PET)
    assert boxed[0, 11, 8].tolist() == list(PET)
    assert boxed[0, 12, 8].tolist() == [255, 255, 255]


def test_a_pair_is_512_wide_with_a_256_square_picture_each() -> None:
    frames = _compose()

    assert frames.shape == (3, 256 + 16, 512, 3)
    assert frames[0, 128, 128].tolist() == list(PET)
    assert frames[0, 128, 256 + 128].tolist() == list(INK)


def test_the_drawing_paws_add_a_second_group_of_strip_rows() -> None:
    frames = _compose(drawing_stance=np.zeros((3, 4), dtype=np.bool_))

    assert frames.shape == (3, 256 + 32, 512, 3)


def test_a_planted_paw_marks_its_strip_row_across_its_frame_column() -> None:
    contacts = np.zeros((3, 4), dtype=np.bool_)
    contacts[1, 0] = True

    frames = _compose(contacts=contacts)

    dark = frames[0, 256 + 1, 512 // 3 + 20].tolist()
    light = frames[0, 256 + 1, 20].tolist()
    assert dark == [64, 64, 64]
    assert light != dark


def test_a_short_pair_loops_from_its_start() -> None:
    clip = np.concatenate((_rgb(1, 8, 8, PET), _rgb(1, 8, 8, (0, 200, 0))))

    frames = _compose(clip_frames=clip, drawing_frames_rgba=_rgba(5, 16, INK), count=4)

    assert [frame[128, 128].tolist() for frame in frames] == [
        list(PET),
        [0, 200, 0],
        list(PET),
        [0, 200, 0],
    ]


def test_a_drawing_that_faces_away_is_flipped_to_face_the_pet() -> None:
    drawing = _rgba(3, 16, INK)
    drawing[:, :, :8, :3] = (0, 200, 0)

    frames = _compose(drawing_frames_rgba=drawing, facing_drawing="left")

    assert frames[0, 128, 256 + 16].tolist() == list(INK)
    assert frames[0, 128, 256 + 240].tolist() == [0, 200, 0]


def test_a_pair_with_no_frames_is_refused() -> None:
    with pytest.raises(ValueError, match="must each hold a frame"):
        _compose(clip_frames=_rgb(0, 8, 8, PET))


def _noise(count: int = 8) -> np.ndarray[Any, Any]:
    return np.random.default_rng(7).integers(0, 256, (count, 64, 96, 3), dtype=np.uint8)


def _stream(path: Path) -> tuple[int, int, float]:
    video = cv2.VideoCapture(str(path))
    try:
        return (
            int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            video.get(cv2.CAP_PROP_FRAME_COUNT),
        )
    finally:
        video.release()


def test_an_encoded_clip_is_silent_h264_without_tags(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"

    size = build_gallery.encode(_noise(), 15.0, path, 30)

    data = path.read_bytes()
    assert size == len(data)
    assert b"avc1" in data
    for absent in (b"soun", b"x264", b"Lavc", b"Lavf", b"\xa9too"):
        assert absent not in data
    assert _stream(path) == (96, 64, 8)


def test_an_encoded_clip_keeps_even_dimensions_that_are_not_multiples_of_sixteen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clip.mp4"

    build_gallery.encode(_noise()[:, :18, :40], 15.0, path, 30)

    assert _stream(path) == (40, 18, 8)


def test_the_first_crf_that_fits_is_used(tmp_path: Path) -> None:
    crf, size = build_gallery.encode_under_budget(_noise(), 15.0, tmp_path / "a.mp4", 10_000_000)

    assert crf == 23
    assert size == (tmp_path / "a.mp4").stat().st_size


def test_the_crf_rises_until_the_clip_is_under_the_limit(tmp_path: Path) -> None:
    frames = _noise()
    limit = build_gallery.encode(frames, 15.0, tmp_path / "probe.mp4", 23)

    crf, size = build_gallery.encode_under_budget(frames, 15.0, tmp_path / "a.mp4", limit)

    assert crf > 23
    assert size < limit
    assert (tmp_path / "a.mp4").stat().st_size == size


def test_a_clip_that_cannot_fit_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"a\.mp4 stays over 10 bytes at CRF 51"):
        build_gallery.encode_under_budget(_noise(), 15.0, tmp_path / "a.mp4", 10)


def test_a_poster_is_a_jpeg_of_the_frame(tmp_path: Path) -> None:
    path = tmp_path / "poster.jpg"

    build_gallery.write_poster(_rgb(1, 8, 12, PET)[0], path)

    with Image.open(path) as poster:
        assert poster.format == "JPEG"
        assert poster.size == (12, 8)


def test_a_tag_build_records_the_tag() -> None:
    env = {"GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v0.1.0", "GITHUB_SHA": "abc123"}

    assert build_gallery.source_ref(env) == "v0.1.0"


def test_a_branch_build_records_the_commit() -> None:
    env = {"GITHUB_REF_TYPE": "branch", "GITHUB_REF_NAME": "main", "GITHUB_SHA": "abc123"}

    assert build_gallery.source_ref(env) == "abc123"


def test_a_local_build_asks_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(command: list[str], **_: Any) -> SimpleNamespace:
        assert command == ["git", "rev-parse", "HEAD"]
        return SimpleNamespace(stdout="deadbeef\n")

    monkeypatch.setattr(build_gallery.subprocess, "run", run)

    assert build_gallery.source_ref({}) == "deadbeef"


def _entry(drawing: str, motion: str) -> Entry:
    return Entry(
        drawing=drawing,
        motion=motion,
        animal="dog",
        gait="walk",
        timing="timing match: strong",
        crf=23,
        file=f"{drawing}__{motion}.mp4",
        poster=f"{drawing}__{motion}.jpg",
        bytes=1000,
    )


def test_the_manifest_names_its_source_and_the_first_clip_as_hero() -> None:
    entries = [_entry("fox", "dog_walk"), _entry("dragon", "dog_walk"), _entry("fox", "cat_trot")]

    manifest = build_gallery.build_manifest("v0.1.0", entries)

    assert manifest["built_from"] == "v0.1.0"
    assert manifest["hero"] == {"file": "hero.mp4", "poster": "hero.jpg"}
    assert manifest["drawings"] == ["dragon", "fox"]
    assert manifest["motions"] == ["cat_trot", "dog_walk"]
    clips = cast("list[dict[str, object]]", manifest["clips"])
    assert clips[0] == {
        "drawing": "fox",
        "motion": "dog_walk",
        "animal": "dog",
        "gait": "walk",
        "timing": "timing match: strong",
        "crf": 23,
        "file": "fox__dog_walk.mp4",
        "poster": "fox__dog_walk.jpg",
        "bytes": 1000,
    }
    assert len(clips) == 3


def test_a_manifest_needs_a_clip() -> None:
    with pytest.raises(ValueError, match="at least one clip"):
        build_gallery.build_manifest("v0.1.0", [])


LIMITS = Budget(outside_clips=100, clip=50, first_view=60, site=400)
MANIFEST_TEXT = json.dumps({"hero": {"file": "hero.mp4", "poster": "hero.jpg"}})
FILES_BYTES = 10 + 10 + 10 + 20 + 5 + 30


def _site(tmp_path: Path, **sizes: int) -> Path:
    """Lay out a site; sizes override the byte count of the named file."""
    site = tmp_path / "site"
    clips = site / "clips"
    clips.mkdir(parents=True)
    defaults = {
        "index": 10,
        "style": 10,
        "hero": 20,
        "poster": 5,
        "extra": 10,
        "other": 30,
    }
    defaults.update(sizes)
    (site / "index.html").write_bytes(bytes(defaults["index"]))
    (site / "style.css").write_bytes(bytes(defaults["style"]))
    (site / "docs").mkdir()
    (site / "docs" / "extra.txt").write_bytes(bytes(defaults["extra"]))
    (clips / "hero.mp4").write_bytes(bytes(defaults["hero"]))
    (clips / "hero.jpg").write_bytes(bytes(defaults["poster"]))
    (clips / "other.mp4").write_bytes(bytes(defaults["other"]))
    (clips / "manifest.json").write_text(MANIFEST_TEXT, encoding="utf-8")
    return site


def test_a_site_inside_every_limit_has_no_problems(tmp_path: Path) -> None:
    assert build_gallery.budget_problems(_site(tmp_path), LIMITS) == []


def test_a_site_needs_a_manifest(tmp_path: Path) -> None:
    problems = build_gallery.budget_problems(tmp_path, LIMITS)

    assert len(problems) == 1
    assert "missing" in problems[0]
    assert "manifest.json" in problems[0]


def test_the_shell_counts_everything_outside_clips(tmp_path: Path) -> None:
    site = _site(tmp_path, extra=90)

    assert build_gallery.budget_problems(site, LIMITS) == ["shell is 110 bytes, limit 100"]


def test_a_clip_at_the_limit_is_over_it(tmp_path: Path) -> None:
    site = _site(tmp_path, other=50)

    assert build_gallery.budget_problems(site, LIMITS) == ["other.mp4 is 50 bytes, limit 50"]


def test_the_first_view_is_the_page_styles_hero_and_poster(tmp_path: Path) -> None:
    site = _site(tmp_path, index=30, extra=0)

    assert build_gallery.budget_problems(site, LIMITS) == [
        "first view is 65 bytes, limit 60",
    ]


def test_a_font_counts_toward_the_first_view_wherever_it_sits(tmp_path: Path) -> None:
    site = _site(tmp_path)
    (site / "docs" / "type.woff2").write_bytes(bytes(30))

    assert build_gallery.budget_problems(site, LIMITS) == ["first view is 75 bytes, limit 60"]


def test_a_missing_hero_file_is_reported(tmp_path: Path) -> None:
    site = _site(tmp_path)
    (site / "clips" / "hero.jpg").unlink()

    assert build_gallery.budget_problems(site, LIMITS) == [
        "first view needs hero.jpg, which is missing"
    ]


def test_the_whole_site_has_a_limit(tmp_path: Path) -> None:
    site = _site(tmp_path)
    total = FILES_BYTES + len(MANIFEST_TEXT)

    assert build_gallery.budget_problems(site, Budget(100, 50, 60, total)) == [
        f"site is {total} bytes, limit {total}"
    ]


def _motion_folder(motions: Path, name: str) -> None:
    folder = motions / name
    folder.mkdir(parents=True)
    MotionFile(
        kp=np.zeros((2, 17, 2), np.float32),
        scores=np.ones((2, 17), np.float32),
        fps=15.0,
        facing="right",
        contacts=np.zeros((2, 4), dtype=np.bool_),
        ground_line_px=np.zeros(2),
        travel_root_x=np.zeros(2, np.float32),
        source="clip.mp4",
    ).save(folder / "motion.npz")
    (folder / "clip.mp4").touch()


def _drawings(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).mkdir(parents=True)
        (directory / name / "char_cfg.yaml").touch()


def _fake_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    def render(char_dir: Path, motion: MotionFile, mode: str, out_dir: Path) -> RenderResult:
        del char_dir, motion, out_dir
        assert mode == "treadmill"
        return RenderResult(
            frames_rgba=_rgba(2, 16, INK),
            fps=15.0,
            joints_per_frame=np.zeros((2, 1, 2), np.float32),
        )

    def clip_frames(clip: Path) -> np.ndarray[Any, Any]:
        assert clip.name == "clip.mp4"
        return _rgb(2, 8, 8, PET)

    monkeypatch.setattr(build_gallery, "render_character", render)
    monkeypatch.setattr(build_gallery, "load_clip_frames", clip_frames)


def test_every_drawing_is_rendered_with_every_motion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    motions = tmp_path / "motions"
    _motion_folder(motions, "dog_walk")
    _motion_folder(motions, "cat_trot")
    drawings = tmp_path / "drawings"
    _drawings(drawings, "dragon", "fox", "no_rig_here")
    (drawings / "no_rig_here" / "char_cfg.yaml").unlink()
    site = tmp_path / "site"
    (site / "clips").mkdir(parents=True)
    (site / "clips" / "stale.mp4").touch()
    _fake_inputs(monkeypatch)
    monkeypatch.setattr(
        build_gallery,
        "load_calibration",
        lambda: (_ for _ in ()).throw(build_gallery.CalibrationError("none")),
    )

    manifest = build_gallery.build_site(motions, drawings, site, "v0.1.0")

    clips = site / "clips"
    assert sorted(path.name for path in clips.glob("*.mp4")) == [
        "dragon__cat_trot.mp4",
        "dragon__dog_walk.mp4",
        "fox__cat_trot.mp4",
        "fox__dog_walk.mp4",
        "hero.mp4",
    ]
    assert sorted(path.name for path in clips.glob("*.jpg")) == [
        "dragon__cat_trot.jpg",
        "dragon__dog_walk.jpg",
        "fox__cat_trot.jpg",
        "fox__dog_walk.jpg",
        "hero.jpg",
    ]
    assert (clips / "hero.mp4").read_bytes() == (clips / "dragon__cat_trot.mp4").read_bytes()
    assert json.loads((clips / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert manifest["built_from"] == "v0.1.0"
    assert manifest["drawings"] == ["dragon", "fox"]
    first = cast("list[dict[str, Any]]", manifest["clips"])[0]
    assert first["animal"] == "cat"
    assert first["timing"] == "timing match: not measured"
    assert first["bytes"] == (clips / first["file"]).stat().st_size
    assert _stream(clips / "fox__dog_walk.mp4") == (512, 256 + 16, 60)


def test_a_motion_without_its_clip_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    motions = tmp_path / "motions"
    _motion_folder(motions, "dog_walk")
    (motions / "dog_walk" / "clip.mp4").unlink()
    _drawings(tmp_path / "drawings", "dragon")
    _fake_inputs(monkeypatch)

    with pytest.raises(FileNotFoundError, match=r"Missing clip\.mp4 for motion dog_walk"):
        build_gallery.build_site(motions, tmp_path / "drawings", tmp_path / "site", "v0.1.0")
