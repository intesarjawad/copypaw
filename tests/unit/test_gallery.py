import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from scripts.build_motions import (
    BVH,
    CLIP,
    CONTACTS_AUTO,
    EVALUATION_SET,
    NPZ,
    THUMB,
    read_evaluation_set,
)
from scripts.check_commons_licences import (
    Row,
    commons_title,
    is_open_licence,
    is_owner_release,
    parse_attribution,
)
from scripts.label_contacts import MERGED_NAME, label_files

MOTIONS = Path(__file__).resolve().parents[2] / "motions"
ATTRIBUTION = "ATTRIBUTION.md"
MAX_CLIP_BYTES = 1_500_000
MAX_CLIP_S = 6.0
CLIP_HEIGHT_PX = 512
LABELLERS = 2


def clip_problems(clip: Path, max_bytes: int = MAX_CLIP_BYTES) -> list[str]:
    """Check what the clip's size, picture and tracks show; embedded tags are not inspected."""
    if not clip.is_file():
        return [f"missing {CLIP}"]
    data = clip.read_bytes()
    problems: list[str] = []
    if len(data) > max_bytes:
        problems.append(f"{CLIP} is {len(data)} bytes, over {max_bytes}")
    if b"avc1" not in data:
        problems.append(f"{CLIP} is not H.264")
    if b"soun" in data:
        problems.append(f"{CLIP} has an audio track")
    video = cv2.VideoCapture(str(clip))
    try:
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)
    finally:
        video.release()
    if height != CLIP_HEIGHT_PX:
        problems.append(f"{CLIP} is {height} px high, not {CLIP_HEIGHT_PX}")
    if fps <= 0 or frames / fps > MAX_CLIP_S:
        problems.append(f"{CLIP} is longer than {MAX_CLIP_S:g} s")
    return problems


def row_problems(row: Row) -> list[str]:
    """Check one attribution row's licence, source and trim."""
    problems: list[str] = []
    from_commons = commons_title(row.source) is not None
    if not (is_open_licence(row.licence) or (is_owner_release(row.licence) and not from_commons)):
        problems.append(f"licence {row.licence!r} is not CC0, CC BY or an owner release")
    if not row.source or not row.author:
        problems.append("source and author are required")
    if not 0 <= row.trim_start_s < row.trim_end_s:
        problems.append("trim must satisfy 0 <= start < end")
    elif row.trim_end_s - row.trim_start_s > MAX_CLIP_S:
        problems.append(f"trim is longer than {MAX_CLIP_S:g} s")
    return problems


def gallery_problems(motions: Path, extra: Path | None = None) -> list[str]:
    """List everything wrong with a motions directory; empty means every folder is complete."""
    table = motions / ATTRIBUTION
    rows = parse_attribution(table.read_text(encoding="utf-8")) if table.is_file() else []
    launch = read_evaluation_set(motions / EVALUATION_SET)
    folders = sorted(path for path in motions.glob("*") if path.is_dir())
    names = {folder.name for folder in folders}
    problems: list[str] = []
    if not table.is_file():
        problems.append(f"missing {ATTRIBUTION}")
    problems.extend(
        f"{name}: in {EVALUATION_SET} but has no folder" for name in sorted(launch - names)
    )
    problems.extend(
        f"{row.motion}: attribution row without a folder" for row in rows if row.motion not in names
    )
    for folder in folders:
        found = [row for row in rows if row.motion == folder.name]
        problems.extend(f"{folder.name}: {problem}" for problem in clip_problems(folder / CLIP))
        if len(found) != 1:
            problems.append(f"{folder.name}: needs exactly one attribution row, found {len(found)}")
        problems.extend(
            f"{folder.name}: {problem}" for row in found[:1] for problem in row_problems(row)
        )
        if folder.name in launch:
            required = [NPZ, BVH, THUMB, MERGED_NAME]
            problems.extend(
                f"{folder.name}: missing {name}"
                for name in required
                if not (folder / name).is_file()
            )
            if len(label_files(folder)) != LABELLERS:
                problems.append(f"{folder.name}: needs {LABELLERS} per-labeller contact files")
        elif extra is not None:
            problems.extend(
                f"{folder.name}: missing generated {name}"
                for name in (NPZ, BVH, THUMB, CONTACTS_AUTO)
                if not (extra / folder.name / name).is_file()
            )
    return problems


def _write_clip(
    path: Path, *, height: int = CLIP_HEIGHT_PX, frames: int = 3, fps: int = 1, audio: bool = False
) -> None:
    ffmpeg: Any = imageio_ffmpeg
    silent = path.with_suffix(".silent.mp4") if audio else path
    with closing(
        ffmpeg.write_frames(
            str(silent),
            (32, height),
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=None,
            ffmpeg_log_level="error",
        )
    ) as writer:
        writer.send(None)
        for index in range(frames):
            writer.send(np.full((height, 32, 3), 40 * index, dtype=np.uint8).tobytes())
    if audio:
        subprocess.run(  # noqa: S603
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-y",
                "-i",
                str(silent),
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=8000:cl=mono",
                "-shortest",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                str(path),
            ],
            check=True,
        )


def _table(*rows: str) -> str:
    header = (
        "| Motion | Source | Licence | Author | Trim start (s) | Trim end (s) | Pet |\n"
        "| --- | --- | --- | --- | ---: | ---: | --- |\n"
    )
    return header + "".join(row + "\n" for row in rows)


COMMONS_ROW = (
    "| dog_walk | https://commons.wikimedia.org/wiki/File:Dog.webm | CC BY 4.0 | A | 0 | 5 | |"
)


def _community(motions: Path, name: str = "dog_walk") -> Path:
    folder = motions / name
    folder.mkdir(parents=True)
    _write_clip(folder / CLIP)
    return folder


def _directory(tmp_path: Path, *rows: str, launch: str = "") -> Path:
    motions = tmp_path / "motions"
    motions.mkdir()
    (motions / ATTRIBUTION).write_text(_table(*rows), encoding="utf-8")
    (motions / EVALUATION_SET).write_text(launch, encoding="utf-8")
    return motions


def test_committed_gallery_is_complete(request: pytest.FixtureRequest) -> None:
    extra = request.config.getoption("--motions-extra")

    assert gallery_problems(MOTIONS, extra) == []


def test_a_community_folder_needs_only_its_clip_and_row(tmp_path: Path) -> None:
    motions = _directory(tmp_path, COMMONS_ROW)
    _community(motions)

    assert gallery_problems(motions) == []


def test_extra_directory_requires_the_generated_files(tmp_path: Path) -> None:
    motions = _directory(tmp_path, COMMONS_ROW)
    _community(motions)
    extra = tmp_path / "built"
    (extra / "dog_walk").mkdir(parents=True)
    for name in (NPZ, BVH, THUMB):
        (extra / "dog_walk" / name).touch()

    assert gallery_problems(motions, extra) == [f"dog_walk: missing generated {CONTACTS_AUTO}"]

    (extra / "dog_walk" / CONTACTS_AUTO).touch()
    assert gallery_problems(motions, extra) == []


def test_a_launch_folder_needs_every_file_and_two_labellers(tmp_path: Path) -> None:
    motions = _directory(tmp_path, COMMONS_ROW, launch="dog_walk\n")
    folder = _community(motions)

    assert gallery_problems(motions) == [
        "dog_walk: missing motion.npz",
        "dog_walk: missing motion.bvh",
        "dog_walk: missing thumb.png",
        "dog_walk: missing contacts_reference.json",
        "dog_walk: needs 2 per-labeller contact files",
    ]

    for name in (NPZ, BVH, THUMB, MERGED_NAME, "contacts_reference.a.json"):
        (folder / name).touch()
    assert gallery_problems(motions) == ["dog_walk: needs 2 per-labeller contact files"]

    (folder / "contacts_reference.b.json").touch()
    assert gallery_problems(motions) == []


def test_a_folder_without_a_row_is_reported(tmp_path: Path) -> None:
    motions = _directory(tmp_path)
    _community(motions)

    assert gallery_problems(motions) == ["dog_walk: needs exactly one attribution row, found 0"]


def test_a_row_without_a_folder_is_reported(tmp_path: Path) -> None:
    motions = _directory(tmp_path, COMMONS_ROW)

    assert gallery_problems(motions) == ["dog_walk: attribution row without a folder"]


def test_a_launch_name_without_a_folder_is_reported(tmp_path: Path) -> None:
    motions = _directory(tmp_path, launch="dog_walk\n")

    assert gallery_problems(motions) == ["dog_walk: in evaluation_set.txt but has no folder"]


def test_a_folder_without_a_clip_is_reported(tmp_path: Path) -> None:
    motions = _directory(tmp_path, COMMONS_ROW)
    (motions / "dog_walk").mkdir()

    assert gallery_problems(motions) == ["dog_walk: missing clip.mp4"]


@pytest.mark.parametrize(
    "licence",
    ["CC BY-SA 4.0", "CC BY-NC 4.0", "all rights reserved", "owner release"],
)
def test_only_open_licences_and_owner_releases_are_accepted(licence: str) -> None:
    row = Row("dog_walk", "https://example.org/a", licence, "A", 0.0, 5.0, "")

    assert row_problems(row) == [f"licence {licence!r} is not CC0, CC BY or an owner release"]


def test_owner_release_and_open_licences_are_accepted() -> None:
    owner = Row("cat_trot", "private submission", "owner release CR-2", "B", 0.0, 5.0, "Miso")
    open_licence = Row("dog_walk", "https://example.org/a", "CC0", "A", 0.0, 5.0, "")

    assert row_problems(owner) == []
    assert row_problems(open_licence) == []


def test_a_commons_source_cannot_claim_an_owner_release() -> None:
    row = Row(
        "dog_walk",
        "https://commons.wikimedia.org/wiki/File:Dog.webm",
        "owner release CR-2",
        "A",
        0.0,
        5.0,
        "",
    )

    assert len(row_problems(row)) == 1


@pytest.mark.parametrize(("start", "end"), [(-1.0, 3.0), (3.0, 3.0), (4.0, 2.0), (0.0, 6.5)])
def test_trim_must_be_a_forward_span_of_at_most_six_seconds(start: float, end: float) -> None:
    row = Row("dog_walk", "https://example.org/a", "CC0", "A", start, end, "")

    assert len(row_problems(row)) == 1


def test_a_six_second_trim_is_allowed() -> None:
    assert row_problems(Row("dog_walk", "https://example.org/a", "CC0", "A", 2.0, 8.0, "")) == []


def test_a_conforming_clip_has_no_problems(tmp_path: Path) -> None:
    clip = tmp_path / CLIP
    _write_clip(clip, frames=6)

    assert clip_problems(clip) == []


def test_a_clip_taller_than_the_limit_is_reported(tmp_path: Path) -> None:
    clip = tmp_path / CLIP
    _write_clip(clip, height=256)

    assert clip_problems(clip) == ["clip.mp4 is 256 px high, not 512"]


def test_a_clip_longer_than_six_seconds_is_reported(tmp_path: Path) -> None:
    clip = tmp_path / CLIP
    _write_clip(clip, frames=7)

    assert clip_problems(clip) == ["clip.mp4 is longer than 6 s"]


def test_a_clip_with_audio_is_reported(tmp_path: Path) -> None:
    clip = tmp_path / CLIP
    _write_clip(clip, audio=True)

    assert clip_problems(clip) == ["clip.mp4 has an audio track"]


def test_an_oversized_clip_is_reported(tmp_path: Path) -> None:
    clip = tmp_path / CLIP
    _write_clip(clip)
    clip.write_bytes(clip.read_bytes() + bytes(MAX_CLIP_BYTES))

    [problem] = clip_problems(clip)
    assert problem.endswith("over 1500000")


def test_a_file_that_is_not_h264_is_reported(tmp_path: Path) -> None:
    clip = tmp_path / CLIP
    clip.write_bytes(b"not a video")

    assert "clip.mp4 is not H.264" in clip_problems(clip)
