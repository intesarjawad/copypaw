import csv
import re
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"})
VIDEO_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".ogv"})
APPLE_LOCATION_KEY = "com.apple.quicktime.location.iso6709"
ID_COLUMNS = ("clip_licence_or_consent_id", "artwork_release_id")
ATTRIBUTION_FILES = (
    "samples/ATTRIBUTION.md",
    "motions/ATTRIBUTION.md",
    "docs/site/clips/manifest.json",
)
BUILT_CLIPS = "docs/site/clips"


def image_exif_tags(path: Path) -> int:
    """Count the EXIF tags an image carries."""
    with Image.open(path) as image:
        return len(image.getexif())


def video_location_keys(path: Path) -> list[str]:
    """Name every location tag in a video's container metadata."""
    result = subprocess.run(  # noqa: S603
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "ffmetadata", "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    keys = [line.partition("=")[0] for line in result.stdout.splitlines() if "=" in line]
    return [
        key
        for key in keys
        if key.lower() in {"location", APPLE_LOCATION_KEY} or key.lower().startswith("location-")
    ]


def evaluation_only_ids(field_test_csv: Path) -> set[str]:
    """Collect the consent IDs of rows that must never be published."""
    with field_test_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row[column].strip()
        for row in rows
        if row["scope"].strip() == "evaluation" or row["person_in_frame"].strip() == "yes"
        for column in ID_COLUMNS
        if row[column].strip()
    }


def ids_named_in(text: str, ids: set[str]) -> set[str]:
    """Return the IDs that appear in text as whole tokens."""
    return {
        found
        for found in ids
        if re.search(rf"(?<![\w-]){re.escape(found)}(?![\w-])", text) is not None
    }


def leaked_ids(root: Path, files: tuple[str, ...], ids: set[str]) -> dict[str, set[str]]:
    """Map each credit file under root to the evaluation-only IDs it names."""
    leaks: dict[str, set[str]] = {}
    for name in files:
        path = root / name
        if path.is_file():
            named = ids_named_in(path.read_text(encoding="utf-8"), ids)
            if named:
                leaks[name] = named
    return leaks


def asset_paths() -> list[Path]:
    """List the committed files, then any clips the gallery build wrote."""
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    listed = subprocess.run(  # noqa: S603
        [git, "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    paths = [ROOT / name for name in listed.split("\0") if name]
    built = ROOT / BUILT_CLIPS
    if built.is_dir():
        paths.extend(sorted(built.rglob("*")))
    return [path for path in paths if path.is_file()]


def test_committed_images_carry_no_exif() -> None:
    images = [path for path in asset_paths() if path.suffix.lower() in IMAGE_SUFFIXES]

    assert images
    assert {path.relative_to(ROOT).as_posix(): image_exif_tags(path) for path in images} == {
        path.relative_to(ROOT).as_posix(): 0 for path in images
    }


def test_committed_videos_carry_no_location_tag() -> None:
    videos = [path for path in asset_paths() if path.suffix.lower() in VIDEO_SUFFIXES]

    assert videos
    assert {path.relative_to(ROOT).as_posix(): video_location_keys(path) for path in videos} == {
        path.relative_to(ROOT).as_posix(): [] for path in videos
    }


def test_no_evaluation_only_asset_is_committed_or_built_into_the_site() -> None:
    ids = evaluation_only_ids(ROOT / "docs/field_test.csv")

    assert leaked_ids(ROOT, ATTRIBUTION_FILES, ids) == {}


def test_image_with_exif_is_detected(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[0x010E] = "kept"
    tagged = tmp_path / "tagged.jpg"
    bare = tmp_path / "bare.jpg"
    Image.new("RGB", (4, 4)).save(tagged, exif=exif)
    Image.new("RGB", (4, 4)).save(bare)

    assert image_exif_tags(tagged) == 1
    assert image_exif_tags(bare) == 0


def _make_video(path: Path, *tags: str) -> None:
    arguments = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:d=0.2:r=10",
        "-pix_fmt",
        "yuv420p",
    ]
    for tag in tags:
        arguments.extend(["-metadata", tag])
    arguments.extend(["-movflags", "use_metadata_tags", str(path)])
    subprocess.run(arguments, check=True)  # noqa: S603


def test_video_location_tags_are_detected(tmp_path: Path) -> None:
    plain = tmp_path / "plain.mp4"
    located = tmp_path / "located.mp4"
    apple = tmp_path / "apple.mov"
    _make_video(plain, "title=walk")
    _make_video(located, "location=+40.7128-074.0060/")
    _make_video(apple, "com.apple.quicktime.location.ISO6709=+40.7128-074.0060/")

    assert video_location_keys(plain) == []
    assert "location" in video_location_keys(located)
    assert "com.apple.quicktime.location.ISO6709" in video_location_keys(apple)


def test_evaluation_only_ids_come_from_scope_and_person_in_frame(tmp_path: Path) -> None:
    table = tmp_path / "field_test.csv"
    table.write_text(
        "clip,clip_licence_or_consent_id,scope,person_in_frame,artwork_release_id\n"
        "a,CR-1,evaluation,no,AR-1\n"
        "b,CR-2,publish,no,AR-2\n"
        "c,CR-3,publish,yes,\n"
        "d,CR-4,publish,no,\n",
        encoding="utf-8",
    )

    assert evaluation_only_ids(table) == {"CR-1", "AR-1", "CR-3"}


def test_credit_files_naming_an_evaluation_only_id_are_reported(tmp_path: Path) -> None:
    (tmp_path / "motions").mkdir()
    (tmp_path / "motions/ATTRIBUTION.md").write_text(
        "| walk | owner | owner release CR-1 | Sam | 0 | 4 | |\n"
        "| trot | owner | owner release CR-10 | Sam | 0 | 4 | |\n",
        encoding="utf-8",
    )
    (tmp_path / "docs/site/clips").mkdir(parents=True)
    (tmp_path / "docs/site/clips/manifest.json").write_text('{"credit": "CR-3"}', encoding="utf-8")

    assert leaked_ids(tmp_path, ATTRIBUTION_FILES, {"CR-1", "CR-3", "CR-4"}) == {
        "motions/ATTRIBUTION.md": {"CR-1"},
        "docs/site/clips/manifest.json": {"CR-3"},
    }
