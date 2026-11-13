import io
import json
import re
import shutil
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright

from copypaw.app.server import STATIC_DIR, LocalApp, Pipeline
from copypaw.cli import animate, capture

ROOT = Path(__file__).resolve().parents[2]
DRAWING = ROOT / "samples/worst_case/plain.png"
GOLDEN_MOTION = ROOT / "tests/golden/dog_walk.motion.npz"
SAMPLE_CLIP = ROOT / "samples/pets/dog_walk.mp4"
RIG_TIMEOUT_MS = 60_000
RENDER_TIMEOUT_MS = 600_000
DOWNLOADS = ("Download GIF", "Download MP4", "Download BVH", "Download character (zip)")

pytestmark = [
    pytest.mark.ui,
    pytest.mark.skipif(
        not (STATIC_DIR / "index.html").is_file(),
        reason="the web page is not built: cd web && npm ci && npm run build",
    ),
]


@pytest.fixture
def local(tmp_path: Path) -> Iterator[LocalApp]:
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    shutil.copyfile(GOLDEN_MOTION, gallery / "dog_walk.motion.npz")
    shutil.copyfile(SAMPLE_CLIP, gallery / "dog_walk.mp4")
    server = LocalApp(Pipeline(capture=capture, animate=animate), gallery_dir=gallery)
    server.start()
    yield server
    server.close()


def test_sample_pair_animates_and_offers_every_download(local: LocalApp) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(accept_downloads=True)
        page.goto(local.url)

        page.locator("#drawing-input").set_input_files(DRAWING)
        expect(page.locator("svg .dot")).to_have_count(16, timeout=RIG_TIMEOUT_MS)
        motion = page.get_by_role("radio", name="Dog walk", exact=True)
        # Without models to track a clip, the lone gallery walk is chosen already and not shown.
        shown = page.locator("fieldset.gallery:not([hidden]), section.motion[hidden]")
        expect(shown).to_have_count(1)
        if motion.is_visible():
            motion.check()
        page.get_by_role("button", name="Animate").click()

        expect(page.locator("#result-video")).to_be_visible(timeout=RENDER_TIMEOUT_MS)
        for name in DOWNLOADS:
            expect(page.get_by_role("link", name=name)).to_have_attribute(
                "href", re.compile(r"^blob:")
            )
        with page.expect_download() as download:
            page.get_by_role("link", name="Download character (zip)").click()
        archive = Path(download.value.path()).read_bytes()
        browser.close()

    with zipfile.ZipFile(io.BytesIO(archive)) as character:
        assert "rig_edits.json" in character.namelist()
        edits = json.loads(character.read("rig_edits.json"))
    assert edits["joints_moved"] == 0
