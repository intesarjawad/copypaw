from urllib.parse import quote

import pytest
from playwright.sync_api import sync_playwright

from scripts import cold_start

pytestmark = pytest.mark.ui


@pytest.mark.parametrize("enabled", [True, False])
def test_gallery_cold_start_requires_a_live_result_from_the_preselected_motion(
    enabled: bool,
) -> None:
    dots = '<circle class="dot" r="5" cx="10" cy="10" />' * cold_start.JOINT_COUNT
    disabled = "" if enabled else "disabled"
    html = f"""<input id="drawing-input" type="file"><svg>{dots}</svg>
        <fieldset hidden><input type="radio" name="motion"></fieldset>
        <div class="result"></div>
        <button {disabled} onclick="
            document.querySelector('.result').replaceChildren(document.createElement('canvas'));
            document.body.dataset.state='result'">Animate</button>"""
    plan = cold_start.Plan(
        cold_start.DRAWING, cold_start.CLIP, None, tier1=True, tier2=False, render=True
    )
    with sync_playwright() as playwright:
        rows, _ = cold_start.run_browsers(
            playwright, ["chromium"], "data:text/html," + quote(html), plan, []
        )
    assert len(rows) == 1
    assert rows[0].failed is not enabled
    if enabled:
        assert rows[0].elapsed_s is not None
        assert rows[0].elapsed_s > 0
