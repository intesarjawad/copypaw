"""Scan the web page with axe-core and drive the whole flow by mouse, touch and keyboard."""

import argparse
import re
import shutil
import sys
import tempfile
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from axe_playwright_python.sync_playwright import Axe  # pyright: ignore[reportMissingTypeStubs]
from playwright.sync_api import Browser, BrowserContext, Locator, Page, expect, sync_playwright
from playwright.sync_api import Error as PlaywrightError

from copypaw.app.server import STATIC_DIR, LocalApp, Pipeline
from copypaw.cli import animate, capture

ROOT = Path(__file__).resolve().parents[1]
DRAWING = ROOT / "samples/worst_case/plain.png"
GOLDEN_MOTION = ROOT / "tests/golden/dog_walk.motion.npz"
SAMPLE_CLIP = ROOT / "samples/pets/dog_walk.mp4"
OUT = ROOT / "build"
MOTION_NAME = "dog_walk"
BLOCKING_IMPACTS = ("serious", "critical")
JOINT_COUNT = 16
RIG_TIMEOUT_MS = 60_000
RENDER_TIMEOUT_MS = 600_000
MAX_TAB_STOPS = 80
MIN_FOCUS_OUTLINE_PX = 2.0
HIDDEN_OUTLINES = ("none", "hidden")
PINS = "svg .dot"

_FOCUS_PROBE = """() => {
  const element = document.activeElement;
  if (element === null || element === document.body) {
    return null;
  }
  const style = getComputedStyle(element);
  const labelled = 'labels' in element && element.labels !== null ? element.labels[0] : undefined;
  const text = element.getAttribute('aria-label') ?? labelled?.textContent ?? element.textContent;
  return {
    tag: element.tagName.toLowerCase(),
    label: (text ?? '').trim().slice(0, 40),
    key: [element.tagName, element.id, element.getAttribute('name'), element.getAttribute('value')]
      .join('|'),
    focusVisible: element.matches(':focus-visible'),
    outlineStyle: style.outlineStyle,
    outlineWidth: style.outlineWidth,
    boxShadow: style.boxShadow,
  };
}"""


@dataclass(frozen=True)
class Profile:
    """How one pass drives the page."""

    name: str
    width: int
    height: int
    touch: bool = False


PROFILES = (
    Profile("desktop", 1280, 800),
    Profile("mobile", 390, 844, touch=True),
    Profile("keyboard", 1280, 800),
)


@dataclass(frozen=True)
class Focus:
    """What the browser reports about the control that holds focus."""

    tag: str
    label: str
    key: str
    focus_visible: bool
    outline_style: str
    outline_width: str
    box_shadow: str


@dataclass(frozen=True)
class Finding:
    """One axe rule that failed on one page state."""

    where: str
    rule: str
    impact: str
    nodes: int
    help: str


@dataclass
class Report:
    """Everything that went wrong, in the order it was found."""

    findings: list[Finding] = field(default_factory=list[Finding])
    failures: list[str] = field(default_factory=list[str])

    @property
    def passed(self) -> bool:
        """True when nothing blocking or broken was recorded."""
        return not self.findings and not self.failures


def blocking(where: str, violations: Sequence[Mapping[str, Any]]) -> list[Finding]:
    """Keep the serious and critical axe violations of one page state."""
    return [
        Finding(
            where=where,
            rule=str(violation["id"]),
            impact=str(violation["impact"]),
            nodes=len(violation["nodes"]),
            help=str(violation["help"]),
        )
        for violation in violations
        if violation["impact"] in BLOCKING_IMPACTS
    ]


def focus_is_visible(focus: Focus) -> bool:
    """True when the focused control draws a ring: an outline of 2px or more, or a shadow."""
    if not focus.focus_visible:
        return False
    if focus.outline_style not in HIDDEN_OUTLINES:
        try:
            if float(focus.outline_width.removesuffix("px")) >= MIN_FOCUS_OUTLINE_PX:
                return True
        except ValueError:
            return False
    return focus.box_shadow != "none"


def slug(text: str, fallback: str) -> str:
    """Return a file-name-safe form of a control's label."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or fallback


def focus_screenshot(directory: Path, profile: Profile, stop: int, focus: Focus) -> Path:
    """Return where the screenshot of the stop-th focused control is stored."""
    return directory / "focus" / profile.name / f"{stop:02d}-{slug(focus.label, focus.tag)}.png"


def read_focus(page: Page) -> Focus | None:
    """Describe the focused control, or None when focus is on the page itself."""
    probe = cast(dict[str, Any] | None, page.evaluate(_FOCUS_PROBE))
    if probe is None:
        return None
    return Focus(
        tag=probe["tag"],
        label=probe["label"],
        key=probe["key"],
        focus_visible=probe["focusVisible"],
        outline_style=probe["outlineStyle"],
        outline_width=probe["outlineWidth"],
        box_shadow=probe["boxShadow"],
    )


def scan(page: Page, where: str, report: Report) -> None:
    """Run axe on the page as it is now and record its blocking violations."""
    response = cast(dict[str, Any], Axe().run(page).response)  # pyright: ignore[reportUnknownMemberType]
    report.findings.extend(blocking(where, response["violations"]))


def tab_to(page: Page, selector: str) -> None:
    """Press Tab until the focused control matches the selector."""
    for _ in range(MAX_TAB_STOPS):
        page.keyboard.press("Tab")
        if cast(
            bool, page.evaluate("(s) => document.activeElement?.matches(s) ?? false", selector)
        ):
            return
    raise AssertionError(f"Tab never reached {selector}")


def walk_focus(page: Page, profile: Profile, directory: Path, report: Report) -> None:
    """Tab through every control once, requiring a visible ring and saving a screenshot of each."""
    page.evaluate("() => { window.scrollTo(0, 0); document.activeElement?.blur(); }")
    seen: set[str] = set()
    for stop in range(1, MAX_TAB_STOPS + 1):
        page.keyboard.press("Tab")
        focus = read_focus(page)
        if focus is None or focus.key in seen:
            return
        seen.add(focus.key)
        target = focus_screenshot(directory, profile, stop, focus)
        target.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=target)
        if not focus_is_visible(focus):
            report.failures.append(
                f"{profile.name}: no visible focus ring on {focus.tag} '{focus.label}'"
            )
    report.failures.append(f"{profile.name}: more than {MAX_TAB_STOPS} tab stops")


def rig_drawing(page: Page, drawing: Path) -> None:
    """Hand the page a drawing; the system file dialog cannot be driven, so set the input."""
    page.locator("#drawing-input").set_input_files(drawing)
    expect(page.locator("svg .dot")).to_have_count(JOINT_COUNT, timeout=RIG_TIMEOUT_MS)


def expect_result(page: Page) -> None:
    expect(page.locator("#result-video")).to_be_visible(timeout=RENDER_TIMEOUT_MS)


def pointer_flow(page: Page, profile: Profile, report: Report) -> None:
    """Finish the flow with the mouse, or with taps on a touch screen."""

    def press(target: Locator) -> None:
        if profile.touch:
            target.tap()
        else:
            target.click()

    press(page.locator(f"{PINS}[data-joint='neck']"))
    press(page.locator(f"input[name=motion][value='{MOTION_NAME}']"))
    scan(page, f"{profile.name}: motion chosen", report)
    press(page.get_by_role("button", name="Animate"))


def keyboard_flow(page: Page, report: Report) -> None:
    """Finish the flow with Tab, arrow keys, Space and Enter only."""
    tab_to(page, PINS)
    first = page.locator(".dot.selected").get_attribute("data-joint")
    before = page.locator(".dot.selected").get_attribute("cx")
    page.keyboard.press("ArrowRight")
    expect(page.locator(".dot.selected")).not_to_have_attribute("cx", before or "")

    page.keyboard.press("Tab")
    if page.locator(".dot.selected").get_attribute("data-joint") == first:
        report.failures.append("keyboard: Tab did not choose the next pin")

    tab_to(page, "input[name=motion]")
    page.keyboard.press("Space")
    scan(page, "keyboard: motion chosen", report)
    tab_to(page, "button.animate")
    page.keyboard.press("Enter")


def run_profile(
    browser: Browser, profile: Profile, url: str, directory: Path, report: Report
) -> None:
    """Drive the whole flow for one profile, then check focus rings on the finished page."""
    context: BrowserContext = browser.new_context(
        viewport={"width": profile.width, "height": profile.height},
        has_touch=profile.touch,
        is_mobile=profile.touch,
    )
    page = context.new_page()
    try:
        page.goto(url)
        scan(page, f"{profile.name}: empty", report)
        rig_drawing(page, DRAWING)
        scan(page, f"{profile.name}: joints", report)
        if profile.name == "keyboard":
            keyboard_flow(page, report)
        else:
            pointer_flow(page, profile, report)
        expect_result(page)
        scan(page, f"{profile.name}: result", report)
        walk_focus(page, profile, directory, report)
    except (AssertionError, PlaywrightError) as error:
        report.failures.append(f"{profile.name}: flow did not finish: {error}")
    finally:
        context.close()


def scan_page(browser: Browser, url: str, where: str, report: Report) -> None:
    """Scan one page that has no flow to drive."""
    page = browser.new_page()
    try:
        page.goto(url)
        scan(page, where, report)
    except PlaywrightError as error:
        report.failures.append(f"{where}: could not load {url}: {error}")
    finally:
        page.close()


def audit(browser: Browser, url: str, try_url: str | None, directory: Path) -> Report:
    """Scan the app, scan the try page when there is one, and run every profile."""
    report = Report()
    if try_url is not None:
        scan_page(browser, try_url, "try", report)
    for profile in PROFILES:
        run_profile(browser, profile, url, directory, report)
    return report


@contextmanager
def local_app() -> Generator[LocalApp]:
    """Serve the built page with one gallery motion, as a launch of the app would."""
    with tempfile.TemporaryDirectory(prefix="copypaw-axe-") as folder:
        gallery = Path(folder)
        shutil.copyfile(GOLDEN_MOTION, gallery / GOLDEN_MOTION.name)
        shutil.copyfile(SAMPLE_CLIP, gallery / SAMPLE_CLIP.name)
        server = LocalApp(
            Pipeline(capture=capture, animate=animate), static_dir=STATIC_DIR, gallery_dir=gallery
        )
        server.start()
        try:
            yield server
        finally:
            server.close()


def print_report(report: Report) -> None:
    for finding in report.findings:
        print(
            f"{finding.where}: {finding.rule} ({finding.impact}) on {finding.nodes} "
            f"element(s): {finding.help}"
        )
    for failure in report.failures:
        print(failure)
    print("passed" if report.passed else "failed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Address of a running app, token fragment included.")
    parser.add_argument("--try-url", help="Address of the try page, served by vite preview.")
    parser.add_argument("--out", type=Path, default=OUT, help="Folder for focus screenshots.")
    args = parser.parse_args(argv)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            if args.url is None:
                with local_app() as server:
                    report = audit(browser, server.url, args.try_url, args.out)
            else:
                report = audit(browser, args.url, args.try_url, args.out)
        finally:
            browser.close()
    print_report(report)
    return int(not report.passed)


if __name__ == "__main__":
    sys.exit(main())
