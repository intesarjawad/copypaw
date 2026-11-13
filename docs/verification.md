# Day-one verification

Initial measurements: 2026-11-05. Upstream provenance checked: 2026-11-06 UTC.
Commands run from the repository root with UV_CACHE_DIR=/tmp/copypaw-uv-cache.
Host: Apple M5 (Mac17,2), 10 cores, 32 GB; macOS 27.2 arm64; CPython 3.12.15.
Hardware was read with system_profiler SPHardwareDataType -json, retaining only
chip_type, machine_model, number_processors and physical_memory.

I recorded these seven initial checks. PASS applies only to the
observed check named in a result. PENDING and PARTIAL require follow-up.
Follow-up status is stated separately from the historical measurements.

| Item | Date | Command | Observed output and result | Follow-up |
| --- | --- | --- | --- | --- |
| 1. Stable pins | 2026-11-05 | `uv --version`; `uv pip index versions numpy`; `uv run python /tmp/copypaw-verification/pins.py`; `uv python list --all-versions` | uv 0.13.7; pip index is an unrecognized subcommand. Registry probes completed. Python package pins match; managed Python 3.12.16 is available versus installed 3.12.15. Vite 8.3.6 supersedes 8.3.5 and CodeQL v4.39.3 supersedes v4.39.2. TypeScript 7.0.2 is newer than the explicitly held 6.0.3. PARTIAL: some pinned versions differed from registry results. | ADR-007 records drift. |
| 2. OpenGL imports and renderer | 2026-11-05 | `uv run python -c "import OpenGL.GL, numpy; print(numpy.__version__)"`; `uv run copypaw doctor` | Import exits 0, prints 2.5.4. Doctor exits 0 and prints `render: none` after its hidden-window probe times out; providers include CPUExecutionProvider. Linux OSMesa was not run on this Mac. PARTIAL. | ADR-004; graphical-session and Linux commands below. |
| 3. Ten-bit portrait HEVC | 2026-11-05 | `uv run python /tmp/copypaw-verification/media.py` | imageio-ffmpeg supplies ffmpeg 9.0 with libx265. Fixture stream is HEVC Main 10, yuv420p10le, 320x180 with 90-degree display rotation. OpenCV 5.0.0 reads 30 frames as uint8 BGR of shape (320, 180, 3). Synthetic decode PASS; real iPhone clip not tested. | ADR-001. This historical measurement covers only the smoke probe, not the separate video regression suite. |
| 4. APT-36K split and species | 2026-11-05 | `curl -fL` on each official OneDrive URL below; `uv run python /tmp/copypaw-verification/apt_split.py /tmp/copypaw-verification/annotations` | Both downloads exit 56 with HTTP 403. No archive or annotation JSON files obtained. Split probe exits 1: `PENDING: no annotation JSON files; obtain the official archive`. Train/val filenames and species counts remain unverified. | ADR-001; obtain the official annotations. |
| 5. Browser codecs | 2026-11-05 | `uv run playwright install chromium webkit firefox` with PLAYWRIGHT_BROWSERS_PATH set to the temporary cache; `uv run python /tmp/copypaw-verification/browser.py` | Browser install exits 0. Chromium, WebKit and Firefox each fail to launch. All H.264 encode, H.264/HEVC WebCodecs decode and HEVC video-element results are PENDING, not unsupported. | ADR-007; commands below. |
| 6. Commons sample | 2026-11-05 | Contact-sheet inspection and ffprobe | I chose a five-second fixed-camera side-view walk for pose and capture tests. The source playback rate is unchanged. | Acceptance is measured on the walk; it does not establish trot accuracy. |
| 7. ViTPose-L CPU timing | 2026-11-05 | SHA-256-checked download below; `uv run python /tmp/copypaw-verification/vitpose.py` | 1234459913 bytes, expected SHA-256 matches. CPUExecutionProvider only; 5 warmups, 20 measured runs; median 110.338 ms on the base M5. PASS for isolated model timing. | ADR-001 holds the complete script and all samples. Whole-clip accurate-mode timing remains unverified here. |

## Reproduce the probes

I keep downloads and generated fixtures in `/tmp/copypaw-verification/`, outside
the repository. Temporary files may be removed by the operating system. Before
every download I run df -h and stop if free space is below 30 GB. The observed initial free space was
56 GiB and the low observed value was 53 GiB while model and browsers were cached.

These probes are embedded here so pending results can be reproduced without a
new dependency. Extract the Python blocks from this file and ADR-001 into the
temporary cache:

```sh
export UV_CACHE_DIR=/tmp/copypaw-uv-cache
export PLAYWRIGHT_BROWSERS_PATH="/tmp/copypaw-verification/browsers"
df -h .
uv run python - <<'PYEXTRACT'
import re
import shutil
from pathlib import Path

if shutil.disk_usage('.').free < 30_000_000_000:
    raise SystemExit('STOP: less than 30 GB free')
cache = Path('/tmp/copypaw-verification')
cache.mkdir(parents=True, exist_ok=True)
(cache / 'annotations').mkdir(exist_ok=True)
for document in (Path('docs/verification.md'), Path('docs/adr/ADR-001.md')):
    for name, source in re.findall(
        r"```python\n# probe: ([\w-]+\.py)\n(.*?)\n```",
        document.read_text(), re.DOTALL,
    ):
        (cache / name).write_text(source + '\n')
PYEXTRACT
```

### 1. Registry versions

```sh
uv --version
uv run python /tmp/copypaw-verification/pins.py
uv python list --all-versions
```

The attempted `uv pip index` command exits 2 in uv 0.13.7. The equivalent probe
uses the package's PyPI JSON info.version or npm latest version and the public
upstream release API; it performs no account mutations. For CodeQL it filters
stable v4 action releases, since releases/latest points at a CodeQL bundle.
The managed Python 3.12 branch and >=3.12,<3.15 range are intentional; the
uv lists stable CPython 3.15.2 outside that range and 3.16.0a1 as a prerelease.
The installed managed 3.12.15 remains within the required branch; uv lists
3.12.16 as a newer downloadable patch. No interpreter was changed. uv also
reported a pre-existing inactive Python 3.13 shim warning while listing versions.

```python
# probe: pins.py
import concurrent.futures
import json
import tomllib
import urllib.request
from pathlib import Path


def fetch(url):
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "Copypaw-verification/0.1"}),
        timeout=45,
    ) as response:
        return json.load(response)


def check(item):
    name, pinned, url, field = item
    try:
        data = fetch(url)
        if field == "pypi":
            latest = data["info"]["version"]
        elif field == "action-v4":
            latest = next(
                release["tag_name"]
                for release in data
                if release["tag_name"].startswith("v4.")
                and not release["prerelease"]
                and not release["draft"]
            )
        else:
            latest = data[field]
        print(
            f"{name}: pinned={pinned} latest={latest} match={pinned.lstrip('v') == latest.lstrip('v')}",
            flush=True,
        )
    except Exception as error:
        print(f"{name}: UNVERIFIED {type(error).__name__}: {error}", flush=True)


project = tomllib.loads(Path("pyproject.toml").read_text())
items = []
for pin in (
    project["project"]["dependencies"]
    + project["dependency-groups"]["dev"]
    + project["build-system"]["requires"]
):
    name, version = pin.split("==")
    items.append((name, version, f"https://pypi.org/pypi/{name}/json", "pypi"))
web = {
    "onnxruntime-web": "1.31.1",
    "delaunator": "5.1.0",
    "mediabunny": "1.65.2",
    "gifenc": "1.0.3",
    "typescript": "6.0.3",
    "typescript-eslint": "8.72.1",
    "eslint": "10.14.0",
    "prettier": "3.9.10",
    "vite": "8.3.5",
    "vitest": "5.1.0",
    "@playwright/test": "1.64.1",
}
for name, version in web.items():
    items.append((name, version, f"https://registry.npmjs.org/{name}/latest", "version"))
actions = {
    "actions/checkout": "v7.0.1",
    "astral-sh/setup-uv": "v10.3.3",
    "actions/cache": "v6.1.0",
    "github/codeql-action": "v4.39.2",
    "pypa/gh-action-pypi-publish": "v1.14.3",
    "actions/upload-artifact": "v7.0.2",
    "actions/download-artifact": "v8.0.2",
    "actions/upload-pages-artifact": "v5.0.0",
    "actions/deploy-pages": "v5.0.1",
    "gitleaks/gitleaks": "v8.30.1",
    "astral-sh/uv": "0.13.7",
}
for name, version in actions.items():
    if name == "github/codeql-action":
        items.append(
            (
                name,
                version,
                f"https://api.github.com/repos/{name}/releases?per_page=30",
                "action-v4",
            )
        )
    else:
        items.append(
            (name, version, f"https://api.github.com/repos/{name}/releases/latest", "tag_name")
        )
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(check, items))
print("Python requirement:", project["project"]["requires-python"], "; managed branch: 3.12")
```

Observed registry output (order normalized for reading):

```text
@playwright/test: pinned=1.64.1 latest=1.64.1 match=True
PyOpenGL: pinned=3.1.10 latest=3.1.10 match=True
PyYAML: pinned=6.0.3 latest=6.0.3 match=True
Python requirement: >=3.12,<3.15 ; managed branch: 3.12
actions/cache: pinned=v6.1.0 latest=v6.1.0 match=True
actions/checkout: pinned=v7.0.1 latest=v7.0.1 match=True
actions/deploy-pages: pinned=v5.0.1 latest=v5.0.1 match=True
actions/download-artifact: pinned=v8.0.2 latest=v8.0.2 match=True
actions/upload-artifact: pinned=v7.0.2 latest=v7.0.2 match=True
actions/upload-pages-artifact: pinned=v5.0.0 latest=v5.0.0 match=True
astral-sh/setup-uv: pinned=v10.3.3 latest=v10.3.3 match=True
astral-sh/uv: pinned=0.13.7 latest=0.13.7 match=True
axe-playwright-python: pinned=0.1.8 latest=0.1.8 match=True
delaunator: pinned=5.1.0 latest=5.1.0 match=True
eslint: pinned=10.14.0 latest=10.14.0 match=True
gifenc: pinned=1.0.3 latest=1.0.3 match=True
github/codeql-action: pinned=v4.39.2 latest=v4.39.3 match=False
gitleaks/gitleaks: pinned=v8.30.1 latest=v8.30.1 match=True
glfw: pinned=2.11.0 latest=2.11.0 match=True
hatchling: pinned=1.33.0 latest=1.33.0 match=True
hypothesis: pinned=6.170.14 latest=6.170.14 match=True
imageio-ffmpeg: pinned=0.8.1 latest=0.8.1 match=True
imageio: pinned=2.38.1 latest=2.38.1 match=True
mediabunny: pinned=1.65.2 latest=1.65.2 match=True
numpy: pinned=2.5.4 latest=2.5.4 match=True
onnxruntime-web: pinned=1.31.1 latest=1.31.1 match=True
onnxruntime: pinned=1.31.1 latest=1.31.1 match=True
opencv-python-headless: pinned=5.0.0.93 latest=5.0.0.93 match=True
pillow: pinned=13.0.0 latest=13.0.0 match=True
playwright: pinned=1.64.1 latest=1.64.1 match=True
pre-commit: pinned=4.7.0 latest=4.7.0 match=True
prettier: pinned=3.9.10 latest=3.9.10 match=True
pypa/gh-action-pypi-publish: pinned=v1.14.3 latest=v1.14.3 match=True
pyright: pinned=1.1.414 latest=1.1.414 match=True
pytest-cov: pinned=7.1.0 latest=7.1.0 match=True
pytest: pinned=9.2.0 latest=9.2.0 match=True
ruff: pinned=0.17.3 latest=0.17.3 match=True
scikit-image: pinned=0.26.0 latest=0.26.0 match=True
scikit-learn: pinned=1.9.1 latest=1.9.1 match=True
scipy-stubs: pinned=1.18.1.1 latest=1.18.1.1 match=True
scipy: pinned=1.18.1 latest=1.18.1 match=True
shapely: pinned=2.2.0 latest=2.2.0 match=True
tqdm: pinned=4.70.1 latest=4.70.1 match=True
typer: pinned=0.27.3 latest=0.27.3 match=True
types-PyYAML: pinned=6.0.12.20261103 latest=6.0.12.20261103 match=True
typescript-eslint: pinned=8.72.1 latest=8.72.1 match=True
typescript: pinned=6.0.3 latest=7.0.2 match=False
vite: pinned=8.3.5 latest=8.3.6 match=False
vitest: pinned=5.1.0 latest=5.1.0 match=True
```

### 2. OpenGL and doctor

```sh
uv run python -c "import OpenGL.GL, numpy; print(numpy.__version__)"
uv run copypaw doctor
```

Observed doctor output, after graphics-service errors and its timeout in a session without a display:

```text
python: CPython 3.12.15
platform: macOS-27.2-arm64-arm-64bit (arm64)
onnxruntime providers: CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider
render: none (on Linux, install libosmesa6 and libosmesa6-dev and set PYOPENGL_PLATFORM=osmesa; on macOS, run in a graphical session)
models lightweight: 0/2 cached (117527476 bytes required)
models balanced: 0/2 cached (461383353 bytes required)
models accurate: 0/2 cached (1335719657 bytes required)
support: supported
```

Rerun the two commands in a graphical macOS session and retain the
actual render: glfw result. On Linux with libosmesa6, libosmesa6-dev and libglu1-mesa installed:

```sh
PYOPENGL_PLATFORM=osmesa uv run python -c 'from OpenGL import osmesa; print("osmesa import: OK")'
PYOPENGL_PLATFORM=osmesa uv run copypaw doctor
```

Those Linux commands were not run here. They are not a completed render test.

### 3. Synthetic HEVC fixture and OpenCV smoke

```sh
uv run python /tmp/copypaw-verification/media.py
```

The script encodes ten-bit HEVC first, then remuxes with display_rotation as an
input option before -i. ffmpeg reported Main 10 and displaymatrix rotation of
90 degrees. OpenCV's auto-rotation yields portrait frame dimensions.

```python
# probe: media.py
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

cache = Path("/tmp/copypaw-verification")
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
print(subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0], flush=True)
subprocess.run(
    [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x180:rate=30:duration=1",
        "-c:v",
        "libx265",
        "-pix_fmt",
        "yuv420p10le",
        "-tag:v",
        "hvc1",
        "-an",
        "-map_metadata",
        "-1",
        str(cache / "hevc-landscape.mp4"),
    ],
    check=True,
)
subprocess.run(
    [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-display_rotation",
        "90",
        "-i",
        str(cache / "hevc-landscape.mp4"),
        "-c",
        "copy",
        str(cache / "hevc-portrait.mp4"),
    ],
    check=True,
)
probe = subprocess.run(
    [ffmpeg, "-hide_banner", "-i", str(cache / "hevc-portrait.mp4"), "-f", "null", "-"],
    capture_output=True,
    text=True,
    check=True,
)
print(probe.stderr, flush=True)
capture = cv2.VideoCapture(str(cache / "hevc-portrait.mp4"))
frames = []
while True:
    ok, frame = capture.read()
    if not ok:
        break
    frames.append(frame)
capture.release()
assert len(frames) == 30, len(frames)
assert frames[0].shape == (320, 180, 3), frames[0].shape
print(
    f"OpenCV {cv2.__version__}: decoded {len(frames)} frames; shape={frames[0].shape}; dtype={frames[0].dtype}"
)
mask = np.full((160, 160), 255, dtype=np.uint8)
mask[40:120, 40:120] = 0
threshold = cv2.adaptiveThreshold(
    mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 8
)
closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
assert closed.shape == mask.shape and closed.dtype == np.uint8 and closed.any()
print("OpenCV adaptive threshold and 3x3 close: smoke passed")
```

Observed output excerpt:

```text
OpenCV 5.0.0: decoded 30 frames; shape=(320, 180, 3); dtype=uint8
OpenCV adaptive threshold and 3x3 close: smoke passed
```

I have not tested an actual ten-bit portrait iPhone clip. Reuse the VideoCapture
read loop above on that local file; do not upload it or commit private input.

### 4. Official annotations

The [APT-36K README at 366ba010](https://github.com/pandorgan/APT-36K/blob/366ba0109bb9586cac793d1b150d5f4261b77cfa/README.md) supplies both links.
Both commands were attempted and failed with curl exit 56, HTTP 403:

```sh
curl -fL --max-time 60 'https://1drv.ms/u/s!AimBgYV7JjTlgTuYdjjtYON3sxEZ?e=5deTDn' -o /tmp/copypaw-verification/apt-annotations.download
curl -fL --max-time 60 'https://1drv.ms/u/s!AimBgYV7JjTlgcZ9zLyl5KnM3dKMgg?e=uaaLz5' -o /tmp/copypaw-verification/apt-full.download
```

No annotation filenames, dog or cat counts, test presence, or separation from
train_annotations_1.json can be confirmed from a failed download. I still need
the official annotations without a paid service or account creation. Extract
only those annotations into the cache's annotations/ folder, then run:

```sh
uv run python /tmp/copypaw-verification/apt_split.py /tmp/copypaw-verification/annotations
```

The probe lists all JSON paths, checks validation differs from the training file
by path, digest and image names, and prints instance and unique-image counts per
species for validation and test when present. It selects split 1 by filename
and stops for ambiguous alternative filenames, so another fold is not
accidentally compared against split 1 training images. It requires dog and cat category
names to be confirmed, rather than silently treating a different schema as zero.
The probe has only been run against the empty folder and remains unvalidated
against the unavailable archive.

```python
# probe: apt_split.py
import collections
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.rglob("*.json"))
if not files:
    raise SystemExit("PENDING: no annotation JSON files; obtain the official archive")
for path in files:
    print(path.relative_to(root))
train = [path for path in files if "train_annotations_1" in path.name]
validation = [path for path in files if "val_annotations_1" in path.name]
if not validation:
    validation = [path for path in files if "val" in path.name.lower()]
if len(validation) != 1:
    raise SystemExit("UNVERIFIED: identify the official validation file for training split 1")
if len(train) != 1 or not validation:
    raise SystemExit("UNVERIFIED: identify the official train and validation split filenames")
train_data = json.loads(train[0].read_text())
train_images = {image["file_name"] for image in train_data["images"]}
with train[0].open("rb") as source:
    train_digest = hashlib.file_digest(source, "sha256").hexdigest()
print("train:", train[0].name, "sha256:", train_digest)
tests = [path for path in files if "test_annotations_1" in path.name]
if not tests:
    tests = [path for path in files if "test" in path.name.lower()]
if len(tests) > 1:
    raise SystemExit("UNVERIFIED: identify the official test file for training split 1")
for path in validation + tests:
    data = json.loads(path.read_text())
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    overlap = train_images & {image["file_name"] for image in data["images"]}
    print(path.name, "sha256:", digest, "train image overlap:", len(overlap))
    assert path != train[0] and digest != train_digest and not overlap
    categories = {category["id"]: category["name"] for category in data["categories"]}
    counts = collections.Counter(annotation["category_id"] for annotation in data["annotations"])
    images_by_species = collections.defaultdict(set)
    for annotation in data["annotations"]:
        images_by_species[annotation["category_id"]].add(annotation["image_id"])
    for category_id, name in sorted(categories.items(), key=lambda pair: pair[1]):
        print(
            f"{name}: instances={counts[category_id]} images={len(images_by_species[category_id])}"
        )
    for target in ("dog", "cat"):
        matches = [key for key, name in categories.items() if name.lower() == target]
        assert matches, f"{target} category absent; inspect category naming before evaluation"
if not any("test" in path.name.lower() for path in files):
    print("test: absent from supplied annotation archive")
```

### 5. Browser WebCodecs and HEVC

After the extraction step, run these exact commands in a graphical
session. The media probe creates the HEVC fixture used here.

```sh
uv run python /tmp/copypaw-verification/media.py
PLAYWRIGHT_BROWSERS_PATH="/tmp/copypaw-verification/browsers" uv run playwright install chromium webkit firefox
PLAYWRIGHT_BROWSERS_PATH="/tmp/copypaw-verification/browsers" uv run python /tmp/copypaw-verification/browser.py
```

Installed package: Playwright 1.64.1. Browser versions below come from the install
output; runtime browser.version was never available because launches failed.

| Browser | Installed version/build | Launch evidence | H.264 encode | H.264 decoder | HEVC decoder | HEVC video element |
| --- | --- | --- | --- | --- | --- | --- |
| Chromium | 156.0.8078.4 / 1248, headless shell | MachPortRendezvousServer bootstrap_check_in: Permission denied (1100) | pending | pending | pending | pending |
| WebKit | 27.2 / 2370 | Abort trap: 6, exitCode 134 | pending | pending | pending | pending |
| Firefox | 157.0 / 1555 | headless launch exits SIGABRT | pending | pending | pending | pending |

The requested H.264 encoder configuration is avc1.42001f at 1024x512. The HEVC
configuration matches the synthetic Main 10 fixture. File URLs provide the
local test page without opening a server; the output also reports secureContext.
A false encoder result from a successfully launched browser selects GIF export
for that browser. No browser is classified GIF-only yet.

```python
# probe: browser.py
import base64
import json
from pathlib import Path

from playwright.sync_api import Error, sync_playwright

cache = Path("/tmp/copypaw-verification")
(cache / "probe.html").write_text("<!doctype html><title>Media capability probe</title>")
fixture = base64.b64encode((cache / "hevc-portrait.mp4").read_bytes()).decode()
probe = """async (base64) => {
  async function support(api, config) {
    if (!api) return {supported: false, reason: 'API absent'};
    try { return {supported: (await api.isConfigSupported(config)).supported}; }
    catch (error) { return {supported: false, reason: String(error)}; }
  }
  const result = {
    secureContext: isSecureContext,
    h264Encode: await support(globalThis.VideoEncoder, {codec: 'avc1.42001f', width: 1024, height: 512}),
    h264Decode: await support(globalThis.VideoDecoder, {codec: 'avc1.42001f', codedWidth: 1024, codedHeight: 512}),
    hevcDecode: await support(globalThis.VideoDecoder, {codec: 'hvc1.2.4.L60.B0', codedWidth: 320, codedHeight: 180})
  };
  const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
  const video = document.createElement('video');
  video.muted = true;
  const url = URL.createObjectURL(new Blob([bytes], {type: 'video/mp4'}));
  result.hevcVideo = await new Promise(resolve => {
    const timer = setTimeout(() => resolve({decoded: false, reason: '10 s timeout'}), 10000);
    video.onloadeddata = () => {clearTimeout(timer); resolve({decoded: true, width: video.videoWidth, height: video.videoHeight});};
    video.onerror = () => {clearTimeout(timer); resolve({decoded: false, error: video.error?.code, message: video.error?.message});};
    video.src = url;
    video.load();
  });
  URL.revokeObjectURL(url);
  return result;
}"""
with sync_playwright() as playwright:
    for name in ("chromium", "webkit", "firefox"):
        try:
            browser = getattr(playwright, name).launch(headless=True, timeout=10000)
        except Error as error:
            print(f"{name}: PENDING launch failed: {error}", flush=True)
            continue
        try:
            page = browser.new_page()
            page.goto((cache / "probe.html").resolve().as_uri())
            print(name, browser.version, json.dumps(page.evaluate(probe, fixture)), flush=True)
        finally:
            browser.close()
```

### 6. Commons walking sample

I chose [a side-view video of a normal dog](https://commons.wikimedia.org/w/index.php?title=File:Quantitative-Phenotyping-of-Duchenne-Muscular-Dystrophy-Dogs-by-Comprehensive-Gait-Analysis-and-pone.0059875.s003.ogv&oldid=1127451745)
for `samples/pets/dog_walk.mp4` because the fixed camera keeps the walking dog
in view throughout the five-second source. The research footage is slowed in
its original form; I preserved its playback rate. The source credit, licence,
conversion details and SHA-256 are in `samples/ATTRIBUTION.md`.

### 7. ViTPose-L download and CPU benchmark

After the extraction step, run:

```sh
df -h .
uv run python -c 'import shutil; assert shutil.disk_usage(".").free >= 30_000_000_000, "STOP: less than 30 GB free"'
curl -fL --max-time 600 'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/e83805274e89428969355ec4afffcbc413e79188/onnx/apt36k/vitpose-l-apt36k.onnx' -o /tmp/copypaw-verification/vitpose-l-apt36k.onnx
uv run python /tmp/copypaw-verification/vitpose.py
```

The benchmark checks byte size and SHA-256 before creating the session. The
complete script, all timings and the measured median are in
[ADR-001](adr/ADR-001.md). It uses no torch or GPU provider.

## Model file provenance

I checked the [publisher's pinned file metadata](https://huggingface.co/api/models/JunkyByte/easy_ViTPose/revision/e83805274e89428969355ec4afffcbc413e79188?blobs=true)
on 2026-11-06 UTC. For each pose file, `size` and `lfs.sha256` supply the values
below. These are publisher metadata, not new local measurements of pose weights.
The original L-model timing above separately verified that download's digest.

| Pinned pose file | Publisher bytes | Publisher SHA-256 |
| --- | ---: | --- |
| [ViTPose-S](https://huggingface.co/JunkyByte/easy_ViTPose/blob/e83805274e89428969355ec4afffcbc413e79188/onnx/apt36k/vitpose-s-apt36k.onnx) | 97307814 | `2a40f76421d87f1b03f1204949d67014dbb876b7572c2bbc60ce2be45f6b4172` |
| [ViTPose-B](https://huggingface.co/JunkyByte/easy_ViTPose/blob/e83805274e89428969355ec4afffcbc413e79188/onnx/apt36k/vitpose-b-apt36k.onnx) | 360123609 | `7f7240941aef7505fd6f57c87ea2305cc8778a7ac5c741120a2e782d66c7aa91` |
| [ViTPose-L](https://huggingface.co/JunkyByte/easy_ViTPose/blob/e83805274e89428969355ec4afffcbc413e79188/onnx/apt36k/vitpose-l-apt36k.onnx) | 1234459913 | `595e2a10cf20a744446b924f0cea7b9a4c7b35e4f730a148510cac863a3d8737` |

The [YOLOX 0.1.1rc0 release metadata](https://api.github.com/repos/Megvii-BaseDetection/YOLOX/releases/tags/0.1.1rc0)
publishes sizes but has `digest: null` for both detector assets. I streamed the
complete versioned downloads on 2026-11-06 UTC, counted bytes and computed
SHA-256 without saving model files. The observed counts match the release API.
A release tag names a version; assets can still be replaced, so the measured
hashes are necessary to identify the exact bytes.

| Release asset | Publisher and measured bytes | Measured SHA-256 |
| --- | ---: | --- |
| [yolox_tiny.onnx](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx) | 20219662 | `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7` |
| [yolox_m.onnx](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx) | 101259744 | `21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f` |

I used this standard-library procedure to reproduce both detector measurements:

```python
# probe: model_provenance.py
import hashlib
import json
import urllib.request

release = "https://api.github.com/repos/Megvii-BaseDetection/YOLOX/releases/tags/0.1.1rc0"
with urllib.request.urlopen(release, timeout=45) as response:
    metadata = json.load(response)
for asset in metadata["assets"]:
    if asset["name"] not in {"yolox_tiny.onnx", "yolox_m.onnx"}:
        continue
    digest = hashlib.sha256()
    count = 0
    with urllib.request.urlopen(asset["browser_download_url"], timeout=45) as response:
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            digest.update(chunk)
    assert count == asset["size"], (count, asset["size"])
    print(asset["name"], count, digest.hexdigest(), "publisher digest:", asset.get("digest"))
```

After extracting the embedded probes, run
`uv run python /tmp/copypaw-verification/model_provenance.py`.
This transfers both detector files again, without installing or retaining them.
[ADR-001](adr/ADR-001.md#upstream-evidence) records the configuration and licence
sources and separates them from the unconfirmed exact-export lineage.

## Not independently confirmed

I have not downloaded the official annotation archive or established validation
and test counts. The pinned upstream training configuration confirms a training
filename only. I have not rerun the real-video acceptance suite, graphical
renderer, browser codecs or the isolated timing during this provenance check.
Publisher digests are unavailable for the two YOLOX assets; their digests above
are my measurements. Exact detector preprocessing and pose export lineage
remain unresolved as described in ADR-001.

## Deferred checks

- Obtain a portrait iPhone input for the real-video regression.
- Obtain official APT-36K annotations and rerun the split probe to record
  validation and test species counts.
- Rerun doctor in a graphical macOS session and check OSMesa on Linux.
- Run the browser probes in a graphical session to check encoder support.
- Check Vite, CodeQL and managed Python patch drift. TypeScript remains held
  at the documented version.
- Record rendered motion quality and whole-clip timing on real footage.
