import re
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SELFTEST_FILES = (
    "copypaw/selftest/dragon.png",
    "copypaw/selftest/dog_walk.mp4",
    "copypaw/selftest/ATTRIBUTION.txt",
)


@pytest.fixture(scope="module")
def distributions(tmp_path_factory: pytest.TempPathFactory) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed")
    out = tmp_path_factory.mktemp("wheel")
    source = tmp_path_factory.mktemp("source")
    for name in ("copypaw", "web", "scripts"):
        shutil.copytree(
            ROOT / name,
            source / name,
            ignore=shutil.ignore_patterns(
                "node_modules",
                "static",
                "__pycache__",
                "test-results",
                "playwright-report",
                "screenshots",
            ),
        )
    for name in (
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "NOTICE",
        "Makefile",
        ".python-version",
        ".gitignore",
    ):
        shutil.copyfile(ROOT / name, source / name)
    git = shutil.which("git")
    assert git is not None
    subprocess.run(  # noqa: S603
        [git, "init", "--quiet", str(source)], check=True, capture_output=True
    )
    subprocess.run(  # noqa: S603
        [uv, "build", "--out-dir", str(out)], cwd=source, check=True, capture_output=True
    )
    return out


@pytest.fixture(scope="module")
def wheel(distributions: Path) -> Iterator[zipfile.ZipFile]:
    built = list(distributions.glob("copypaw-*.whl"))
    assert len(built) == 1
    with zipfile.ZipFile(built[0]) as archive:
        yield archive


def test_wheel_carries_the_self_test_pair_and_its_credit(wheel: zipfile.ZipFile) -> None:
    names = set(wheel.namelist())

    assert set(SELFTEST_FILES) <= names


def test_wheel_credit_names_the_clip_licence_and_source(wheel: zipfile.ZipFile) -> None:
    credit = wheel.read("copypaw/selftest/ATTRIBUTION.txt").decode()

    assert "CC BY 2.5" in credit
    assert "https://commons.wikimedia.org/wiki/File:" in credit


def test_wheel_notice_names_the_clip(wheel: zipfile.ZipFile) -> None:
    [notice] = [name for name in wheel.namelist() if name.endswith(".dist-info/licenses/NOTICE")]

    assert "copypaw/selftest/dog_walk.mp4" in wheel.read(notice).decode()


def test_wheel_carries_the_built_app_and_its_assets(wheel: zipfile.ZipFile) -> None:
    names = set(wheel.namelist())
    assert "copypaw/app/static/index.html" in names
    assert any(
        name.startswith("copypaw/app/static/assets/") and name.endswith(".js") for name in names
    )


def test_sdist_carries_the_app_and_rebuild_sources(distributions: Path) -> None:
    [source] = distributions.glob("*.tar.gz")
    with tarfile.open(source) as archive:
        names = {name.split("/", 1)[-1] for name in archive.getnames()}
    assert "copypaw/app/static/index.html" in names
    assert "web/package-lock.json" in names
    assert "web/src/main.ts" in names
    assert "scripts/bench.py" in names
    assert not any("node_modules/" in name for name in names)


def test_installed_wheel_serves_the_app_outside_the_checkout(
    distributions: Path, tmp_path: Path
) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    [built] = distributions.glob("*.whl")
    environment = tmp_path / "venv"
    subprocess.run([uv, "venv", str(environment)], check=True, capture_output=True, cwd=tmp_path)  # noqa: S603
    subprocess.run(  # noqa: S603
        [uv, "pip", "install", "--python", str(environment / "bin/python"), str(built)],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )
    with (tmp_path / "server.log").open("w+") as log:
        server = subprocess.Popen(  # noqa: S603
            [str(environment / "bin/copypaw"), "app", "--no-browser"],
            cwd=tmp_path,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                log.seek(0)
                output = log.read()
                match = re.search(r"is running at (http://\S+)", output)
                if match:
                    with urllib.request.urlopen(match[1], timeout=10) as response:  # noqa: S310
                        assert response.status == 200
                        assert b'<script type="module"' in response.read()
                    break
                assert server.poll() is None, output
                time.sleep(0.1)
            else:
                pytest.fail("Installed app did not start within 300 seconds")
        finally:
            server.terminate()
            server.wait(timeout=15)
