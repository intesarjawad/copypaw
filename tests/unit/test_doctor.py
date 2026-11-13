import pytest
from typer.testing import CliRunner

import copypaw.cli
from copypaw.cli import app
from copypaw.doctor import report


def test_missing_render_backend_gives_fix() -> None:
    output = report(backend_probe=lambda: "none")

    assert "render: none" in output
    assert "PYOPENGL_PLATFORM=osmesa" in output
    assert "graphical session" in output


def test_doctor_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(copypaw.cli, "report", lambda: report(backend_probe=lambda: "none"))
    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "render:" in result.stdout
    assert "support:" in result.stdout
