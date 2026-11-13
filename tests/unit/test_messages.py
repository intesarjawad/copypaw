import importlib
import json
import pkgutil
import re
import string
from pathlib import Path

import pytest

import copypaw
from copypaw.errors import CopypawError, InternalError, message_table
from scripts.copy_lint import violations

MESSAGES = Path(copypaw.__file__).parent / "messages.json"
STRINGS = Path(__file__).resolve().parents[2] / "web" / "src" / "strings.ts"
ISSUES_URL = "https://github.com/intesarjawad/copypaw/issues/new?template=bug.yml"

# Format arguments each code's text names, written by hand from messages.json.
ARGUMENTS: dict[str, set[str]] = {
    "internal": set(),
    "capture_anatomy": {"problem"},
    "video_open": {"path"},
    "video_frame_rate": {"path"},
    "video_timestamps": {"path"},
    "video_no_frames": {"path"},
    "video_decode": {"path", "reason"},
    "no_animal_found": set(),
    "model_checksum": {"filename", "expected", "actual"},
    "model_size": {"filename", "expected", "actual"},
    "model_download": {"filename", "reason"},
    "model_cache": {"path", "reason"},
    "output_directory": {"path", "reason"},
    "mask_too_small": set(),
    "mask_too_large": set(),
    "mask_multiple_creatures": set(),
    "drawing_unreadable": {"path"},
    "joints_invalid": {"path", "problem"},
    "character_missing": {"path", "name"},
    "motion_unreadable": {"path"},
    "motion_missing_key": {"key"},
    "motion_version": {"found", "expected"},
    "motion_invalid": {"key", "problem"},
    "app_static_missing": {"path"},
    "app_port": {"port", "reason"},
    "app_request": {"problem"},
    "app_body_too_large": {"limit_mb"},
}


def _subclasses(base: type[CopypawError]) -> set[type[CopypawError]]:
    found: set[type[CopypawError]] = set()
    for child in base.__subclasses__():
        found.add(child)
        found |= _subclasses(child)
    return found


def _error_classes() -> set[type[CopypawError]]:
    for module in pkgutil.walk_packages(copypaw.__path__, "copypaw."):
        if ".vendor" not in module.name:
            importlib.import_module(module.name)
    return {cls for cls in _subclasses(CopypawError) if cls.__module__.startswith("copypaw.")}


def _fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}


def _control_names() -> set[str]:
    if not STRINGS.is_file():
        return set()
    return set(re.findall(r"""["'`]([^"'`\n]+)["'`]""", STRINGS.read_text(encoding="utf-8")))


def points_to_next_step(text: str, controls: set[str]) -> bool:
    return bool(
        re.search(r"`[^`]+`", text)
        or re.search(r"https?://\S+", text)
        or any(control in text for control in controls)
    )


def test_every_error_class_has_codes_in_the_table() -> None:
    classes = _error_classes()
    assert {"VideoReadError", "ModelChecksumError", "NoAnimalFoundError", "MaskError"} <= {
        cls.__name__ for cls in classes
    }
    for cls in classes:
        assert cls.codes, cls.__name__
        assert set(cls.codes) <= set(message_table()), cls.__name__


def test_no_message_is_unused() -> None:
    declared = {code for cls in _error_classes() for code in cls.codes}
    assert declared == set(message_table())


def test_internal_message_matches_the_contract() -> None:
    entry = message_table()["internal"]
    assert entry["what"] == "Copypaw hit a bug it did not expect."
    assert entry["next"] == (
        "Run `copypaw doctor` and open an issue with its output: " + ISSUES_URL
    )
    assert InternalError.codes == ("internal",)


@pytest.mark.parametrize("code", sorted(ARGUMENTS))
def test_entry_is_complete(code: str) -> None:
    entry = message_table()[code]
    assert entry["what"].strip()
    assert points_to_next_step(entry["next"], _control_names())
    assert _fields(entry["what"]) | _fields(entry["next"]) == ARGUMENTS[code]


def test_table_and_expected_arguments_list_the_same_codes() -> None:
    assert set(message_table()) == set(ARGUMENTS)


def test_each_entry_has_only_what_and_next() -> None:
    assert all(set(entry) == {"what", "next"} for entry in message_table().values())


def test_next_step_check_accepts_command_control_or_link_only() -> None:
    assert points_to_next_step("Run `copypaw doctor`.", set())
    assert points_to_next_step("See https://example.org/help", set())
    assert points_to_next_step("Press Choose another clip.", {"Choose another clip"})
    assert not points_to_next_step("Try again later.", {"Choose another clip"})


def test_messages_pass_the_copy_lint() -> None:
    assert violations(MESSAGES.read_text(encoding="utf-8")) == []


def test_messages_file_is_valid_json_with_text_entries() -> None:
    data = json.loads(MESSAGES.read_text(encoding="utf-8"))
    assert all(isinstance(value, str) for entry in data.values() for value in entry.values())


def test_error_renders_its_arguments() -> None:
    error = InternalError()
    assert str(error) == "Copypaw hit a bug it did not expect."
    assert error.next.startswith("Run `copypaw doctor`")


def test_error_rejects_a_code_it_does_not_own() -> None:
    with pytest.raises(ValueError, match="cannot raise code"):
        InternalError("video_open", path="clip.mp4")
    with pytest.raises(TypeError, match="needs a code"):
        CopypawError()
