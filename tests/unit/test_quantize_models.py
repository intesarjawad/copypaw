from pathlib import Path

import pytest

from scripts.quantize_models import Variant, choose_variant, make_variant, worst_drops

LABELS: tuple[str, ...] = ("raw", "+ flip test", "+ identity", "+ identity + flip test")
DETECTOR_BYTES = 20_000_000


def _variant(precision: str, size_mb: int, ap: float, pck: float) -> Variant:
    scores = dict.fromkeys(LABELS, (ap, pck))
    return Variant(precision, Path(f"{precision}.onnx"), size_mb * 1_000_000, scores)


def test_smallest_variant_within_limits_is_kept() -> None:
    variants = [
        _variant("fp32", 97, 74.0, 90.0),
        _variant("fp16", 48, 73.8, 89.9),
        _variant("int8", 25, 73.5, 89.5),
    ]

    chosen = choose_variant(variants, DETECTOR_BYTES)

    assert chosen is not None
    assert chosen.precision == "int8"


def test_a_drop_of_one_point_or_more_rejects_the_variant() -> None:
    variants = [
        _variant("fp32", 97, 74.0, 90.0),
        _variant("fp16", 48, 73.5, 89.5),
        _variant("int8", 25, 73.5, 89.0),
    ]

    chosen = choose_variant(variants, DETECTOR_BYTES)

    assert chosen is not None
    assert chosen.precision == "fp16"


def test_one_bad_condition_rejects_the_variant() -> None:
    baseline = _variant("fp32", 97, 74.0, 90.0)
    int8 = _variant("int8", 25, 74.0, 90.0)
    int8 = Variant("int8", int8.path, int8.size_bytes, {**int8.scores, "+ identity": (72.5, 90.0)})

    assert worst_drops(baseline, int8) == (1.5, 0.0)
    assert choose_variant([baseline, int8], DETECTOR_BYTES) is None


def test_files_must_stay_under_the_budget_together() -> None:
    variants = [_variant("fp32", 97, 74.0, 90.0), _variant("fp16", 50, 74.0, 90.0)]

    # 50 MB plus the 20 MB detector reaches 70 MB, which is not under the budget.
    assert choose_variant(variants, DETECTOR_BYTES) is None
    assert choose_variant(variants, DETECTOR_BYTES - 1) is not None


def test_fp32_is_the_source_and_unknown_precisions_fail(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"

    assert make_variant(source, "fp32", tmp_path) == source
    with pytest.raises(ValueError, match="int4"):
        make_variant(source, "int4", tmp_path)
