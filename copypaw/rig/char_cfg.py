"""Write a drawing and its skeleton in the drawing engine's character layout."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from copypaw.rig.mask import segment
from copypaw.rig.template import Joint, Joints

__all__ = ["CharacterDir", "write_character"]


@dataclass(frozen=True)
class CharacterDir:
    """Paths to one character directory."""

    directory: Path
    texture: Path
    mask: Path
    char_cfg: Path
    manifest: Path


def _png(path: Path, image: NDArray[np.uint8]) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Cannot encode PNG: {path}")
    path.write_bytes(np.asarray(encoded, dtype=np.uint8).tobytes())


def _mirror(joints: Joints, width: int) -> Joints:
    return Joints(
        {name: Joint(width - 1 - joint.x, joint.y, joint.parent) for name, joint in joints.items()},
        legs=joints.legs,
        tails=tuple(
            ((width - 1 - base[0], base[1]), (width - 1 - tip[0], tip[1]))
            for base, tip in joints.tails
        ),
        legs_guessed=joints.legs_guessed,
    )


def _pixel(joint: Joint, width: int, height: int) -> list[int]:
    x = min(max(round(joint.x), 0), width - 1)
    y = min(max(round(joint.y), 0), height - 1)
    return [x, y]


def write_character(image: NDArray[np.uint8], joints: Joints, out_dir: Path) -> CharacterDir:
    """Write texture, mask, skeleton config and sidecar metadata.

    The input is uint8 BGR like the segmenter's. A creature whose neck lies
    left of its hip is mirrored to face right, and the mirroring is recorded.
    Tails that reach down among the legs are recorded from base to tip.
    """
    drawing = np.ascontiguousarray(image)
    mask = segment(drawing)
    if mask.shape != drawing.shape[:2]:
        mask = cv2.resize(
            mask, (drawing.shape[1], drawing.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    mirrored = joints["neck"].x < joints["hip"].x
    placed = joints
    if mirrored:
        drawing = np.ascontiguousarray(drawing[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
        placed = _mirror(joints, drawing.shape[1])

    height, width = drawing.shape[:2]
    texture = np.dstack((drawing, np.full((height, width), 255, dtype=np.uint8)))
    locations = {name: _pixel(joint, width, height) for name, joint in placed.items()}
    locations["root"] = locations["hip"]
    skeleton = [
        {"name": name, "loc": locations[name], "parent": joint.parent}
        for name, joint in placed.items()
    ]

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    texture_path = directory / "texture.png"
    mask_path = directory / "mask.png"
    config_path = directory / "char_cfg.yaml"
    metadata_path = directory / "copypaw.yaml"
    _png(texture_path, np.asarray(texture, dtype=np.uint8))
    _png(mask_path, np.asarray(mask, dtype=np.uint8))
    config_path.write_text(
        yaml.safe_dump(
            {"height": height, "width": width, "skeleton": skeleton},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    metadata: dict[str, object] = {"legs": placed.legs}
    if mirrored:
        metadata["mirrored"] = True
    if placed.legs_guessed:
        metadata["legs_guessed"] = True
    if placed.tails:
        metadata["tails"] = [[list(base), list(tip)] for base, tip in placed.tails]
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return CharacterDir(directory, texture_path, mask_path, config_path, metadata_path)
