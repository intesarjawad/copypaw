from importlib import import_module
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

bvh = import_module("copypaw.vendor.animated_drawings.model.bvh")

BVH_PATH = Path(__file__).resolve().parents[1] / "fixtures/upstream/bvh/fair1/zombie.bvh"


@runtime_checkable
class ChannelJoint(Protocol):
    channel_order: list[str]

    def get_children(self) -> list[object]: ...


def channel_order(joint: object) -> list[str]:
    assert isinstance(joint, ChannelJoint)
    channels = list(joint.channel_order)
    for child in joint.get_children():
        if isinstance(child, bvh.BVH_Joint):
            channels.extend(channel_order(child))
    return channels


def test_load_bvh_twice_preserves_channel_masks_and_motion() -> None:
    first = bvh.BVH.from_file(str(BVH_PATH))
    first_channels = channel_order(first.root_joint)
    first_mask = np.array(["rotation" in channel for channel in first_channels])
    first_mask[:3] = True

    second = bvh.BVH.from_file(str(BVH_PATH))
    second_channels = channel_order(second.root_joint)
    second_mask = np.array(["rotation" in channel for channel in second_channels])
    second_mask[:3] = True

    assert first_channels == second_channels
    source_channels = [
        channel
        for line in BVH_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("CHANNELS ")
        for channel in line.split()[2:]
    ]
    assert first_channels == source_channels
    assert len(first.pos_data) > 0
    np.testing.assert_array_equal(first_mask, second_mask)
    np.testing.assert_array_equal(first.pos_data, second.pos_data)
    np.testing.assert_array_equal(first.rot_data, second.rot_data)


def test_joint_channel_defaults_are_independent() -> None:
    first = bvh.BVH_Joint(widget=False)
    second = bvh.BVH_Joint(widget=False)
    first.channel_order.append("Xrotation")
    assert second.channel_order == []
