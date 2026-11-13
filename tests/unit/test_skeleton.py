from collections import Counter

import pytest

from copypaw.motion.skeleton import BONES, HELPER_JOINTS, ROOT_KP, SIDE_OFFSET_RATIO


def test_distal_keypoints_cover_every_joint_except_root_and_eyes() -> None:
    assert ROOT_KP == 4
    assert Counter(bone.dist_kp for bone in BONES) == Counter(set(range(2, 17)) - {ROOT_KP})


def test_hierarchy_has_unique_names_and_connected_parents() -> None:
    names = {bone.name for bone in BONES}
    assert len(names) == len(BONES)
    by_name = {bone.name: bone for bone in BONES}
    visited: set[str] = set()
    for bone in BONES:
        if bone.parent is None:
            assert bone.prox_kp == ROOT_KP
        else:
            assert bone.parent in names
            assert bone.parent in visited
            assert bone.prox_kp == by_name[bone.parent].dist_kp
        visited.add(bone.name)


@pytest.mark.parametrize(
    ("side", "names", "keypoints"),
    [
        ("centre", ("Neck", "Nose"), (4, 3, 2)),
        ("near", ("Near_Shoulder", "Near_Elbow", "Near_F_Paw"), (3, 5, 6, 7)),
        ("far", ("Far_Shoulder", "Far_Elbow", "Far_F_Paw"), (3, 8, 9, 10)),
        ("near", ("Near_Hip", "Near_Knee", "Near_B_Paw"), (4, 11, 12, 13)),
        ("far", ("Far_Hip", "Far_Knee", "Far_B_Paw"), (4, 14, 15, 16)),
    ],
)
def test_identity_resolved_chains(
    side: str, names: tuple[str, ...], keypoints: tuple[int, ...]
) -> None:
    by_name = {bone.name: bone for bone in BONES}
    for index, name in enumerate(names):
        bone = by_name[name]
        assert bone.side == side
        assert (bone.prox_kp, bone.dist_kp) == keypoints[index : index + 2]
        expected_parent = names[index - 1] if index else ("Neck" if keypoints[0] == 3 else None)
        assert bone.parent == expected_parent


def test_forward_helpers_are_separate_static_root_children() -> None:
    assert HELPER_JOINTS == (("Fwd_Near", 0.5), ("Fwd_Far", -0.5))
    assert {name for name, _ in HELPER_JOINTS}.isdisjoint(bone.name for bone in BONES)
    assert SIDE_OFFSET_RATIO == 0.08
