# Evaluation

## Gait fidelity

The bundled sample has no eligible gait score. Its captured contacts do not contain
enough landings on every paw for `is_locomotion`, and `fidelity.json` therefore has
`locomotion: false` and `gait_r: null`. The pet clips in `samples/pets/` and
`copypaw/selftest/` are copies of the same source; there is no other gallery clip.
Gait fidelity and matching an individual pet's gait have not been demonstrated.

The sample does measure paw-contact timing. The light table's generated record,
[`sheet.json`](../web/src/lightTable/sheet.json), reports `contactMatch` from the
corrected rendered footfalls and `uncorrectedContactF1` from the joint motion
before foot planting. Neither is a gait score or a pose-accuracy measurement.
Regenerate these with `uv run python scripts/build_light_table.py --name camel`.
The capture, rig and animate commands in the README also write `footfall.json`
and `fidelity.json` for this sample.

No gallery measurement has been generated in this checkout. `scripts/eval_motion.py`
writes this section from `motions/` and the stored drawing renders, and fails
clearly when either is absent.

## Pose

No pose measurement has been generated in this checkout. `scripts/eval_pose.py --apt36k DIR`
writes this section from a local copy of the APT-36K validation split, and fails clearly
when the annotations are absent. The dataset is never committed.

## What the speed table includes

Per-frame stages exclude model opening and video decoding. Capture pipeline rows include both,
identity, smoothing, contacts and motion exports; they exclude CLI startup and cache checksum checks.
Render includes the animate command, video/GIF encoding,
footfalls and fidelity exports. End to end runs capture → rig → animate with verified cached models;
it excludes model downloads. Gallery rendering times `render_character` with frame read-back,
excluding sheet composition and video/GIF exports, and reports the slowest available gallery motion.
It reports `n/a` when a rigged gallery drawing or motion files are absent; that stage is implemented.
