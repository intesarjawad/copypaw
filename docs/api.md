# Public API

Two packages are public: `copypaw.capture` and `copypaw.motion`. The names each
package lists in its `__all__` are the whole public surface, and they are listed
below. Everything else, including every other module and every name that starts
with an underscore, is internal and can change in any release.

## Versioning promise

The public names follow [semantic versioning](https://semver.org/spec/v2.0.0.html)
from 0.1.0. While the version is 0.x, a breaking change to a public name ships only
in a minor release (0.1 to 0.2), never in a patch release, and the
[CHANGELOG](../CHANGELOG.md) announces it. From 1.0 on, breaking changes need a major
release. Adding a name, or a keyword argument with a default, is not a breaking change.

`tests/unit/test_public_api.py` compares each package's `__all__` with the lists on this
page, so a rename or a removal fails CI until this page and the CHANGELOG change with it.

The command line, the JSON in `quality.json` and `fidelity.json`, and the local app's
HTTP routes are not covered by this promise. The `motion.npz` file carries its own
`version` field, and `MotionFile.load` refuses a version it does not read.

## `copypaw.capture`

Extract a motion from a clip.

- `Frames`: sampled RGB frames with their presentation times.
- `NoAnimalFoundError`: raised when fewer than half the frames hold a detected animal.
- `PoseStage`: the callable type of a pose estimator bound to its model files.
- `QualityReport`: joint coverage, interpolated frames, corrected swaps, warnings and errors.
- `VideoReadError`: raised when a clip cannot be opened or decoded.
- `capture_motion`: write `motion.npz`, `motion.bvh` and `quality.json` for a clip's frames.
- `load_stages`: open the detector for a mode and bind the pose estimator to its models.
- `read_frames`: sample a clip's frames by presentation time.

## `copypaw.motion`

Store a motion and write it as BVH.

- `BONES`: the skeleton's bones.
- `Bone`: one bone, named for its distal joint.
- `BvhSummary`: frame count, frame time and scale of a written BVH.
- `MOTION_FILE_VERSION`: the archive version this release reads and writes.
- `MotionFile`: a captured motion, with `save` and `load`.
- `MotionFileError`: raised when an archive is unreadable or invalid.
- `write_bvh`: write planar keypoints as a BVH file.
