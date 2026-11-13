# Known limits

- A tail drawn down to the paws' level is held lifted away from the legs so it does not read as a fifth leg. It turns only a share of the hind legs' swing and has no motion of its own, so it looks stiffer than a real tail.
- Single-line legs are rigged, but only when they hang below the body, start within 24 px of it and are at least half as long as the body is tall. Shorter stubs and legs drawn as separate floating lines are left out of the mask. The mask thickens each leg to 21 px so the mesh can bend it; the drawn line itself is not changed.
- A drawing that faces left is mirrored for rigging and flipped back after rendering, and the frames are then flipped to face the way the clip does unless you pass `--keep-facing`.
- `fidelity.json` for the sample pair reports `"locomotion": false` and no gait score, so there is no measured gait fidelity for it yet.
  The sample measures paw-contact timing, not how closely the drawing copies a pet's gait.
  The light table's contact score comes from the corrected rendered footfalls; `fidelity.json` measures the uncorrected joint motion separately.
  No eligible gait example exists in the bundled material. See [evaluation](docs/evaluation.md) for the measured scope and reproduction commands.
- Pose accuracy on a fixed-camera trot, real iPhone HEVC clips and Linux rendering are unmeasured ([ADR-005](docs/adr/ADR-005.md), [verification](docs/verification.md)).
- Front and rear views lose legs; film from the side.
