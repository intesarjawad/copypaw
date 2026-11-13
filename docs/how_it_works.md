# How it works

Copypaw copies the motion of a pet in a side-view clip onto a drawing of a
four-legged animal. It keeps the drawing's own strokes: the drawing is deformed,
not redrawn. The whole path runs on a CPU.

The stages are, in order: 2D pose, limb identity, a planar BVH, contact-aware
retargeting, and a check of how well the drawing's gait matches the clip's. Where
this page names a threshold, it is the value in the code at this revision. Where a
choice is not yet measured, the page says so and points to the
[architecture notes](architecture.md) and the [ADRs](adr/ADR-001.md).

## 2D pose

`copypaw.capture` reads the clip with OpenCV, which applies the rotation flag of
portrait clips. It samples frames by presentation time at up to 30 frames per second
and stops at 8 seconds.

1. **Detect and track.** YOLOX finds boxes. Only the COCO animal classes count, so a
   person never wins the largest-box choice. The tracker keeps the box that overlaps
   the previous one by at least 0.3 (intersection over union), reuses a box for up to
   three missed frames, and fails with a message if fewer than half the frames hold the
   animal.
2. **Estimate pose.** ViTPose, trained on APT-36K, runs through ONNX Runtime on the CPU
   and returns 17 joints in the AP-10K order. By default it also runs a mirrored pass
   and averages the heatmaps (the flip test). The `lightweight`, `balanced` (default)
   and `accurate` modes pick the detector and model size. See [ADR-001](adr/ADR-001.md).
3. **Clean.** Joints below a 0.3 score are treated as missing, filled by linear
   interpolation in seconds, and smoothed with a short polynomial filter over 2/15 of a
   second. The filter is limited so that gait-rate motion keeps at least 90% of its
   amplitude. Quality is measured on the raw scores before this step, so filling cannot
   hide gaps.

The 2D choice is deliberate: a side-view clip already holds the bone angles, the depth
order and the root movement that the drawing engine uses, and a 3D pipeline would add
cost and rig requirements for no v1 need. The price is that a front or rear view
loses legs; see [filming_guide.md](filming_guide.md).

## Identity

A pose model can swap left and right labels when two legs cross, and on a side view
left and right are not anatomical anyway. `resolve_identity` treats each limb pair
as two observed chains, each frame either keeping or swapping them, and finds the
labelling of the whole clip that costs least. It does this exactly with a two-state
Viterbi pass, not a greedy frame-by-frame choice.

Positions cannot decide a symmetric crossing, so the cost compares velocities fitted on
clear spans before and after each ambiguous interval, and uses positions and scores only
to place frames inside the interval. The full argument and the noise estimates are in
[ADR-006](adr/ADR-006.md). The step also finds which way the animal faces and mirrors
the keypoints so that every stored motion faces right. The number of corrected swaps is
reported in `quality.json`.

## Planar BVH

The motion file `motion.npz` stores the cleaned keypoints, their scores, the frame rate,
the original facing, the paw contacts, the ground line and the root's horizontal
travel. `motion.bvh` is derived from the same data.

The BVH has a 14-bone skeleton. Each bone has a driver joint above it with rotation
channels. The pose is planar, so near-side limbs are offset by 0.08 body lengths toward
the camera and far-side limbs by the same amount away, which gives the engine a depth
order. Lengths are in body lengths (the median Root to Neck distance), and a rest
direction is the average of the per-frame unit vectors so that a long observation does
not outweigh a short one. When a drawing is animated, `bake_bvh` rebuilds the file with
that drawing's own leg lengths.

## Contacts and root travel

A paw is in contact when it is within 0.06 body lengths of the ground and moving slower
than 0.3 body lengths per second, vertically and forward. The ground is a line in image
space, fitted from the confident, vertically quiet paws across the clip. Forward swings
are rejected, runs shorter than two frames are dropped and one-frame gaps are filled.
These thresholds are initial values; the sweep that would tune them is not done
([ADR-005](adr/ADR-005.md)).

Root travel comes from planted paws: the root's displacement minus the paw's cancels
camera translation, so the camera need not be perfectly still. The default **treadmill**
mode removes that travel and keeps the vertical bob, so the drawing stays in frame.
**Travel** mode keeps it.

## Standing on the ground

The engine would turn each drawn bone to the direction of the pet's bone, which gives a
drawing the pet's anatomy as well as its gait. `bake_bvh` therefore writes the motion the
drawing's own legs would make: a planted leg keeps its drawn shape and swings from the hip,
and a lifted leg bends at the knee only to clear the ground as the pet's paw does.

A hook between the engine's joint positions and its as-rigid-as-possible solve sends each
frame to `copypaw.animate.contact_ik`. It lowers, raises and tips the body until the
contact legs reach the ground, then slides each contact leg at its hip so the paw stands on
its own ground line without the leg bending. Analytic two-bone IK handles a paw that drags
out of reach. The hook adds handles along and beside the leg bones, past the neck and on a
tail, so limbs stay straight and bend at their joints, and a tail sways behind the hind legs
without sinking. In travel mode the drawing advances by its own planted paws. A drawing
whose leg joints are a guess keeps its legs as drawn and only bobs. The camera is then placed
so the whole motion fills the frame. The reasons are in [ADR-005](adr/ADR-005.md), and the
engine changes are recorded in the project's `NOTICE`. Rendering runs in a spawned process
because OpenGL needs the main thread; see [ADR-004](adr/ADR-004.md).

## Evaluation

`copypaw.evaluate` measures how well the drawing follows the clip. It uses the drawing's
uncorrected joints, because foot planting is already conditioned on the clip's contacts:

- **Uncorrected contact F1 (`fidelity.json`).** Footfall onsets from uncorrected retargeted joints,
  matched within one analysis frame and averaged over paws.
- **Displayed contact F1 (`sheet.json` / `contactMatch`).** The same onset metric computed on
  the corrected render's `footfall.json` tracks, cropped to the displayed frames. This measures
  the displayed correction, not independent gait accuracy. `uncorrectedContactF1` retains the diagnostic.
  The sample is 0.5833 for displayed corrected tracks and 0.35 before correction; it has no gait score.
- **Duty factor error.** The mean difference in the fraction of time each paw is planted.
- **Gait correlation.** The Pearson correlation of leg-length-normalised paw heights,
  and the period error from the first autocorrelation peak. Both are reported only when
  every paw steps at least twice while the root advances fast enough.

A correlation turns into plain words ("strong", "fair" or "weak") by comparison with the
95th percentile of the correlation between mismatched clip and drawing pairs. That
percentile is written from the gallery; in this checkout the calibration file is empty, so
no words are produced. Pose accuracy (OKS average precision and PCK) is measured on APT-36K
by `scripts/eval_pose.py`, and [evaluation.md](evaluation.md) records only results that
were run. The held-out check is the pre-registered [field test](field_test.md).

## Failures

What goes wrong, and what the code does about it:

- **No animal.** Fewer than half the frames hold a detection: the run stops with a
  message to film from the side and keep the pet in view.
- **Poor coverage.** If fewer than 12 of 17 joints are confident in more than 30% of
  the frames, `quality.json` carries a warning. A required landmark (neck, tail root, any
  paw) that is never confident is listed as an error.
- **Identity swaps.** Corrected swaps are counted. An ambiguity with no clear span on
  either side falls back to the adjacent-frame distance and can still be wrong.
- **Bad pages.** A drawing that is too small, too large, or holds more than one creature
  stops the rig with a message and a link to examples. See `samples/worst_case/`.
- **Not locomotion.** Clips without repeating steps get no gait score and no timing words.
- **Unmeasured.** Pose on a fixed-camera trot, rendered motion quality on the walking
  sample, the contact thresholds, the browser codecs and real iPhone HEVC clips are
  recorded as unverified in [ADR-005](adr/ADR-005.md), [ADR-007](adr/ADR-007.md) and
  [verification.md](verification.md).

## Related work

I describe each work below from its title and abstract; I have not run any of them.
The question for each is what it does differently from Copypaw, which keeps the strokes
of a 2D drawing, takes its motion from a pet clip, runs on a CPU and is open source.

| Work | What it does | What differs |
| --- | --- | --- |
| [AnimatedDrawings](https://github.com/facebookresearch/AnimatedDrawings) (Smith et al., ACM Transactions on Graphics, 2023; MIT licence) | Animates a child's drawing of a human figure by deforming it with motion-capture BVH files. Copypaw vendors its deformation and rendering engine ([ADR-002](adr/ADR-002.md), `NOTICE`). | Its motion comes from a BVH library, not a clip. Copypaw adds pose from video, limb identity, a four-legged rig fitted to the drawing and contact IK. |
| SketchAnim (Rai, Gupta and Sharma, Computer Graphics Forum, 2024; [doi:10.1111/cgf.15176](https://doi.org/10.1111/cgf.15176)) | Real-time transfer of motion from a video onto a sketch. | I have not read its pipeline closely enough to compare it point by point; see the comparison procedure below. |
| Video-Driven Sketch Animation via Cyclic Reconstruction Mechanism (Xie, Mo and Gao, ICME 2024; [doi:10.1109/ICME57554.2024.10688206](https://doi.org/10.1109/ICME57554.2024.10688206)) | Animates a sketch from a driving video, using a cyclic reconstruction mechanism. | As above: no side-by-side has been made. |
| Kirin (ECCV 2026; [arXiv:2609.01823](https://arxiv.org/abs/2609.01823)) | Reconstructs 3D animal motion from in-the-wild video, learns motion priors from it, generates motion from text and an image, and applies it to 3D meshes that an image-to-3D model rigs. Its [project page](https://kirin-ani.github.io/) links [released training and inference code](https://github.com/briannlongzhao/kirin). | Targets rigged 3D meshes rather than deformation of a 2D drawing's strokes. |
| TopoCap (SIGGRAPH 2026; [arXiv:2606.12153](https://arxiv.org/abs/2606.12153)) | Extracts motion from monocular video and retargets it to 3D characters with arbitrary, unseen skeletons, with no test-time optimization. | Targets 3D rigs. Copypaw targets one fixed 2D quadruped template and keeps the drawing's strokes. |
| Video motion-transfer generators | Synthesize new video frames of a subject that follows a reference clip's motion. | They generate pixels, so the drawing's strokes are redrawn, not kept, and they commonly need a GPU. |

Kirin's project page and linked code were rechecked on 2026-11-12. The other paper
summaries retain the 2026-11-07 abstract check; none is a measured side-by-side.
Copypaw's own code is MIT. The downloaded detector weights' exact licensing is still
unconfirmed; see [ADR-001](adr/ADR-001.md#weight-identity-and-licences).

### Comparison with SketchAnim

For any SketchAnim source clip that carries a licence, run this pipeline on that clip and
publish the result with a link to SketchAnim's published figure beside it. The figure is
never reproduced here, so each side stays under its own licence. No such comparison has
been made yet, and none appears on this page until it is.
