# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-12

### Added

- Clip capture with YOLOX and ViTPose on the CPU, identity correction, smoothing,
  paw contacts, ground estimation and BVH/motion/quality exports.
- Drawing segmentation and a draggable quadruped rig; contact-aware animation
  with treadmill and travel modes, video/GIF, footfall and fidelity exports.
- A local browser app and a browser-only gallery animation path using a worker
  and WebGL2, plus the light-table example built from a real pipeline run.
- Verified model downloads, runtime diagnostics and a packaged sample-pair self-test.
- Built web assets in the Python wheel and source distribution.

### Changed

- A drawing is animated with its own leg shape: planted legs stay as drawn and
  swing from the hip, contact paws stand on the ground without bending the
  leg, and a tail drags instead of sinking through the ground.
- The rendered drawing fills its panel for the whole motion, viewed straight on.

- Leg bones stay straight and bend only at the knee; a tail sways behind the
  hind legs; in travel mode the drawing advances by its own planted paws.
- A drawing whose leg joints are a guess keeps its legs as drawn instead of
  shearing.

### Fixed

- The drawing's outline is kept at full weight and darkness; the mask stopped
  inside the stroke and the texture was sampled without mipmaps.
- Body, neck and hip joints are placed inside the outline, so the body and
  head follow the motion.
- `copypaw rig` detects which way a drawing faces instead of assuming right.

