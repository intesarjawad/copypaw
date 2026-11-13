# The command line, step by step

The CLI takes a clip of your own pet and writes a side-by-side video and GIF.
Every step below was run on the clips and drawing that ship in `copypaw/selftest/`.

```sh
uv run copypaw capture copypaw/selftest/dog_walk.mp4 -o out/motion
```

```text
Warning: Smoothing is reduced or skipped at sampling rates below 30 Hz to preserve at least 90% of 4 Hz limb motion.
Wrote motion.npz, motion.bvh and quality.json to out/motion
```

```sh
uv run copypaw rig copypaw/selftest/dragon.png -o out/dragon
```

```text
Wrote the rigged drawing to out/dragon
```

`rig` also prints the 16 joints as JSON, and `--joints FILE` replaces the snapped joints with yours.

```sh
uv run copypaw animate out/dragon out/motion/motion.npz --clip copypaw/selftest/dog_walk.mp4 -o out/result
```

```text
 Writing video to: /.../animation.gif
100%|...| 74/74 [...]
Wrote animation and side-by-side videos to out/result
```

The renderer prints its temporary GIF destination and a frame progress bar; the variable
path and terminal progress are abbreviated above. Numerical warnings are not expected.

`out/result` holds `animation.mp4`, `animation.gif`, `side_by_side.mp4`, `side_by_side.gif`, `footfall.json` and `fidelity.json`.
The side-by-side video carries a footfall strip under the frames.
`--mode travel` lets the drawing advance on its own paws instead of the default treadmill.
`capture --mode lightweight|balanced|accurate` picks the detector and pose model pair.

After installing the tool, run `copypaw app` to open the packaged interface on
127.0.0.1 with a session token. A source checkout needs `make setup` to build that interface.
For a good clip, read [docs/filming_guide.md](docs/filming_guide.md).

## Building and checking

`make check` runs web lint, build and serialized web tests (one retry for runner variance) before Python format, lint,
types and tests. `make build` rebuilds the web app and puts it into both the wheel and sdist.
The Hatch hook also builds the app for direct `uv build` and wheels rebuilt from an sdist.
Building from source requires Node.js and npm; installing a wheel does not.
The packaging tests install the wheel in a fresh venv outside the checkout and check that
`copypaw app --no-browser` answers with its built index page.
