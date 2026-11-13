# How Copypaw compares

Checked from the linked primary pages on 2026-11-12. Each of these does something Copypaw does not.

| | What it does | What it asks of you | What it does better |
| --- | --- | --- | --- |
| [AnimatedDrawings](https://github.com/facebookresearch/AnimatedDrawings) (open source, MIT) | Animates a child's drawing, mainly of a human figure, from any BVH motion file; has quadruped example configs | Python 3.8 and conda, plus TorchServe to rig a new image; archived on 2025-09-03 | Trained detection and pose models for human figures, a bigger motion library and the paper behind it. Copypaw vendors its engine. |
| [Meta's Animated Drawings demo](https://sketch.metademolab.com/) | A browser demo with preset animations for an uploaded drawing | Nothing to install | Zero setup and a polished flow. Its page currently serves a JavaScript app shell; its interactive flow and current terms were not reverified. |
| [Viggle](https://viggle.ai/pricing) | Hosted generative character and motion video | Free credits refresh daily (the free amount is unspecified), watermarked output and 7-day storage; Pro is listed at $7.99 a month | Generates new video of any character, including 3D worlds, and needs no local compute. It redraws pixels, so strokes are not preserved. |
| [Procreate Dreams](https://procreate.com/dreams) | Frame-by-frame and keyframe 2D animation, with video import | $12.99 once, iPad only | A full animation studio with hand control over every frame. It does not copy a pet's motion for you. |

Copypaw covers one narrow job: a four-legged drawing that keeps its strokes and takes its motion from a pet clip,
on a CPU. Its code is MIT; detector-weight licensing is qualified below. The wider related-work list is in [docs/how_it_works.md](docs/how_it_works.md).
