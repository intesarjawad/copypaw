# Data and licences

Code is MIT ([LICENSE](LICENSE)); the vendored AnimatedDrawings engine (MIT, Meta Platforms) and the rtmlib subset
(Apache-2.0) keep their own licences, listed in [NOTICE](NOTICE). `copypaw models fetch` downloads detector and pose weights separately;
they are not covered by Copypaw's MIT licence. The pinned [ViTPose model card](https://huggingface.co/JunkyByte/easy_ViTPose/blob/e83805274e89428969355ec4afffcbc413e79188/README.md) points to
Apache-2.0. YOLOX's [repository licence](https://github.com/Megvii-BaseDetection/YOLOX/blob/e1052df71842031413f6030723c3607b839c80ce/LICENSE) publishes Apache-2.0 terms, but its [0.1.1rc0 release](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0) does not
explicitly state terms for the exact `yolox_m.onnx` and `yolox_tiny.onnx` assets. Their weight
licence remains unconfirmed; this is not a claim that the full downloaded pipeline has an
open licence. See the dated primary-source record in [ADR-001](docs/adr/ADR-001.md#weight-identity-and-licences). The self-test clip `dog_walk.mp4` is CC BY 2.5, credited in
[samples/ATTRIBUTION.md](samples/ATTRIBUTION.md). The web page's typefaces, BBH Bogle and Mozilla Headline, are under the
SIL Open Font License 1.1. Photos and clips you process stay on your machine.
