# Worst-case pages

`scripts/make_worst_case.py` writes the synthetic pages in this folder from
fixed geometry, so two runs produce identical bytes. `expected.yaml` records the
segmenter result for each file: an accepted coverage range, or the
`MaskError` kind. Regenerate them with:

```sh
uv run python scripts/make_worst_case.py
```

Real photographs and released drawings are not generated. Place
those files in `real/`.
