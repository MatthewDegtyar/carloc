# depthbench

Monocular depth models, scored in metres against real ground truth, bucketed by
range, out to 50 m.

## Why it is built this way

**Each model runs in its own virtualenv, as a subprocess.** These models disagree
about torch, timm, transformers and xformers versions; one environment holding all
of them either does not resolve or silently downgrades something and you are then
benchmarking the downgrade. Adapters therefore talk over files, not imports:

    manifest.json  ->  runner (own venv)  ->  predictions.json  ->  scorer

The scorer never imports a model. Adding a model means adding a runner and an
environment spec, and nothing else moves.

## Ground truth

nuScenes v1.0-mini: real camera, real intrinsics, 3-D boxes with true dimensions.
Two depths are computed per object and both are reported, because they answer
different questions:

- **surface depth** — the near face of the 3-D box. This is what a depth model is
  actually predicting: the distance to the visible surface. Primary metric.
- **centroid depth** — the object's centre. This is what a geolocation pipeline
  wants. Reported alongside, because the gap between them is a real systematic
  offset (about half an object's length) that would otherwise be silently charged
  to the models.

## Results

Six configurations completed on 74 nuScenes images / 324 objects, 5.5-48.9 m.
`reports/depthbench.md` (surface depth) and `reports/depthbench_centroid.md`.

**Metric3D v2 with real intrinsics is the only model usable across 0-50 m**:
1.33 m median, 95% within 25%, 1.02 m error at 5-10 m rising to only 2.96 m at
45-50 m, and it never reverses.

Two findings worth keeping:

- **Intrinsics are worth 3.5x.** The same Metric3D checkpoint scores 4.66 m on a
  default 1000 px focal against 1.33 m on the true one. Rank correlation is 0.97
  either way, so the model reads the scene correctly and the focal only sets the
  scale.
- **Three models reverse past ~45 m**, reporting further objects as nearer
  (YOLO26n, YOLO26x, DAv2-relative). Worse than saturating: a saturating model
  still orders objects, so a filter can rank them.

Depth Pro was cancelled mid-run; its earlier HuggingFace-port run scored 2.51 m.

## Running

    uv run python -m depthbench.cli manifest --out depthbench/data/manifest.json
    uv run python -m depthbench.cli setup   --model <name>     # builds its venv
    uv run python -m depthbench.cli run     --model <name>
    uv run python -m depthbench.cli score   --out reports/depthbench.md
