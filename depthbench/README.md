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

## Running

    uv run python -m depthbench.cli manifest --out depthbench/data/manifest.json
    uv run python -m depthbench.cli setup   --model <name>     # builds its venv
    uv run python -m depthbench.cli run     --model <name>
    uv run python -m depthbench.cli score   --out reports/depthbench.md
