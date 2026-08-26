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

## On-device (iOS / CoreML)

`scripts/export_metric3d_coreml.py` converts Metric3D to CoreML. No published
build exists because conversion hits four blockers; the last looks architectural
and is not. Full write-up in that file. The key one:

> Core ML caps tensors at rank 5, and RAFT convex upsampling builds rank 7 --
> `mask.view(N, 1, 9, f, f, H, W)`. It only splits the two upsample axes so it can
> recombine them into the output grid at the end, which is exactly what
> `pixel_shuffle` does. Rewritten within rank 5 and verified bit-identical.

Converted ViT-small, scored on the same 324 objects:

| | ViT-large PyTorch | ViT-small CoreML |
|---|---|---|
| median | 1.33 m | **1.16 m** |
| delta<1.25 | 95% | 95% |
| usable to | 50 m | 50 m |
| p90 | 3.83 m | 5.92 m |
| 40-50 m | 2.81 m | 5.85 m |
| size | 1.6 GB | **72 MB** |

**The Neural Engine will not take it.** CPU_ONLY 442 ms vs ALL 421 ms -- the ANE
contributes nothing, because `unfold` and the iterative decoder are rejected op by
op. So it is a ~420 ms CPU model: outside a per-frame perception budget, inside a
500 ms ranging loop. For ANE-resident depth, Apple ships CoreML Depth Anything V2
(25 ms, 50 MB) -- faster, but the sub-15 m specialist.

## Running

    uv run python -m depthbench.cli manifest --out depthbench/data/manifest.json
    uv run python -m depthbench.cli setup   --model <name>     # builds its venv
    uv run python -m depthbench.cli run     --model <name>
    uv run python -m depthbench.cli score   --out reports/depthbench.md
