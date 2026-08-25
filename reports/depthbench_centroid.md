# Monocular depth benchmark, 0-50 m

74 images, 324 objects, 5.9-49.9 m. Ground truth: nuscenes v1.0-mini scene-0655,scene-0757,scene-0103,scene-0553, real intrinsics, 3-D box annotations.

Scored against **centroid depth m**. Depth models predict the distance to the visible *surface*; a geolocation pipeline wants the object *centroid*. Those differ by about half an object's length -- a median of 1.0 m here -- which is comparable to the errors being measured, so the benchmark computes both and neither is charged to a model silently. Re-run with `--depth centroid` for the other view.

## Results

| model | n | coverage | median abs err | p90 abs err | median rel err | delta<1.25 | rank corr | s/image |
|---|---|---|---|---|---|---|---|---|
| Metric3D v2 (real intrinsics) | 324 | 100% | **1.74 m** | 4.88 m | 7% | 94% | 0.97 | 4.62 |
| YOLO26 depth (yolo26n-depth.pt) | 324 | 100% | **2.98 m** | 14.25 m | 14% | 64% | 0.83 | 0.09 |
| Depth Anything V2 (relative + reference rescale) | 324 | 100% | **3.35 m** | 11.68 m | 16% | 64% | 0.95 | 1.45 |
| Depth Anything V2 (metric-outdoor-large) | 324 | 100% | **4.61 m** | 9.96 m | 20% | 63% | 0.95 | 1.45 |
| YOLO26 depth (yolo26x-depth.pt) | 324 | 100% | **5.37 m** | 15.72 m | 23% | 41% | 0.90 | 0.39 |
| Metric3D v2 (default intrinsics) | 324 | 100% | **5.81 m** | 12.18 m | 25% | 21% | 0.97 | 4.79 |

`delta<1.25` is the standard depth metric: the fraction of objects whose predicted and true depth are within 25% of each other. `rank corr` is Spearman against truth -- if it is near zero the metric error is noise, and a negative value means the depth map was read upside down.

## Usable range

Furthest range at which median error stays within 2 m or 15%, and the range beyond which predictions start *decreasing* as true depth increases.

| model | usable to | reverses at |
|---|---|---|
| Metric3D v2 (real intrinsics) | 50 m | never |
| YOLO26 depth (yolo26n-depth.pt) | 30 m | **50 m** |
| Depth Anything V2 (relative + reference rescale) | 15 m | **50 m** |
| Depth Anything V2 (metric-outdoor-large) | -- | never |
| YOLO26 depth (yolo26x-depth.pt) | 15 m | **50 m** |
| Metric3D v2 (default intrinsics) | 10 m | never |

Reversal is worse than saturation and is the reason both are reported. A saturating model still orders objects correctly, so a downstream filter can rank them; a reversing one reports distant objects as nearer than close ones, and there is nothing left to use.

## Error by range

Median absolute error in metres:

| model | 5.0-10.0 m | 10.0-15.0 m | 15.0-20.0 m | 20.0-25.0 m | 25.0-30.0 m | 30.0-35.0 m | 35.0-40.0 m | 40.0-45.0 m | 45.0-50.0 m |
|---|---|---|---|---|---|---|---|---|---|
| Metric3D v2 (real intrinsics) | 0.59 | 1.09 | 1.10 | 1.67 | 1.94 | 2.44 | 2.79 | 2.76 | 4.22 |
| YOLO26 depth (yolo26n-depth.pt) | 0.64 | 1.48 | 2.13 | 2.93 | 3.70 | 5.56 | 9.14 | 6.46 | 26.15 |
| Depth Anything V2 (relative + reference rescale) | 0.57 | 1.84 | 2.82 | 3.67 | 4.09 | 6.24 | 7.86 | 7.44 | 18.93 |
| Depth Anything V2 (metric-outdoor-large) | 7.14 | 3.32 | 4.25 | 4.69 | 5.15 | 3.90 | 3.12 | 3.56 | 4.33 |
| YOLO26 depth (yolo26x-depth.pt) | 0.65 | 1.91 | 3.81 | 5.62 | 6.41 | 8.35 | 10.70 | 11.19 | 25.69 |
| Metric3D v2 (default intrinsics) | 1.03 | 3.43 | 4.38 | 5.78 | 6.88 | 8.64 | 10.00 | 10.60 | 13.16 |

## Saturation

The characteristic failure of monocular depth: predictions compress into a narrow band, so the model cannot distinguish near from far even though its average error looks tolerable. `span ratio` is the predicted spread across buckets divided by the true spread -- 1.0 tracks range correctly, near 0 means the answer barely changes with distance.

| model | median pred @0-10 m | @40-50 m | span ratio |
|---|---|---|---|
| Metric3D v2 (real intrinsics) | 8.09 m | 43.52 m | **0.89** |
| YOLO26 depth (yolo26n-depth.pt) | 7.09 m | 22.27 m | **0.38** |
| Depth Anything V2 (relative + reference rescale) | 6.87 m | 29.86 m | **0.58** |
| Depth Anything V2 (metric-outdoor-large) | 14.39 m | 49.62 m | **0.88** |
| YOLO26 depth (yolo26x-depth.pt) | 7.03 m | 21.77 m | **0.37** |
| Metric3D v2 (default intrinsics) | 6.46 m | 34.74 m | **0.71** |

## Speed

| model | s/image | device |
|---|---|---|
| YOLO26 depth (yolo26n-depth.pt) | 0.09 | cpu |
| YOLO26 depth (yolo26x-depth.pt) | 0.39 | cpu |
| Depth Anything V2 (metric-outdoor-large) | 1.45 | mps |
| Depth Anything V2 (relative + reference rescale) | 1.45 | mps |
| Metric3D v2 (real intrinsics) | 4.62 | mps |
| Metric3D v2 (default intrinsics) | 4.79 | mps |

## Did not run

- **Depth Pro** — cancelled by request -- the official-checkpoint run was too slow to finish (~10 s/image on MPS, 74 images). An earlier full run via the HuggingFace port scored 2.51 m median / 13% relative / delta<1.25 73%, usable to ~30

## What this says about picking one

The overall median hides the thing that decides usability, which is where the model stops tracking range at all. Read the saturation table with the per-bucket table: a model can post a good average by being excellent up close and simply returning a constant beyond 30 m.

## Method

- Each model runs in its own virtualenv as a subprocess, talking to the harness over JSON. They disagree about torch, timm, transformers and numpy versions; one shared environment silently downgrades something and then you are benchmarking the downgrade.
- Depth is sampled as the median over the central 50% of each ground-truth box. A whole box contains background through windows and around outlines, which sits tens of metres further away.
- Sampling is identical for every model, in one shared helper, so the comparison is between models rather than between sampling choices.
- Objects are filtered to >=60% visibility, fully in front of the camera, at least 24 px, and inside 50 m.
- Predictions that come back non-finite are recorded as misses, not dropped, so a model cannot look accurate by declining the hard objects. That is what `coverage` reports.

- **Metric3D v2 (real intrinsics)**: focal used: per-image from manifest; mmcv shim in use (logging only); 0 hard-coded cuda device literals patched for CPU/MPS
- **Metric3D v2 (default intrinsics)**: focal used: 1000 px fixed; mmcv shim in use (logging only); 0 hard-coded cuda device literals patched for CPU/MPS
- **Depth Anything V2 (metric-outdoor-large)**: Metric checkpoint, output used as metres with no rescaling.
- **Depth Anything V2 (relative + reference rescale)**: Inverse-depth assumed; rescaled per image from one known-size object. 0 of 74 images had no usable reference.
- **YOLO26 depth (yolo26n-depth.pt)**: depth read from Results.depth
- **YOLO26 depth (yolo26x-depth.pt)**: depth read from Results.depth
