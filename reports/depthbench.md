# Monocular depth benchmark, 0-50 m

74 images, 324 objects, 5.5-48.9 m. Ground truth: nuscenes v1.0-mini scene-0655,scene-0757,scene-0103,scene-0553, real intrinsics, 3-D box annotations.

Scored against **surface depth m**. Depth models predict the distance to the visible *surface*; a geolocation pipeline wants the object *centroid*. Those differ by about half an object's length -- a median of 1.0 m here -- which is comparable to the errors being measured, so the benchmark computes both and neither is charged to a model silently. Re-run with `--depth centroid` for the other view.

## Results

| model | n | coverage | median abs err | p90 abs err | median rel err | delta<1.25 | rank corr | s/image |
|---|---|---|---|---|---|---|---|---|
| Metric3D v2 (real intrinsics) | 324 | 100% | **1.33 m** | 3.83 m | 7% | 95% | 0.97 | 4.62 |
| Depth Anything V2 (relative + reference rescale) | 324 | 100% | **2.04 m** | 10.78 m | 10% | 76% | 0.95 | 1.45 |
| YOLO26 depth (yolo26n-depth.pt) | 324 | 100% | **2.84 m** | 13.37 m | 14% | 69% | 0.83 | 0.09 |
| YOLO26 depth (yolo26x-depth.pt) | 324 | 100% | **3.85 m** | 14.67 m | 18% | 55% | 0.91 | 0.39 |
| Metric3D v2 (default intrinsics) | 324 | 100% | **4.66 m** | 11.08 m | 21% | 46% | 0.97 | 4.79 |
| Depth Anything V2 (metric-outdoor-large) | 324 | 100% | **5.96 m** | 11.07 m | 27% | 44% | 0.95 | 1.45 |

`delta<1.25` is the standard depth metric: the fraction of objects whose predicted and true depth are within 25% of each other. `rank corr` is Spearman against truth -- if it is near zero the metric error is noise, and a negative value means the depth map was read upside down.

## Usable range

Furthest range at which median error stays within 2 m or 15%, and the range beyond which predictions start *decreasing* as true depth increases.

| model | usable to | reverses at |
|---|---|---|
| Metric3D v2 (real intrinsics) | 50 m | never |
| Depth Anything V2 (relative + reference rescale) | 30 m | **50 m** |
| YOLO26 depth (yolo26n-depth.pt) | 30 m | **45 m** |
| YOLO26 depth (yolo26x-depth.pt) | 15 m | **45 m** |
| Metric3D v2 (default intrinsics) | 15 m | never |
| Depth Anything V2 (metric-outdoor-large) | -- | never |

Reversal is worse than saturation and is the reason both are reported. A saturating model still orders objects correctly, so a downstream filter can rank them; a reversing one reports distant objects as nearer than close ones, and there is nothing left to use.

## Error by range

Median absolute error in metres:

| model | 5.0-10.0 m | 10.0-15.0 m | 15.0-20.0 m | 20.0-25.0 m | 25.0-30.0 m | 30.0-35.0 m | 35.0-40.0 m | 40.0-45.0 m | 45.0-50.0 m |
|---|---|---|---|---|---|---|---|---|---|
| Metric3D v2 (real intrinsics) | 1.02 | 1.21 | 1.14 | 1.33 | 1.04 | 1.57 | 1.90 | 2.06 | 2.96 |
| Depth Anything V2 (relative + reference rescale) | 0.22 | 0.33 | 1.20 | 2.48 | 3.18 | 5.50 | 7.05 | 7.89 | 18.17 |
| YOLO26 depth (yolo26n-depth.pt) | 0.75 | 1.64 | 1.77 | 3.11 | 3.84 | 5.08 | 6.55 | 10.90 | 25.12 |
| YOLO26 depth (yolo26x-depth.pt) | 0.46 | 0.79 | 2.63 | 4.70 | 6.43 | 6.94 | 9.24 | 14.64 | 24.29 |
| Metric3D v2 (default intrinsics) | 0.55 | 1.75 | 2.83 | 5.03 | 5.94 | 7.84 | 9.09 | 9.28 | 11.77 |
| Depth Anything V2 (metric-outdoor-large) | 7.39 | 5.77 | 6.84 | 5.58 | 5.97 | 4.94 | 4.04 | 4.30 | 5.67 |

## Saturation

The characteristic failure of monocular depth: predictions compress into a narrow band, so the model cannot distinguish near from far even though its average error looks tolerable. `span ratio` is the predicted spread across buckets divided by the true spread -- 1.0 tracks range correctly, near 0 means the answer barely changes with distance.

| model | median pred @0-10 m | @40-50 m | span ratio |
|---|---|---|---|
| Metric3D v2 (real intrinsics) | 8.16 m | 43.80 m | **0.90** |
| Depth Anything V2 (relative + reference rescale) | 6.97 m | 29.86 m | **0.58** |
| YOLO26 depth (yolo26n-depth.pt) | 7.20 m | 22.27 m | **0.38** |
| YOLO26 depth (yolo26x-depth.pt) | 7.07 m | 21.77 m | **0.37** |
| Metric3D v2 (default intrinsics) | 6.51 m | 34.96 m | **0.72** |
| Depth Anything V2 (metric-outdoor-large) | 14.32 m | 52.28 m | **0.96** |

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

- **depth_pro** — FileNotFoundError: [Errno 2] No such file or directory: 'sessions/nuscenes/samples/CAM_FRONT/n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385092112404.jpg'

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
