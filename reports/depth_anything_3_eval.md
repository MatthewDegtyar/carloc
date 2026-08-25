# Depth Anything 3 as a range sensor — evaluated and rejected

**Question:** the iPhone's LiDAR dies at ~5 m. Could a monocular depth model
(`depth-anything-3`, `DA3METRIC-LARGE`, 334M params) give metric range at 150 m
instead?

**Answer: no, and it is worse than the free baseline already in the repo at every
range.** It is not integrated. This note exists so the question does not get
asked twice.

## Method

307 annotated objects across 11 keyframes of nuScenes `scene-0655`
(boston-seaport), filtered to >=60% visibility, spanning **17–188 m** of true
z-depth. Depth sampled as the median of the central 50% of each ground-truth box,
to avoid reading background through gaps. Real camera intrinsics passed to the
model. Compared against `range/size_prior.py`, which infers range from bounding
box height and a class height prior — no model, no inference cost.

## Result

Median absolute relative error:

| true depth | n | size prior (existing) | DA3 metric | winner |
|---|---|---|---|---|
| 0–25 m | 15 | **10%** | 28% | size prior |
| 25–50 m | 74 | **13%** | 31% | size prior |
| 50–75 m | 84 | **12%** | 54% | size prior |
| 75–100 m | 51 | **13%** | 65% | size prior |
| 100–200 m | 83 | **13%** | 73% | size prior |
| overall | 307 | **13%** | 58% | size prior |

## Why it fails: saturation

The failure is structural, not a tuning problem. What each method predicts as
truth increases:

| true depth | size prior says | DA3 says |
|---|---|---|
| 20–30 m | 23.2 m | 16.8 m |
| 45–55 m | 46.5 m | 26.8 m |
| 95–190 m | **107.5 m** | **34.5 m** |

DA3 compresses everything beyond ~30 m into a narrow band. It cannot distinguish
75 m from 190 m. That is inherent to inferring depth from learned scene context:
the cues saturate with distance. Apparent size does not — it is a clean `1/R`
relationship, which is why the size prior's error stays flat at ~13% across the
whole range rather than growing.

The output is also not metric despite the "metric" checkpoint: global
`median(true/pred) = 2.38`. Even granting an oracle global scale fit — which you
cannot do without already knowing the answer — error only improves to 46%, and
the saturation remains.

## Cost

7.1 s/frame on CPU (M-series, 334M params). The fast-loop budget is 100 ms. Even
a 10x speedup on MPS leaves it ~7x over budget for a result that is 4.5x worse
than a calculation costing nothing.

## Caveats, stated

- Both methods were given ground-truth boxes. That favours the size prior, whose
  input *is* the box height, more than DA3, which only uses the box to choose
  where to sample. With noisy detector boxes the gap would narrow — but not by
  4.5x, and the saturation evidence is independent of box quality.
- Only `DA3METRIC-LARGE` was tested, on one scene, at `process_res=504`.
- Installation needs `--no-deps`: the package requires `xformers`, which has no
  macOS-ARM wheel and fails to build against Apple clang.

## Where a monocular model would still earn its place

Not as a range sensor at distance, but:

- **Objects with no known size class.** The size prior returns *invalid* for
  anything not in its height table; a depth model returns something for anything.
- **Dense depth** for free space and ground-plane estimation, rather than
  per-object range.

Both are close-range uses, and neither is on the critical path. **The thing that
actually delivers range at 150 m is perpendicular baseline** — 10 m of sideways
motion gives ~4% range error at 150 m, and it costs nothing but walking.
