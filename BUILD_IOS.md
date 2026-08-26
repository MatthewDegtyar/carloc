# iOS build plan

Goal: geolocated tracks with honest uncertainty, running live on an iPhone 16 Pro.
Depth inference every few seconds to start, with detection and filtering running
continuously alongside it.

Everything below carries its provenance. **Measured** means measured, on the
hardware named. **Estimated** means derived from a measurement plus a published
spec ratio, and is bracketed. The first milestone exists to replace the estimates
with device numbers before anything is built on top of them.

---

## The budget

Three loops, as in the Python pipeline. The whole point is that nothing blocks on
the slow thing.

| loop | cadence | work | M2 measured | iPhone estimate |
|---|---|---|---|---|
| **capture** | 60 Hz | ARKit pose + pixel buffer | — | free (ARKit does it) |
| **perception** | 10 Hz | YOLO11n detect | 11 ms | ~5 ms |
| **fuse** | 10 Hz | bearings, association, EKF, batch refine | 11 ms | ~11 ms (CPU) |
| **ranging** | **0.3 Hz** | Metric3D depth | 116 ms @448x784 | ~71 ms (52-194) |
| **decision** | 0.2 Hz | surfacing policy | <1 ms | <1 ms |

At 0.3 Hz the depth model is a **2-6% duty cycle**. That is the reason to start
there: it makes the hardest component the least likely to be the problem, so the
first build measures the things nobody has measured yet (ARKit pose quality,
thermals, sustained throughput) rather than re-measuring inference.

Per-second load at these cadences: perception 10x5 = 50 ms, fuse 10x11 = 110 ms,
ranging 0.3x71 = 21 ms. Roughly **180 ms of work per second**, ~18% of one core's
wall clock, spread across three queues. There is headroom; the risk is thermal and
scheduling, not raw throughput.

## What depth is actually for

Worth being explicit, because it decides how hard to push the depth budget.

**Triangulation is the primary range source** and it is far better: measured
sub-1% of range at 100-200 m on real nuScenes data, against Metric3D's 1.16 m
median inside 50 m. It costs nothing but lateral motion, and `sigma_R = sqrt(2)
R^2 sigma_theta / B` says 10 m of sideways walk buys ~4% range accuracy at 150 m.

Depth earns its place in exactly two cases:

1. **Degenerate geometry** -- the operator is walking straight at the subject, so
   there is no perpendicular baseline and triangulation cannot converge. The
   pipeline already detects this (recall 1.00 on the forward-motion scenario).
2. **Track initialisation** -- a depth prior places a new track within a factor of
   two immediately, instead of waiting several frames for parallax.

So depth is a *fallback and a prior*, not the main estimator. Running it at 0.3 Hz
is not a compromise forced by the hardware; it is what the role calls for.

---

## Milestone 0 -- measure before building

**Do this first. It is two minutes of work and it settles three open questions.**

Xcode's Core ML Performance Report runs an `.mlpackage` on a connected device and
reports per-op device placement and real latency.

- [ ] Run `m3d_small_448x784.mlpackage` and `m3d_small_chunked_616x1064.mlpackage`
      through it on a physical iPhone 16 Pro
- [ ] Run `yolo11n.mlpackage` through it
- [ ] Record: latency, and the fraction of ops landing on the Neural Engine

**Why it matters:** the iPhone estimates in this document are brackets, not
numbers. The A18 Pro has 2.2x the ANE compute of the M2 I benchmarked on but only
0.6x the memory bandwidth, and the workload is measurably memory-bound at high
token counts. Those ratios pull in opposite directions: 448x784 could be 52 ms or
194 ms. Everything downstream is sized off that number.

**Accept:** real latency for all three models, and a decision on which depth
resolution to ship.

## Milestone 1 -- capture and pose

No inference at all. Prove the input is trustworthy first.

- [ ] `ARSession` with `ARWorldTrackingConfiguration`, 60 Hz
- [ ] Per frame: `camera.transform` (pose), `camera.intrinsics`, `capturedImage`
- [ ] Convert ARKit pose to the pipeline's ENU convention
- [ ] Log pose, intrinsics and timestamp to disk in the Stray Scanner layout
- [ ] Run `scripts/validate_capture.py` against the result

**The axis conversion already exists and is tested** -- see
`src/geoloc_agent/io/stray_scanner.py`. ARKit is y-up with the camera looking down
its own -z; ours is z-up looking down +z. Getting it wrong produces a
self-consistent and completely incorrect map, which is why the validator checks it
explicitly rather than trusting it.

**Accept:** `validate_capture.py` passes, including the motion check -- it fails
below 35% perpendicular motion, which is the single capture mistake that wastes a
session.

## Milestone 2 -- detection on-device

- [ ] `yolo11n.mlpackage` via Vision or direct Core ML
- [ ] **`CVPixelBuffer` input, not `MLMultiArray`.** ARKit hands back a pixel
      buffer; converting it to a float array per frame is a pointless copy of
      several MB. Re-export with `ct.ImageType` if the current model does not
      take one.
- [ ] Letterbox, not squash -- see `CoreMLDetector._letterbox`. A 1.78x vertical
      stretch is not what the network was trained on, and object *height* is what
      the size prior reads.
- [ ] Frame-drop policy: always process the newest frame, discard anything queued
      behind it. Never let a backlog build.

**Accept:** sustained 10 Hz detection with no frame backlog, thermal state nominal
after 5 minutes.

## Milestone 3 -- the filter in Swift

This is the largest port and the one with no shortcuts. Roughly 1,500 lines.

| module | ports? | notes |
|---|---|---|
| `contracts.py` | yes | structs; keep the validation, it catches convention bugs |
| `geometry.py` | yes | `bearing_from_pixel` is load-bearing -- port its unit tests too |
| `range/triangulation.py` | yes | 3x3 solves; `simd` or Accelerate |
| `range/size_prior.py` | yes | trivial arithmetic |
| `fuse/ekf.py` | yes | 3x3 EKF, Joseph form |
| `fuse/tracker.py` | yes | association needs a Hungarian implementation |
| `fuse/degenerate.py` | yes | small, and it is what keeps the output honest |
| `eval/`, `depthbench/` | **no** | stays in Python, offline |
| `agent/` | later | policy is ~200 lines; the LLM path is not on-device |

Port the **tests** alongside, not afterwards. The synthetic scenarios in
`io/synthetic.py` need no camera and would run in a unit test target -- the same
fixtures caught a transposed rotation, an overconfident filter, and a
self-reinforcing class posterior in the Python build.

**Accept:** the same synthetic scenario produces the same track positions in Swift
as in Python, to within float tolerance.

## Milestone 4 -- depth on the ranging loop

- [ ] Metric3D at the resolution Milestone 0 selected
- [ ] Fire every ~3 s, on the newest frame, on its own serial queue
- [ ] Feed results in as `range_prior` for new tracks, and as a `RangeMeas` only
      where geometry is flagged degenerate
- [ ] Never block perception or fuse on it

**Do not fuse depth as a measurement everywhere.** It is worse than triangulation
wherever there is baseline, and folding it in every frame would drag good fixes
toward a weaker estimator. The prior/measurement distinction is already in the
contracts (`Observation.range_prior` vs `Observation.range`) for exactly this
reason.

**Accept:** depth inference never delays a detection; tracks in degenerate
geometry converge faster than without it.

## Milestone 5 -- thermal behaviour

The one that decides whether it is usable in the field, and the one a Mac
benchmark cannot tell you anything about.

- [ ] Observe `ProcessInfo.processInfo.thermalState`
- [ ] Back off on `.serious`: drop ranging to 0.1 Hz, perception to 5 Hz
- [ ] Stop ranging entirely on `.critical`; keep detection and filtering
- [ ] Log thermal state alongside tracks so a degraded run is identifiable later
- [ ] Measure: time to `.fair`, to `.serious`, and steady-state cadence

**Accept:** 20 minutes of continuous operation without reaching `.critical`, and
graceful degradation rather than dropped frames when it warms.

---

## Risks

**Thermal, and it is the top risk.** Sustained ANE load on a phone is a different
proposition to the same load on a laptop. Mitigation is the low ranging cadence,
which is why the baseline starts at once every few seconds rather than pushing for
frame rate.

**ARKit pose quality is unmeasured.** Every accuracy number in this project so far
used nuScenes ego-pose, which is far better than handheld VIO. ARKit drift over a
10 m walk is perhaps 1-2%; at a 2 m baseline that is 10% of the baseline and
propagates straight into range. Longer lateral walks help twice over. Milestone 1
exists to find out before it is load-bearing.

**Association, not geometry, is the dominant error source.** Measured on real
data: pure tracks 1.4 m median, impure 17 m, worst case one track absorbing
observations from 14 distinct objects. Epipolar gating, a size prior and a class
penalty took the worst case from 948 m to 118 m, and it is still the leading
error. A busy street will be worse than nuScenes, not better.

**The iPhone may be slower than the Mac at full resolution.** 2.2x compute against
0.6x bandwidth on a memory-bound workload. Milestone 0 settles it.

## Open questions

- Real device latency for all three models (Milestone 0)
- ARKit VIO drift over a realistic capture (Milestone 1)
- Sustained thermal envelope (Milestone 5)
- Whether 448x784 depth (2.64 m median, 20 m usable) is enough, or whether the
  50 m config is needed despite costing ~4x
