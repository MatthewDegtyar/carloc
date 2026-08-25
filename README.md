# geoloc-agent

Turns posed video into geolocated object tracks with honest uncertainty, then
lets an agent decide which of those tracks are worth an operator's attention.

The deliverable is the **framework and the eval harness**. The computer vision is
commodity and is treated as such — there is no detector tuning anywhere in here.

```bash
uv sync --extra plot
uv run pytest                 # 150 tests
uv run geoloc-eval            # -> reports/eval.md + plots + CSVs
uv run geoloc-agent-eval      # -> reports/agent_eval.md
uv run geoloc-render          # -> reports/geoloc_demo.mp4
```

## Video

```bash
uv run geoloc-render                                    # synthetic, two-part
uv run geoloc-render --source nuscenes                   # real imagery + real detector
uv run geoloc-render --source nuscenes --detector truth  # real imagery, oracle detector
```

The real-detector path needs a one-time model export — see `scripts/export_yolo.py`.

### Real data (nuScenes)

Two clips, same scene, differing only in the detector:

- `reports/geoloc_real_yolo.mp4` — **fully real**: real imagery, real ego pose,
  YOLO11n running on the Apple Neural Engine. No ground truth anywhere in the loop.
- `reports/geoloc_real_nuscenes.mp4` — same, but with `TruthProjectionDetector`,
  an **oracle** that projects the annotated 3-D boxes. It exists so geometry and
  filter error can be measured without detector error mixed in.

Keeping both is the point: the difference between them *is* the detector's
contribution, and it is measured rather than asserted.

| detector | detections | false pos | pure tracks | median err | p90 | class acc |
|---|---|---|---|---|---|---|
| truth projection (oracle) | 575 | 6 | 53 | **0.56 m** | 4.84 m | 110/110 |
| YOLO11n (real) | 244 | 24 | 11 | **1.03 m** | 7.86 m | 47/58 |

A real detector roughly doubles median error, triples p90, quadruples false
positives and finds under half as many objects — mostly by missing small distant
ones. Every frame states which detector produced it.

**Perception latency: mean 8.8 ms, p95 11.4 ms** against a 100 ms fast-loop
budget, benchmarked after warm-up (the first inference pays model compilation and
would overstate latency by an order of magnitude). The three-loop budget holds
with room to spare.

### Synthetic

`geoloc-render` also produces a two-part clip: the camera view with class, range and
1-sigma range uncertainty on every box, beside a top-down map with error ellipses.

Part 1 is lateral motion (good geometry) — ranges converge and the ellipses
collapse to centimetres. Part 2 is the same objects, detector and filter under
forward motion — the ellipses stretch into streaks *along the line of sight* and
every box reads `RANGE UNRELIABLE`. Putting them in one file is deliberate: the
well-conditioned result only means something next to the degenerate one.

In the synthetic clip **every pixel is synthesised.** A `SyntheticSession` has no imagery; the ground
grid and object bodies are drawn from true geometry through the real camera
matrix, and the frame is labelled `SYNTHETIC RENDER` throughout. What is *not*
synthesised is the output: boxes come from the same projection the tracker
consumes, class and confidence from the class posterior, and range +/- sigma from
the filter covariance projected onto the line of sight. Range is never shown as a
bare number, and a sigma of 0.03 m prints as `0.030` rather than `0.0` — "0.0 m"
reads as certainty, which is the one claim this system must not make.

## What it does

```
session ──► detector ──► bearings ──► ranger ──► EKF + batch ──► tracks ──► agent ──► CoT / Lattice
(pose)      (fast)       (world)      (middle)   (fuse)          (+cov)     (slow)
```

Three decoupled loops: perception per frame, ranging on a slower cadence,
decision slower still. Nothing blocks on the slow thing.

## The two results worth knowing before trusting any coordinate

**1. Degenerate geometry is a hard limit, not a tuning problem.** A camera moving
straight forward gets almost no perpendicular baseline on anything near its own
optical axis, so range is essentially unobservable. Triangulation still returns
*a* number, and that number looks like every other number in the output.

The system detects this with **recall 1.00** on the `forward_motion` scenario and
reports those tracks with inflated covariance rather than hiding them. Median
error under forward motion is **10.4 m** against **0.10 m** for well-conditioned
geometry — a hundredfold difference that is entirely geometry, not tuning.

**2. A constant heading bias defeats the covariance.** With a 2° yaw error the
median error is 2.5 m while mean NEES reaches **~14,800** against a nominal 3. The
filter is not merely wrong, it is *confidently* wrong, because a constant bias is
unobservable to a filter that models zero-mean noise. This needs an independent
heading reference; no covariance bookkeeping fixes it.

Both are reproduced by `uv run geoloc-eval` and written into `reports/eval.md`.

## Design rules, and where they are enforced

| Rule | Enforcement |
|---|---|
| No bare point estimates | `RangeMeas` rejects a valid measurement without a positive sigma (`contracts.py`) |
| Covariance reaches the agent | `query_tracks` always returns `sigma_horizontal_m` and `cep50_m`; asserted in `test_agent.py` |
| Three decoupled loops | Explicit cadences in `pipeline.py`; no stage reaches into another |
| Config-driven | Scenarios and sweeps are YAML in `configs/`; unknown keys raise rather than being ignored |
| Truth flows through eval mode | `Observation.truth_id` / `truth_position`, `PipelineResult.track_records` |

## Layout

```
src/geoloc_agent/
  contracts.py      Frame, Detection, RangeMeas, Observation, TrackState, Decision
  geometry.py       bearing_from_pixel and the angular bookkeeping around it
  geo.py            local ENU <-> WGS84, and the documented map origins
  noise.py          the injected noise model the sweeps range over
  pipeline.py       end-to-end replay, three loops at explicit cadences
  io/               synthetic | nuscenes | stray_scanner, one interface
  detect/           base | stub (scripted JSON) | coreml (YOLO11n on the ANE)
  range/            triangulation: linear init + ML angular refinement
  fuse/             ekf | tracker | degenerate
  eval/             scenario | metrics | runner | report | cli
  agent/            tools | policy | patterns | backend | eval | report | cli
  viz/              render | cli — pipeline run to mp4/gif
  publish/cot/      Cursor-on-Target to a local TAK server
  publish/lattice/  Lattice SDK — SEPARATE TREE, see licensing below
```

## Notable engineering decisions

**Tracks localise by triangulation, not from a birth prior.** Seeding a bearings-only
track at a nominal range and folding bearings into it makes the filter
catastrophically overconfident: a single update collapsed range sigma from 25 m to
4 m while the error was still 45 m (NEES 250). The cause is that a Cartesian
Gaussian cannot represent "range is somewhere between 5 and 200 m". Tracks now
accumulate bearings and localise from a triangulation whose covariance is
validated against the analytic CRLB (matches to 0.01% in the small-angle regime).

**Triangulation refines on the angular residual, not perpendicular distance.**
The perpendicular-distance objective is convenient but not statistically optimal;
it leaves the estimator's true error ~20-30% above the covariance it reports.
The linear solve is used only as an initial guess for a Gauss-Newton pass on the
actual angular measurement.

**Degeneracy is judged on the linear stage, before refinement.** With near-parallel
bearings the linear solve returns a minimum-norm point near the cameras, and
Gauss-Newton from there converges somewhere arbitrary — where those same widely
spaced cameras subtend a *large* angle. Testing the refined information matrix
would report a tight covariance on exactly the geometry that has none.

**Two of the degeneracy tests do not depend on the estimate.** Observation count
and absolute perpendicular baseline. The covariance-based tests are evaluated at
the estimated position, so a short track that has landed badly wrong assesses its
geometry at that wrong place and concludes everything is fine. Adding the two
estimate-independent gates cut RMSE from 12.3 m to 0.76 m at 2.5 px bearing noise.

**Process noise defaults to zero.** These tracks model static objects, and process
noise on a static object is pure covariance inflation — it drove NEES to 1.36
against a nominal 3, which is the filter discarding information.

## Real-data results (nuScenes v1.0-mini, scene-0655, boston-seaport)

Same `fuse/` code as the synthetic runs, unchanged — the Phase 2 acceptance
criterion. Scored against the annotated 3-D boxes, 41 keyframes, 114 tracks:

| | median error | p90 |
|---|---|---|
| pure tracks, good geometry | **0.56 m** | 4.84 m |
| tracks flagged degenerate | 11.7 m | — |

Getting there surfaced two real problems that synthetic data never exercised:

**Scoring only surviving tracks measured almost nothing.** On a driving scene
objects sweep through the field of view, so `final_tracks` held 3 of 95. The
tracker now retains tracks at death.

**Association, not geometry, was the dominant error.** Pure tracks had median
1.4 m error while impure ones had 17 m and a 948 m worst case — one track had
absorbed observations from **14 distinct objects**. The cause was that an
unlocalised track carries a deliberately wide range prior, which made its
Mahalanobis gate large enough to swallow any nearby detection. Three fixes, all
of which use information the tracker already had:

- **Epipolar gating** for unlocalised tracks. Such a track knows its bearing ray
  exactly; requiring a new bearing to be consistent with *some* range along it is
  a 1-D constraint instead of a loose 2-D one.
- **A size prior** (`range/size_prior.py`) from bounding-box height, which places
  a new track within a factor of two instead of at an arbitrary default. Used for
  initialisation and gating only, never fused as a measurement.
- **A class penalty**, not a class veto. Vetoing made the posterior
  self-reinforcing — a track that drifted to "car" would reject every pedestrian
  observation and could never discover it was wrong. The synthetic test suite
  caught that immediately.

Worst-case error fell from 948 m to 118 m and pure tracks rose from 33 to 63.
Association remains the leading error source and is reported, not hidden.

## Calibration status

NEES is the honesty metric: `eᵀP⁻¹e`, chi-square with 3 DOF, so the mean should
be 3. On pure, well-conditioned tracks the filter measures **3.32, inside the 95%
band**. Pooled over all confirmed tracks it sits at ~4.2 — the excess is
association error, which is a real error source the covariance does not model, so
it is reported rather than tuned away. Track purity is reported alongside.

## Capturing your own (Stray Scanner, iPhone)

A normal iPhone video is useless here: the Camera app records pixels and no pose,
and without per-frame 6-DoF pose there is no geolocation, only detection. Stray
Scanner records ARKit's pose alongside the video.

```bash
uv run python scripts/validate_capture.py sessions/my_capture
```

Run that first. It checks the conventions that fail *silently* — axis handedness,
principal point, intrinsics-vs-video resolution — and the one capture mistake
that quietly wastes the whole shoot:

**Move sideways, not forwards.** Walking toward your subject gives near-zero
perpendicular baseline, and range stays unobservable no matter how long you
record. Strafe across the scene, or arc around it. The validator reports what
fraction of your motion was perpendicular to the look direction and fails below
35%.

Two things a phone capture cannot give you:

- **No ground truth**, so geolocation error is not scoreable. The harness reports
  coverage, convergence and self-consistency for this source, not accuracy it
  cannot measure.
- **No north and no origin.** ARKit is metric and gravity-aligned but starts at an
  arbitrary position with arbitrary yaw. Pass `origin=` and `heading_offset_deg=`
  explicitly; until you do, output is metric-relative and emits no lat/lon.

The iPhone 16 Pro also writes `depth/` (LiDAR), but it is good to roughly 5 m —
useful for close objects, irrelevant at street distance. `RangeMethod.LIDAR`
exists for it and is not yet wired up.

### Working at 150 m

Two separate constraints, and they bind in the opposite order to what you would
expect.

**Geometry is the easy one.** `sigma_R = sqrt(2) R^2 sigma_theta / B`, so range
accuracy is bought with sideways walk. At 150 m with a phone camera:

| sideways walk | 1-sigma range error @150 m |
|---|---|
| 2 m | 21 m (14%) |
| 5 m | 8.5 m (5.7%) |
| 10 m | 4.2 m (2.8%) |
| 20 m | 2.1 m (1.4%) |

**Detection is the hard one.** A person at 150 m is 17 px tall in a 1920-wide
frame. A 640-input detector downscales it 3x, so it arrives at under 6 px and is
simply not there. No ranging cleverness recovers an object that was never detected.

`detect/tiled.py` fixes it by running the same detector over overlapping
*native-resolution* crops, so a 17 px object stays 17 px. Measured on nuScenes:

| | detections >50 m | 125-150 m | latency p95 |
|---|---|---|---|
| full frame | 96 | 8 | 8.1 ms |
| tiled | **227** | **24** | 29.4 ms |

Tiles are restricted to a band around the horizon, because that is where distant
objects image — 3 tiles instead of 12, which is what keeps it inside the 100 ms
fast-loop budget. The full-frame pass still runs, so near objects are unaffected.

With tiling, geolocation error at 100-200 m came out at **under 1% of range**
(0.46-0.94 m median) — though on only 3 tracks per bin after the purity filter,
so read that as indicative rather than precise.

**Monocular depth was evaluated and rejected.** `depth-anything-3`
(`DA3METRIC-LARGE`) was tested against 307 annotated nuScenes objects spanning
17-188 m. It is worse than the free bbox-height size prior at *every* range (58%
vs 13% median error overall) because learned depth cues saturate: it predicts
34.5 m for objects genuinely 95-190 m away, and cannot distinguish 75 m from
190 m. Apparent size is a clean 1/R relationship and does not saturate. Full
writeup and caveats in `reports/depth_anything_3_eval.md`.

**How far can a given camera actually work?**

```bash
uv run python scripts/range_envelope.py --capture sessions/my_capture
```

Detection recall was measured against nuScenes annotations (>=60% visibility,
four boston-seaport scenes) and indexed by apparent *pixel height*, which is what
transfers between cameras — range depends on focal length, pixels on target do
not. For an iPhone wide camera (fx ~1500 px) tracking a person:

| range | px tall | P(track) full-frame | P(track) tiled | range err @10 m walk |
|---|---|---|---|---|
| 50 m | 51 | 100% | 100% | 0.6 m (1%) |
| 100 m | 26 | 98% | 100% | 2.4 m (2%) |
| 150 m | 17 | 41% | **91%** | 5.3 m (4%) |
| 200 m | 13 | 12% | 71% | 9.4 m (5%) |

Tiling is what makes 150 m viable — it flips track formation from unlikely to
likely. `P(track)` assumes detections are independent between frames, which they
are not (misses correlate with pose and occlusion), so it is an upper bound.

## Limitations, stated plainly

- **Bounding-box centroid drift.** Bearings go through the centroid of a 2-D box,
  which is not the centroid of the 3-D object and moves with viewing aspect. A
  bias, not noise: it does not average out and is not in the covariance.
- **Degenerate geometry under forward motion.** Detected and reported, not solved.
- **Constant heading bias is invisible to the filter.** Needs an external reference.
- **Linearised covariance is local.** For short tracks on small baselines it can be
  small and wrong together. Mitigated by the estimate-independent gates above;
  not eliminated.
- **nuScenes poses are map-local, not WGS84.** There is no georeference anywhere in
  nuScenes. Any lat/lon this emits is relative to a documented assumed origin and
  is accurate *relatively*, not absolutely. Every published event carries the
  assumption in its remarks.
- **Synthetic numbers are an upper bound.** A geometric simulator with a perfect
  detector isolates geometry and filter error, which is what it is for.

## Licensing — the two publishers are deliberately separate trees

The `anduril-lattice-sdk` licence permits use only for building against Lattice.
So: `publish/lattice/` imports nothing from `publish/cot/` and vice versa, no SDK
code is vendored here, and the SDK import happens inside the function that needs
it so that importing `geoloc_agent` never loads it. Both rules are enforced by
tests in `test_publish.py`, not just by comment. Credentials come from the
environment and are never committed.

## Status against the build plan

| Phase | State |
|---|---|
| 0 — skeleton, synthetic | **Done.** `pytest` green; synthetic object geolocated to 0.00 m (accept: < 0.5 m) |
| 1 — geometry and filter | **Done.** Covariance shrinks monotonically with baseline; matches analytic prediction within 20% |
| 2 — nuScenes loader | **Done and validated.** Median 0.56 m on real data (see below) |
| 3 — eval harness | **Done.** One command produces `reports/eval.md`; regression bounds pinned |
| 4 — real detector | **Done.** YOLO11n on the ANE, p95 11.4 ms against a 100 ms budget |
| — mono-depth ranger | **Added.** Size prior from bbox height; initialisation and gating only |
| 5 — publish | **CoT done** (round-trips over a real socket). **Lattice written**, needs credentials |
| 6 — agent layer | **Done.** `reports/agent_eval.md` with the pattern comparison |

### Agent pattern comparison (from `reports/agent_eval.md`)

| pattern | task success | trajectory quality | tool calls/track | missed escalations | adversarial |
|---|---|---|---|---|---|
| `graph` | 1.000 | 1.000 | 2.11 | 0 | 1.000 |
| `planner_executor` | 1.000 | 0.917 | 2.26 | 0 | 1.000 |
| `linear` | 0.881 | 0.405 | 1.37 | **3** | **0.333** |

`linear` is genuinely cheaper and that is exactly what it buys: it decides from
the track listing alone, so it never learns a track's class *entropy*. On the
adversarial slice — top-class confidence 0.55, posterior nearly even between car
and pedestrian, position known only to 22 m — it sees a confident class over a
wide fix and **silently suppresses** it. That is the quieter failure and the worse
one: a wrong marker on a map gets questioned; a track that never appears does not.

The default agent backend is a deterministic rule policy, so the harness runs in
CI with no API key and no run-to-run variance — and it sets the floor. A pattern
that cannot beat a hundred lines of thresholds does not justify a model call.
`ClaudeBackend` runs the same tools and scenarios through `claude-opus-5` when
`ANTHROPIC_API_KEY` is set.
