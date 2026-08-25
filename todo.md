# geoloc-agent — build plan

## What this is

A pipeline that turns posed video into geolocated object tracks with honest
uncertainty, then lets an LLM agent decide which of those tracks are worth
surfacing to an operator. Published to Anduril Lattice.

**The deliverable is the framework and the eval harness.** The computer vision
is commodity and should be treated as such. Do not spend time tuning detectors.

## Data sources, in order

1. **Synthetic** (numpy, no video) — validates filter math. Phase 0.
2. **nuScenes v1.0-mini** — primary real-data source. Phase 1 onward.
3. **Own capture via Stray Scanner (iPhone 16 Pro)** — arriving later today.
   Build the loader interface so this drops in without touching anything else.
4. CARLA — optional, only if injected-noise sweeps need more control than
   nuScenes allows.

---

## Why nuScenes first

- `pip install nuscenes-devkit`, v1.0-mini is ~4 GB, 10 scenes, no scraping.
- Every `sample_data` has an `ego_pose` (translation + quaternion) and a
  `calibrated_sensor` (intrinsics + extrinsics). Real pose, free.
- `sample_annotation` gives 3D boxes in the global frame. **That is ground
  truth object position**, which means geolocation error is directly scorable.
- Boston Seaport is a US urban scene with parked cars, buildings, street
  furniture. Close enough to the target domain.

### nuScenes gotchas to handle up front

- **Poses are in a per-map local frame in meters, not WGS84.** There is no
  lat/lon in the pose. Pick a documented origin per map, convert with a local
  ENU transform, and state the assumption in the report. Do not fake precision.
- Annotations are at 2 Hz keyframes; camera sweeps run at 12 Hz and are
  unannotated. Use keyframes for scoring, sweeps for extra bearings.
- CAM_FRONT is 1600x900. Six cameras total; start with CAM_FRONT only.
- **Motion is mostly forward.** Objects straight ahead have near-zero
  perpendicular baseline and will triangulate badly. This is a feature: it
  exercises the degenerate-geometry detector on real data. Parked cars to the
  side are the good-geometry case. Report both separately.

---

## Repo layout

```
src/
  io/          session loaders: synthetic, nuscenes, stray_scanner, video_only
  detect/      detector interface + stub + coreml/yolo impl
  range/       rangers: triangulation, mono_depth, lidar, (streetscape later)
  fuse/        EKF, association, track management, covariance
  publish/
    cot/       Cursor-on-Target -> local TAK
    lattice/   anduril-lattice-sdk        <-- SEPARATE TREE, see licensing
  agent/       tools, orchestration patterns, policies
  eval/        scenario runner, metrics, report generation
configs/       noise models, scenarios, sensor params (YAML)
sessions/      data (gitignored)
reports/
```

## Core contracts — write these before any implementation

```
Frame        image, timestamp, intrinsics, pose (R, t), pose_cov
Detection    bbox, class, score, frame_id
RangeMeas    value, sigma, method, valid
Observation  bearing (world unit vector), range (optional RangeMeas), t
TrackState   mean (x,y,z or lat,lon,alt), cov 3x3, class posterior, n_obs, age
Decision     track_id, action (surface|suppress|escalate), rationale
```

Every module sits behind these. Ship stub implementations on day one so no
downstream work blocks on upstream work.

---

## Phase 0 — skeleton, synthetic only

Do this first even though nuScenes is the target. It takes an hour and it
catches the bugs that are miserable to find on real data.

- [ ] Repo, uv or poetry, ruff, pytest
- [ ] Contracts as dataclasses with validation
- [ ] `SyntheticSession`: generates a walk path and true object positions.
      No images at all, just geometry.
- [ ] `StubDetector`: emits scripted detections from JSON
- [ ] End-to-end smoke test, no ML anywhere in the loop

**Accept:** `pytest` green. One synthetic object geolocated to under 0.5 m.

## Phase 1 — geometry and filter

- [ ] `bearing_from_pixel()`: intrinsics + extrinsics + ego pose -> world
      unit vector. Unit-test against hand-computed cases.
- [ ] `TriangulationRanger`: N bearings -> position estimate + covariance
- [ ] EKF track: initialize from first observation with wide covariance,
      update per observation
- [ ] Track management: birth, association (Mahalanobis gate), death
- [ ] Degenerate-geometry detector: flag when the perpendicular component of
      the baseline is too small relative to range

**Accept:** on synthetic data, covariance shrinks monotonically as baseline
grows, and error matches the analytic prediction `R^2 * sigma_theta / B`
within 20%.

## Phase 2 — nuScenes loader

- [ ] `NuScenesSession` implementing the same interface as `SyntheticSession`
- [ ] Extract ego_pose, calibrated_sensor intrinsics, camera frames
- [ ] Extract sample_annotation 3D boxes as ground truth object positions
- [ ] Map-frame to local-ENU conversion with a documented origin
- [ ] Score: run the phase 1 filter against real poses, compare to annotations

**Accept:** the same pipeline from phase 0 runs on nuScenes with no changes to
`fuse/`. Report geolocation error split by good vs degenerate geometry.

## Phase 3 — eval harness (this is the actual deliverable)

- [ ] Scenario spec in YAML: object layout, path, injected noise
- [ ] Noise injection: `gps_sigma`, `heading_bias`, `heading_sigma`,
      `range_sigma`, `bearing_sigma`, `detection_dropout`, `false_positive_rate`
- [ ] Metrics: geolocation RMSE, covariance calibration (NEES), track purity,
      track fragmentation, time-to-converge, degenerate-geometry recall
- [ ] Sweep runner: grid over noise parameters -> CSV + plots
- [ ] Regression suite with pinned expected ranges

**Accept:** one command produces `reports/eval.md` containing error-vs-noise
curves, and NEES falls inside chi-square bounds (the filter is honest about
its own uncertainty rather than overconfident).

## Phase 4 — real detector

- [ ] `CoreMLDetector`: YOLO11n exported via coremltools, running on the ANE
- [ ] Benchmark: latency per stage, confirm the three-loop budget holds
- [ ] Note: COCO has no infrastructure classes. Cars and people only for now.
      Fine-tuning is out of scope for this build.

## Phase 5 — publish

- [ ] `publish/cot/` first: Cursor-on-Target XML to a local TAK server
- [ ] `publish/lattice/`: `anduril-lattice-sdk`, entity publish, 30-minute
      token refresh, retry with backoff. Credentials from env, never committed.
- [ ] **Licensing: keep these trees completely separate.** The Lattice SDK
      license permits use only for building against Lattice. Do not import it
      into the CoT path, and do not vendor SDK code into the repo.

**Accept:** tracks from a replayed nuScenes scene appear on a Lattice map.

## Phase 6 — agent layer

- [ ] Tools: `query_tracks`, `get_track_detail`, `classify`, `publish_entity`,
      `escalate_to_human`, `request_better_geometry`
- [ ] Orchestration patterns behind one interface: linear, planner/executor,
      small agent graph
- [ ] Policy: given N tracks with covariances and class posteriors, decide
      what surfaces, in what order, and when to interrupt
- [ ] Eval: task success, tool-call correctness, trajectory quality
- [ ] Adversarial slice: ambiguous class with wide covariance must escalate,
      not guess
- [ ] Head-to-head comparison table across patterns on identical scenarios

**Accept:** `reports/agent_eval.md` with the pattern comparison table.

---

## Design rules

1. **Three decoupled loops.** Perception in milliseconds, ranging in hundreds
   of milliseconds, decision in seconds. Nothing blocks on the slow thing.
   Build it this way from the start; retrofitting is miserable.
2. **Every measurement carries a sigma.** No bare point estimates anywhere in
   the codebase.
3. **Covariance reaches the agent.** It is the input that makes the surfacing
   decision non-trivial. Do not collapse tracks to points before the agent
   sees them.
4. **Config-driven.** Scenarios are YAML, not Python.
5. **Ground truth flows through the whole pipeline in eval mode** so any stage
   can be scored in isolation.

## Writeup targets — write these as you go

- Geolocation error vs injected noise, per stage
- Filter calibration (NEES) plot
- Good-geometry vs degenerate-geometry error, split out
- Agent pattern comparison table
- Limitations section, stated plainly: bounding-box centroid drift on the
  object, degenerate geometry under forward motion, nuScenes poses being
  map-local rather than WGS84