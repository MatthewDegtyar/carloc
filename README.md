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

`geoloc-render` produces a two-part clip: the camera view with class, range and
1-sigma range uncertainty on every box, beside a top-down map with error ellipses.

Part 1 is lateral motion (good geometry) — ranges converge and the ellipses
collapse to centimetres. Part 2 is the same objects, detector and filter under
forward motion — the ellipses stretch into streaks *along the line of sight* and
every box reads `RANGE UNRELIABLE`. Putting them in one file is deliberate: the
well-conditioned result only means something next to the degenerate one.

**Every pixel is synthesised.** A `SyntheticSession` has no imagery; the ground
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

## Calibration status

NEES is the honesty metric: `eᵀP⁻¹e`, chi-square with 3 DOF, so the mean should
be 3. On pure, well-conditioned tracks the filter measures **3.32, inside the 95%
band**. Pooled over all confirmed tracks it sits at ~4.2 — the excess is
association error, which is a real error source the covariance does not model, so
it is reported rather than tuned away. Track purity is reported alongside.

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
| 2 — nuScenes loader | **Written, not validated.** Needs the ~4 GB `v1.0-mini` download |
| 3 — eval harness | **Done.** One command produces `reports/eval.md`; regression bounds pinned |
| 4 — real detector | **Interface + benchmark only.** Needs a YOLO11n CoreML export |
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
