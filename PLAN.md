# carloc — every car in a scene, once, with a real lat/lon

## What this is

Take drone video of a parking area. Produce one row per physical car:

```
id   lat          lon           class  conf   seen_in_frames
1    28.295961    112.999723    car    0.87   12
2    28.295974    112.999688    car    0.91   9
```

Three requirements, in order of difficulty:

1. **Real coordinates.** Not pixels, not "arbitrary metres" — WGS84 that drops a
   pin on Google Maps at the right spot.
2. **No duplicates.** A car seen in 40 frames is one car, not 40. This is the
   part that is actually hard and the part most demos quietly skip.
3. **Simple scene.** Every failure below is easier to see and diagnose when the
   picture is not a city.

## Data

**AirZoo-Real**, site `jiaxiao`, split `12-14` — Changsha, Hunan.
`28.29596 N, 112.99972 E`.

Chosen after looking at all four sites: it is the only one with an open lot of
orderly parked cars rather than dense housing, farmland, or a motorway. Frames
around 430-560 keep that lot in view.

What comes with it, and why each matters:

| | |
|---|---|
| RGB, 1008x756 | the picture |
| 6-DoF pose per frame, EPSG:4547 metres | where the camera was |
| per-frame intrinsics | quoted for 4032x3024, so scale by 0.25 |
| GNSS lat/lon per frame | **ground truth for the georeferencing** |
| DSM, 0.5 m/px, same datum | the ground to intersect rays with |

The GNSS is the reason this dataset and not a stock drone clip: the location is
*known*, so putting it on Google Maps is a **verification** rather than a guess.
A clip off the internet would make step 1 unfalsifiable.

Kept in `sessions/`. Everything else from the previous build is in `archive/v1/`.

## How

### 1. Pixel to ground

Ray from the camera centre through the box, intersected with the DSM. No depth
network, no parallax, no triangulation — a single frame is enough for a
downward-looking camera over known terrain.

Carries a sigma from attitude and surface error: for `R = h/sin(theta)`,
`dR/dtheta = -R/tan(theta)`, so a ray near the horizon is unusable and must be
refused rather than answered.

### 2. Ground to WGS84

Fit an affine from EPSG:4547 metres to lat/lon using the paired pose and GNSS
samples the dataset already provides. Grid north is **not** true north — the
meridian convergence here is about 0.48 deg, which displaces a point ~5 m over a
600 m flight. Fitting it from data rather than assuming a tangent plane also
gives a residual to report, which naming an EPSG code does not.

### 3. Dedupe

The real work. One physical car generates one detection per frame it appears in.

Cluster ground fixes: a *parked* car projects to the same ground point from every
viewpoint, so camera motion is irrelevant and association is nearest-neighbour in
metres. This is why image-plane trackers are not used here — measured on the
previous build, a static point moves 48.7 px between frames against a 12-20 px
box, so inter-frame IoU is exactly zero and IoU-based association cannot work.

Then merge. Two failure modes, in tension:

- **over-merge** — adjacent bays are ~2.5 m apart and the fix sigma is ~0.5 m, so
  a loose radius silently fuses two cars into one
- **under-merge** — a car whose fixes scatter across passes becomes two rows

The scatter of a cluster's own fixes measures which is happening, and it needs no
ground truth: if the spread far exceeds the predicted sigma, the cluster is
holding more than one car.

### 4. Report

CSV plus a map. Every row carries its uncertainty and how many frames supported
it. Rows the geometry cannot support are marked, not dropped.

## Verification

1. **Location** — drop the computed lat/lon on Google Maps satellite and confirm
   the lot, road and buildings match the video frame.
2. **Georeferencing** — affine residual against the held-out GNSS fixes, in
   metres.
3. **Dedupe** — count of unique cars against a hand count on a still frame. The
   only number here that needs a human, and it is worth the ten minutes.
4. **Uncertainty** — cluster scatter against predicted sigma. Near 1.0 means the
   error bars are honest.

## Not doing

Licence plates. Occupancy timing. Payment integration. Tracking moving vehicles.
Anything on-device. Those were explored in `archive/v1/`; none of them is needed
to put a car on a map exactly once, and each one hid whether that part worked.

## Blocked: AirZoo cannot support this task

Built the chain end to end and it refuses every detection, correctly.

Position error is dominated by attitude, and `dR/dtheta = -R/tan(theta)` means it
depends on how steeply the camera looks down. Telling neighbouring cars apart
needs sigma under ~1.2 m, half the ~2.5 m spacing of a bay. At AirZoo's 130 m AGL:

| depression | slant range | sigma | |
|---|---|---|---|
| 30 deg (its median) | 260 m | 7.10 m | |
| 45 deg (its best) | 184 m | 2.92 m | |
| 65 deg | 143 m | 1.11 m | OK |
| 90 deg (nadir) | 130 m | 0.32 m | OK |

**AirZoo has zero frames above 45 deg** in either split checked. It is an oblique
reconstruction dataset, so this is what it is for; it is simply the wrong tool.

Before the cap was added the pipeline "worked" and reported 14 cars. That number
was an artefact: the radius sweep showed no plateau at all, climbing 2 -> 31 cars
as the merge radius went 1 -> 6 m, which is what it looks like when the threshold
rather than the data decides the answer. Fixes of one car were landing further
apart than neighbouring cars are.

Note what does **not** help: flying lower. At nadir the attitude term vanishes
entirely (`1/tan(90) = 0`), so sigma is 0.32 m at 40 m and at 250 m alike. The
requirement is **look angle, not altitude** -- above about 65 degrees.

That is what parking-lot drone footage normally is, and CARPK (Phantom 3 at 40 m
over four lots) is exactly it. Its download is a dead 2017 Google Drive link, as
is the Busy Parking Lot video's. Sourcing near-nadir footage is the open item.

## Status

- [x] session loader — and it caught that AirZoo's camera-axis convention
      **alternates within a single flight** (424/752 frames need the flip).
      Detected per frame from the physical rule that a survey camera looks down;
      a hardcoded convention silently inverts half the poses.
- [x] DSM + ray-cast ranging, refusing when the geometry cannot support a fix
- [x] WGS84 fit — held-out residual **0.1 cm**, grid convergence +0.4757 deg
- [x] detection — tiled, since full-frame finds nothing at ~20 px
- [x] dedupe and merge audit, incl. the radius sweep that exposed the problem
- [x] CSV + map
- [ ] **near-nadir footage** — blocking everything below
- [ ] rerun on data that can support it
- [ ] Google Maps confirmation
