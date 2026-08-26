# AirZoo-Real: what the files actually contain

Every convention below was established by measuring against the data. None of it
is documented upstream, and three of my initial assumptions were wrong in ways
that produce plausible numbers rather than errors — which is the dangerous kind.

Recorded because rediscovering it costs a day, and because a silently wrong
convention here corrupts every coordinate downstream without ever raising.

Source: [`RingoWRW97/AirZoo-Real`](https://huggingface.co/datasets/RingoWRW97/AirZoo-Real)
on Hugging Face, gated (`gated: auto` — accept terms with an account). Paper:
[AirZoo, arXiv:2604.26567](https://arxiv.org/abs/2604.26567). SAW Lab, National
University of Defense Technology — see *Provenance* at the end.

## Scale

38.2 GB, 28,753 files. Do not clone it whole.

```
airzoo_real_recon/
  {jiaxiao,xuexiao,guangchang,biandian}/     4 sites
    {06-08,12-14,18-20,22-24}/               time-of-day splits, ~2 GB each
      images/{i}_{0,1,2}.png                 ~679 poses x 3 channels
      poses/poses/t_pose.txt
      poses/intrinsics/t_intrinsic.txt
      poses/times/time.txt
  dom/
    fcw_hangtian_DSM.tif   (69 MB)  + .tfw + .prj   <- surface model
    fcw_hangtian_DOM.tif   (52 MB)  + .tfw + .prj   <- orthophoto
    allfeicuiwan/DEM/...   (1.9 GB)                 <- skippable
    metadata_china.json
Scene1/, Scene2/                                    raw DJI JPG + RTK/RINEX
```

A working slice is **one split plus the `fcw_hangtian` rasters ≈ 2.3 GB**.

## Poses — `t_pose.txt`

```
<path> qw qx qy qz tx ty tz
```

**World-to-camera, quaternion first, `wxyz` order.** The camera centre is

```
C = -R(q)ᵀ · t
```

The raw translation is *not* a position: its components span ±3.1×10⁶ and mix
axes. That magnitude is the tell — it is `-R·C` for a `C` with large projected
coordinates.

Confirmed by elimination: `wxyz` puts every camera centre inside the DSM extent
(X 400829–401403, Y 3131141–3131764) with Z 157.62–157.81 m, matching the
telemetry altitude of 157.65 m. The `xyzw` reading scatters centres across
millions of metres.

## Camera axes — **OpenGL, not OpenCV**

The optical axis is **−Z**, with **+Y up** in the image. A pixel ray is

```
d_cam = [ (u−cx)/fx , −(v−cy)/fy , −1 ]
```

Read as OpenCV (`+Z` forward, `+Y` down), the optical axis points at the sky:
its world Z component comes out at **+0.707** where the gimbal reports 45°
**down**. In a ray-cast against the terrain that convention scored **0 hits out
of 900** — which at least fails loudly. The subtler variant, flipping only `Z`,
*does* hit and gives ranges spanning the same interval, but assigns them to
mirrored pixels: correlation against the depth render is +0.59 rather than the
correct −0.93.

`io/airzoo.py` applies `diag(1, −1, −1)` once at load, so nothing downstream
carries the distinction.

| convention | DSM hits | corr. vs depth |
|---|---|---|
| OpenCV `[+x,+y,+z]` | 0 / 900 | — |
| **OpenGL `[+x,−y,−z]`** | **900 / 900** | **−0.93** |
| flip Z only `[+x,+y,−z]` | 900 / 900 | +0.59 |
| flip Y only `[+x,−y,+z]` | 0 / 900 | — |

## Intrinsics — `t_intrinsic.txt`

```
<path> PINHOLE 4032.0 3024.0 3528.0 3307.5 2016.0 1512.0
       model    W      H      fx     fy     cx     cy
```

**Quoted for 4032×3024; the delivered PNGs are 1008×756.** Scale by 0.25.
Taking the quoted values scales every bearing by four — an error that reads as
an unusual lens rather than a fault. The loader reads a real PNG rather than
trusting the header, and refuses a non-uniform scale instead of guessing.

Note `fx ≠ fy` (3528.0 vs 3307.5) even before scaling.

## World frame — EPSG:4547

CGCS2000 / 3-degree Gauss-Krüger CM 114E. Metres. Poses and rasters share it.

Coordinates run ~4×10⁵ east by ~3.1×10⁶ north, so `AirZooSession` subtracts a
local origin by default. **That moves the cameras and not the terrain** — use
`session.local_dsm`, never `session.dsm`, with frames from that session. Getting
this wrong makes every ray start outside the raster, and it surfaces as a ranger
that refuses *everything* rather than as a crash.

## Image channels

| file | content |
|---|---|
| `{i}_0.png` | RGB, 1008×756 |
| `{i}_1.png` | inverse depth, 8-bit greyscale |
| `{i}_2.png` | semantic labels, RGB rendering |

### The depth render carries no global scale

The most consequential finding, and the easiest to get wrong.

Fitting `1/range = a·png + b` against DSM ray-casts **on one frame** gives
**r = 0.964** over 2955 samples. That looks like a global calibration.

It is not. Across 120 frames the same fit falls to **r = 0.709**, and per-frame
fits show why:

| | median | spread |
|---|---|---|
| slope `a` | 2.22×10⁻⁵ | 1.4× |
| intercept `b` | 9.32×10⁻⁴ | **12×** |
| png max | 255 | 255 in every frame |

A twelvefold intercept swing with the maximum pinned at 255 is a per-image
stretch. Publishing a constant would yield plausible metres that are wrong by a
frame-dependent amount, so no constant is published.

`inverse_depth_at()` returns the raw values, documented as relative — larger is
nearer, good for structure, occlusion ordering and masking. For a distance,
ray-cast the DSM. `depth_scale_against_surface()` fits one frame when you want
to compare the two, but it needs the DSM anyway, so it is a cross-check rather
than a ranging path.

### Semantic layer

An RGB rendering, not class indices. Classes per `metadata_china.json`: `ROAD`,
`BUILDING_FACADE`, `GREEN_LANDS`, `CONSTRUCTION`, `COAST_ZONES`, `OTHERS`,
`BLD_ROOF`. Water is identifiable by blue dominance and is worth masking — it is
a **hole in the DSM**, not a surface, since water has no stable texture to
reconstruct from. Overall DSM coverage is 63.9%.

## Telemetry — `time.txt`

```
<name> lon lat alt c4 c5 c6
```

Despite the filename it carries no timestamps. `lon`/`lat` are WGS84, which is
how `session.origin` gets a **measured** geographic origin rather than an assumed
one — the thing nuScenes could not provide.

Columns 4–6 are angles, not what they look like. **I initially read `c6` as
relative altitude** (its first value, 99.1, sits plausibly close to a flight
height) and concluded there was a 35 m datum error against the DSM. There is no
such error: `c6` runs to **−355.9**, which no altitude does. It is yaw. `c5`
takes only `{−30, 0, 45}` — gimbal pitch — and `c4` only `{−1, 0}`.

There is no AGL column. Height above ground comes from the DSM: median **134.7 m**
for `guangchang/12-14`.

## Surface model

`fcw_hangtian_DSM.tif`, float32, 2859×5292 at **0.5 m/px**, georeferenced by an
ESRI world file (`.tfw`) rather than an embedded GeoTIFF header — so no GDAL is
required. Nodata is `-9999`; `io/dsm.py` masks anything below −1000 to NaN,
because a sentinel averaged into a height puts the surface a kilometre
underground.

Bounds: X 400201.8–402847.8, Y 3130401.8–3131831.3.

It is **2.5-D**: one height per cell, so no overhangs, no bridge decks with road
beneath, no building facades. A ray that should pass under a bridge stops on top
of it. For downward-looking work that is nearly always the right trade.

## What is not here

**No object annotations.** All 52 non-image files are poses, intrinsics and
terrain. Per-object geolocation error is therefore *not* scorable on this
dataset; the surface model is the reference instead. A detector supplies the
objects.

## Provenance

Both AirZoo and the unreleased SkyPin come from SAW Lab, **National University of
Defense Technology** — a PLA-affiliated institution on the US Commerce Entity
List since 2015. The licence is a custom "Saw Lab" one, not standard OSS, and the
sites are in Changsha, Hunan (`metadata_china.json`).

Flagged as a fact to weigh, not a legal opinion. For anything defence-facing,
check it properly rather than taking this note's word for it.

For reference, [SkyPin](https://doi.org/10.3390/drones10070500) — the same group's
benchmark that *does* carry RTK-annotated ground targets — remains unreleased:
its repository holds a 933-byte README promising publication-time release, the
paper published 2026-06-30, and nothing has shipped.
