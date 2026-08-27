"""Locate a dashcam car, and the parked cars it passes, from video alone.

The satellite survey answers *which cars are in a paid zone right now*. It cannot
answer *how long one has been there*, because satellite imagery is a single
instant months stale. Overstay needs two passes separated in time, and a patrol
vehicle with a camera is how a parking authority already gets them.

So the question this module tests is whether one pass can be placed on the map
accurately enough for the zone verdict to survive. The chain is:

    video -> odometry -> map-matched trajectory (4 Hz, with a CI)
          -> parked-car detections -> world positions -> zone verdict

**There is no GPS.** The clip carries no telemetry (it is a YouTube re-encode,
`encoder: Google`), so every position here is derived from pixels plus
OpenStreetMap. There is also no Google Maps API key available, so Street View
matching is not the anchor source; landmarks identified in the imagery are.

Two scales are unknown and both are self-calibrated rather than assumed:

* **`f`, pixels per radian.** Recovered from a turn whose true angle is known
  from the street grid -- a left turn from a north-south avenue onto an
  east-west street is 90 degrees, and the yaw integral over it is `f * pi/2`.
* **`K`, metres per unit of ground flow.** Recovered from the distance between
  two anchors. It cannot be predicted from `f * h` because the flow statistic is
  biased low by features that are not on the road surface, which is also why
  `_ground_speed` fits the ground plane explicitly instead of taking a median.

The bias that `K` absorbs depends on scene content -- a street of tall buildings
and a bridge deck do not bias it equally -- so `K` fitted on one stretch should
not be trusted far outside it. That is the main reason the demo is scoped to a
single street rather than the whole clip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EARTH_R = 6_378_137.0

CAMERA_HEIGHT_M = 1.35
"""Windscreen dashcam above the road. Only used for sanity checks: the speed
scale comes from anchors, not from this."""

HORIZON_FRAC = 0.52
"""Row of the horizon as a fraction of image height. A dashcam sits near level,
so the principal point is close to the centre; refined by `fit_horizon`."""


@dataclass
class Odometry:
    """Per-sample motion, in image units. Neither series is metric yet."""

    dt: float
    yaw_px: np.ndarray
    """Horizontal image shift per sample. Divided by `f` this is radians."""

    flow: np.ndarray
    """Ground-plane flow coefficient `c` where `dy = c * (y - cy)**2`.

    Constant across the ground plane by construction, which is what makes it a
    speed proxy: for a pinhole camera at height `h`, a ground point at range `Z`
    projects to `y - cy = f*h/Z`, so `dy/dt = v * (y-cy)**2 / (f*h)`.
    """

    cy: float
    n_ground: np.ndarray
    """Inliers behind each flow sample. A collapse here means the road was not
    visible -- stopped in traffic, or a vehicle filling the frame -- and the
    sample should not be trusted."""

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.flow)) * self.dt


@dataclass(frozen=True)
class Anchor:
    """A place in the video whose ground truth is known from the map."""

    t: float
    lat: float
    lon: float
    sigma_m: float
    note: str


@dataclass(frozen=True)
class Fix:
    """Where the camera was, and how well that is known."""

    t: float
    lat: float
    lon: float
    heading_deg: float
    speed_ms: float
    sigma_along_m: float
    sigma_cross_m: float

    @property
    def sigma_m(self) -> float:
        """Circularised, for callers that want one number."""
        return math.hypot(self.sigma_along_m, self.sigma_cross_m)


def _frame(lat: float) -> tuple[float, float]:
    return (math.pi / 180) * EARTH_R * math.cos(math.radians(lat)), \
           (math.pi / 180) * EARTH_R


def _ground_speed(a: np.ndarray, b: np.ndarray, cy: float,
                  min_pts: int = 12) -> tuple[float, int]:
    """Fit `dy = c * (y - cy)**2` over tracked points, robustly.

    A median of `dy / (y-cy)**2` -- the obvious thing -- is badly biased: most
    strong corners in a street scene are on buildings, parked cars and other
    traffic, none of which lie on the road plane, and they drag the median down
    by more than an order of magnitude. Fitting the model and keeping only what
    agrees with it rejects them, because a point off the ground plane does not
    follow the square law.
    """
    y = a[:, 1]
    dy = b[:, 1] - y
    keep = (y > cy + 20) & (dy > 0.02)
    if keep.sum() < min_pts:
        return float("nan"), 0
    x = (y[keep] - cy) ** 2
    d = dy[keep]
    c = float(np.median(d / x))
    n_inlier = int(keep.sum())
    for _ in range(3):
        residual = d - c * x
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-9
        inlier = np.abs(residual) < 2.5 * scale
        n_inlier = int(inlier.sum())
        if n_inlier < min_pts:
            break
        c = float(np.sum(d[inlier] * x[inlier]) / np.sum(x[inlier] ** 2))
    return c, n_inlier


def extract_odometry(video_path: str, stride: int = 2,
                     progress=None) -> Odometry:
    """Track features frame to frame and reduce them to yaw and ground flow.

    `stride` skips frames: at 30 fps every second frame is still only 67 ms of
    motion, which keeps displacements inside the Lucas-Kanade window while
    halving the work.
    """
    import cv2

    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    feature = dict(maxCorners=600, qualityLevel=0.01, minDistance=6, blockSize=7)
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    ok, first = capture.read()
    if not ok:
        raise RuntimeError(f"cannot read {video_path}")
    previous = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    height, _ = previous.shape
    cy = height * HORIZON_FRAC

    yaw: list[float] = []
    flow: list[float] = []
    counts: list[int] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        index += 1
        if index % stride == 0:
            p0 = cv2.goodFeaturesToTrack(previous, mask=None, **feature)
            w = float("nan")
            c = float("nan")
            n = 0
            if p0 is not None and len(p0) > 20:
                p1, status, _ = cv2.calcOpticalFlowPyrLK(previous, grey, p0, None, **lk)
                if p1 is not None:
                    keep = status.ravel() == 1
                    a, b = p0[keep, 0], p1[keep, 0]
                    if len(a) > 20:
                        # Yaw from a rigid 2-D fit: a rotation of the camera
                        # shifts the whole image sideways, while forward motion
                        # mostly scales it, so the fitted translation isolates
                        # the turn.
                        model, _ = cv2.estimateAffinePartial2D(
                            a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                        if model is not None:
                            w = float(model[0, 2] + model[0, 0] * 0.0)
                        c, n = _ground_speed(a, b, cy)
            yaw.append(w)
            flow.append(c)
            counts.append(n)
            if progress and len(flow) % 500 == 0:
                progress(len(flow))
        previous = grey
    capture.release()
    return Odometry(dt=stride / fps, yaw_px=np.array(yaw), flow=np.array(flow),
                    cy=cy, n_ground=np.array(counts))


def _fill(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    bad = ~np.isfinite(a)
    if bad.all():
        return np.zeros_like(a)
    a[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), a[~bad])
    return a


def smooth(a: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return a
    kernel = np.ones(window) / window
    return np.convolve(_fill(a), kernel, mode="same")


def calibrate_yaw_scale(odom: Odometry, t0: float, t1: float,
                        true_turn_deg: float) -> float:
    """Pixels per radian, from a turn whose angle the street grid already fixes.

    A left turn off a north-south avenue onto an east-west street is 90 degrees
    whatever the camera's field of view, so the yaw integral across it measures
    `f` directly. This removes the need to know the lens.
    """
    i0, i1 = int(t0 / odom.dt), int(t1 / odom.dt)
    integral = float(np.sum(_fill(odom.yaw_px)[i0:i1]))
    return abs(integral) / abs(math.radians(true_turn_deg))


def calibrate_speed_scale(odom: Odometry, t0: float, t1: float,
                          distance_m: float) -> float:
    """Metres per unit of ground flow, from a known distance between anchors.

    Not derivable from `f * h`: the flow statistic retains a scene-dependent
    bias even after the ground-plane fit, and this absorbs it. Valid only near
    the stretch it was fitted on.
    """
    i0, i1 = int(t0 / odom.dt), int(t1 / odom.dt)
    travelled = float(np.sum(_fill(odom.flow)[i0:i1])) * odom.dt
    if travelled <= 0:
        raise ValueError("no forward flow between those times")
    return distance_m / travelled
