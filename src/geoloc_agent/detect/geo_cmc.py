"""Camera motion compensation from known pose and terrain, not from pixels.

`roboflow/trackers` compensates for a moving camera by estimating a homography
between consecutive frames with Lucas-Kanade optical flow, exposed through
``CoordinatesTransformation``. That is the right abstraction and the only
practical estimator when all you have is video.

When the platform knows where it is, you can compute the same transformation
instead of estimating it, and the differences are not marginal:

* **No drift.** An estimated transform is chained frame to frame, so error
  accumulates without bound. This one is absolute -- every frame maps to the
  same ground coordinates, independently.
* **Parallax is handled.** A homography is exact only for a planar scene or a
  pure rotation. Here the buildings stand 40 m tall at 193 m range, which is
  real parallax, and no homography fits both the rooftops and the street.
* **Metres, not pixels.** The output is ground coordinates, so displacements
  have physical meaning and the tracker's gates can be set in metres.
* **Texture is irrelevant.** Optical flow needs corners. Roughly forty percent
  of these frames is a lake, which has none.

Measured on AirZoo `guangchang/12-14`, where a static ground point moves 48.7 px
between frames against a 12-20 px car box -- so inter-frame IoU is exactly zero
and every IoU-based tracker fails outright. BoT-SORT with optical-flow CMC
recovers 30 of 527 detections; without it, 4.

The interface is deliberately theirs: ``abs_to_rel`` / ``rel_to_abs`` over
``(N, 2)`` arrays. Ground coordinates are two-dimensional, so a surface-backed
transform fits it exactly, with no API change.

Accuracy, measured as a round trip -- pixels to ground and back -- over 2039
samples across 8 frames:

    p50    0.000 px      83.1% under 1 px
    p75    0.291 px      90.0% under 5 px
    p90    4.928 px
    p95   24.055 px
    max  207.979 px

Exact for most of the frame, with a bad tail. The tail is not noise and not
distributed randomly: it sits on building edges, and it is the 2.5-D limitation
of a height field. A ray grazing a facade stops on the roof, and the ground point
it returns can be a cell whose height is the street below, so the round trip
lands a storey away in image space. Interiors of roofs and open ground are
exact; silhouettes are not.

That matters for how this should be used. It is reliable for association over
open ground and unreliable exactly where tall structures occlude, so a consumer
should treat a large round-trip residual as a signal to distrust the mapping for
that point rather than as a position. A true 3-D surface would fix it; a height
field cannot.
"""

from __future__ import annotations

import numpy as np

from geoloc_agent.contracts import Frame


class GeoGroundTransformation:
    """Maps image points to metric ground coordinates and back.

    Implements the same contract as `trackers.CoordinatesTransformation`
    (``abs_to_rel`` / ``rel_to_abs`` over ``(N, 2)``), so it drops into any
    tracker that accepts one. Subclassing is avoided so this module imports
    without the ``tracking`` extra installed; `as_trackers_transformation()`
    adapts it when the package is present.

    "Absolute" here means ground coordinates in the session's world frame, in
    metres. "Relative" means pixels in this frame.
    """

    def __init__(self, frame: Frame, surface, max_distance: float = 2000.0,
                 step: float = 1.0) -> None:
        self.frame = frame
        self.surface = surface
        self.max_distance = float(max_distance)
        self.step = float(step)

    # -- image -> ground ---------------------------------------------------

    def rel_to_abs(self, points: np.ndarray) -> np.ndarray:
        """Pixels -> ground (x, y) in metres. NaN where the ray misses the surface.

        NaN rather than an extrapolated guess: a ray above the horizon, or one
        crossing only nodata, has no ground point, and inventing one would put a
        track at a confident wrong position. Callers must handle absence -- which
        is the same contract the rangers hold to.
        """
        from geoloc_agent.geometry import bearing_from_pixel

        points = np.asarray(points, dtype=float).reshape(-1, 2)
        out = np.full((len(points), 2), np.nan)
        for index, (u, v) in enumerate(points):
            bearing = bearing_from_pixel(u, v, self.frame.intrinsics, self.frame.pose)
            distance = self.surface.raycast(self.frame.pose.t, bearing,
                                            max_distance=self.max_distance, step=self.step)
            if np.isfinite(distance) and distance > 0:
                out[index] = (self.frame.pose.t + bearing * distance)[:2]
        return out

    # -- ground -> image ---------------------------------------------------

    def abs_to_rel(self, points: np.ndarray) -> np.ndarray:
        """Ground (x, y) -> pixels. NaN for anything behind the camera.

        Height comes from the surface model, so this is the true inverse of
        ``rel_to_abs`` rather than a planar approximation of it.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        heights = np.asarray(self.surface.height_at(points[:, 0], points[:, 1]),
                             dtype=float).reshape(-1)
        world = np.column_stack([points, heights])

        pose = self.frame.pose
        camera = (pose.R.T @ (world - pose.t).T).T
        intrinsics = self.frame.intrinsics
        out = np.full((len(points), 2), np.nan)
        in_front = camera[:, 2] > 1e-6
        out[in_front, 0] = intrinsics.fx * camera[in_front, 0] / camera[in_front, 2] + intrinsics.cx
        out[in_front, 1] = intrinsics.fy * camera[in_front, 1] / camera[in_front, 2] + intrinsics.cy
        out[~np.isfinite(heights)] = np.nan
        return out

    # -- convenience -------------------------------------------------------

    def reproject_into(self, other: Frame, points: np.ndarray) -> np.ndarray:
        """Pixels in this frame -> pixels in ``other``, via the ground.

        The operation a tracker actually wants: where did this box go. Composing
        two absolute transforms means the answer never depends on the frames
        being adjacent, so a track can survive an occlusion of arbitrary length
        without its position drifting in the meantime.
        """
        ground = self.rel_to_abs(points)
        return type(self)(other, self.surface, self.max_distance, self.step).abs_to_rel(ground)


def as_trackers_transformation(transformation: GeoGroundTransformation):
    """Wrap as a `trackers.CoordinatesTransformation` for their tracker classes."""
    from trackers import CoordinatesTransformation

    class _Adapter(CoordinatesTransformation):
        def abs_to_rel(self, points: np.ndarray) -> np.ndarray:
            return transformation.abs_to_rel(points)

        def rel_to_abs(self, points: np.ndarray) -> np.ndarray:
            return transformation.rel_to_abs(points)

    return _Adapter()
