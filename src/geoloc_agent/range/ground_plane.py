"""Ranging by intersecting the bearing with the ground.

For a downward-looking airborne camera this replaces both triangulation and
monocular depth. A camera at a known height above a known surface needs no
parallax and no learned depth: the range to whatever a pixel is looking at is
fixed by where that ray meets the ground.

That is a much better-conditioned problem than the ground-vehicle case this
project started with. Triangulation needs a perpendicular baseline, and forward
motion supplies none -- the failure that `fuse/degenerate.py` exists to catch.
Here the baseline is irrelevant; a single frame is enough.

It brings its own degeneracy, though, and it is just as sharp. For a flat plane
at height `h` below the camera, a ray at depression angle `theta` below the
horizon meets the ground at

    R = h / sin(theta)

so as the ray flattens towards the horizon the range runs away to infinity. Near
the horizon a fraction of a degree of attitude error moves the intersection by
hundreds of metres, and at or above the horizon there is no intersection at all.
This module treats that the same way the triangulation path treats a missing
baseline: it is detected, it is reported with a reason, and no number is
returned. The failure mode is different, the discipline is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from geoloc_agent.contracts import Observation, RangeMeas, RangeMethod
from geoloc_agent.fuse.degenerate import DEFAULT_MAX_RELATIVE_RANGE_SIGMA
from geoloc_agent.range.base import Ranger

DEFAULT_ATTITUDE_SIGMA_RAD = np.radians(0.9)
"""Attitude uncertainty of the platform, one sigma per axis.

The dominant error term for airborne ranging, and the reason the abstract of
every paper on this subject leads with pose rather than perception. A consumer
airframe with a good IMU and RTK holds attitude to a few tenths of a degree, and
0.5 deg was the original defensible guess.

**0.9 deg is measured, not guessed.** Repeated ground fixes of the same static
object scatter by an amount that is itself a measurement of the total error, and
comparing that scatter to the analytic prediction calibrates this constant with
no ground truth at all -- see `analysis/track_geo.py`. Over 59 tracks on AirZoo
`guangchang/12-14` the empirical-to-analytic ratio crosses 1.0 here:

    0.50 deg -> 1.82   optimistic
    0.75 deg -> 1.21
    0.90 deg -> ~1.0   consistent
    1.25 deg -> 0.73   pessimistic

Read it as an *effective* sigma, not the platform's true attitude error. It is
the one free parameter in the model, so every unmodelled error lands on it --
bounding-box centroid drift, surface-model error, and association slop included.
That makes it right for producing honest covariances and wrong for quoting as an
IMU specification."""

DEFAULT_ALTITUDE_SIGMA_M = 0.10
"""Height above the surface, one sigma. RTK vertical is centimetres; the surface
model it is measured against is not, so this is dominated by the DSM."""

DEFAULT_SURFACE_SIGMA_M = 0.30
"""Vertical accuracy of the surface model itself. Photogrammetric DSMs run tens
of centimetres over hard ground and far worse over vegetation and water."""

MIN_DEPRESSION_RAD = np.radians(1.0)
"""Below this the ray is treated as parallel to the ground regardless of what the
error propagation says. Guards the arithmetic, not the statistics: at a tenth of
a degree `1/sin(theta)` is numerically fine and physically meaningless."""


@dataclass(frozen=True)
class GroundGeometry:
    """What the ranger resolved, kept for reporting and for the degeneracy note."""

    range_m: float
    sigma_m: float
    depression_rad: float
    height_m: float
    point: np.ndarray
    """Intersection in world coordinates."""


def range_sigma(range_m: float, height_m: float, depression_rad: float,
                attitude_sigma: float, altitude_sigma: float,
                surface_sigma: float) -> float:
    """First-order propagation of pose and surface error into range.

    From ``R = h / sin(theta)``:

        dR/dh     =  1 / sin(theta)      =  R / h
        dR/dtheta = -h cos(theta) / sin^2(theta) = -R / tan(theta)

    The two terms say different things and it is worth keeping them apart. The
    height term scales linearly with range -- twice as far, twice the error, and
    nothing to be done about it. The attitude term scales as ``1 / tan(theta)``,
    which is the one that explodes: at 45 degrees it equals the range, at 5
    degrees it is eleven times the range. Vertical error in the surface enters
    exactly as height error does, because only their difference is observable.
    """
    if depression_rad <= 0:
        return float("inf")
    d_height = range_m / max(height_m, 1e-6)
    d_attitude = range_m / max(np.tan(depression_rad), 1e-9)
    height_var = altitude_sigma**2 + surface_sigma**2
    return float(np.hypot(d_height * np.sqrt(height_var), d_attitude * attitude_sigma))


def intersect_plane(origin: np.ndarray, bearing: np.ndarray,
                    ground_z: float) -> tuple[float, np.ndarray] | None:
    """Where a ray meets the horizontal plane ``z = ground_z``.

    Returns None when the ray points away from the plane -- level, upward, or
    already below it. All three are "no answer", not "a large answer".
    """
    origin = np.asarray(origin, dtype=float)
    bearing = np.asarray(bearing, dtype=float)
    height = float(origin[2] - ground_z)
    if height <= 0:
        return None
    descent = -float(bearing[2])          # positive when the ray goes downward
    if descent <= 1e-9:
        return None
    distance = height / descent
    return distance, origin + bearing * distance


class GroundPlaneRanger(Ranger):
    """Range by intersecting the bearing with a flat ground plane.

    The simplest surface model, and the right one to start from: it needs only a
    ground height, it has a closed-form solution, and its error propagation is
    analytic rather than sampled. A DSM-backed version handles relief, but it
    reduces to this locally, so getting this right first is not wasted work.
    """

    method = RangeMethod.GROUND_PLANE

    def __init__(
        self,
        ground_z: float = 0.0,
        attitude_sigma: float = DEFAULT_ATTITUDE_SIGMA_RAD,
        altitude_sigma: float = DEFAULT_ALTITUDE_SIGMA_M,
        surface_sigma: float = DEFAULT_SURFACE_SIGMA_M,
        max_relative_sigma: float = DEFAULT_MAX_RELATIVE_RANGE_SIGMA,
        max_range_m: float | None = None,
    ) -> None:
        self.ground_z = float(ground_z)
        self.attitude_sigma = float(attitude_sigma)
        self.altitude_sigma = float(altitude_sigma)
        self.surface_sigma = float(surface_sigma)
        self.max_relative_sigma = float(max_relative_sigma)
        self.max_range_m = max_range_m

    def solve(self, origin: np.ndarray, bearing: np.ndarray) -> GroundGeometry | None:
        hit = intersect_plane(origin, bearing, self.ground_z)
        if hit is None:
            return None
        distance, point = hit
        height = float(np.asarray(origin, dtype=float)[2] - self.ground_z)
        # Depression below the horizon, from the vertical component of the unit
        # bearing. Equivalent to asin(h / R) and cheaper.
        depression = float(np.arcsin(np.clip(-bearing[2], -1.0, 1.0)))
        sigma = range_sigma(distance, height, depression, self.attitude_sigma,
                            self.altitude_sigma, self.surface_sigma)
        return GroundGeometry(range_m=distance, sigma_m=sigma, depression_rad=depression,
                              height_m=height, point=point)

    def range_for(self, obs: Observation, history: Sequence[Observation]) -> RangeMeas:
        """History is unused: a single frame determines the answer."""
        del history

        origin = np.asarray(obs.origin, dtype=float)
        if origin[2] <= self.ground_z:
            return RangeMeas.invalid(
                self.method,
                f"camera at z={origin[2]:.1f} m is at or below the ground plane "
                f"z={self.ground_z:.1f} m",
            )

        geometry = self.solve(origin, np.asarray(obs.bearing, dtype=float))
        if geometry is None:
            return RangeMeas.invalid(
                self.method, "bearing does not descend; ray never meets the ground"
            )

        if geometry.depression_rad < MIN_DEPRESSION_RAD:
            return RangeMeas.invalid(
                self.method,
                f"depression {np.degrees(geometry.depression_rad):.2f} deg is below the "
                f"{np.degrees(MIN_DEPRESSION_RAD):.1f} deg floor; ray is effectively "
                f"parallel to the ground",
            )

        relative = geometry.sigma_m / geometry.range_m
        if relative > self.max_relative_sigma:
            return RangeMeas.invalid(
                self.method,
                f"range {geometry.range_m:.0f} m +/- {geometry.sigma_m:.0f} m is "
                f"{relative:.0%} of range, over the {self.max_relative_sigma:.0%} bar "
                f"-- {np.degrees(geometry.depression_rad):.1f} deg depression is too "
                f"shallow at this height",
            )

        if self.max_range_m is not None and geometry.range_m > self.max_range_m:
            return RangeMeas.invalid(
                self.method,
                f"range {geometry.range_m:.0f} m is beyond the declared "
                f"{self.max_range_m:.0f} m envelope",
            )

        return RangeMeas(
            value=geometry.range_m,
            sigma=geometry.sigma_m,
            method=self.method,
            reason=(f"ground intersection at {np.degrees(geometry.depression_rad):.1f} deg "
                    f"depression, {geometry.height_m:.0f} m AGL"),
        )


def usable_depression(height_m: float, max_range_m: float,
                      attitude_sigma: float = DEFAULT_ATTITUDE_SIGMA_RAD,
                      altitude_sigma: float = DEFAULT_ALTITUDE_SIGMA_M,
                      surface_sigma: float = DEFAULT_SURFACE_SIGMA_M,
                      max_relative_sigma: float = DEFAULT_MAX_RELATIVE_RANGE_SIGMA) -> float:
    """Shallowest depression angle that still meets the relative-sigma bar.

    The planning counterpart of the check inside the ranger: given a flight
    height, it says how far from nadir the camera may look before ranging stops
    being trustworthy. Useful for choosing an altitude rather than discovering
    the limit in the air.

    Returns the angle in radians, or ``inf`` if no angle up to nadir qualifies.
    """
    for degrees in np.arange(1.0, 90.01, 0.25):
        theta = np.radians(degrees)
        distance = height_m / np.sin(theta)
        if distance > max_range_m:
            continue
        sigma = range_sigma(distance, height_m, theta, attitude_sigma,
                            altitude_sigma, surface_sigma)
        if sigma / distance <= max_relative_sigma:
            return float(theta)
    return float("inf")


class DsmRanger(GroundPlaneRanger):
    """Range by intersecting the bearing with a digital surface model.

    The flat-plane ranger with its one assumption removed. Everything else --
    the error model, the near-horizon refusal, the relative-sigma bar -- carries
    over unchanged, because none of it depended on the surface being flat.

    Two things do change. The intersection is found by marching rather than in
    closed form, so it costs more; and the local surface slope now matters. A
    ray landing on a slope tilted towards the camera meets it sooner and at a
    steeper effective angle than the horizontal-plane formula predicts, so the
    depression used for error propagation is measured from the geometry actually
    solved rather than from the bearing alone.
    """

    def __init__(self, dsm, max_distance: float = 2000.0, step: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dsm = dsm
        self.max_distance = float(max_distance)
        self.step = float(step)

    def solve(self, origin: np.ndarray, bearing: np.ndarray) -> GroundGeometry | None:
        origin = np.asarray(origin, dtype=float)
        bearing = np.asarray(bearing, dtype=float)
        distance = self.dsm.raycast(origin, bearing, max_distance=self.max_distance,
                                    step=self.step)
        if not np.isfinite(distance) or distance <= 0:
            return None
        point = origin + bearing * distance
        height = float(origin[2] - point[2])
        if height <= 0:
            return None
        # Effective depression of the solved geometry, not of the raw bearing: on
        # sloping ground these differ, and the error model wants the one that
        # actually produced this range.
        depression = float(np.arcsin(np.clip(height / distance, -1.0, 1.0)))
        sigma = range_sigma(distance, height, depression, self.attitude_sigma,
                            self.altitude_sigma, self.surface_sigma)
        return GroundGeometry(range_m=distance, sigma_m=sigma, depression_rad=depression,
                              height_m=height, point=point)

    def range_for(self, obs: Observation, history: Sequence[Observation]) -> RangeMeas:
        del history
        geometry = self.solve(np.asarray(obs.origin, dtype=float),
                              np.asarray(obs.bearing, dtype=float))
        if geometry is None:
            return RangeMeas.invalid(
                self.method,
                "ray does not meet the surface model within range -- pointing above "
                "the terrain, or crossing only nodata (water is a hole in a "
                "photogrammetric DSM, not a surface)",
            )
        if geometry.depression_rad < MIN_DEPRESSION_RAD:
            return RangeMeas.invalid(
                self.method,
                f"effective depression {np.degrees(geometry.depression_rad):.2f} deg "
                f"is below the {np.degrees(MIN_DEPRESSION_RAD):.1f} deg floor",
            )
        relative = geometry.sigma_m / geometry.range_m
        if relative > self.max_relative_sigma:
            return RangeMeas.invalid(
                self.method,
                f"range {geometry.range_m:.0f} m +/- {geometry.sigma_m:.0f} m is "
                f"{relative:.0%} of range, over the {self.max_relative_sigma:.0%} bar",
            )
        if self.max_range_m is not None and geometry.range_m > self.max_range_m:
            return RangeMeas.invalid(
                self.method,
                f"range {geometry.range_m:.0f} m is beyond the declared "
                f"{self.max_range_m:.0f} m envelope",
            )
        return RangeMeas(
            value=geometry.range_m, sigma=geometry.sigma_m, method=self.method,
            reason=(f"surface intersection at {np.degrees(geometry.depression_rad):.1f} deg "
                    f"effective depression, {geometry.height_m:.0f} m below camera"),
        )
