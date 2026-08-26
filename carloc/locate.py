"""Pixel to ground, with an error bar that refuses when it should.

For a camera at height `h` looking down at depression `theta`, a ray meets flat
ground at `R = h / sin(theta)`. Differentiating gives two error terms that behave
completely differently:

    dR/dh     =  R / h        scales with range; unavoidable
    dR/dtheta = -R / tan(theta)   explodes as the ray flattens

At 45 degrees the attitude term equals the range. At 5 degrees it is eleven times
it. Near the horizon a fraction of a degree moves the intersection by hundreds of
metres, and at or above the horizon there is no intersection at all.

So this refuses rather than answers when the geometry cannot support a fix. That
is the same discipline as a triangulation ranger declining without a baseline;
only the failing quantity differs.

The actual intersection uses the terrain, not a plane -- the closed form above is
for reasoning about error, not for computing the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ATTITUDE_SIGMA_RAD = np.radians(0.9)
"""Platform attitude uncertainty, one sigma, and the dominant error term.

0.9 deg rather than the 0.5 a good IMU spec suggests, because it is the model's
one free parameter and every unmodelled error lands on it: box centroid drift,
terrain error, calibration slop. Measured on the previous build by comparing the
scatter of repeated fixes of static objects against this prediction -- the two
agree at 0.9. Right for producing honest covariances, wrong to quote as an IMU
specification."""

ALTITUDE_SIGMA_M = 0.10
SURFACE_SIGMA_M = 0.30
"""Vertical error in the height field. Tens of centimetres over hard ground and
much worse over vegetation, which is why it is not smaller."""

MIN_DEPRESSION_RAD = np.radians(5.0)
"""Below this, refuse outright. At 5 degrees the attitude term is already eleven
times the range; the arithmetic still works and the answer is meaningless."""

MAX_RELATIVE_SIGMA = 0.15
"""A fix whose sigma exceeds this fraction of its range is not reported."""

MAX_ABSOLUTE_SIGMA_M = 1.2
"""Hard cap in metres, and the one that actually binds here.

The relative bar is a statement about the geometry; this is a statement about the
*task*. Cars in a lot sit about 2.5 m apart, so a fix carrying more error than
half that cannot say which car it belongs to -- fixes of one car land further
apart than neighbouring cars do, and no clustering radius can recover the
difference.

It bites hard on this data. At 200 m slant range the sigma runs 5.5 m at the
30 deg depression these flights mostly use and 3.2 m at their best 45 deg, so
most detections are refused. That is the correct outcome: the alternative is a
car count that is an artefact of the merge threshold. Shorter range or a steeper
look fixes it -- sigma scales with range -- and no amount of tuning does."""


@dataclass(frozen=True)
class Fix:
    """One ground position with its uncertainty, or a refusal."""

    x: float
    y: float
    z: float
    range_m: float
    sigma_m: float
    depression_deg: float
    valid: bool
    reason: str = ""

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @classmethod
    def refused(cls, reason: str, depression_deg: float = float("nan")) -> Fix:
        return cls(np.nan, np.nan, np.nan, np.nan, np.inf, depression_deg, False, reason)


def range_sigma(range_m: float, height_m: float, depression_rad: float) -> float:
    """First-order propagation of attitude and height error into range."""
    if depression_rad <= 0:
        return float("inf")
    d_height = range_m / max(height_m, 1e-6)
    d_attitude = range_m / max(np.tan(depression_rad), 1e-9)
    height_variance = ALTITUDE_SIGMA_M**2 + SURFACE_SIGMA_M**2
    return float(np.hypot(d_height * np.sqrt(height_variance),
                          d_attitude * ATTITUDE_SIGMA_RAD))


def locate(frame, terrain, u: float, v: float, max_distance: float = 2000.0,
           step: float = 1.0) -> Fix:
    """Where on the ground is the object at pixel (u, v)?"""
    bearing = frame.bearing(u, v)
    depression = float(np.arcsin(np.clip(-bearing[2], -1.0, 1.0)))
    degrees = float(np.degrees(depression))

    if depression <= 0:
        return Fix.refused("ray points at or above the horizon", degrees)
    if depression < MIN_DEPRESSION_RAD:
        return Fix.refused(
            f"depression {degrees:.1f} deg is below the "
            f"{np.degrees(MIN_DEPRESSION_RAD):.0f} deg floor", degrees)

    distance = terrain.raycast(frame.centre, bearing, max_distance=max_distance, step=step)
    if not np.isfinite(distance) or distance <= 0:
        return Fix.refused(
            "ray does not meet the surface within range -- over the terrain edge, "
            "or crossing only nodata (water is a hole, not a surface)", degrees)

    point = frame.centre + bearing * distance
    height = float(frame.centre[2] - point[2])
    if height <= 0:
        return Fix.refused("intersection is above the camera", degrees)

    # Effective depression of the geometry actually solved. On sloping ground
    # this differs from the bearing's own angle, and the error model wants the
    # one that produced this range.
    effective = float(np.arcsin(np.clip(height / distance, -1.0, 1.0)))
    sigma = range_sigma(distance, height, effective)
    relative = sigma / distance
    if relative > MAX_RELATIVE_SIGMA:
        return Fix.refused(
            f"{distance:.0f} m +/- {sigma:.0f} m is {relative:.0%} of range, "
            f"over the {MAX_RELATIVE_SIGMA:.0%} bar", float(np.degrees(effective)))
    if sigma > MAX_ABSOLUTE_SIGMA_M:
        return Fix.refused(
            f"+/- {sigma:.1f} m is coarser than the {MAX_ABSOLUTE_SIGMA_M:.1f} m needed "
            f"to tell neighbouring cars apart", float(np.degrees(effective)))

    return Fix(float(point[0]), float(point[1]), float(point[2]),
               distance, sigma, float(np.degrees(effective)), True)
