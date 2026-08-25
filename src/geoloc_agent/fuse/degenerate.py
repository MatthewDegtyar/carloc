"""Degenerate-geometry detection.

The failure this catches: a camera moving straight forward gets almost no
perpendicular baseline on anything near its own optical axis, so range is
essentially unobservable. Triangulation still returns *a* number, and that number
looks like every other number in the output. Flagging it is the difference
between a track an operator can act on and a track that is a guess wearing a
coordinate.

Two independent tests, because they fail in different situations:

1. **Parallax.** Geometric and cheap. Directly measures whether the camera ever
   moved perpendicular to the line of sight.
2. **Relative along-range sigma.** Comes from the filter's own covariance, so it
   also catches cases where the geometry looked adequate but the measurements
   were too noisy to exploit it.

Both of those are evaluated at the *estimated* position, which is the weakness
they share: a short track that has landed somewhere badly wrong will assess its
geometry at that wrong place and conclude everything is fine. A two-observation
fix on a 1 m baseline that should have been 95 m away but landed at 17 m reports
a metre of uncertainty and is off by seventy -- and nothing above catches it,
because at 17 m a 1 m baseline really would be adequate.

So there are two further tests that do not depend on the estimate at all:

3. **Observation count.** Two bearings give one degree of redundancy. There is
   not enough information to notice that the fix is wrong.
4. **Absolute perpendicular baseline.** A camera that has moved a metre sideways
   has not earned a confident fix on anything at street distance, whatever its
   covariance happens to say.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from geoloc_agent.geometry import parallax_angle, perpendicular_projector

DEFAULT_MIN_PARALLAX_DEG = 1.0
DEFAULT_MAX_RELATIVE_RANGE_SIGMA = 0.25
DEFAULT_MIN_OBS = 3
DEFAULT_MIN_PERP_BASELINE_M = 2.0


@dataclass
class GeometryReport:
    perp_baseline: float
    range_m: float
    parallax_rad: float
    along_range_sigma: float
    relative_range_sigma: float
    degenerate: bool
    reason: str = ""

    @property
    def parallax_deg(self) -> float:
        return float(np.degrees(self.parallax_rad))


def max_perpendicular_baseline(origins: Sequence[np.ndarray], target: np.ndarray) -> float:
    """Largest camera-centre separation perpendicular to the line of sight.

    Uses the mean line of sight rather than any single bearing, so a long track
    that curves gets credit for the whole sweep.
    """
    origins = [np.asarray(o, dtype=float) for o in origins]
    if len(origins) < 2:
        return 0.0
    target = np.asarray(target, dtype=float)
    centre = np.mean(origins, axis=0)
    line_of_sight = target - centre
    norm = np.linalg.norm(line_of_sight)
    if norm < 1e-9:
        return 0.0
    projector = perpendicular_projector(line_of_sight / norm)
    projected = np.array([projector @ o for o in origins])
    # Max pairwise distance among the projected centres.
    best = 0.0
    for i in range(len(projected)):
        for j in range(i + 1, len(projected)):
            best = max(best, float(np.linalg.norm(projected[i] - projected[j])))
    return best


def assess_geometry(
    origins: Sequence[np.ndarray],
    mean: np.ndarray,
    cov: np.ndarray,
    min_parallax_deg: float = DEFAULT_MIN_PARALLAX_DEG,
    max_relative_range_sigma: float = DEFAULT_MAX_RELATIVE_RANGE_SIGMA,
    n_obs: int | None = None,
    min_obs: int = DEFAULT_MIN_OBS,
    min_perp_baseline: float = DEFAULT_MIN_PERP_BASELINE_M,
) -> GeometryReport:
    mean = np.asarray(mean, dtype=float)
    origins = [np.asarray(o, dtype=float) for o in origins]
    latest = origins[-1] if origins else np.zeros(3)
    delta = mean - latest
    range_m = float(np.linalg.norm(delta))

    perp = max_perpendicular_baseline(origins, mean)
    parallax = parallax_angle(perp, range_m)

    if range_m > 1e-9:
        direction = delta / range_m
        along_sigma = float(np.sqrt(max(direction @ np.asarray(cov) @ direction, 0.0)))
    else:
        along_sigma = float("inf")
    relative = along_sigma / max(range_m, 1e-9)

    reasons = []
    if len(origins) < 2:
        reasons.append("single view, range unobservable")
    # Estimate-independent tests first: these still hold when the fix is wrong.
    observed = len(origins) if n_obs is None else n_obs
    if 2 <= observed < min_obs:
        reasons.append(f"only {observed} observations, too few to detect a bad fix")
    if len(origins) >= 2 and perp < min_perp_baseline:
        reasons.append(
            f"perpendicular baseline {perp:.2f} m below the {min_perp_baseline:.2f} m floor"
        )
    if np.degrees(parallax) < min_parallax_deg:
        reasons.append(
            f"parallax {np.degrees(parallax):.2f} deg < {min_parallax_deg:.2f} deg "
            f"(perp baseline {perp:.2f} m at {range_m:.1f} m)"
        )
    if relative > max_relative_range_sigma:
        reasons.append(
            f"along-range sigma {along_sigma:.1f} m is {relative * 100:.0f}% of range"
        )

    return GeometryReport(
        perp_baseline=perp,
        range_m=range_m,
        parallax_rad=parallax,
        along_range_sigma=along_sigma,
        relative_range_sigma=relative,
        degenerate=bool(reasons),
        reason="; ".join(reasons),
    )
