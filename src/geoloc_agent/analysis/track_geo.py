"""Object position and its uncertainty, from a track's history of ground fixes.

Each observation of a tracked object gives one ground position: project the box
through the terrain and you get metres. One observation also gives an *analytic*
uncertainty, propagated from attitude and surface error through
``range/ground_plane.py``.

Several observations of the same object give something the single fix cannot: an
**empirical** uncertainty. A static object seen from a moving platform should
project to the same ground point every time, so the scatter of those points is a
direct measurement of the error -- no ground truth required, only identity, which
is exactly what an image-plane tracker supplies.

That makes the two numbers comparable, and comparing them is the point:

* If the empirical scatter matches the analytic prediction, the uncertainty is
  honest and can be published.
* If the scatter is larger, something unmodelled dominates -- pose error beyond
  the assumed sigma, association mixing two objects, or terrain the surface model
  does not describe.
* If the scatter is smaller, the analytic sigma is pessimistic and range is being
  thrown away.

The ratio is a chi-square statistic, so it is testable rather than a vibe. This
is the same discipline `eval/metrics.py` applies through NEES against ground
truth, except it needs none: it is self-calibration from redundancy alone.

The catch, stated plainly: scatter measures **precision, not accuracy**. Errors
common to every observation -- a constant attitude bias, a datum shift, a
systematically wrong surface height -- move all the fixes together and leave the
scatter untouched. This detects noise and blunders, never bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_FIXES = 3
"""Fewest ground fixes before a covariance is estimated.

Two points define a line and produce a singular covariance; three is the minimum
that says anything about spread in both axes, and even then it says little. The
count travels with the estimate so a consumer can weigh it."""

CHI2_95 = 5.991
"""Chi-square, 2 dof, 95%. The bar a fix must clear to count as an inlier."""


@dataclass(frozen=True)
class TrackGeoEstimate:
    """Where a tracked object is, how sure we are, and whether that is believable."""

    track_id: int
    mean_xy: np.ndarray
    cov_xy: np.ndarray
    """Empirical 2x2 covariance of the ground fixes, in metres squared."""

    analytic_sigma_m: float
    """Median single-fix sigma the ranger predicted for these observations."""

    n_fixes: int
    span_m: float
    """Platform baseline across the observations. Scatter from a stationary
    platform measures almost nothing -- the same error repeats."""

    @property
    def sigma_major_m(self) -> float:
        return float(np.sqrt(max(np.linalg.eigvalsh(self.cov_xy)[-1], 0.0)))

    @property
    def sigma_minor_m(self) -> float:
        return float(np.sqrt(max(np.linalg.eigvalsh(self.cov_xy)[0], 0.0)))

    @property
    def empirical_sigma_m(self) -> float:
        """Single number for the scatter: the radius of the equivalent circle."""
        return float(np.sqrt(max(np.trace(self.cov_xy) / 2.0, 0.0)))

    @property
    def consistency(self) -> float:
        """Empirical spread over analytic prediction. 1.0 is a calibrated filter.

        Above 1 the reported uncertainty is optimistic -- the dangerous direction,
        because it is a confident wrong position. Below 1 it is pessimistic, which
        wastes range but never misleads.
        """
        if self.analytic_sigma_m <= 0:
            return float("nan")
        return self.empirical_sigma_m / self.analytic_sigma_m

    @property
    def verdict(self) -> str:
        ratio = self.consistency
        if not np.isfinite(ratio):
            return "no analytic prediction to compare against"
        if self.n_fixes < MIN_FIXES:
            return f"only {self.n_fixes} fixes; scatter is not yet meaningful"
        if self.span_m < 1.0:
            return "platform barely moved; scatter repeats one error rather than sampling it"
        if ratio > 2.0:
            return (f"scatter {ratio:.1f}x the predicted sigma -- the reported "
                    f"uncertainty is optimistic")
        if ratio < 0.5:
            return f"scatter {ratio:.1f}x predicted -- uncertainty is pessimistic"
        return f"scatter {ratio:.1f}x predicted -- consistent"

    def mahalanobis(self, point_xy) -> float:
        """Squared Mahalanobis distance of a point from this estimate.

        The question an operator asks of a map: could this object be *there*.
        Compare against CHI2_95.
        """
        delta = np.asarray(point_xy, dtype=float).reshape(2) - self.mean_xy
        try:
            return float(delta @ np.linalg.solve(self.cov_xy, delta))
        except np.linalg.LinAlgError:
            return float("inf")

    def ellipse(self, sigma: float = 1.0, points: int = 64) -> np.ndarray:
        """The covariance ellipse as a polygon, for drawing on a map."""
        values, vectors = np.linalg.eigh(self.cov_xy)
        values = np.clip(values, 0.0, None)
        angle = np.linspace(0.0, 2 * np.pi, points)
        unit = np.column_stack([np.cos(angle), np.sin(angle)])
        return self.mean_xy + (unit * sigma * np.sqrt(values)) @ vectors.T


def estimate_from_fixes(track_id: int, fixes: np.ndarray, sigmas: np.ndarray,
                        origins: np.ndarray, trim: bool = True) -> TrackGeoEstimate | None:
    """Combine repeated ground fixes of one object into a position and a spread.

    ``trim`` drops fixes beyond the 95% ellipse of a first pass and refits. A
    single blunder -- one frame where the box jumped to a neighbouring object, or
    the ray grazed a rooftop -- otherwise inflates the covariance enough to hide
    a genuinely good track, and blunders here are not rare: the surface model is
    2.5-D and silhouettes are exactly where it fails.

    Trimming is done once, not iterated, and never removes more than a third of
    the fixes. Iterating shrinks the covariance toward whatever subset agrees,
    which manufactures confidence.
    """
    fixes = np.asarray(fixes, dtype=float).reshape(-1, 2)
    sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
    origins = np.asarray(origins, dtype=float).reshape(-1, 3)

    good = np.isfinite(fixes).all(1) & np.isfinite(sigmas)
    fixes, sigmas, origins = fixes[good], sigmas[good], origins[good]
    if len(fixes) < MIN_FIXES:
        return None

    mean = fixes.mean(axis=0)
    cov = np.cov(fixes.T, ddof=1)

    if trim and len(fixes) >= 5:
        delta = fixes - mean
        try:
            distance = np.einsum("ij,jk,ik->i", delta, np.linalg.inv(cov), delta)
        except np.linalg.LinAlgError:
            distance = np.zeros(len(fixes))
        keep = distance <= CHI2_95
        if keep.sum() >= max(MIN_FIXES, int(np.ceil(len(fixes) * 2 / 3))):
            fixes, sigmas, origins = fixes[keep], sigmas[keep], origins[keep]
            mean = fixes.mean(axis=0)
            cov = np.cov(fixes.T, ddof=1)

    cov = np.atleast_2d(cov)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return None
    # The covariance of the *mean* is what a consumer wants, not of one sample.
    cov_of_mean = cov / len(fixes)

    span = 0.0
    if len(origins) > 1:
        span = float(np.max(np.linalg.norm(origins[:, None, :] - origins[None, :, :], axis=-1)))

    return TrackGeoEstimate(
        track_id=track_id,
        mean_xy=mean,
        cov_xy=cov_of_mean,
        analytic_sigma_m=float(np.median(sigmas)) / np.sqrt(len(fixes)),
        n_fixes=len(fixes),
        span_m=span,
    )
