"""Triangulation from N bearings, in two stages.

**Stage 1 -- linear initialisation.** Weighted least squares on the
perpendicular residual. For observation ``i`` with camera centre ``o_i`` and unit
bearing ``b_i``, the residual of a candidate point ``x`` is ``P_i (x - o_i)``
where ``P_i = I - b_i b_i^T``. That residual has standard deviation
``r_i * sigma_i``, so the weight is ``1 / (r_i sigma_i)^2``. Weighting by range
matters: unweighted midpoint triangulation lets a distant, nearly-parallel
bearing drag the estimate as hard as a close, well-conditioned one.

**Stage 2 -- maximum likelihood refinement.** The perpendicular-distance
objective is convenient but not statistically optimal: it minimises a metric
distance where the actual noise is angular. The difference is not academic. It
leaves the estimator's true error variance roughly 20-30% above the covariance
it reports, which shows up directly as NEES sitting above the chi-square band --
the filter claiming to be better than it is. So the linear solution is used only
as an initial guess, and Gauss-Newton then minimises the *angular* residual,
which is the real measurement. The information matrix of that second stage is
the Cramer-Rao bound, and reporting it is what makes the covariance honest.

The degenerate case falls out for free either way: when all bearings are
near-parallel the information along the line of sight goes to zero and the
covariance inflates in exactly that direction, instead of quietly returning a
confident wrong answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from geoloc_agent.contracts import Observation, RangeMeas, RangeMethod
from geoloc_agent.geometry import (
    angular_difference,
    azimuth_elevation,
    perpendicular_projector,
)
from geoloc_agent.range.base import Ranger

MAX_COND = 1e12
MIN_RANGE = 0.5  # m; floors the weighting range so a collapsed solve cannot explode weights


@dataclass
class TriangulationResult:
    position: np.ndarray
    cov: np.ndarray
    n_obs: int
    ok: bool
    reason: str = ""
    condition_number: float = float("inf")
    max_parallax_rad: float = 0.0

    @property
    def sigma_xyz(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, None))


def _angular_information(
    position: np.ndarray, observations: Sequence[Observation]
) -> tuple[np.ndarray, np.ndarray]:
    """Fisher information and gradient of the angular log-likelihood at ``position``."""
    from geoloc_agent.fuse.ekf import bearing_jacobian

    information = np.zeros((3, 3))
    gradient = np.zeros(3)
    for obs in observations:
        delta = position - obs.origin
        range_m = float(np.linalg.norm(delta))
        if range_m < 1e-6:
            continue
        az_pred, el_pred = azimuth_elevation(delta / range_m)
        az_meas, el_meas = azimuth_elevation(obs.bearing)
        residual = np.array(
            [angular_difference(az_meas, az_pred), angular_difference(el_meas, el_pred)]
        )
        H = bearing_jacobian(delta)
        cos_el = max(abs(np.cos(el_pred)), 1e-6)
        R = np.diag([(obs.bearing_sigma / cos_el) ** 2, obs.bearing_sigma**2])
        R = R + H @ obs.origin_cov @ H.T
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            continue
        information += H.T @ R_inv @ H
        gradient += H.T @ R_inv @ residual
    return information, gradient


def _refine_angular(
    position: np.ndarray, observations: Sequence[Observation], iterations: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Newton on the angular residual. Returns (position, information)."""
    x = np.asarray(position, dtype=float).copy()
    information = np.zeros((3, 3))
    for _ in range(iterations):
        information, gradient = _angular_information(x, observations)
        try:
            step = np.linalg.solve(information + 1e-12 * np.eye(3), gradient)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        # Cap the step so a poorly conditioned early iteration cannot throw the
        # estimate somewhere the linearisation is meaningless.
        step_norm = float(np.linalg.norm(step))
        max_step = max(0.5 * float(np.linalg.norm(x - observations[0].origin)), 1.0)
        if step_norm > max_step:
            step = step * (max_step / step_norm)
        x = x + step
        if step_norm < 1e-9:
            break
    information, _ = _angular_information(x, observations)
    return x, information


def triangulate(
    observations: Sequence[Observation],
    prior_range: float = 30.0,
    iterations: int = 3,
    ridge: float = 1e-9,
) -> TriangulationResult:
    """N bearings -> position estimate + 3x3 covariance.

    Iterates because the weights depend on the per-observation range, which
    depends on the estimate. Three passes is plenty; it converges immediately
    once the ranges are roughly right.
    """
    obs = list(observations)
    if len(obs) < 2:
        return TriangulationResult(
            position=np.full(3, np.nan), cov=np.full((3, 3), np.inf), n_obs=len(obs),
            ok=False, reason="need at least two bearings to triangulate",
        )

    origins = np.array([o.origin for o in obs])
    bearings = np.array([o.bearing for o in obs])
    sigmas = np.array([o.bearing_sigma for o in obs])
    projectors = np.array([perpendicular_projector(b) for b in bearings])

    ranges = np.full(len(obs), float(prior_range))
    position = np.full(3, np.nan)
    information = np.zeros((3, 3))

    for _ in range(max(1, iterations)):
        information = np.zeros((3, 3))
        rhs = np.zeros(3)
        for i in range(len(obs)):
            # Perpendicular residual sigma; add the camera-centre uncertainty
            # projected perpendicular to the ray, so GPS noise widens the fix.
            perp_var = (ranges[i] * sigmas[i]) ** 2
            origin_perp = projectors[i] @ obs[i].origin_cov @ projectors[i].T
            perp_var += max(np.trace(origin_perp) / 2.0, 0.0)
            weight = 1.0 / max(perp_var, 1e-12)
            information += weight * projectors[i]
            rhs += weight * (projectors[i] @ origins[i])

        regularised = information + ridge * np.eye(3)
        try:
            position = np.linalg.solve(regularised, rhs)
        except np.linalg.LinAlgError:
            return TriangulationResult(
                position=np.full(3, np.nan), cov=np.full((3, 3), np.inf), n_obs=len(obs),
                ok=False, reason="information matrix is singular (all bearings parallel)",
            )
        # Refresh the per-observation ranges for the next weighting pass.
        ranges = np.maximum(np.linalg.norm(position - origins, axis=1), MIN_RANGE)

    # Largest parallax between any pair of *measured* bearings. This is intrinsic
    # to the geometry, so unlike anything computed at an estimated position it
    # cannot be flattered by a bad estimate.
    max_parallax = 0.0
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            cos_angle = float(np.clip(bearings[i] @ bearings[j], -1.0, 1.0))
            max_parallax = max(max_parallax, float(np.arccos(cos_angle)))

    # Degeneracy is judged on the *linear* stage, before any refinement.
    #
    # This ordering is load-bearing. With near-parallel bearings the linear
    # information matrix is rank deficient and the regularised solve returns the
    # minimum-norm point, which sits near the cameras rather than anywhere along
    # the rays. Gauss-Newton started from there converges to an arbitrary nearby
    # point -- and at a nearby point those same widely-spaced cameras subtend a
    # large angle, so the refined information matrix looks *well* conditioned.
    # Testing the refined matrix would therefore report a tight covariance on
    # exactly the geometry that has none. Test the honest one first.
    linear_eigenvalues = np.linalg.eigvalsh(information)
    condition = float(linear_eigenvalues.max() / max(linear_eigenvalues.min(), 1e-30))
    if linear_eigenvalues.min() <= 0 or condition > MAX_COND:
        return TriangulationResult(
            position=position, cov=np.full((3, 3), np.inf), n_obs=len(obs), ok=False,
            reason=f"degenerate geometry, condition number {condition:.2e}",
            condition_number=condition, max_parallax_rad=max_parallax,
        )
    if not np.all(np.isfinite(position)):
        return TriangulationResult(
            position=position, cov=np.full((3, 3), np.inf), n_obs=len(obs), ok=False,
            reason="linear stage produced a non-finite position",
            max_parallax_rad=max_parallax,
        )
    behind = [i for i in range(len(obs)) if (position - origins[i]) @ bearings[i] <= 0]
    if behind:
        return TriangulationResult(
            position=position, cov=np.full((3, 3), np.inf), n_obs=len(obs), ok=False,
            reason=f"solution lies behind {len(behind)} camera(s)",
            condition_number=condition, max_parallax_rad=max_parallax,
        )

    # Stage 2: maximum-likelihood refinement on the angular residual. This is
    # what makes the reported covariance match the estimator's real spread.
    position, information = _refine_angular(position, obs)

    eigenvalues = np.linalg.eigvalsh(information)
    condition = float(eigenvalues.max() / max(eigenvalues.min(), 1e-30))
    if eigenvalues.min() <= 0 or condition > MAX_COND:
        return TriangulationResult(
            position=position, cov=np.full((3, 3), np.inf), n_obs=len(obs), ok=False,
            reason=f"degenerate geometry after refinement, condition {condition:.2e}",
            condition_number=condition, max_parallax_rad=max_parallax,
        )

    cov = np.linalg.inv(information)
    cov = 0.5 * (cov + cov.T)  # kill asymmetry from round-off

    behind = [i for i in range(len(obs)) if (position - origins[i]) @ bearings[i] <= 0]
    if behind:
        return TriangulationResult(
            position=position, cov=cov, n_obs=len(obs), ok=False,
            reason=f"refined solution lies behind {len(behind)} camera(s)",
            condition_number=condition, max_parallax_rad=max_parallax,
        )

    return TriangulationResult(
        position=position, cov=cov, n_obs=len(obs), ok=True,
        condition_number=condition, max_parallax_rad=max_parallax,
    )


class TriangulationRanger(Ranger):
    """Wraps ``triangulate`` behind the Ranger interface."""

    method = RangeMethod.TRIANGULATION

    def __init__(
        self, prior_range: float = 30.0, min_parallax_rad: float = np.radians(0.5)
    ) -> None:
        self.prior_range = prior_range
        self.min_parallax_rad = min_parallax_rad

    def range_for(self, obs: Observation, history: Sequence[Observation]) -> RangeMeas:
        result = triangulate([*history, obs], prior_range=self.prior_range)
        if not result.ok:
            return RangeMeas.invalid(self.method, result.reason)
        if result.max_parallax_rad < self.min_parallax_rad:
            return RangeMeas.invalid(
                self.method,
                f"parallax {np.degrees(result.max_parallax_rad):.3f} deg below "
                f"{np.degrees(self.min_parallax_rad):.3f} deg threshold",
            )
        delta = result.position - obs.origin
        value = float(np.linalg.norm(delta))
        if value <= 0:
            return RangeMeas.invalid(self.method, "non-positive range")
        # Project the position covariance onto the line of sight.
        direction = delta / value
        sigma = float(np.sqrt(max(direction @ result.cov @ direction, 1e-12)))
        return RangeMeas(value=value, sigma=sigma, method=self.method, valid=True)
