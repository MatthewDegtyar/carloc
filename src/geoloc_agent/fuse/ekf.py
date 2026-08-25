"""EKF over object position, updated by bearings and optional ranges.

State is the object's 3D position in the world ENU frame. The motion model is
"static plus slow drift": objects of interest here are parked cars, people and
street furniture, and the process noise absorbs both genuine slow motion and
bounding-box centroid drift on the object.

Two details carry most of the honesty of this filter:

* **Pose uncertainty enters the measurement covariance, not the state.** A GPS
  error at the camera and a bearing error at the object are indistinguishable
  from a single view, so camera-centre covariance is projected through the
  measurement Jacobian and added to R. Heading error adds straight into the
  azimuth variance, because a yaw error rotates the bearing one-for-one.
* **Joseph-form covariance update.** It costs a matrix multiply and it keeps the
  covariance symmetric positive-definite over long tracks instead of slowly
  rotting into an indefinite matrix that produces nonsense NEES.
"""

from __future__ import annotations

import numpy as np

from geoloc_agent.contracts import Observation, Pose
from geoloc_agent.geometry import angular_difference, azimuth_elevation, perpendicular_projector


def bearing_jacobian(delta: np.ndarray) -> np.ndarray:
    """d(azimuth, elevation) / d(object position). ``delta`` is object - camera."""
    dx, dy, dz = (float(v) for v in delta)
    horizontal_sq = dx * dx + dy * dy
    horizontal = np.sqrt(max(horizontal_sq, 1e-18))
    range_sq = max(horizontal_sq + dz * dz, 1e-18)
    return np.array(
        [
            [-dy / max(horizontal_sq, 1e-18), dx / max(horizontal_sq, 1e-18), 0.0],
            [
                -dz * dx / (range_sq * horizontal),
                -dz * dy / (range_sq * horizontal),
                horizontal / range_sq,
            ],
        ]
    )


def initial_state(
    obs: Observation, prior_range: float = 30.0, range_sigma: float = 25.0
) -> tuple[np.ndarray, np.ndarray]:
    """Seed a track from a single bearing.

    One bearing constrains two of three degrees of freedom. The covariance says
    so explicitly: tight across the ray, deliberately enormous along it. Seeding
    with an isotropic blob instead is the classic way to make a filter
    overconfident from birth and never recover.
    """
    if obs.has_range:
        range_estimate = obs.range.value
        along_sigma = obs.range.sigma
    else:
        range_estimate = float(prior_range)
        along_sigma = float(range_sigma)

    mean = obs.origin + obs.bearing * range_estimate
    perpendicular_sigma = max(range_estimate * obs.bearing_sigma, 1e-3)
    along = np.outer(obs.bearing, obs.bearing) * along_sigma**2
    across = perpendicular_projector(obs.bearing) * perpendicular_sigma**2
    cov = along + across + obs.origin_cov
    return mean, 0.5 * (cov + cov.T)


class PositionEKF:
    """Extended Kalman filter on a 3D point with bearing and range updates."""

    def __init__(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        process_noise_per_s: float = 0.05,
        max_mahalanobis: float = 9.21,  # chi-square 2 dof, 99%
    ) -> None:
        self.mean = np.asarray(mean, dtype=float).reshape(3)
        self.cov = np.asarray(cov, dtype=float).reshape(3, 3)
        self.process_noise_per_s = float(process_noise_per_s)
        self.max_mahalanobis = float(max_mahalanobis)

    # -- prediction -------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Static object; drift only. Never let dt go negative on out-of-order data."""
        if dt <= 0:
            return
        self.cov = self.cov + np.eye(3) * (self.process_noise_per_s**2 * dt)

    # -- measurement models ----------------------------------------------

    def predicted_bearing(self, origin: np.ndarray) -> tuple[float, float, np.ndarray]:
        delta = self.mean - np.asarray(origin, dtype=float)
        az, el = azimuth_elevation(delta / max(np.linalg.norm(delta), 1e-12))
        return az, el, delta

    def innovation(
        self, obs: Observation, pose: Pose | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (residual, innovation covariance S, Jacobian H)."""
        az_pred, el_pred, delta = self.predicted_bearing(obs.origin)
        az_meas, el_meas = azimuth_elevation(obs.bearing)
        residual = np.array(
            [angular_difference(az_meas, az_pred), angular_difference(el_meas, el_pred)]
        )
        H = bearing_jacobian(delta)

        # Angular measurement noise. Azimuth uncertainty inflates as 1/cos(el)
        # because a fixed angular error spans more azimuth near the poles.
        cos_el = max(abs(np.cos(el_meas)), 1e-6)
        R = np.diag([(obs.bearing_sigma / cos_el) ** 2, obs.bearing_sigma**2])
        if pose is not None:
            rot_cov = pose.rotation_cov
            R = R + np.diag([rot_cov[2, 2], rot_cov[0, 0]])
        # Camera-centre uncertainty is indistinguishable from bearing error.
        R = R + H @ obs.origin_cov @ H.T

        S = H @ self.cov @ H.T + R
        return residual, 0.5 * (S + S.T), H

    def mahalanobis(self, obs: Observation, pose: Pose | None = None) -> float:
        residual, S, _ = self.innovation(obs, pose)
        try:
            return float(residual @ np.linalg.solve(S, residual))
        except np.linalg.LinAlgError:
            return float("inf")

    # -- updates ----------------------------------------------------------

    def update_bearing(
        self, obs: Observation, pose: Pose | None = None, iterations: int = 2
    ) -> bool:
        """Iterated EKF bearing update. Returns False if gated out.

        The iteration re-evaluates the Jacobian at the updated estimate. For
        bearings-only geometry that matters: the measurement is strongly
        nonlinear in range, so a Jacobian evaluated at a stale mean both moves
        the estimate to the wrong place and misstates how much was learned.
        """
        residual, S, H = self.innovation(obs, pose)
        try:
            distance = float(residual @ np.linalg.solve(S, residual))
        except np.linalg.LinAlgError:
            return False
        if not np.isfinite(distance) or distance > self.max_mahalanobis:
            return False

        prior_mean = self.mean.copy()
        prior_cov = self.cov.copy()
        estimate = prior_mean.copy()
        K = np.zeros((3, 2))
        R = S - H @ prior_cov @ H.T

        for _ in range(max(1, iterations)):
            saved = self.mean
            self.mean = estimate
            residual, S, H = self.innovation(obs, pose)
            self.mean = saved
            R = S - H @ prior_cov @ H.T
            try:
                K = prior_cov @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return False
            # The (prior - estimate) term is what makes this an iterated update
            # rather than repeated independent updates from a moving prior.
            estimate = prior_mean + K @ (residual - H @ (prior_mean - estimate))

        self.mean = estimate
        self.cov = prior_cov
        self._joseph(K, H, R)
        return True

    def update_range(self, obs: Observation) -> bool:
        if not obs.has_range:
            return False
        delta = self.mean - obs.origin
        predicted = float(np.linalg.norm(delta))
        if predicted < 1e-6:
            return False
        H = (delta / predicted).reshape(1, 3)
        R = np.array([[obs.range.sigma**2]])
        S = H @ self.cov @ H.T + R
        residual = np.array([obs.range.value - predicted])
        if float(residual[0] ** 2 / S[0, 0]) > self.max_mahalanobis:
            return False
        K = self.cov @ H.T @ np.linalg.inv(S)
        self.mean = self.mean + (K @ residual).reshape(3)
        self._joseph(K, H, R)
        return True

    def _joseph(self, K: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Joseph form: stays symmetric positive-definite where (I-KH)P does not."""
        A = np.eye(3) - K @ H
        cov = A @ self.cov @ A.T + K @ R @ K.T
        self.cov = 0.5 * (cov + cov.T)

    # -- diagnostics ------------------------------------------------------

    def nees(self, truth: np.ndarray) -> float:
        """Normalised estimation error squared. Should be chi-square, 3 dof.

        This is the number that says whether the filter is honest. Consistently
        above the chi-square bound means overconfident; consistently below means
        the covariance is inflated and the filter is throwing away information.
        """
        error = self.mean - np.asarray(truth, dtype=float).reshape(3)
        try:
            return float(error @ np.linalg.solve(self.cov, error))
        except np.linalg.LinAlgError:
            return float("inf")
