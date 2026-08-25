"""EKF behaviour, including the failure modes that motivated its design."""

import numpy as np
import pytest

from geoloc_agent.contracts import Observation, Pose, RangeMeas, RangeMethod
from geoloc_agent.fuse.ekf import PositionEKF, bearing_jacobian, initial_state


def bearing_obs(origin, target, sigma=1e-3, **kw):
    bearing = np.asarray(target, float) - np.asarray(origin, float)
    return Observation(
        t=0.0, frame_id=0, origin=np.asarray(origin, float), bearing=bearing,
        bearing_sigma=sigma, **kw,
    )


def test_bearing_jacobian_matches_numerical_differentiation():
    from geoloc_agent.geometry import azimuth_elevation

    delta = np.array([3.0, 20.0, 2.0])
    analytic = bearing_jacobian(delta)
    step = 1e-6
    numeric = np.zeros((2, 3))
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        plus = np.array(azimuth_elevation((delta + offset) / np.linalg.norm(delta + offset)))
        minus = np.array(azimuth_elevation((delta - offset) / np.linalg.norm(delta - offset)))
        numeric[:, axis] = (plus - minus) / (2 * step)
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_initial_state_is_wide_along_the_ray_and_tight_across_it():
    """One bearing constrains two of three degrees of freedom, and says so."""
    obs = bearing_obs(np.zeros(3), np.array([0.0, 50.0, 0.0]))
    mean, cov = initial_state(obs, prior_range=30.0, range_sigma=25.0)
    assert np.allclose(mean, [0.0, 30.0, 0.0])
    along = np.array([0.0, 1.0, 0.0])
    across = np.array([1.0, 0.0, 0.0])
    assert np.sqrt(along @ cov @ along) == pytest.approx(25.0, rel=1e-6)
    assert np.sqrt(across @ cov @ across) < 0.1
    assert np.sqrt(along @ cov @ along) > 100 * np.sqrt(across @ cov @ across)


def test_initial_state_uses_a_range_measurement_when_one_exists():
    obs = bearing_obs(
        np.zeros(3), np.array([0.0, 50.0, 0.0]),
        range=RangeMeas(value=50.0, sigma=1.0, method=RangeMethod.LIDAR),
    )
    mean, cov = initial_state(obs, prior_range=30.0, range_sigma=25.0)
    assert np.allclose(mean, [0.0, 50.0, 0.0])
    along = np.array([0.0, 1.0, 0.0])
    assert np.sqrt(along @ cov @ along) == pytest.approx(1.0, rel=1e-6)


def test_covariance_stays_symmetric_positive_definite_over_many_updates():
    """Joseph form: the reason a long track does not rot into an indefinite matrix."""
    target = np.array([2.0, 40.0, 1.0])
    obs = bearing_obs(np.array([-12.0, 0.0, 0.0]), target)
    mean, cov = initial_state(obs, prior_range=40.0, range_sigma=30.0)
    ekf = PositionEKF(mean, cov)
    rng = np.random.default_rng(0)
    for i in range(200):
        origin = np.array([-12.0 + 24.0 * i / 200, rng.normal(0, 0.1), 0.0])
        ekf.update_bearing(bearing_obs(origin, target))
        assert np.allclose(ekf.cov, ekf.cov.T, atol=1e-12)
        assert np.linalg.eigvalsh(ekf.cov).min() > 0


def test_range_update_shrinks_uncertainty_along_the_ray():
    obs = bearing_obs(np.zeros(3), np.array([0.0, 50.0, 0.0]))
    mean, cov = initial_state(obs, prior_range=50.0, range_sigma=30.0)
    ekf = PositionEKF(mean, cov)
    along = np.array([0.0, 1.0, 0.0])
    before = np.sqrt(along @ ekf.cov @ along)
    ranged = bearing_obs(
        np.zeros(3), np.array([0.0, 50.0, 0.0]),
        range=RangeMeas(value=50.0, sigma=2.0, method=RangeMethod.LIDAR),
    )
    assert ekf.update_range(ranged)
    assert np.sqrt(along @ ekf.cov @ along) < before
    assert np.sqrt(along @ ekf.cov @ along) == pytest.approx(2.0, rel=0.05)


def test_gate_rejects_a_wild_observation():
    obs = bearing_obs(np.zeros(3), np.array([0.0, 50.0, 0.0]))
    mean, cov = initial_state(obs, prior_range=50.0, range_sigma=1.0)
    ekf = PositionEKF(mean, cov)
    # A bearing 90 degrees away cannot belong to this track.
    assert not ekf.update_bearing(bearing_obs(np.zeros(3), np.array([50.0, 0.0, 0.0])))


def test_pose_attitude_uncertainty_widens_the_innovation_covariance():
    obs = bearing_obs(np.zeros(3), np.array([0.0, 50.0, 0.0]))
    mean, cov = initial_state(obs, prior_range=50.0, range_sigma=10.0)
    ekf = PositionEKF(mean, cov)
    _, s_clean, _ = ekf.innovation(obs, Pose(R=np.eye(3), t=np.zeros(3)))
    noisy_cov = np.zeros((6, 6))
    noisy_cov[5, 5] = np.radians(5.0) ** 2  # 5 degrees of heading uncertainty
    _, s_noisy, _ = ekf.innovation(obs, Pose(R=np.eye(3), t=np.zeros(3), cov=noisy_cov))
    assert s_noisy[0, 0] > s_clean[0, 0]


def test_camera_position_uncertainty_widens_the_innovation_covariance():
    """GPS error and bearing error are indistinguishable from a single view."""
    clean = bearing_obs(np.zeros(3), np.array([0.0, 50.0, 0.0]))
    noisy = bearing_obs(
        np.zeros(3), np.array([0.0, 50.0, 0.0]), origin_cov=np.eye(3) * 25.0
    )
    mean, cov = initial_state(clean, prior_range=50.0, range_sigma=10.0)
    ekf = PositionEKF(mean, cov)
    _, s_clean, _ = ekf.innovation(clean)
    _, s_noisy, _ = ekf.innovation(noisy)
    assert s_noisy[0, 0] > s_clean[0, 0]


def test_predict_ignores_non_positive_dt():
    ekf = PositionEKF(np.zeros(3), np.eye(3), process_noise_per_s=1.0)
    before = ekf.cov.copy()
    ekf.predict(-1.0)
    ekf.predict(0.0)
    assert np.allclose(ekf.cov, before)
    ekf.predict(4.0)
    assert ekf.cov[0, 0] == pytest.approx(1.0 + 4.0)
