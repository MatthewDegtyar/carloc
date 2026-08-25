"""Contract validation. These types are the seams between every module, so the
invariants are enforced at construction rather than trusted."""

import numpy as np
import pytest

from geoloc_agent.contracts import (
    Action,
    Decision,
    Detection,
    Intrinsics,
    Observation,
    Pose,
    RangeMeas,
    RangeMethod,
    TrackState,
)


def test_pose_rejects_non_orthonormal_rotation():
    with pytest.raises(ValueError, match="orthonormal"):
        Pose(R=np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]]), t=np.zeros(3))


def test_pose_rejects_left_handed_rotation():
    # A reflection is orthonormal but flips handedness, which silently mirrors
    # every bearing. det must be +1.
    with pytest.raises(ValueError, match="right-handed"):
        Pose(R=np.diag([1.0, 1.0, -1.0]), t=np.zeros(3))


def test_pose_rejects_indefinite_covariance():
    cov = np.zeros((6, 6))
    cov[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semi-definite"):
        Pose(R=np.eye(3), t=np.zeros(3), cov=cov)


def test_pose_transforms_round_trip():
    pose = Pose(R=np.eye(3), t=np.array([1.0, 2.0, 3.0]))
    point = np.array([4.0, 5.0, 6.0])
    assert np.allclose(pose.cam_to_world(pose.world_to_cam(point)), point)


def test_pose_compose_chains_extrinsic_onto_ego():
    # Sensor 1 m ahead of the body; body at (10, 0, 0) yawed 90 degrees.
    yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    sensor = Pose(R=np.eye(3), t=np.array([1.0, 0.0, 0.0]))
    body = Pose(R=yaw, t=np.array([10.0, 0.0, 0.0]))
    composed = Pose.compose(sensor, body)
    # The sensor's +x offset is rotated by the body yaw into +y.
    assert np.allclose(composed.t, [10.0, 1.0, 0.0])
    assert np.allclose(composed.R, yaw)


def test_intrinsics_inverse_is_a_real_inverse():
    intr = Intrinsics(fx=800.0, fy=810.0, cx=640.0, cy=360.0, width=1280, height=720)
    assert np.allclose(intr.K @ intr.K_inv, np.eye(3))


def test_detection_rejects_inverted_bbox():
    with pytest.raises(ValueError, match="positive extent"):
        Detection(bbox=np.array([10.0, 10.0, 5.0, 20.0]), cls="car", score=0.5, frame_id=0)


def test_detection_rejects_out_of_range_score():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Detection(bbox=np.array([0.0, 0.0, 5.0, 5.0]), cls="car", score=1.5, frame_id=0)


def test_valid_range_must_carry_a_positive_sigma():
    """The no-bare-point-estimates rule, enforced."""
    with pytest.raises(ValueError, match="positive sigma"):
        RangeMeas(value=10.0, sigma=0.0, method=RangeMethod.LIDAR)


def test_invalid_range_carries_a_reason_and_no_value():
    meas = RangeMeas.invalid(RangeMethod.TRIANGULATION, "degenerate geometry")
    assert not meas.valid
    assert meas.reason == "degenerate geometry"
    assert np.isnan(meas.value)


def test_observation_normalises_bearing_and_rejects_zero():
    obs = Observation(
        t=0.0, frame_id=0, origin=np.zeros(3), bearing=np.array([3.0, 4.0, 0.0]), bearing_sigma=1e-3
    )
    assert np.allclose(obs.bearing, [0.6, 0.8, 0.0])
    with pytest.raises(ValueError, match="non-zero"):
        Observation(
            t=0.0, frame_id=0, origin=np.zeros(3), bearing=np.zeros(3), bearing_sigma=1e-3
        )


def test_observation_requires_positive_bearing_sigma():
    with pytest.raises(ValueError, match="bearing_sigma"):
        Observation(
            t=0.0, frame_id=0, origin=np.zeros(3), bearing=np.array([0.0, 0.0, 1.0]),
            bearing_sigma=0.0,
        )


def test_track_state_reports_horizontal_sigma_and_cep():
    state = TrackState(track_id=1, mean=np.zeros(3), cov=np.diag([4.0, 9.0, 1.0]))
    assert state.sigma_horizontal == pytest.approx(np.sqrt(13.0))
    assert state.cep50 == pytest.approx(1.1774 * np.sqrt(6.5))


def test_class_entropy_is_zero_when_certain_and_maximal_when_uniform():
    certain = TrackState(track_id=1, mean=np.zeros(3), cov=np.eye(3), class_posterior={"car": 1.0})
    assert certain.class_entropy == pytest.approx(0.0)
    assert certain.top_class == ("car", 1.0)
    ambiguous = TrackState(
        track_id=2, mean=np.zeros(3), cov=np.eye(3),
        class_posterior={"car": 0.5, "pedestrian": 0.5},
    )
    assert ambiguous.class_entropy == pytest.approx(np.log(2))


def test_decision_requires_a_rationale():
    with pytest.raises(ValueError, match="rationale"):
        Decision(track_id=1, action=Action.SURFACE, rationale="   ")
