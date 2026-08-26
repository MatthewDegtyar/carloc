"""Ranging by ground intersection: the airborne counterpart to triangulation.

The tests that matter here are the refusals. A ground-plane ranger will happily
return a number for a ray a hair below the horizon, and that number will be
enormous and wrong, so most of what follows checks that it says no instead.
"""

import numpy as np
import pytest

from geoloc_agent.contracts import Observation, RangeMethod
from geoloc_agent.range.ground_plane import (
    DEFAULT_ATTITUDE_SIGMA_RAD,
    GroundPlaneRanger,
    intersect_plane,
    range_sigma,
    usable_depression,
)


def looking(origin, depression_deg, azimuth_deg=0.0) -> Observation:
    """A bearing at a given depression below the horizon."""
    theta = np.radians(depression_deg)
    phi = np.radians(azimuth_deg)
    bearing = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi),
                        -np.sin(theta)])
    return Observation(t=0.0, frame_id=0, origin=np.asarray(origin, float),
                       bearing=bearing, bearing_sigma=1e-3, cls="car")


# -- intersection geometry --------------------------------------------------

def test_nadir_range_is_the_height():
    distance, point = intersect_plane(np.array([10.0, 20.0, 80.0]),
                                      np.array([0.0, 0.0, -1.0]), 0.0)
    assert distance == pytest.approx(80.0)
    assert point == pytest.approx([10.0, 20.0, 0.0])


def test_range_follows_h_over_sin_theta():
    height = 80.0
    for degrees in (10.0, 30.0, 45.0, 75.0):
        obs = looking([0.0, 0.0, height], degrees)
        distance, _ = intersect_plane(obs.origin, obs.bearing, 0.0)
        assert distance == pytest.approx(height / np.sin(np.radians(degrees)), rel=1e-9)


def test_a_level_or_upward_ray_has_no_intersection():
    origin = np.array([0.0, 0.0, 80.0])
    assert intersect_plane(origin, np.array([1.0, 0.0, 0.0]), 0.0) is None
    assert intersect_plane(origin, np.array([1.0, 0.0, 0.3]), 0.0) is None


def test_a_camera_below_the_plane_has_no_intersection():
    assert intersect_plane(np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, -1.0]), 0.0) is None


def test_ground_height_offsets_the_solution():
    """Terrain 20 m above datum brings the intersection 20 m closer at nadir."""
    origin = np.array([0.0, 0.0, 80.0])
    down = np.array([0.0, 0.0, -1.0])
    assert intersect_plane(origin, down, 20.0)[0] == pytest.approx(60.0)


# -- error propagation ------------------------------------------------------

def test_sigma_grows_as_the_ray_flattens():
    height = 80.0
    previous = 0.0
    for degrees in (75.0, 45.0, 30.0, 15.0, 5.0):
        theta = np.radians(degrees)
        distance = height / np.sin(theta)
        relative = range_sigma(distance, height, theta,
                               DEFAULT_ATTITUDE_SIGMA_RAD, 0.1, 0.3) / distance
        assert relative > previous
        previous = relative


def test_nadir_is_the_best_conditioned_case():
    """Straight down, attitude error contributes nothing: 1/tan(90 deg) is zero."""
    sigma = range_sigma(80.0, 80.0, np.radians(90.0), DEFAULT_ATTITUDE_SIGMA_RAD, 0.1, 0.3)
    assert sigma == pytest.approx(np.hypot(0.1, 0.3), rel=1e-6)


def test_attitude_error_dominates_at_shallow_angles():
    """The reason pose accuracy, not perception, sets the achievable error."""
    height, theta = 80.0, np.radians(10.0)
    distance = height / np.sin(theta)
    attitude_only = range_sigma(distance, height, theta, DEFAULT_ATTITUDE_SIGMA_RAD, 0.0, 0.0)
    height_only = range_sigma(distance, height, theta, 0.0, 0.1, 0.3)
    assert attitude_only > 5 * height_only


def test_surface_and_altitude_error_are_interchangeable():
    """Only the difference between camera height and ground height is observable."""
    a = range_sigma(160.0, 80.0, np.radians(30.0), 0.0, 0.4, 0.0)
    b = range_sigma(160.0, 80.0, np.radians(30.0), 0.0, 0.0, 0.4)
    assert a == pytest.approx(b)


def test_doubling_height_almost_but_not_quite_doubles_sigma():
    """Only the attitude term scales with height, and that is worth knowing.

    The height term is ``(R/h) * sigma_h``, and ``R/h`` is ``1/sin(theta)`` --
    independent of altitude. So flying twice as high doubles the attitude
    contribution while leaving the height contribution where it was, and the
    total grows by slightly less than two. The gap closes as attitude comes to
    dominate, which is the regime any real flight is in.
    """
    theta = np.radians(30.0)
    one = range_sigma(80.0 / np.sin(theta), 80.0, theta, DEFAULT_ATTITUDE_SIGMA_RAD, 0.1, 0.3)
    two = range_sigma(160.0 / np.sin(theta), 160.0, theta, DEFAULT_ATTITUDE_SIGMA_RAD, 0.1, 0.3)
    assert 1.9 < two / one < 2.0

    # With a very accurate surface and altitude the height term vanishes and the
    # scaling becomes exactly linear.
    clean_one = range_sigma(80.0 / np.sin(theta), 80.0, theta, DEFAULT_ATTITUDE_SIGMA_RAD, 0, 0)
    clean_two = range_sigma(160.0 / np.sin(theta), 160.0, theta, DEFAULT_ATTITUDE_SIGMA_RAD, 0, 0)
    assert clean_two / clean_one == pytest.approx(2.0, rel=1e-9)


# -- the ranger's refusals --------------------------------------------------

def test_a_well_conditioned_look_returns_a_range_with_a_reason():
    ranger = GroundPlaneRanger(ground_z=0.0)
    measurement = ranger.range_for(looking([0.0, 0.0, 80.0], 45.0), [])
    assert measurement.valid
    assert measurement.method is RangeMethod.GROUND_PLANE
    assert measurement.value == pytest.approx(80.0 / np.sin(np.radians(45.0)), rel=1e-6)
    assert measurement.sigma > 0
    assert "depression" in measurement.reason


def test_a_near_horizon_ray_is_refused_not_answered():
    """The headline failure: the number would be huge, confident and wrong."""
    ranger = GroundPlaneRanger(ground_z=0.0)
    measurement = ranger.range_for(looking([0.0, 0.0, 80.0], 0.4), [])
    assert not measurement.valid
    assert "parallel to the ground" in measurement.reason


def test_a_shallow_but_not_horizontal_ray_fails_on_the_sigma_bar():
    """Between the angle floor and good geometry, the relative-sigma bar decides."""
    ranger = GroundPlaneRanger(ground_z=0.0)
    measurement = ranger.range_for(looking([0.0, 0.0, 80.0], 2.5), [])
    assert not measurement.valid
    assert "of range" in measurement.reason
    assert "too shallow" in measurement.reason


def test_an_upward_ray_is_refused():
    ranger = GroundPlaneRanger(ground_z=0.0)
    obs = looking([0.0, 0.0, 80.0], -10.0)
    assert not ranger.range_for(obs, []).valid


def test_a_camera_below_the_surface_is_refused_with_a_clear_reason():
    ranger = GroundPlaneRanger(ground_z=100.0)
    measurement = ranger.range_for(looking([0.0, 0.0, 80.0], 45.0), [])
    assert not measurement.valid
    assert "below the ground plane" in measurement.reason


def test_the_envelope_is_enforced_when_declared():
    ranger = GroundPlaneRanger(ground_z=0.0, max_range_m=100.0)
    measurement = ranger.range_for(looking([0.0, 0.0, 80.0], 20.0), [])
    assert not measurement.valid
    assert "envelope" in measurement.reason


def test_an_invalid_measurement_never_carries_a_usable_number():
    """RangeMeas.invalid must not be mistakable for a fix."""
    ranger = GroundPlaneRanger(ground_z=0.0)
    measurement = ranger.range_for(looking([0.0, 0.0, 80.0], 0.4), [])
    assert not np.isfinite(measurement.value)
    assert not np.isfinite(measurement.sigma)


def test_ranging_needs_no_history():
    """The whole point against triangulation: one frame is enough."""
    ranger = GroundPlaneRanger(ground_z=0.0)
    assert ranger.range_for(looking([0.0, 0.0, 80.0], 45.0), []).valid


def test_forward_motion_does_not_degrade_it():
    """The nuScenes failure mode has no analogue here.

    Three frames flying straight at the target -- zero perpendicular baseline,
    which makes triangulation unobservable -- and every one still ranges.
    """
    ranger = GroundPlaneRanger(ground_z=0.0)
    for y in (0.0, 25.0, 50.0):
        obs = looking([0.0, y, 80.0], 40.0)
        assert ranger.range_for(obs, []).valid


# -- planning ---------------------------------------------------------------

def test_usable_depression_gets_shallower_as_you_fly_lower():
    """Lower flight buys reach: the same ground range sits at a steeper angle."""
    low = usable_depression(30.0, 400.0)
    high = usable_depression(120.0, 400.0)
    assert low < high


def test_usable_depression_reports_no_answer_rather_than_a_bad_one():
    """A 400 m envelope cannot be met from 2 km up at any angle."""
    assert not np.isfinite(usable_depression(2000.0, 400.0))


def test_the_planner_agrees_with_the_ranger():
    """Whatever `usable_depression` promises, the ranger must actually accept."""
    height = 80.0
    theta = usable_depression(height, 400.0)
    assert np.isfinite(theta)
    ranger = GroundPlaneRanger(ground_z=0.0, max_range_m=400.0)
    # A hair steeper than the reported limit must be accepted.
    obs = looking([0.0, 0.0, height], np.degrees(theta) + 0.5)
    assert ranger.range_for(obs, []).valid
