"""Hand-computed checks on the geometry primitives.

Everything downstream inherits these conventions, so they are pinned against
values worked out by hand rather than against the implementation's own output.
"""

import numpy as np
import pytest

from geoloc_agent.contracts import Intrinsics, Pose
from geoloc_agent.geometry import (
    angular_difference,
    azimuth_elevation,
    bearing_from_pixel,
    expected_range_sigma,
    parallax_angle,
    perpendicular_baseline,
    perpendicular_projector,
    pixel_sigma_to_angular,
    unit_from_azimuth_elevation,
)
from geoloc_agent.io.synthetic import look_along

INTR = Intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0, width=100, height=80)
IDENTITY = Pose(R=np.eye(3), t=np.zeros(3))


def test_principal_point_maps_to_optical_axis():
    # Camera frame is x right, y down, z forward; identity pose means the
    # optical axis is world +z.
    bearing = bearing_from_pixel(50.0, 40.0, INTR, IDENTITY)
    assert np.allclose(bearing, [0.0, 0.0, 1.0])


def test_one_focal_length_right_of_centre_is_45_degrees():
    # u = cx + fx  ->  camera ray (1, 0, 1) -> 45 degrees to the right.
    bearing = bearing_from_pixel(150.0, 40.0, INTR, IDENTITY)
    assert np.allclose(bearing, [np.sqrt(0.5), 0.0, np.sqrt(0.5)])


def test_below_principal_point_is_positive_camera_y():
    # +v is down in the image and +y is down in the camera frame.
    bearing = bearing_from_pixel(50.0, 140.0, INTR, IDENTITY)
    assert np.allclose(bearing, [0.0, np.sqrt(0.5), np.sqrt(0.5)])


def test_bearing_is_rotated_into_the_world_frame():
    # Facing north (+y): the optical axis must come out as world +y, and a
    # pixel to the right of centre must bear to the east (+x).
    pose = Pose(R=look_along(np.array([0.0, 1.0, 0.0])), t=np.array([1.0, 2.0, 3.0]))
    assert np.allclose(bearing_from_pixel(50.0, 40.0, INTR, pose), [0.0, 1.0, 0.0], atol=1e-12)
    right = bearing_from_pixel(150.0, 40.0, INTR, pose)
    assert np.allclose(right, [np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-12)


def test_bearing_ignores_camera_position():
    """Translation must not change a bearing -- only rotation may."""
    a = bearing_from_pixel(70.0, 20.0, INTR, IDENTITY)
    b = bearing_from_pixel(70.0, 20.0, INTR, Pose(R=np.eye(3), t=np.array([10.0, -5.0, 3.0])))
    assert np.allclose(a, b)


def test_bearing_round_trips_through_projection():
    pose = Pose(R=look_along(np.array([0.3, 1.0, -0.1])), t=np.array([4.0, -2.0, 1.5]))
    point = np.array([12.0, 40.0, 2.0])
    p_cam = pose.world_to_cam(point)
    uv = INTR.K @ p_cam
    bearing = bearing_from_pixel(uv[0] / uv[2], uv[1] / uv[2], INTR, pose)
    expected = point - pose.t
    expected /= np.linalg.norm(expected)
    assert np.allclose(bearing, expected, atol=1e-12)


def test_azimuth_elevation_round_trip():
    for az, el in [(0.0, 0.0), (1.2, 0.3), (-2.5, -0.8), (np.pi - 0.01, 0.1)]:
        vec = unit_from_azimuth_elevation(az, el)
        az2, el2 = azimuth_elevation(vec)
        assert angular_difference(az, az2) == pytest.approx(0.0, abs=1e-12)
        assert el2 == pytest.approx(el, abs=1e-12)


def test_angular_difference_wraps():
    assert angular_difference(0.1, 2 * np.pi - 0.1) == pytest.approx(0.2, abs=1e-12)
    assert angular_difference(-np.pi + 0.1, np.pi - 0.1) == pytest.approx(0.2, abs=1e-12)


def test_perpendicular_projector_annihilates_the_bearing():
    b = np.array([0.6, 0.8, 0.0])
    P = perpendicular_projector(b)
    assert np.allclose(P @ b, np.zeros(3), atol=1e-12)
    # It is a projector: applying it twice changes nothing.
    assert np.allclose(P @ P, P)
    # And it preserves anything already perpendicular.
    perp = np.array([-0.8, 0.6, 0.0])
    assert np.allclose(P @ perp, perp)


def test_perpendicular_baseline_ignores_motion_along_the_bearing():
    bearing = np.array([0.0, 1.0, 0.0])
    # Pure forward motion: no useful baseline at all. This is the degenerate case.
    forward = perpendicular_baseline(np.zeros(3), np.array([0.0, 10.0, 0.0]), bearing)
    assert forward == pytest.approx(
        0.0, abs=1e-12
    )
    # Pure lateral motion: all of it counts.
    assert perpendicular_baseline(np.zeros(3), np.array([3.0, 0.0, 0.0]), bearing) == pytest.approx(
        3.0
    )
    # Mixed: only the perpendicular part counts.
    assert perpendicular_baseline(np.zeros(3), np.array([3.0, 9.0, 4.0]), bearing) == pytest.approx(
        5.0
    )


def test_parallax_and_expected_range_sigma():
    assert parallax_angle(1.0, 100.0) == pytest.approx(np.arctan2(1.0, 100.0))
    assert parallax_angle(1.0, 0.0) == 0.0
    # Range error grows as R^2 and falls as 1/B.
    assert expected_range_sigma(100.0, 1e-3, 10.0) == pytest.approx(np.sqrt(2) * 1.0)
    assert expected_range_sigma(50.0, 1e-3, 0.0) == float("inf")


def test_off_axis_pixels_subtend_less_angle():
    """The correction that keeps the filter from overstating its bearing noise."""
    on_axis = pixel_sigma_to_angular(2.0, INTR, INTR.cx, INTR.cy)
    off_axis = pixel_sigma_to_angular(2.0, INTR, INTR.cx + 100.0, INTR.cy)
    assert on_axis == pytest.approx(2.0 / 100.0, rel=1e-9)
    assert off_axis < on_axis
    # 45 degrees off-axis: x = 1, rho = sqrt(2), so the per-axis sensitivities are
    # sqrt(1 - 1/2)/(sqrt(2) f) in azimuth and 1/(sqrt(2) f) in elevation.
    d_az = np.sqrt(0.5) / (np.sqrt(2) * 100.0)
    d_el = 1.0 / (np.sqrt(2) * 100.0)
    assert off_axis == pytest.approx(2.0 * np.sqrt(0.5 * (d_az**2 + d_el**2)), rel=1e-9)
