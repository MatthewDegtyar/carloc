"""World geometry: pixels to bearings, and the angular bookkeeping around them.

The single most load-bearing function here is ``bearing_from_pixel``. If its
sign or transpose convention is wrong, every downstream number is wrong in a way
that still looks plausible, so it is unit-tested against hand-computed cases.
"""

from __future__ import annotations

import numpy as np

from geoloc_agent.contracts import Detection, Frame, Intrinsics, Observation, Pose, RangeMeas


def bearing_from_pixel(u: float, v: float, intrinsics: Intrinsics, pose: Pose) -> np.ndarray:
    """Pixel -> world-frame unit bearing vector.

    Backprojects through the intrinsics into the camera frame (OpenCV: x right,
    y down, z forward), then rotates into the world with the camera->world
    rotation. Returns a unit vector pointing from the camera centre toward
    whatever produced that pixel.
    """
    ray_cam = intrinsics.K_inv @ np.array([float(u), float(v), 1.0])
    ray_cam /= np.linalg.norm(ray_cam)
    ray_world = pose.R @ ray_cam
    return ray_world / np.linalg.norm(ray_world)


def pixel_sigma_to_angular(
    sigma_px: float, intrinsics: Intrinsics, u: float | None = None, v: float | None = None
) -> float:
    """Convert a pixel-space centroid sigma to an angular sigma in radians.

    Near the optical axis one pixel subtends 1/f radians. Off-axis it subtends
    measurably less, because the ray is longer and more oblique: for a
    normalised ray ``(x, y, 1)`` of length ``rho``, a unit step in ``x`` turns
    the bearing by ``sqrt(1 - x^2/rho^2) / rho`` radians rather than 1.

    Ignoring that is not harmless. It tells the filter its bearings are worse
    than they are, which shows up as a covariance that is too wide on exactly
    the off-axis objects that have the best geometry. Passing the pixel location
    corrects it; omitting it falls back to the on-axis approximation.
    """
    if u is None or v is None:
        focal = 0.5 * (intrinsics.fx + intrinsics.fy)
        return float(sigma_px / focal)

    x = (float(u) - intrinsics.cx) / intrinsics.fx
    y = (float(v) - intrinsics.cy) / intrinsics.fy
    rho_sq = x * x + y * y + 1.0
    rho = np.sqrt(rho_sq)
    # Angular sensitivity to a one-pixel step along each image axis.
    d_az = np.sqrt(max(1.0 - x * x / rho_sq, 0.0)) / (rho * intrinsics.fx)
    d_el = np.sqrt(max(1.0 - y * y / rho_sq, 0.0)) / (rho * intrinsics.fy)
    return float(sigma_px * np.sqrt(0.5 * (d_az**2 + d_el**2)))


def azimuth_elevation(direction: np.ndarray) -> tuple[float, float]:
    """World unit vector -> (azimuth from +x toward +y, elevation from horizontal)."""
    d = np.asarray(direction, dtype=float)
    az = float(np.arctan2(d[1], d[0]))
    el = float(np.arctan2(d[2], np.hypot(d[0], d[1])))
    return az, el


def unit_from_azimuth_elevation(az: float, el: float) -> np.ndarray:
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def angular_difference(a: float, b: float) -> float:
    """Wrap ``a - b`` into [-pi, pi). Azimuth residuals must never wrap wrong."""
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def perpendicular_projector(bearing: np.ndarray) -> np.ndarray:
    """``I - b b^T``: projects out the along-bearing component.

    This is the operator that makes the whole degenerate-geometry story precise.
    Displacement that survives it is useful baseline; displacement it annihilates
    tells you nothing about range.
    """
    b = np.asarray(bearing, dtype=float).reshape(3)
    return np.eye(3) - np.outer(b, b)


def perpendicular_baseline(
    origin_a: np.ndarray, origin_b: np.ndarray, bearing: np.ndarray
) -> float:
    """Component of the camera-centre displacement perpendicular to the bearing."""
    delta = np.asarray(origin_a, dtype=float) - np.asarray(origin_b, dtype=float)
    return float(np.linalg.norm(perpendicular_projector(bearing) @ delta))


def parallax_angle(perp_baseline: float, range_m: float) -> float:
    """The angle the object subtends between two viewpoints. The real currency."""
    if range_m <= 1e-9:
        return 0.0
    return float(np.arctan2(perp_baseline, range_m))


def expected_range_sigma(range_m: float, bearing_sigma: float, perp_baseline: float) -> float:
    """Analytic along-range 1-sigma for a two-view fix.

    The exact two-view CRLB is ``sqrt(2) * R^2 * sigma_theta / B``. The familiar
    rule of thumb ``R^2 * sigma_theta / B`` is the same expression without the
    sqrt(2) that comes from having two independently noisy bearings rather than
    one. We report the exact form; the rule of thumb is recovered by dividing by
    sqrt(2).
    """
    if perp_baseline <= 1e-9:
        return float("inf")
    return float(np.sqrt(2.0) * range_m**2 * bearing_sigma / perp_baseline)


def observation_from_detection(
    detection: Detection,
    frame: Frame,
    bearing_sigma_px: float = 2.0,
    range_meas: RangeMeas | None = None,
    truth_position: np.ndarray | None = None,
    truth_id: str | None = None,
    range_prior: RangeMeas | None = None,
) -> Observation:
    """Detection + posed frame -> the world-frame Observation that ``fuse/`` eats.

    The bearing is taken through the bbox centroid. That is a real modelling
    assumption and a real error source: the centroid of a 2D box is not the
    centroid of the 3D object, and it drifts with viewing aspect. It is called
    out in the limitations section rather than hidden in the sigma.
    """
    u, v = detection.centroid
    bearing = bearing_from_pixel(u, v, frame.intrinsics, frame.pose)
    angular_sigma = pixel_sigma_to_angular(bearing_sigma_px, frame.intrinsics, u, v)
    return Observation(
        t=frame.timestamp,
        frame_id=frame.frame_id,
        origin=frame.pose.t,
        bearing=bearing,
        bearing_sigma=angular_sigma,
        cls=detection.cls,
        score=detection.score,
        track_hint=detection.track_hint,
        range=range_meas,
        range_prior=range_prior,
        origin_cov=frame.pose.position_cov,
        truth_position=truth_position,
        truth_id=truth_id,
    )


def bearing_measurement_cov(pose: Pose, bearing_sigma: float) -> np.ndarray:
    """2x2 (azimuth, elevation) measurement covariance including pose attitude error.

    A yaw error rotates the bearing one-for-one in azimuth, so it adds directly.
    Position error is handled separately, in the filter, because it depends on
    the current range estimate.
    """
    rot_cov = pose.rotation_cov
    return np.diag([bearing_sigma**2 + rot_cov[2, 2], bearing_sigma**2 + rot_cov[0, 0]])
