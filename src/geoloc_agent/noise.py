"""Injected noise model.

Phase 3 sweeps over exactly these knobs. Both the synthetic session and any real
session (nuScenes, Stray Scanner) are wrapped by the same injectors, so an
error-vs-noise curve means the same thing on real data as on synthetic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from geoloc_agent.contracts import Detection, Frame, Pose


@dataclass(frozen=True)
class NoiseModel:
    """All knobs default to zero: the no-noise model is the identity."""

    gps_sigma: float = 0.0  # m, per-axis position noise on the camera centre
    gps_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)  # m, constant offset
    heading_bias: float = 0.0  # rad, constant yaw offset -- the nasty one
    heading_sigma: float = 0.0  # rad, per-frame yaw noise
    tilt_sigma: float = 0.0  # rad, per-frame roll/pitch noise
    range_sigma: float = 0.0  # m, extra noise on any RangeMeas
    bearing_sigma: float = 0.0  # rad, extra angular noise, applied in pixels
    detection_dropout: float = 0.0  # probability a true detection is missed
    false_positive_rate: float = 0.0  # expected spurious detections per frame
    bbox_jitter_px: float = 0.0  # px, centroid jitter (models centroid drift)

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"{f.name} must be non-negative, got {value}")
        for p in (self.detection_dropout, self.false_positive_rate):
            if p > 1.0 and p is self.detection_dropout:
                raise ValueError("detection_dropout is a probability in [0, 1]")

    @property
    def is_identity(self) -> bool:
        return all(
            (np.all(np.asarray(v) == 0) if isinstance(v, tuple) else v == 0)
            for v in asdict(self).values()
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> NoiseModel:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown noise parameters: {sorted(unknown)}")
        data = dict(data)
        if "gps_bias" in data:
            data["gps_bias"] = tuple(data["gps_bias"])
        return cls(**data)

    def label(self) -> str:
        active = {k: v for k, v in asdict(self).items() if not (np.all(np.asarray(v) == 0))}
        if not active:
            return "clean"
        return ", ".join(f"{k}={v}" for k, v in active.items())


def _yaw_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _small_rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    """Exact exponential map, so large injected biases stay orthonormal."""
    v = np.array([rx, ry, rz], dtype=float)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


class PoseNoiseInjector:
    """Corrupts poses, and reports the covariance the corruption implies.

    The reported covariance covers the *random* terms only. A constant heading
    bias is deliberately not represented in it -- that is the point of the
    heading-bias sweep: it is the error mode a filter cannot see in its own
    covariance, and NEES should blow up accordingly.
    """

    def __init__(self, noise: NoiseModel, rng: np.random.Generator) -> None:
        self.noise = noise
        self.rng = rng

    def pose_covariance(self) -> np.ndarray:
        n = self.noise
        cov = np.zeros((6, 6))
        cov[0, 0] = cov[1, 1] = cov[2, 2] = max(n.gps_sigma**2, 1e-12)
        cov[3, 3] = cov[4, 4] = max(n.tilt_sigma**2, 1e-12)
        cov[5, 5] = max(n.heading_sigma**2, 1e-12)
        return cov

    def apply(self, frame: Frame) -> Frame:
        n = self.noise
        if n.is_identity:
            return frame
        t = frame.pose.t + np.asarray(n.gps_bias, dtype=float)
        if n.gps_sigma > 0:
            t = t + self.rng.normal(0.0, n.gps_sigma, size=3)
        R = frame.pose.R
        yaw = n.heading_bias + (
            self.rng.normal(0.0, n.heading_sigma) if n.heading_sigma > 0 else 0.0
        )
        if yaw != 0.0:
            R = _yaw_matrix(yaw) @ R  # yaw about world up, applied to camera->world
        if n.tilt_sigma > 0:
            rx, ry = self.rng.normal(0.0, n.tilt_sigma, size=2)
            R = _small_rotation(rx, ry, 0.0) @ R
        cov = frame.pose.cov + self.pose_covariance()
        return Frame(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            intrinsics=frame.intrinsics,
            pose=Pose(R=R, t=t, cov=cov),
            image=frame.image,
            source=frame.source,
            is_keyframe=frame.is_keyframe,
        )


class DetectionNoiseInjector:
    """Drops true detections, jitters boxes, and invents false positives."""

    FP_CLASSES = ("car", "pedestrian", "clutter")

    def __init__(self, noise: NoiseModel, rng: np.random.Generator) -> None:
        self.noise = noise
        self.rng = rng

    def apply(self, detections: list[Detection], frame: Frame) -> list[Detection]:
        n = self.noise
        out: list[Detection] = []
        for det in detections:
            if n.detection_dropout > 0 and self.rng.random() < n.detection_dropout:
                continue
            bbox = det.bbox.copy()
            if n.bbox_jitter_px > 0:
                shift = self.rng.normal(0.0, n.bbox_jitter_px, size=2)
                bbox += np.array([shift[0], shift[1], shift[0], shift[1]])
            # bearing_sigma is angular; convert to a pixel shift via focal length.
            if n.bearing_sigma > 0:
                fx, fy = frame.intrinsics.fx, frame.intrinsics.fy
                du = self.rng.normal(0.0, n.bearing_sigma) * fx
                dv = self.rng.normal(0.0, n.bearing_sigma) * fy
                bbox += np.array([du, dv, du, dv])
            out.append(
                Detection(bbox=bbox, cls=det.cls, score=det.score, frame_id=det.frame_id,
                          track_hint=det.track_hint)
            )
        if n.false_positive_rate > 0:
            for _ in range(self.rng.poisson(n.false_positive_rate)):
                out.append(self._false_positive(frame))
        return out

    def _false_positive(self, frame: Frame) -> Detection:
        intr = frame.intrinsics
        u = self.rng.uniform(0.05 * intr.width, 0.95 * intr.width)
        v = self.rng.uniform(0.45 * intr.height, 0.95 * intr.height)
        half_w = self.rng.uniform(15.0, 60.0)
        half_h = self.rng.uniform(15.0, 60.0)
        return Detection(
            bbox=np.array([u - half_w, v - half_h, u + half_w, v + half_h]),
            cls=str(self.rng.choice(self.FP_CLASSES)),
            score=float(self.rng.uniform(0.30, 0.65)),
            frame_id=frame.frame_id,
            track_hint=None,
        )
