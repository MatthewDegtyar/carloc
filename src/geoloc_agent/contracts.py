"""Core data contracts.

Every module in the pipeline sits behind these types. Two rules hold everywhere:

1. No bare point estimates. Anything measured carries a sigma or a covariance.
2. Ground truth rides alongside the estimate in eval mode so any stage can be
   scored in isolation.

Frame conventions
-----------------
World frame is a local ENU tangent plane: +x east, +y north, +z up, metres.
Camera frame is OpenCV pinhole: +x right, +y down, +z forward along the optical
axis. ``Pose.R`` maps camera -> world and ``Pose.t`` is the camera centre
expressed in world coordinates, so ``p_world = R @ p_cam + t``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# small validation helpers
# --------------------------------------------------------------------------


def _as_vec(value: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have {n} elements, got shape {np.shape(value)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {arr}")
    return arr


def _as_mat(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _check_cov(cov: np.ndarray, name: str) -> np.ndarray:
    """Covariances must be symmetric and positive semi-definite.

    A filter that quietly carries an indefinite covariance produces confident
    nonsense, so this is checked at construction rather than at use.
    """
    if not np.allclose(cov, cov.T, atol=1e-8):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues.min() < -1e-8:
        raise ValueError(f"{name} must be positive semi-definite, min eig {eigenvalues.min():.3e}")
    return cov


def _unit(vec: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError(f"{name} must be a non-zero vector")
    return vec / norm


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics. No distortion model yet; loaders undistort upstream."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=float
        )

    @property
    def K_inv(self) -> np.ndarray:
        return np.array(
            [
                [1.0 / self.fx, 0.0, -self.cx / self.fx],
                [0.0, 1.0 / self.fy, -self.cy / self.fy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    @classmethod
    def from_matrix(cls, K: Any, width: int, height: int) -> Intrinsics:
        K = _as_mat(K, (3, 3), "K")
        return cls(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], width=width, height=height)

    def contains(self, u: float, v: float) -> bool:
        return 0.0 <= u < self.width and 0.0 <= v < self.height


@dataclass(frozen=True)
class Pose:
    """Camera pose in the world frame, with uncertainty.

    ``cov`` is 6x6 over ``[tx, ty, tz, rx, ry, rz]``, where the rotation block is
    a small-angle perturbation in world axes (radians^2). Heading error lives in
    the ``rz`` term for a level camera.
    """

    R: np.ndarray
    t: np.ndarray
    cov: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))

    def __post_init__(self) -> None:
        R = _as_mat(self.R, (3, 3), "Pose.R")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
            raise ValueError("Pose.R must be orthonormal")
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
            raise ValueError("Pose.R must be right-handed (det = +1)")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", _as_vec(self.t, 3, "Pose.t"))
        cov = _as_mat(self.cov, (6, 6), "Pose.cov")
        object.__setattr__(self, "cov", _check_cov(cov, "Pose.cov"))

    @property
    def position_cov(self) -> np.ndarray:
        return self.cov[:3, :3]

    @property
    def rotation_cov(self) -> np.ndarray:
        return self.cov[3:, 3:]

    def cam_to_world(self, p_cam: Any) -> np.ndarray:
        return self.R @ _as_vec(p_cam, 3, "p_cam") + self.t

    def world_to_cam(self, p_world: Any) -> np.ndarray:
        return self.R.T @ (_as_vec(p_world, 3, "p_world") - self.t)

    @classmethod
    def compose(cls, sensor_to_body: Pose, body_to_world: Pose, cov: Any | None = None) -> Pose:
        """Chain an extrinsic onto an ego pose (the nuScenes / Stray layout)."""
        R = body_to_world.R @ sensor_to_body.R
        t = body_to_world.R @ sensor_to_body.t + body_to_world.t
        return cls(R=R, t=t, cov=body_to_world.cov if cov is None else cov)


@dataclass(frozen=True)
class Frame:
    """One posed image. ``image`` is None for geometry-only sessions."""

    frame_id: int
    timestamp: float
    intrinsics: Intrinsics
    pose: Pose
    image: np.ndarray | None = None
    source: str = ""
    is_keyframe: bool = True

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")

    @property
    def pose_cov(self) -> np.ndarray:
        return self.pose.cov


# --------------------------------------------------------------------------
# perception
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """Axis-aligned image-space detection, ``bbox`` as (x1, y1, x2, y2) pixels."""

    bbox: np.ndarray
    cls: str
    score: float
    frame_id: int
    track_hint: int | None = None

    def __post_init__(self) -> None:
        bbox = _as_vec(self.bbox, 4, "Detection.bbox")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError(f"bbox must have positive extent, got {bbox}")
        object.__setattr__(self, "bbox", bbox)
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")

    @property
    def centroid(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


class RangeMethod(str, Enum):
    TRIANGULATION = "triangulation"
    MONO_DEPTH = "mono_depth"
    LIDAR = "lidar"
    GROUND_PLANE = "ground_plane"
    STREETSCAPE = "streetscape"
    NONE = "none"


@dataclass(frozen=True)
class RangeMeas:
    """A range estimate along a bearing. Never used without checking ``valid``."""

    value: float
    sigma: float
    method: RangeMethod = RangeMethod.NONE
    valid: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", RangeMethod(self.method))
        if self.valid:
            if not np.isfinite(self.value) or self.value <= 0:
                raise ValueError(f"a valid range must be positive and finite, got {self.value}")
            if not np.isfinite(self.sigma) or self.sigma <= 0:
                raise ValueError(f"a valid range must carry a positive sigma, got {self.sigma}")

    @classmethod
    def invalid(cls, method: RangeMethod, reason: str) -> RangeMeas:
        return cls(
            value=float("nan"), sigma=float("inf"), method=method, valid=False, reason=reason
        )


@dataclass(frozen=True)
class Observation:
    """A bearing from a known camera centre, optionally with a range.

    This is the only thing ``fuse/`` ever consumes. Anything upstream -- a
    detector, a lidar return, a scripted stub -- reduces to this.
    """

    t: float
    frame_id: int
    origin: np.ndarray
    bearing: np.ndarray
    bearing_sigma: float
    cls: str = "unknown"
    score: float = 1.0
    range: RangeMeas | None = None
    range_prior: RangeMeas | None = None
    """A weak range guess, e.g. from apparent object size.

    Deliberately separate from ``range``. A prior is used to initialise a track
    and to tighten its association gate; a measurement is fused into the state.
    Treating a size prior as a measurement would fold the same class-size
    assumption into the estimate on every single frame, and the filter would
    converge confidently onto whatever that assumption happened to be."""

    origin_cov: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    truth_position: np.ndarray | None = None
    truth_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _as_vec(self.origin, 3, "Observation.origin"))
        bearing = _as_vec(self.bearing, 3, "Observation.bearing")
        object.__setattr__(self, "bearing", _unit(bearing, "Observation.bearing"))
        if not np.isfinite(self.bearing_sigma) or self.bearing_sigma <= 0:
            raise ValueError("bearing_sigma must be positive and finite (radians)")
        object.__setattr__(
            self,
            "origin_cov",
            _check_cov(_as_mat(self.origin_cov, (3, 3), "Observation.origin_cov"), "origin_cov"),
        )
        if self.truth_position is not None:
            object.__setattr__(self, "truth_position", _as_vec(self.truth_position, 3, "truth"))

    @property
    def has_range(self) -> bool:
        return self.range is not None and self.range.valid


# --------------------------------------------------------------------------
# tracks and decisions
# --------------------------------------------------------------------------


class TrackStatus(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    DEAD = "dead"


@dataclass
class TrackState:
    """A geolocated object with its uncertainty and class posterior.

    The covariance is the point of the whole pipeline; it is what makes the
    agent's surfacing decision non-trivial. It is never collapsed to a point
    before the agent sees it.
    """

    track_id: int
    mean: np.ndarray
    cov: np.ndarray
    class_posterior: dict[str, float] = field(default_factory=dict)
    n_obs: int = 0
    age: float = 0.0
    first_t: float = 0.0
    last_t: float = 0.0
    status: TrackStatus = TrackStatus.TENTATIVE
    misses: int = 0
    degenerate: bool = False
    degeneracy_reason: str = ""
    max_perp_baseline: float = 0.0
    truth_id: str | None = None

    def __post_init__(self) -> None:
        self.mean = _as_vec(self.mean, 3, "TrackState.mean")
        self.cov = _check_cov(_as_mat(self.cov, (3, 3), "TrackState.cov"), "TrackState.cov")
        self.status = TrackStatus(self.status)

    @property
    def sigma_xyz(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, None))

    @property
    def sigma_horizontal(self) -> float:
        """1-sigma radius in the ground plane -- the number an operator cares about."""
        return float(np.sqrt(max(self.cov[0, 0] + self.cov[1, 1], 0.0)))

    @property
    def cep50(self) -> float:
        """Circular error probable, 50%. Standard way to state a geolocation fix."""
        eigenvalues = np.linalg.eigvalsh(self.cov[:2, :2])
        return float(1.1774 * np.sqrt(max(eigenvalues.mean(), 0.0)))

    @property
    def top_class(self) -> tuple[str, float]:
        if not self.class_posterior:
            return ("unknown", 0.0)
        cls = max(self.class_posterior, key=self.class_posterior.get)
        return (cls, float(self.class_posterior[cls]))

    @property
    def class_entropy(self) -> float:
        """Nats. High entropy plus wide covariance is the escalate case."""
        probabilities = np.array([p for p in self.class_posterior.values() if p > 0])
        if probabilities.size == 0:
            return 0.0
        return float(-np.sum(probabilities * np.log(probabilities)))

    def copy(self) -> TrackState:
        return TrackState(
            track_id=self.track_id,
            mean=self.mean.copy(),
            cov=self.cov.copy(),
            class_posterior=dict(self.class_posterior),
            n_obs=self.n_obs,
            age=self.age,
            first_t=self.first_t,
            last_t=self.last_t,
            status=self.status,
            misses=self.misses,
            degenerate=self.degenerate,
            degeneracy_reason=self.degeneracy_reason,
            max_perp_baseline=self.max_perp_baseline,
            truth_id=self.truth_id,
        )


class Action(str, Enum):
    SURFACE = "surface"
    SUPPRESS = "suppress"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class Decision:
    track_id: int
    action: Action
    rationale: str
    priority: float = 0.0
    tool_calls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", Action(self.action))
        if not self.rationale.strip():
            raise ValueError("every decision must carry a rationale")
