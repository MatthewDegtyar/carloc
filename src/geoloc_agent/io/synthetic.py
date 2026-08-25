"""Synthetic session: pure geometry, no images, no ML.

This exists to validate the filter math. It is deliberately the first thing that
runs, because the bugs it catches -- a transposed rotation, a sign flip in the
camera frame, a covariance that never shrinks -- are miserable to find on real
data where a dozen other things are also wrong.

Paths are chosen to exercise the two geometry regimes explicitly:

``straight``  forward motion. Objects near the optical axis have almost no
              perpendicular baseline and must triangulate badly. This is the
              degenerate case, and it is the common case for a vehicle.
``lateral``   the camera strafes across its own view direction. Maximum
              perpendicular baseline; this is the well-conditioned case.
``arc``       an orbit, which sweeps bearing continuously.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from geoloc_agent.contracts import Detection, Frame, Intrinsics, Pose
from geoloc_agent.geo import GeoOrigin
from geoloc_agent.io.base import Session, TruthObject

DEFAULT_INTRINSICS = Intrinsics(fx=1266.4, fy=1266.4, cx=800.0, cy=450.0, width=1600, height=900)
"""CAM_FRONT-like intrinsics, so synthetic pixel errors mean the same thing as
nuScenes pixel errors."""


def look_along(forward: np.ndarray, world_up: np.ndarray | None = None) -> np.ndarray:
    """Camera->world rotation for a camera pointing along ``forward``.

    Camera axes are OpenCV: x right, y down, z forward.
    """
    world_up = np.array([0.0, 0.0, 1.0]) if world_up is None else np.asarray(world_up, float)
    z = np.asarray(forward, dtype=float)
    z = z / np.linalg.norm(z)
    right = np.cross(z, world_up)
    norm = np.linalg.norm(right)
    if norm < 1e-9:  # looking straight up or down; pick an arbitrary roll
        right = np.cross(z, np.array([1.0, 0.0, 0.0]))
        norm = np.linalg.norm(right)
    right = right / norm
    down = np.cross(z, right)
    return np.column_stack([right, down, z])


@dataclass
class SyntheticScenario:
    """Declarative description of a synthetic run. Config-driven, not code."""

    name: str = "synthetic"
    path: str = "lateral"  # straight | lateral | arc
    n_frames: int = 40
    rate_hz: float = 10.0
    speed_mps: float = 5.0
    camera_height_m: float = 1.6
    start: tuple[float, float] = (0.0, 0.0)
    heading_deg: float = 90.0  # 90 deg = facing north (+y)
    arc_radius_m: float = 25.0
    objects: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path not in {"straight", "lateral", "arc"}:
            raise ValueError(f"unknown path type: {self.path}")
        if self.n_frames < 2:
            raise ValueError("need at least 2 frames to have any baseline")
        if not self.objects:
            self.objects = default_object_layout()

    @classmethod
    def from_dict(cls, data: dict) -> SyntheticScenario:
        data = dict(data)
        if "start" in data:
            data["start"] = tuple(data["start"])
        return cls(**data)


def default_object_layout() -> list[dict]:
    """A layout that contains both geometry regimes, on purpose.

    With the default ``lateral`` path the camera moves east along y=0 facing
    north, so objects far up the +y axis sit near the optical axis (degenerate)
    and objects offset in x are well-conditioned.
    """
    return [
        {"id": "car_left", "pos": [-12.0, 30.0, 0.8], "cls": "car", "size": [1.9, 4.5, 1.6]},
        {"id": "car_right", "pos": [14.0, 26.0, 0.8], "cls": "car", "size": [1.9, 4.5, 1.6]},
        {"id": "ped_near", "pos": [5.0, 18.0, 0.9], "cls": "pedestrian", "size": [0.7, 0.7, 1.8]},
        {"id": "car_ahead", "pos": [0.5, 70.0, 0.8], "cls": "car", "size": [1.9, 4.5, 1.6]},
        {"id": "ped_far_ahead", "pos": [-1.0, 95.0, 0.9], "cls": "pedestrian",
         "size": [0.7, 0.7, 1.8]},
    ]


class SyntheticSession(Session):
    """Generates a walk path and true object positions. No images at all."""

    def __init__(
        self,
        scenario: SyntheticScenario | None = None,
        intrinsics: Intrinsics = DEFAULT_INTRINSICS,
        origin: GeoOrigin | None = None,
    ) -> None:
        self.scenario = scenario or SyntheticScenario()
        self.name = self.scenario.name
        self.intrinsics = intrinsics
        self._origin = origin or GeoOrigin(42.336849, -71.05785, 0.0, "synthetic", "chosen")
        self._truth = {
            o["id"]: TruthObject(
                obj_id=o["id"],
                position=np.asarray(o["pos"], dtype=float),
                cls=o.get("cls", "unknown"),
                size=tuple(o.get("size", (1.0, 1.0, 1.0))),
            )
            for o in self.scenario.objects
        }
        self._frames = self._build_frames()

    # -- Session interface ------------------------------------------------

    @property
    def origin(self) -> GeoOrigin:
        return self._origin

    def frames(self) -> Iterator[Frame]:
        yield from self._frames

    def truth(self) -> dict[str, TruthObject]:
        return self._truth

    def scripted_detections(self) -> dict[int, list[Detection]]:
        """Noise-free projections of the truth. Noise is injected downstream."""
        table: dict[int, list[Detection]] = {}
        for frame in self._frames:
            dets = []
            for obj in self._truth.values():
                det = self._project(obj, frame)
                if det is not None:
                    dets.append(det)
            table[frame.frame_id] = dets
        return table

    # -- geometry ---------------------------------------------------------

    def _build_frames(self) -> list[Frame]:
        s = self.scenario
        dt = 1.0 / s.rate_hz
        step = s.speed_mps * dt
        heading = np.radians(s.heading_deg)
        facing = np.array([np.cos(heading), np.sin(heading), 0.0])
        # Right-hand side of the facing direction, in the ground plane.
        strafe = np.array([facing[1], -facing[0], 0.0])

        frames: list[Frame] = []
        for i in range(s.n_frames):
            if s.path == "straight":
                centre = np.array([s.start[0], s.start[1], s.camera_height_m]) + facing * (step * i)
                R = look_along(facing)
            elif s.path == "lateral":
                centre = np.array([s.start[0], s.start[1], s.camera_height_m]) + strafe * (step * i)
                R = look_along(facing)
            else:  # arc -- a constant-radius left turn, tangent to `heading`
                # Integrating the unit tangent [cos(h+a), sin(h+a)] over arc
                # length gives this closed form, which starts exactly at `start`
                # facing exactly `heading` rather than at an arbitrary rotation.
                angle = (s.speed_mps * dt * i) / s.arc_radius_m
                centre = np.array(
                    [
                        s.start[0] + s.arc_radius_m * (np.sin(heading + angle) - np.sin(heading)),
                        s.start[1] - s.arc_radius_m * (np.cos(heading + angle) - np.cos(heading)),
                        s.camera_height_m,
                    ]
                )
                R = look_along(np.array([np.cos(heading + angle), np.sin(heading + angle), 0.0]))
            frames.append(
                Frame(
                    frame_id=i,
                    timestamp=i * dt,
                    intrinsics=self.intrinsics,
                    pose=Pose(R=R, t=centre, cov=np.zeros((6, 6))),
                    image=None,
                    source=f"{self.name}:{s.path}",
                    is_keyframe=True,
                )
            )
        return frames

    def _project(self, obj: TruthObject, frame: Frame) -> Detection | None:
        p_cam = frame.pose.world_to_cam(obj.position)
        if p_cam[2] < 0.5:  # behind the camera or on top of it
            return None
        uv = frame.intrinsics.K @ p_cam
        u, v = uv[0] / uv[2], uv[1] / uv[2]
        if not frame.intrinsics.contains(u, v):
            return None
        rng = float(p_cam[2])
        half_w = 0.5 * frame.intrinsics.fx * max(obj.size[0], obj.size[1]) / rng
        half_h = 0.5 * frame.intrinsics.fy * obj.size[2] / rng
        return Detection(
            bbox=np.array([u - half_w, v - half_h, u + half_w, v + half_h]),
            cls=obj.cls,
            score=0.9,
            frame_id=frame.frame_id,
            track_hint=None,
        )

    # -- interop ----------------------------------------------------------

    def write_detection_script(self, path: str | Path) -> Path:
        """Dump scripted detections to JSON so StubDetector can replay them."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session": self.name,
            "frames": {
                str(fid): [
                    {"bbox": d.bbox.tolist(), "cls": d.cls, "score": d.score,
                     "track_hint": d.track_hint}
                    for d in dets
                ]
                for fid, dets in self.scripted_detections().items()
            },
        }
        path.write_text(json.dumps(payload, indent=2))
        return path
