"""Stray Scanner (iPhone) session loader.

Written against the interface, not against the data, so that when a capture
arrives it drops in without touching anything downstream. Everything that could
differ between app versions -- file names, column order, whether depth exists --
is isolated in this file.

Stray Scanner exports per session:

    camera_matrix.csv   3x3 intrinsics for the RGB camera
    odometry.csv        per-frame ARKit pose: timestamp, frame, x,y,z, qx,qy,qz,qw
    rgb.mp4             the video
    depth/              optional 16-bit depth, millimetres (LiDAR models)
    confidence/         optional per-pixel depth confidence

Two conversions matter:

**ARKit's camera convention is not OpenCV's.** ARKit cameras look down their
own -z with +y up; ours look down +z with +y down. The fix is a 180-degree
rotation about the camera x-axis, applied once here. Getting this wrong flips
every bearing vertically and yields a beautifully self-consistent, entirely
wrong map.

**ARKit poses are metric but arbitrary-origin.** The session frame starts wherever
tracking started, gravity-aligned but with an arbitrary yaw. Without an external
reference there is no true north, so ``heading_offset_deg`` is exposed as an
explicit, documented input rather than being silently assumed to be zero.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from geoloc_agent.contracts import Frame, Intrinsics, Pose
from geoloc_agent.geo import GeoOrigin
from geoloc_agent.io.base import Session, TruthObject

# ARKit camera (x right, y up, z backward) -> OpenCV camera (x right, y down,
# z forward). A 180-degree rotation about x.
ARKIT_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def quaternion_xyzw_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _yaw(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class StrayScannerSession(Session):
    """An iPhone capture, exposed through the standard Session interface."""

    def __init__(
        self,
        path: str | Path,
        origin: GeoOrigin | None = None,
        heading_offset_deg: float = 0.0,
        position_sigma: float = 0.10,
        heading_sigma_deg: float = 1.0,
        load_images: bool = False,
        max_frames: int | None = None,
        truth: dict[str, TruthObject] | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no Stray Scanner session at {self.path}")
        self.name = f"stray:{self.path.name}"
        self.load_images = load_images
        self.max_frames = max_frames
        self._origin = origin
        self._truth = truth or {}
        # ARKit VIO drift is small and smooth over a short capture; these are
        # defaults to be overridden from a scenario, not measurements.
        self._pose_cov = np.zeros((6, 6))
        var = max(position_sigma**2, 1e-12)
        self._pose_cov[0, 0] = self._pose_cov[1, 1] = self._pose_cov[2, 2] = var
        self._pose_cov[3, 3] = self._pose_cov[4, 4] = 1e-12
        self._pose_cov[5, 5] = max(np.radians(heading_sigma_deg) ** 2, 1e-12)

        self._heading = _yaw(np.radians(heading_offset_deg))
        self.intrinsics = self._read_intrinsics()
        self._frames = self._read_odometry()

    # -- Session interface ------------------------------------------------

    @property
    def origin(self) -> GeoOrigin | None:
        return self._origin

    def frames(self) -> Iterator[Frame]:
        yield from self._frames

    def truth(self) -> dict[str, TruthObject]:
        """Empty unless surveyed points are supplied.

        A phone capture has no ground truth. Geolocation error is therefore not
        scorable on this source, and the eval harness reports coverage and
        self-consistency for it rather than accuracy it cannot measure.
        """
        return self._truth

    def scripted_detections(self) -> None:
        return None

    # -- parsing ----------------------------------------------------------

    def _read_intrinsics(self) -> Intrinsics:
        matrix_path = self.path / "camera_matrix.csv"
        if not matrix_path.exists():
            raise FileNotFoundError(f"expected {matrix_path}")
        rows = [
            [float(v) for v in row]
            for row in csv.reader(matrix_path.open())
            if row and not row[0].lstrip().startswith("#")
        ]
        K = np.array(rows, dtype=float).reshape(3, 3)
        width, height = self._frame_size()
        return Intrinsics.from_matrix(K, width=width, height=height)

    def _frame_size(self) -> tuple[int, int]:
        """Stray Scanner records at 1920x1440 by default; overridden if metadata exists."""
        import json

        for candidate in ("metadata.json", "info.json"):
            meta_path = self.path / candidate
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                width = meta.get("width") or meta.get("frameWidth")
                height = meta.get("height") or meta.get("frameHeight")
                if width and height:
                    return int(width), int(height)
        return 1920, 1440

    def _read_odometry(self) -> list[Frame]:
        odometry_path = self.path / "odometry.csv"
        if not odometry_path.exists():
            raise FileNotFoundError(f"expected {odometry_path}")

        frames: list[Frame] = []
        with odometry_path.open() as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            # Header is optional across app versions; if the first row parses as
            # numbers, it was data.
            if header and _looks_numeric(header):
                handle.seek(0)
                reader = csv.reader(handle)
            t0: float | None = None
            for index, row in enumerate(reader):
                if not row or len(row) < 8:
                    continue
                values = [float(v) for v in row[:8]]
                timestamp, _frame_index, x, y, z, qx, qy, qz = values
                qw = float(row[8]) if len(row) > 8 else _recover_w(qx, qy, qz)
                if t0 is None:
                    t0 = timestamp
                frames.append(
                    Frame(
                        frame_id=index,
                        timestamp=timestamp - t0,
                        intrinsics=self.intrinsics,
                        pose=self._make_pose(np.array([x, y, z]), (qx, qy, qz, qw)),
                        image=None,
                        source=self.name,
                        is_keyframe=True,
                    )
                )
                if self.max_frames and len(frames) >= self.max_frames:
                    break
        return frames

    def _make_pose(self, translation: np.ndarray, quaternion: tuple) -> Pose:
        """ARKit world/camera conventions -> ours, then apply the heading offset."""
        R_arkit = quaternion_xyzw_to_matrix(*quaternion)
        # ARKit's world frame is y-up; ours is z-up (ENU). Map (x, y, z)_arkit to
        # (x, -z, y)_enu, then rotate the camera axes into OpenCV convention.
        arkit_world_to_enu = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        R = self._heading @ arkit_world_to_enu @ R_arkit @ ARKIT_TO_OPENCV
        t = self._heading @ arkit_world_to_enu @ np.asarray(translation, dtype=float)
        return Pose(R=R, t=t, cov=self._pose_cov)

    @property
    def georeference_note(self) -> str:
        if self._origin is None:
            return (
                "ARKit poses are metric but start at an arbitrary origin with arbitrary "
                "yaw. No origin was supplied, so this session is metric-only and emits no "
                "lat/lon."
            )
        return (
            f"ARKit session frame anchored to an assumed origin "
            f"({self._origin.lat:.6f}, {self._origin.lon:.6f}) with a manually supplied "
            f"heading offset. Relative geometry is trustworthy; absolute position and "
            f"bearing are only as good as those two inputs."
        )


def _looks_numeric(row: list[str]) -> bool:
    try:
        [float(v) for v in row[:4]]
    except ValueError:
        return False
    return True


def _recover_w(x: float, y: float, z: float) -> float:
    """Some exports drop the scalar term; recover it assuming a unit quaternion."""
    return float(np.sqrt(max(0.0, 1.0 - (x * x + y * y + z * z))))
