"""Reading AirZoo-Real: posed frames, terrain, and a fitted route to WGS84.

Every convention below was established by measuring against the data, not read
off a spec, and three of them are the opposite of the obvious guess. They are
recorded because getting one wrong produces plausible coordinates rather than an
error.

* ``t_pose.txt`` is ``path qw qx qy qz tx ty tz``, **world-to-camera**. The camera
  centre is ``C = -R^T t``; the raw translation is not a position and runs to
  millions of metres.
* Camera axes are **OpenGL** -- ``-Z`` forward, ``+Y`` up in the image. Read as
  OpenCV the optical axis points at the sky, which scores zero terrain hits.
* Intrinsics are quoted for 4032x3024 while the PNGs ship at 1008x756. Taking
  them as given scales every bearing by four, which reads as an odd lens.
* The world frame is EPSG:4547 (CGCS2000 Gauss-Kruger), metres, shared with the
  terrain. Coordinates near 4e5 by 3.1e6, so a local origin is subtracted.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from carloc.terrain import Terrain

AXIS_FLIP = np.diag([1.0, -1.0, -1.0])
"""Half turn between the two camera-axis conventions in this dataset.

Determinant +1, so it composes into a pose without breaking orthonormality.

**Which one applies is detected per split, not hardcoded**, because AirZoo is not
internally consistent. Verified against the depth render, which is derived from
the same geometry the imagery is:

    guangchang/12-14   telemetry pitch +45   OpenGL (-Z fwd)   corr +0.75
    jiaxiao/12-14      telemetry pitch -30   OpenCV (+Z fwd)   corr +0.95

Under the wrong convention each scores *zero* terrain hits, so the failure is at
least loud once a ray is cast -- but a pipeline that trusted a hardcoded flip
would have produced confident, plausible, wrong coordinates for one of the two.
The rule used is the one that cannot be got wrong: a survey camera looks down, so
take the sign that points the optical axis at the ground."""


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class Frame:
    frame_id: int
    rotation: np.ndarray
    """Camera-to-world, already in the +Z-forward convention."""

    centre: np.ndarray
    """Camera centre in the session's local metric frame."""

    intrinsics: Intrinsics
    name: str
    longitude: float
    latitude: float
    altitude: float
    path: Path | None = None

    def bearing(self, u: float, v: float) -> np.ndarray:
        """Pixel -> world-frame unit bearing."""
        intr = self.intrinsics
        ray = np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
        world = self.rotation @ (ray / np.linalg.norm(ray))
        return world / np.linalg.norm(world)

    def image(self) -> np.ndarray | None:
        if self.path is None or not self.path.exists():
            return None
        from PIL import Image

        with Image.open(self.path) as handle:
            return np.array(handle.convert("RGB"))


def _quaternion_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Session:
    """One site at one time of day."""

    def __init__(self, root: str | Path = "sessions/airzoo", site: str = "jiaxiao",
                 split: str = "12-14", terrain_name: str = "fcw_hangtian_DSM.tif") -> None:
        self.root = Path(root)
        self.site = site
        self.split = split
        self.base = self.root / site / split
        self._terrain_path = self.root / "dom" / terrain_name
        self._terrain: Terrain | None = None
        self._local_terrain: Terrain | None = None

        if not (self.base / "poses").is_dir():
            raise FileNotFoundError(f"no split at {self.base}")
        self._frames = self._read()
        self.axis_note = (f"{self._flipped}/{len(self._frames)} frames needed the axis flip "
                          f"to point the camera at the ground")
        # Projected coordinates near 3.1e6 waste float precision in the
        # arithmetic downstream, and only differences are ever needed.
        self.origin = self._frames[0].centre.copy() if self._frames else np.zeros(3)
        self._frames = [
            Frame(f.frame_id, f.rotation, f.centre - self.origin, f.intrinsics,
                  f.name, f.longitude, f.latitude, f.altitude, f.path)
            for f in self._frames
        ]

    # -- parsing -----------------------------------------------------------

    def _image_size(self) -> tuple[int, int]:
        """Measured from a delivered PNG, never taken from the intrinsics file.

        The two disagree by a factor of four in this release.
        """
        for candidate in sorted((self.base / "images").glob("*_0.png")):
            from PIL import Image

            with Image.open(candidate) as handle:
                return handle.size
        return (1008, 756)

    def _read(self) -> list[Frame]:
        poses = (self.base / "poses" / "t_pose.txt").read_text().split("\n")
        intrinsic_lines = (self.base / "poses" / "t_intrinsic.txt").read_text().split("\n")
        time_path = self.base / "poses" / "time.txt"

        declared: dict[str, tuple[float, ...]] = {}
        for line in intrinsic_lines:
            parts = line.split()
            if len(parts) >= 8:
                declared[Path(parts[0]).name] = tuple(float(v) for v in parts[2:8])

        telemetry: dict[str, tuple[float, float, float]] = {}
        if time_path.exists():
            for line in time_path.read_text().split("\n"):
                parts = line.split()
                if len(parts) >= 4:
                    telemetry[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[3]))

        width, height = self._image_size()
        frames: list[Frame] = []
        raw: list[tuple] = []
        for index, line in enumerate(poses):
            parts = line.split()
            if len(parts) < 8:
                continue
            name = Path(parts[0]).name
            spec = declared.get(name)
            if spec is None:
                continue
            declared_w, declared_h, fx, fy, cx, cy = spec
            scale = width / declared_w
            if abs(height / declared_h - scale) > 1e-6:
                raise ValueError(
                    f"{name}: {width}x{height} is not a uniform scaling of "
                    f"{declared_w:.0f}x{declared_h:.0f}"
                )

            world_to_camera = _quaternion_to_matrix(*(float(v) for v in parts[1:5]))
            translation = np.array([float(v) for v in parts[5:8]])
            raw.append((index, name, world_to_camera, translation,
                        Intrinsics(fx * scale, fy * scale, cx * scale, cy * scale,
                                   width, height)))
            continue

        # Resolved PER FRAME, not per split. The gimbal pitch changes during a
        # flight -- jiaxiao/12-14 alone carries -30, -28.6, 0 and +45 -- and the
        # sign that reads as "down" changes with it. Taking a split-wide median
        # picks the majority and silently inverts the rest.
        self._flipped = 0
        for index, name, world_to_camera, translation, intrinsics in raw:
            camera_to_world = world_to_camera.T
            if (camera_to_world @ np.array([0.0, 0.0, 1.0]))[2] > 0:
                # Optical axis is above the horizon. A survey camera looks down,
                # so this frame is the other convention.
                camera_to_world = camera_to_world @ AXIS_FLIP
                self._flipped += 1
            longitude, latitude, altitude = telemetry.get(name, (np.nan,) * 3)
            image_path = self.base / "images" / f"{index}_0.png"
            frames.append(Frame(
                frame_id=index,
                rotation=camera_to_world,
                centre=-world_to_camera.T @ translation,
                intrinsics=intrinsics,
                name=name, longitude=longitude, latitude=latitude, altitude=altitude,
                path=image_path if image_path.exists() else None,
            ))
        return frames

    # -- access ------------------------------------------------------------

    def frames(self, with_images: bool = False) -> Iterator[Frame]:
        for frame in self._frames:
            if with_images and frame.path is None:
                continue
            yield frame

    def available(self) -> list[Frame]:
        """Frames whose imagery is on disk. The release is 38 GB; this is a slice."""
        return [f for f in self._frames if f.path is not None]

    @property
    def terrain(self) -> Terrain:
        if self._terrain is None:
            if not self._terrain_path.exists():
                raise FileNotFoundError(f"no terrain at {self._terrain_path}")
            self._terrain = Terrain.load(self._terrain_path)
        return self._terrain

    @property
    def local_terrain(self) -> Terrain:
        """Terrain in this session's frame. Always use this, never ``terrain``."""
        if self._local_terrain is None:
            self._local_terrain = self.terrain.translated(self.origin)
        return self._local_terrain

    def gnss_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Local (x, y) against WGS84 (lon, lat), for fitting and for scoring."""
        local, geographic = [], []
        for frame in self._frames:
            if np.isfinite(frame.longitude) and np.isfinite(frame.latitude):
                local.append(frame.centre[:2])
                geographic.append([frame.longitude, frame.latitude])
        return np.array(local), np.array(geographic)
