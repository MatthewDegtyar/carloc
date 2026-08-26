"""AirZoo-Real: posed aerial imagery over a georeferenced surface model.

Real DJI flights over four sites, each flown at several times of day, with 6-DoF
poses, per-frame intrinsics, a semantic layer, an inverse-depth render, and a
half-metre DSM covering the whole area. Everything the pipeline needs except
object annotations, which a detector supplies.

Why this is a better fit for the geometry than the ground-vehicle data: a camera
looking down at 45 degrees ranges by intersecting the terrain, so a single frame
suffices. The forward-motion degeneracy that dominates dashcam sequences -- no
perpendicular baseline, therefore no observable range -- simply does not arise.
It is replaced by a different one, the near-horizon ray, which `range/ground_plane.py`
detects on the same terms.

Conventions, all established by measurement against the data rather than assumed
from the file names -- see `docs/airzoo_format.md` for how each was pinned down:

* ``t_pose.txt``      ``path qw qx qy qz tx ty tz``, **world-to-camera**. The camera
                      centre is ``C = -R^T t``; the raw translation is not a position
                      and is millions of metres in magnitude.
* camera axes         **OpenGL** -- ``-Z`` forward, ``+Y`` up in the image. Reading
                      them as OpenCV points the optical axis at the sky.
* world frame         EPSG:4547 (CGCS2000 Gauss-Kruger), metres, sharing its datum
                      with the DSM. Coordinates are ~4e5 by ~3.1e6, so a local
                      origin is subtracted by default.
* ``t_intrinsic.txt`` quoted for 4032x3024; the delivered PNGs are 1008x756, so
                      intrinsics are scaled by the ratio rather than trusted as-is.
* ``images/i_0``      RGB. ``i_1`` inverse depth, 8-bit. ``i_2`` semantic labels.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geoloc_agent.contracts import Frame, Intrinsics, Pose
from geoloc_agent.geo import GeoOrigin
from geoloc_agent.io.base import Session, TruthObject
from geoloc_agent.io.dsm import DigitalSurfaceModel

OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0])
"""Flip taking AirZoo's camera axes to the pipeline's.

AirZoo looks down ``-Z`` with ``+Y`` up in the image; every ranger and projection
here assumes ``+Z`` forward with ``+Y`` down. The two differ by a half turn about
the optical axis' horizontal, which is a proper rotation (determinant +1), so it
composes into the pose without breaking orthonormality."""

DEPTH_IS_PER_FRAME_NORMALISED = True
"""The ``i_1`` render is normalised per frame, so it carries no absolute scale.

Worth recording because the opposite is the natural assumption and it is wrong in
a way that hides. Fitting ``1/range = a*png + b`` against DSM ray-casts on a
single frame gives a correlation of 0.96, which looks like a global calibration
has been found. Extending the same fit across 120 frames drops it to 0.71, and
fitting each frame separately shows why: the slope moves by 1.4x between frames
but the intercept moves by **12x**, and the maximum value is pinned at 255 in
every frame. That is a per-image stretch, not a shared encoding.

So a constant conversion would be wrong by a frame-dependent amount -- plausible
numbers, no error, quietly meaningless. There is no global constant to publish,
and the DSM is both metric and more accurate, so ranging goes through that and
this layer stays what it honestly is: relative structure."""


@dataclass(frozen=True)
class AirZooPose:
    name: str
    rotation: np.ndarray
    """Camera-to-world, already converted to the pipeline's axis convention."""

    centre: np.ndarray
    intrinsics: Intrinsics
    longitude: float = float("nan")
    latitude: float = float("nan")
    altitude: float = float("nan")


def _quaternion_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class AirZooSession(Session):
    """One site at one time of day."""

    name = "airzoo"

    def __init__(
        self,
        dataroot: str | Path = "sessions/airzoo",
        site: str = "guangchang",
        split: str = "12-14",
        load_images: bool = True,
        local_origin: bool = True,
        dsm_name: str = "fcw_hangtian_DSM.tif",
        max_frames: int | None = None,
        image_channel: int = 0,
    ) -> None:
        self.dataroot = Path(dataroot)
        self.site = site
        self.split = split
        self.load_images = load_images
        self.image_channel = image_channel
        self.max_frames = max_frames
        self._dsm_path = self.dataroot / "dom" / dsm_name
        self._dsm: DigitalSurfaceModel | None = None
        self._local_dsm: DigitalSurfaceModel | None = None

        self.root = self.dataroot / site / split
        if not (self.root / "poses").is_dir():
            raise FileNotFoundError(
                f"no AirZoo split at {self.root}. Expected "
                f"{self.root}/poses/{{t_pose.txt,t_intrinsic.txt,time.txt}}"
            )
        self.poses = self._read_poses()
        if self.max_frames is not None:
            self.poses = self.poses[: self.max_frames]

        # Large projected coordinates carried through covariance arithmetic lose
        # precision for no benefit; the geometry only ever needs differences.
        self.frame_origin = (
            self.poses[0].centre.copy() if (local_origin and self.poses) else np.zeros(3)
        )

    # -- parsing -----------------------------------------------------------

    def _read_poses(self) -> list[AirZooPose]:
        pose_lines = (self.root / "poses" / "t_pose.txt").read_text().split("\n")
        intr_lines = (self.root / "poses" / "t_intrinsic.txt").read_text().split("\n")

        intrinsics: dict[str, tuple[float, ...]] = {}
        for line in intr_lines:
            parts = line.split()
            if len(parts) < 8:
                continue
            key = Path(parts[0]).name
            intrinsics[key] = tuple(float(v) for v in parts[2:8])

        telemetry: dict[str, tuple[float, float, float]] = {}
        time_path = self.root / "poses" / "time.txt"
        if time_path.exists():
            for line in time_path.read_text().split("\n"):
                parts = line.split()
                if len(parts) >= 4:
                    telemetry[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[3]))

        width, height = self._image_size()
        poses: list[AirZooPose] = []
        for line in pose_lines:
            parts = line.split()
            if len(parts) < 8:
                continue
            key = Path(parts[0]).name
            quaternion = [float(v) for v in parts[1:5]]
            translation = np.array([float(v) for v in parts[5:8]])

            world_to_camera = _quaternion_to_matrix(*quaternion)
            centre = -world_to_camera.T @ translation
            rotation = world_to_camera.T @ OPENGL_TO_OPENCV

            declared = intrinsics.get(key)
            if declared is None:
                continue
            declared_w, declared_h, fx, fy, cx, cy = declared
            scale = width / declared_w if declared_w else 1.0
            if abs(height / declared_h - scale) > 1e-6:
                raise ValueError(
                    f"{key}: image {width}x{height} is not a uniform scaling of the "
                    f"declared {declared_w:.0f}x{declared_h:.0f}; a non-uniform scale "
                    "would need separate fx and fy factors and is not handled"
                )
            lon, lat, alt = telemetry.get(key, (float("nan"),) * 3)
            poses.append(AirZooPose(
                name=key, rotation=rotation, centre=centre,
                intrinsics=Intrinsics(fx=fx * scale, fy=fy * scale, cx=cx * scale,
                                      cy=cy * scale, width=width, height=height),
                longitude=lon, latitude=lat, altitude=alt,
            ))
        return poses

    def _image_size(self) -> tuple[int, int]:
        """Read one delivered PNG rather than trusting the quoted intrinsics.

        The two disagree by a factor of four in this release, and taking the
        quoted size would scale every bearing by the same factor -- an error that
        looks like a plausible lens rather than an obvious fault.
        """
        for candidate in sorted((self.root / "images").glob("*_0.png")):
            from PIL import Image

            with Image.open(candidate) as image:
                return image.size
        return (1008, 756)

    # -- Session -----------------------------------------------------------

    @property
    def dsm(self) -> DigitalSurfaceModel:
        if self._dsm is None:
            if not self._dsm_path.exists():
                raise FileNotFoundError(f"no surface model at {self._dsm_path}")
            self._dsm = DigitalSurfaceModel.load(self._dsm_path)
        return self._dsm

    @property
    def local_dsm(self) -> DigitalSurfaceModel:
        """The surface model in this session's frame, ready to range against.

        Always prefer this over ``dsm`` when working with frames from this
        session: ``dsm`` is in the raw projected coordinates and will not line up
        with poses once a local origin has been subtracted.
        """
        if self._local_dsm is None:
            self._local_dsm = self.dsm.translated(self.frame_origin)
        return self._local_dsm

    @property
    def origin(self) -> GeoOrigin | None:
        """Geographic origin, taken from the telemetry of the first frame.

        AirZoo carries real WGS84 longitude and latitude per image alongside the
        projected pose, so the local frame can be tied to the globe without a
        reprojection library and without inventing a datum. This is the one thing
        nuScenes could not offer -- its poses are map-local with no georeference at
        all, and every coordinate this project published from it carried an assumed
        origin. Here the origin is measured.
        """
        if not self.poses or not np.isfinite(self.poses[0].latitude):
            return None
        first = self.poses[0]
        return GeoOrigin(lat=first.latitude, lon=first.longitude, alt=first.altitude,
                         name=f"airzoo/{self.site}", provenance="DJI GNSS telemetry")

    def frames(self) -> Iterator[Frame]:
        for index, pose in enumerate(self.poses):
            image = self._load_image(index) if self.load_images else None
            yield Frame(
                frame_id=index,
                timestamp=float(index),
                intrinsics=pose.intrinsics,
                pose=Pose(R=pose.rotation, t=pose.centre - self.frame_origin),
                image=image,
                source=f"airzoo/{self.site}/{self.split}/{pose.name}",
            )

    def truth(self) -> dict[str, TruthObject]:
        """Empty: AirZoo annotates geometry, not object instances.

        Stated rather than faked. Geolocation error cannot be scored per-object
        here; it is scored against the surface model instead, which is what
        `eval` uses.
        """
        return {}

    # -- extras beyond the Session contract --------------------------------

    def _channel_path(self, index: int, channel: int) -> Path:
        return self.root / "images" / f"{index}_{channel}.png"

    def _load_image(self, index: int) -> np.ndarray | None:
        path = self._channel_path(index, self.image_channel)
        if not path.exists():
            return None
        from PIL import Image

        with Image.open(path) as image:
            return np.array(image.convert("RGB"))

    def inverse_depth_at(self, index: int) -> np.ndarray | None:
        """The depth render, as delivered: per-frame-normalised inverse depth.

        **Not metres, and not convertible to metres without per-frame
        calibration** -- see DEPTH_IS_PER_FRAME_NORMALISED. Larger values are
        nearer. Useful for relative structure, occlusion ordering and masking;
        useless as a distance. For a distance, ray-cast ``local_dsm``.
        """
        path = self._channel_path(index, 1)
        if not path.exists():
            return None
        from PIL import Image

        with Image.open(path) as image:
            return np.array(image.convert("L")).astype(float)

    def depth_scale_against_surface(self, index: int, samples: int = 13
                                    ) -> tuple[float, float] | None:
        """Fit this one frame's ``1/range = a*png + b`` against the surface model.

        The only honest way to get metres out of the render, and it needs the DSM
        anyway -- so it exists for checking the two against each other, not as a
        ranging path.
        """
        from geoloc_agent.geometry import bearing_from_pixel

        png = self.inverse_depth_at(index)
        if png is None or index >= len(self.poses):
            return None
        frame = next(f for f in self.frames() if f.frame_id == index)
        surface = self.local_dsm
        height, width = png.shape
        values, ranges = [], []
        for u in np.linspace(50, width - 50, samples).astype(int):
            for v in np.linspace(50, height - 50, samples).astype(int):
                bearing = bearing_from_pixel(u, v, frame.intrinsics, frame.pose)
                distance = surface.raycast(frame.pose.t, bearing)
                if np.isfinite(distance) and distance > 0:
                    values.append(png[v, u])
                    ranges.append(distance)
        if len(values) < 30:
            return None
        slope, offset = np.polyfit(np.array(values), 1.0 / np.array(ranges), 1)
        return float(slope), float(offset)

    def labels_at(self, index: int) -> np.ndarray | None:
        """The semantic layer, as delivered (an RGB rendering, not class indices)."""
        path = self._channel_path(index, 2)
        if not path.exists():
            return None
        from PIL import Image

        with Image.open(path) as image:
            return np.array(image.convert("RGB"))

    def available_frames(self) -> list[int]:
        """Indices whose imagery is actually on disk.

        The full release is 38 GB and this works on a subset, so the pose table is
        routinely longer than the imagery beside it.
        """
        return [i for i in range(len(self.poses)) if self._channel_path(i, 0).exists()]
