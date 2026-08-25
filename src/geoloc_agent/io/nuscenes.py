"""nuScenes v1.0-mini session loader.

Implements the same ``Session`` interface as ``SyntheticSession``, so the Phase 0
pipeline runs on it with no change to ``fuse/``. The devkit is an optional
dependency; importing this module without it raises with an install hint rather
than failing at some confusing point later.

The three things that bite, handled explicitly:

**Poses are map-local, not WGS84.** ``ego_pose`` is metres in a per-map frame
with no georeference anywhere in the dataset. We attach a documented origin per
map and convert with a local ENU transform. Everything downstream is therefore
accurate *relative* to that origin and no better than roughly city-block
absolute. That assumption is recorded on the session and reproduced in the
report; it is never presented as a survey fix.

**Annotations are 2 Hz, camera sweeps are 12 Hz.** Keyframes carry the 3-D boxes
and are the only frames that can be scored. Sweeps carry no annotations but do
carry real pose, so they are extra bearings. ``include_sweeps`` controls whether
they are loaded; scoring always uses keyframes.

**Forward motion dominates.** The ego vehicle mostly drives straight, so objects
near the optical axis have almost no perpendicular baseline. This is not a
dataset defect to work around -- it is the degenerate case occurring naturally on
real data, and results are reported split by geometry class.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from geoloc_agent.contracts import Frame, Intrinsics, Pose
from geoloc_agent.geo import NUSCENES_ORIGINS, GeoOrigin
from geoloc_agent.io.base import Session, TruthObject

# nuScenes camera frames are already OpenCV convention (x right, y down,
# z forward), so no axis permutation is needed between it and our camera frame.
DEFAULT_CAMERA = "CAM_FRONT"


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """nuScenes stores rotations as (w, x, y, z)."""
    w, x, y, z = (float(v) for v in q)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class NuScenesSession(Session):
    """One nuScenes scene, exposed through the standard Session interface."""

    def __init__(
        self,
        dataroot: str | Path,
        scene: str | int = 0,
        version: str = "v1.0-mini",
        camera: str = DEFAULT_CAMERA,
        include_sweeps: bool = False,
        load_images: bool = False,
        gps_sigma: float = 0.0,
        heading_sigma: float = 0.0,
        classes: tuple[str, ...] = ("vehicle", "human"),
        min_visibility: int = 1,
        nusc=None,
    ) -> None:
        self.dataroot = Path(dataroot)
        self.camera = camera
        self.include_sweeps = include_sweeps
        self.load_images = load_images
        self.classes = classes
        # nuScenes visibility: 1 = 0-40% visible, 4 = 80-100%. The default keeps
        # everything; raise it when measuring detector recall, where counting
        # near-invisible annotations as misses measures the annotation policy
        # rather than the detector.
        self.min_visibility = int(min_visibility)
        self._pose_cov = np.zeros((6, 6))
        position_var = max(gps_sigma**2, 1e-12)
        self._pose_cov[0, 0] = self._pose_cov[1, 1] = self._pose_cov[2, 2] = position_var
        self._pose_cov[3, 3] = self._pose_cov[4, 4] = 1e-12
        self._pose_cov[5, 5] = max(heading_sigma**2, 1e-12)

        self.nusc = nusc if nusc is not None else self._load_devkit(version)
        self._scene = self._resolve_scene(scene)
        self.name = f"nuscenes:{self._scene['name']}"

        log = self.nusc.get("log", self._scene["log_token"])
        self.map_name = log["location"]
        self._origin = NUSCENES_ORIGINS.get(
            self.map_name, GeoOrigin(0.0, 0.0, 0.0, self.map_name, "unknown map, origin at (0,0)")
        )

        self._frames: list[Frame] = []
        self._truth: dict[str, TruthObject] = {}
        self._build()

    # -- Session interface ------------------------------------------------

    @property
    def origin(self) -> GeoOrigin:
        return self._origin

    def frames(self) -> Iterator[Frame]:
        yield from self._frames

    def truth(self) -> dict[str, TruthObject]:
        return self._truth

    def scripted_detections(self) -> None:
        """Real imagery: a real detector is required. No scripted shortcut."""
        return None

    # -- construction -----------------------------------------------------

    def _load_devkit(self, version: str):
        """Prefer the bundled JSON reader; fall back to the devkit if installed.

        The reader answers the same two questions and avoids a dependency chain
        (shapely/GEOS, opencv, scikit-learn) that is heavy for reading nine JSON
        files. Pass ``nusc=NuScenes(...)`` explicitly to use the devkit instead.
        """
        from geoloc_agent.io.nuscenes_tables import NuScenesTables

        return NuScenesTables(self.dataroot, version)

    def _resolve_scene(self, scene: str | int) -> dict:
        if isinstance(scene, int):
            return self.nusc.scene[scene]
        for candidate in self.nusc.scene:
            if candidate["name"] == scene or candidate["token"] == scene:
                return candidate
        raise KeyError(f"scene {scene!r} not found in this nuScenes split")

    def _pose_for(self, sample_data: dict) -> Pose:
        """Compose sensor extrinsics onto the ego pose to get camera->world."""
        ego = self.nusc.get("ego_pose", sample_data["ego_pose_token"])
        calibrated = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        sensor_to_body = Pose(
            R=quaternion_to_matrix(np.asarray(calibrated["rotation"])),
            t=np.asarray(calibrated["translation"], dtype=float),
        )
        body_to_world = Pose(
            R=quaternion_to_matrix(np.asarray(ego["rotation"])),
            t=np.asarray(ego["translation"], dtype=float),
        )
        return Pose.compose(sensor_to_body, body_to_world, cov=self._pose_cov)

    def _intrinsics_for(self, sample_data: dict) -> Intrinsics:
        calibrated = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        return Intrinsics.from_matrix(
            np.asarray(calibrated["camera_intrinsic"], dtype=float),
            width=sample_data["width"],
            height=sample_data["height"],
        )

    def _keep_class(self, name: str) -> bool:
        return any(name.startswith(prefix) for prefix in self.classes)

    def _build(self) -> None:
        frame_id = 0
        t0: float | None = None
        sample_token = self._scene["first_sample_token"]

        while sample_token:
            sample = self.nusc.get("sample", sample_token)
            keyframe_data = self.nusc.get("sample_data", sample["data"][self.camera])

            # Keyframe, then optionally the unannotated sweeps that follow it.
            chain = [keyframe_data]
            if self.include_sweeps:
                nxt = keyframe_data["next"]
                while nxt:
                    candidate = self.nusc.get("sample_data", nxt)
                    if candidate["is_key_frame"]:
                        break
                    chain.append(candidate)
                    nxt = candidate["next"]

            for sample_data in chain:
                timestamp = sample_data["timestamp"] * 1e-6
                if t0 is None:
                    t0 = timestamp
                image = None
                if self.load_images:
                    image = self._read_image(sample_data)
                self._frames.append(
                    Frame(
                        frame_id=frame_id,
                        timestamp=timestamp - t0,
                        intrinsics=self._intrinsics_for(sample_data),
                        pose=self._pose_for(sample_data),
                        image=image,
                        source=f"{self.name}:{self.camera}",
                        is_keyframe=bool(sample_data["is_key_frame"]),
                    )
                )
                if sample_data["is_key_frame"]:
                    self._collect_truth(sample, frame_id)
                frame_id += 1

            sample_token = sample["next"]

    def _collect_truth(self, sample: dict, frame_id: int) -> None:
        """3-D boxes in the global frame. This is what makes error directly scorable."""
        for annotation_token in sample["anns"]:
            annotation = self.nusc.get("sample_annotation", annotation_token)
            if not self._keep_class(annotation["category_name"]):
                continue
            visibility = int(annotation.get("visibility_token", 4))
            if visibility < self.min_visibility:
                continue
            instance = annotation["instance_token"]
            position = np.asarray(annotation["translation"], dtype=float)
            rotation = quaternion_to_matrix(np.asarray(annotation["rotation"], dtype=float))
            width, length, height = (float(v) for v in annotation["size"])
            if instance not in self._truth:
                self._truth[instance] = TruthObject(
                    obj_id=instance,
                    position=position,
                    cls=_simplify_class(annotation["category_name"]),
                    size=(width, length, height),
                    rotation=rotation,
                )
            self._truth[instance].positions[frame_id] = position
            self._truth[instance].rotations[frame_id] = rotation
            self._truth[instance].visibilities[frame_id] = visibility

    def _read_image(self, sample_data: dict):  # pragma: no cover - needs real data
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to load nuScenes imagery") from exc
        return np.asarray(Image.open(self.dataroot / sample_data["filename"]))

    # -- geo --------------------------------------------------------------

    def to_wgs84(self, position: np.ndarray) -> tuple[float, float, float]:
        """Map-local metres -> lat/lon, under the assumed origin. Stated, not implied."""
        return self._origin.enu_to_wgs84(position)

    @property
    def georeference_note(self) -> str:
        return (
            f"nuScenes poses are metres in the '{self.map_name}' map frame and carry no "
            f"georeference. Coordinates are converted through an assumed origin "
            f"({self._origin.lat:.6f}, {self._origin.lon:.6f}, provenance: "
            f"{self._origin.provenance}). Relative accuracy is preserved; absolute "
            f"accuracy is no better than the origin assumption."
        )


def _simplify_class(category_name: str) -> str:
    """`vehicle.car` -> `car`, `human.pedestrian.adult` -> `pedestrian`."""
    parts = category_name.split(".")
    if parts[0] == "human":
        return "pedestrian"
    return parts[-1] if len(parts) > 1 else parts[0]
