"""Build a benchmark manifest from nuScenes.

Two depths per object, because they are different questions and conflating them
would quietly charge every model a bias it did not earn:

``surface_depth_m``   depth to the near face of the 3-D box, which is what the
                      camera can see and therefore what a depth model predicts.
``centroid_depth_m``  depth to the object centre, which is what a geolocation
                      pipeline needs. For a 4.5 m car viewed end-on these differ
                      by over 2 m -- comparable to the error being measured.

Objects are filtered to what can honestly be scored: sufficiently visible, fully
in front of the camera, a big enough box to sample, and inside the declared range.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from depthbench.schema import GtObject, Manifest, RefObject, Sample

from geoloc_agent.io.nuscenes import NuScenesSession

MIN_BOX_PX = 24.0
MIN_DEPTH_M = 2.0
SCORED_CLASSES = ("car", "truck", "bus", "pedestrian")

# Reference objects for relative-depth rescaling must have a *reliable* real
# height. A car roof line varies little between models; a "truck" in nuScenes
# spans vans to articulated lorries, so it is never used as the reference.
REFERENCE_CLASSES = ("car", "pedestrian")


def build_manifest(
    dataroot: str | Path,
    scenes: list[str],
    version: str = "v1.0-mini",
    max_range_m: float = 50.0,
    stride: int = 2,
    min_visibility: int = 3,
) -> Manifest:
    # Absolute. Runners execute as subprocesses and at least one of them (Depth
    # Pro) chdirs to find its own checkpoint, at which point a relative image path
    # silently stops resolving.
    dataroot = Path(dataroot).resolve()
    manifest = Manifest(source=f"nuscenes {version} {','.join(scenes)}", max_range_m=max_range_m)

    for scene in scenes:
        session = NuScenesSession(
            dataroot=dataroot, scene=scene, version=version, min_visibility=min_visibility
        )
        truth = session.truth()
        tables = session.nusc

        filenames = []
        token = session._scene["first_sample_token"]
        while token:
            sample = tables.get("sample", token)
            filenames.append(tables.get("sample_data", sample["data"]["CAM_FRONT"])["filename"])
            token = sample["next"]

        frames = list(session.frames())
        for frame, filename in list(zip(frames, filenames, strict=True))[::stride]:
            objects = _objects_in(frame, truth, max_range_m)
            if not objects:
                continue
            manifest.samples.append(
                Sample(
                    image=str(Path(dataroot) / filename),
                    width=frame.intrinsics.width,
                    height=frame.intrinsics.height,
                    K=frame.intrinsics.K.tolist(),
                    objects=objects,
                    reference=_pick_reference(objects, truth),
                )
            )
    return manifest


def _objects_in(frame, truth, max_range_m: float) -> list[GtObject]:
    intr = frame.intrinsics
    out: list[GtObject] = []
    for obj_id, obj in truth.items():
        if obj.cls not in SCORED_CLASSES or frame.frame_id not in obj.positions:
            continue
        corners = obj.corners(frame.frame_id)
        cam = (frame.pose.R.T @ (corners - frame.pose.t).T).T
        # Whole box in front: a box straddling the image plane projects to a hull
        # spanning the frame, and its sampled depth would be meaningless.
        if np.any(cam[:, 2] < MIN_DEPTH_M):
            continue

        centre_cam = frame.pose.R.T @ (obj.at(frame.frame_id) - frame.pose.t)
        centroid_depth = float(centre_cam[2])
        surface_depth = float(cam[:, 2].min())
        if centroid_depth > max_range_m:
            continue

        uv = (intr.K @ cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        x1, y1 = uv.min(axis=0)
        x2, y2 = uv.max(axis=0)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if not intr.contains(cx, cy):
            continue
        x1, x2 = np.clip([x1, x2], 0.0, intr.width - 1.0)
        y1, y2 = np.clip([y1, y2], 0.0, intr.height - 1.0)
        if (x2 - x1) < MIN_BOX_PX or (y2 - y1) < MIN_BOX_PX:
            continue

        out.append(
            GtObject(
                obj_id=f"{obj_id[:10]}@{frame.frame_id}",
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                cls=obj.cls,
                surface_depth_m=surface_depth,
                centroid_depth_m=centroid_depth,
                height_m=float(obj.size[2]),
                visibility=obj.visibility_at(frame.frame_id),
            )
        )
    return out


def _pick_reference(objects: list[GtObject], truth) -> RefObject | None:
    """The largest well-understood object in the frame.

    Largest because rescaling divides by the reference's apparent size, so a
    small reference multiplies its own pixel error into every other estimate in
    the image.
    """
    candidates = [o for o in objects if o.cls in REFERENCE_CLASSES and o.height_m > 0]
    if not candidates:
        return None
    best = max(candidates, key=lambda o: o.bbox[3] - o.bbox[1])
    return RefObject(bbox=list(best.bbox), height_m=best.height_m, cls=best.cls)
