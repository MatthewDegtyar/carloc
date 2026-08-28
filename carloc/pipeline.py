"""End-to-end parked-car counting from a video segment.

Consolidates what the per-experiment `research/` scripts each re-implemented:
extract frames -> detect kerb-side vehicles -> track across frames -> triangulate
each car's along-street position -> slot into atomic physical cars. Returns one
record per parked car with its relative along-street position, side, appearance
and support. Absolute lat/lon is a separate, anchor-dependent step (see the
`localise` helpers and `research/*_pipeline.py`), because that is where the
accuracy actually varies -- see the README.
"""

from __future__ import annotations

import glob
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

W, H = 1920, 1080
F_PX = 687.0                 # px per radian at 1920 wide (~109 deg HFOV dashcam)
MIN_BEARING_DEG = 20.0
MIN_BOX_H = 80


@dataclass
class ParkedCar:
    """One physical parked car, counted once."""

    along_m: float           # position along the street, from the segment start
    sigma_m: float           # 1-sigma along-street uncertainty
    side: str                # "left" | "right" (kerb, relative to travel)
    vehicle_class: str
    color: str
    detections: int          # frames it was tracked across
    tracklets: int           # >1 means rebuilt from occlusion-split pieces


def extract_frames(video: str, t0: float, t1: float, out_dir: str, fps: float = 4.0,
                   width: int = W, height: int = H) -> list[str]:
    """Decode [t0, t1) of `video` to JPEG frames at `fps`. Returns sorted paths."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", str(t0), "-to", str(t1), "-i", video,
         "-vf", f"fps={fps},scale={width}:{height}", "-pix_fmt", "yuvj420p",
         f"{out_dir}/f_%05d.jpg"],
        check=True)
    return sorted(glob.glob(f"{out_dir}/f_*.jpg"))


def _detect_frames(files, model, lateral_m, both_sides, on_progress=None):
    """RF-DETR over the frames; keep kerb-side vehicles with a bearing tag."""
    import cv2
    import numpy as np

    from carloc.appearance import classify_colour, dominant_rgb
    from carloc.rfdetr_detect import COCO_VEHICLES

    # camera along-position per frame, from residual scene motion (relative)
    mags = []
    pg = None
    for path in files:
        g = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), (640, 360))
        v = 0.0
        if pg is not None:
            p0 = cv2.goodFeaturesToTrack(pg, maxCorners=500, qualityLevel=0.01,
                                         minDistance=6, blockSize=7)
            if p0 is not None and len(p0) > 20:
                lk = dict(winSize=(21, 21), maxLevel=3, criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
                p1, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, p0, None, **lk)
                if p1 is not None:
                    s = st.ravel() == 1
                    a, b = p0[s, 0], p1[s, 0]
                    if len(a) > 20:
                        M, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                           ransacReprojThreshold=2.0)
                        if M is not None:
                            v = float(np.percentile(
                                np.linalg.norm(b - ((a @ M[:, :2].T) + M[:, 2]), axis=1), 80))
        mags.append(v)
        pg = g
    smoothed = np.convolve(mags, np.ones(5) / 5, mode="same")
    cum = np.concatenate([[0.0], np.cumsum(smoothed)])
    total = float(cum[-1]) or 1.0

    left, right = [], []
    for i, path in enumerate(files):
        s_cam = float(cum[min(i, len(cum) - 1)])
        arr = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        res = model.predict(arr, threshold=0.45)
        cls = np.array(res.class_id)
        box = np.array(res.xyxy)
        keep = np.isin(cls, list(COCO_VEHICLES))
        for (x1, y1, x2, y2), cid in zip(box[keep], cls[keep], strict=False):
            if (y2 - y1) < MIN_BOX_H:
                continue
            cx = (x1 + x2) / 2
            bearing = math.degrees(math.atan((cx - W / 2) / F_PX))
            if abs(bearing) < MIN_BEARING_DEG:
                continue
            if bearing > 0 and not both_sides:
                continue
            crop = arr[int(max(y1, 0)):int(y2), int(max(x1, 0)):int(x2)]
            d = {"frame": i, "s_cam": s_cam, "bearing": -abs(bearing),
                 "cx": float(cx), "cy": float((y1 + y2) / 2), "bh": float(y2 - y1),
                 "bw": float(x2 - x1), "color": classify_colour(dominant_rgb(crop)),
                 "cls": COCO_VEHICLES.get(int(cid), "car")}
            (left if bearing < 0 else right).append(d)
        if on_progress and i % 50 == 0:
            on_progress(i, len(files))
    return left, right, total


def _slot_side(dets, lateral_m):
    from carloc.tracking import associate, slot, triangulate
    tracks = [t for t in associate(dets, W=W) if len(t.dets) >= 2]
    recs = []
    for t in tracks:
        S, sig, k = triangulate(t, lateral_m)
        if not math.isfinite(S):
            continue
        recs.append({"S": S, "sigma_S": max(sig, 1.0), "color": t.color, "cls": t.cls,
                     "ndet": k, "t_first": t.dets[0]["frame"], "t_last": t.dets[-1]["frame"]})
    return slot(recs, min_sep_m=4.5)


def count_parked(video: str, t0: float, t1: float, lateral_m: float = 7.0,
                 both_sides: bool = True, fps: float = 4.0, on_progress=None
                 ) -> list[ParkedCar]:
    """Detect, track and atomically count parked cars in [t0, t1) of a video."""
    from rfdetr import RFDETRBase

    model = RFDETRBase()
    with tempfile.TemporaryDirectory() as tmp:
        files = extract_frames(video, t0, t1, tmp, fps=fps)
        left, right, _ = _detect_frames(files, model, lateral_m, both_sides, on_progress)

    cars: list[ParkedCar] = []
    for side, dets in (("left", left), ("right", right)):
        for c in _slot_side(dets, lateral_m):
            cars.append(ParkedCar(along_m=round(c["S"], 1), sigma_m=round(c["sigma_S"], 1),
                                   side=side, vehicle_class=c["cls"], color=c["color"],
                                   detections=c["ndet"], tracklets=c["n_tracklets"]))
    cars.sort(key=lambda c: c.along_m)
    return cars
