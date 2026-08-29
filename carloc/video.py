"""Video in, parked cars out — the one call most users need.

``count_parked(video, start, end)`` runs the whole vision pipeline over a segment
and returns one :class:`~carloc.types.ParkedCar` per physical car, positioned
along the street with its uncertainty and the times it was seen. That output is
*relative*; hand it and a :class:`~carloc.trajectory.Trajectory` to
:func:`carloc.geolocate.geolocate` to get absolute, timestamped cars.

Compose it your own way if you prefer: swap the detector (any object with
``predict(rgb_array) -> boxes`` in RF-DETR's shape, or pass your own detections
to :func:`track_parked`), or change the frame rate and kerb geometry.
"""

from __future__ import annotations

import glob
import math
import subprocess
import tempfile
from pathlib import Path

from carloc.types import ParkedCar

W, H = 1920, 1080
F_PX = 687.0                 # px per radian at 1920 wide (~109 deg HFOV dashcam)
MIN_BEARING_DEG = 20.0
MIN_BOX_H = 80


def extract_frames(video: str, t0: float, t1: float, out_dir: str, fps: float = 4.0,
                   width: int = W, height: int = H) -> list[str]:
    """Decode ``[t0, t1)`` of ``video`` to JPEG frames at ``fps``."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", str(t0), "-to", str(t1), "-i", video,
         "-vf", f"fps={fps},scale={width}:{height}", "-pix_fmt", "yuvj420p",
         f"{out_dir}/f_%05d.jpg"], check=True)
    return sorted(glob.glob(f"{out_dir}/f_*.jpg"))


def _scene_motion(files):
    """Relative camera along-track distance per frame, from residual scene motion."""
    import cv2
    import numpy as np

    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    mags, pg = [], None
    for path in files:
        g = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), (640, 360))
        v = 0.0
        if pg is not None:
            p0 = cv2.goodFeaturesToTrack(pg, maxCorners=500, qualityLevel=0.01,
                                         minDistance=6, blockSize=7)
            if p0 is not None and len(p0) > 20:
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
    return np.concatenate([[0.0], np.cumsum(smoothed)])


def _camera_along(files, t0, fps, speed_mps, trajectory):
    """Metric camera along-track distance per frame.

    With a trajectory (e.g. GPS), integrate its real positions — exact. Without
    one, scale the *shape* of the scene-motion profile (which captures stops and
    speed-ups) to a total of ``speed_mps * duration`` — approximate, but metric,
    which is what the triangulation and the atomic-merge need. Counting without a
    scale over-counts; this is the minimum to make it correct.
    """
    import numpy as np

    if trajectory is not None:
        import math
        pts = [trajectory.position_at(t0 + i / fps) for i in range(len(files))]
        s = [0.0]
        for (la0, lo0, _), (la1, lo1, _) in zip(pts, pts[1:], strict=False):
            my = 111_320.0
            mx = my * math.cos(math.radians(la0))
            s.append(s[-1] + math.hypot((la1 - la0) * my, (lo1 - lo0) * mx))
        return np.array(s)
    cum = _scene_motion(files)
    total = speed_mps * (len(files) / fps)
    return cum / (cum[-1] or 1.0) * total


def _detect(files, model, t0, fps, both_sides, speed_mps=7.0, trajectory=None):
    """RF-DETR over frames -> kerb-side detection dicts, tagged with camera s_cam."""
    import cv2
    import numpy as np

    from carloc.appearance import classify_colour, dominant_rgb
    from carloc.detect import COCO_VEHICLES

    cum = _camera_along(files, t0, fps, speed_mps, trajectory)
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
            if abs(bearing) < MIN_BEARING_DEG or (bearing > 0 and not both_sides):
                continue
            crop = arr[int(max(y1, 0)):int(y2), int(max(x1, 0)):int(x2)]
            d = {"frame": i, "t": t0 + i / fps, "s_cam": s_cam, "bearing": -abs(bearing),
                 "cx": float(cx), "cy": float((y1 + y2) / 2), "bh": float(y2 - y1),
                 "bw": float(x2 - x1), "color": classify_colour(dominant_rgb(crop)),
                 "cls": COCO_VEHICLES.get(int(cid), "car")}
            (left if bearing < 0 else right).append(d)
    return left, right


def track_parked(detections: list[dict], side: str, lateral_m: float = 7.0,
                 min_frames: int = 2) -> list[ParkedCar]:
    """Associate → triangulate → slot a list of detection dicts into parked cars.

    ``min_frames`` is the frame-confidence threshold: a car is only counted if it
    was seen (detected and tracked) in at least this many frames across the whole
    pass. Raising it trades recall for precision — 2 keeps every triangulable car,
    higher values drop the brief, one-glimpse detections most likely to be spurious
    or moving. (Two detections are always needed to triangulate, so 2 is the floor.)

    ``detections`` need the keys produced internally (frame, t, s_cam, bearing,
    cx, cy, bh, bw, color, cls). Most users call :func:`count_parked` instead;
    this is the composable seam for a custom detector.
    """
    from carloc.tracking import associate, slot, triangulate

    tracks = [t for t in associate(detections, W=W) if len(t.dets) >= 2]
    recs = []
    for t in tracks:
        S, sig, k = triangulate(t, lateral_m)
        if not math.isfinite(S):
            continue
        abeam = max(t.dets, key=lambda d: abs(d["bearing"]))
        recs.append({"S": S, "sigma_S": max(sig, 1.0), "color": t.color, "cls": t.cls,
                     "ndet": k, "t_first": t.dets[0]["frame"], "t_last": t.dets[-1]["frame"],
                     "abeam_t": abeam["t"], "first_t": t.dets[0]["t"], "last_t": t.dets[-1]["t"]})
    cars = []
    for c in slot(recs, min_sep_m=4.5):
        if c["ndet"] < min_frames:               # frame-confidence threshold
            continue
        near = min((r for r in recs if r["color"] == c["color"]),
                   key=lambda r: abs(r["S"] - c["S"]), default=None)
        conf = round(1.0 - 0.6 ** (c["ndet"] - 1), 2)   # 2 frames->0.4, 5->0.87, 8->0.98
        cars.append(ParkedCar(
            along_m=round(c["S"], 1), side=side, sigma_along_m=round(c["sigma_S"], 1),
            abeam_t=round(near["abeam_t"], 2) if near else 0.0,
            first_t=round(near["first_t"], 2) if near else 0.0,
            last_t=round(near["last_t"], 2) if near else 0.0,
            vehicle_class=c["cls"], color=c["color"],
            n_detections=c["ndet"], n_tracklets=c["n_tracklets"], confidence=conf))
    return cars


def detect_segment(video: str, start: float, end: float, both_sides: bool = True,
                   fps: float = 4.0, detector=None, speed_mps: float = 7.0,
                   trajectory=None) -> tuple[list[dict], list[dict]]:
    """Detect kerb-side vehicles across a segment, returning raw (left, right) dicts.

    The detection half of :func:`count_parked`, exposed so it can be run once and
    then tracked at several ``min_frames`` thresholds (see
    :mod:`carloc.confidence`) without re-running the model.
    """
    if detector is None:
        from rfdetr import RFDETRBase
        detector = RFDETRBase()
    with tempfile.TemporaryDirectory() as tmp:
        files = extract_frames(video, start, end, tmp, fps=fps)
        return _detect(files, detector, start, fps, both_sides,
                       speed_mps=speed_mps, trajectory=trajectory)


def count_parked(video: str, start: float, end: float, lateral_m: float = 7.0,
                 both_sides: bool = True, fps: float = 4.0, detector=None,
                 speed_mps: float = 7.0, trajectory=None,
                 min_frames: int = 2) -> list[ParkedCar]:
    """Count the parked cars in ``[start, end)`` seconds of ``video``.

    Returns one :class:`ParkedCar` per physical car, positioned along the street.
    Pass ``detector`` (anything with ``predict(rgb) -> {class_id, xyxy}``) to swap
    the model; by default RF-DETR is used.

    ``min_frames`` is the frame-confidence threshold — the fewest frames a car must
    be tracked across to be counted (see :func:`track_parked`). Higher = stricter.

    The along-street positions are metric only up to a scale: pass ``trajectory``
    (a GPS track is ideal) and it is used to scale the camera's motion exactly, or
    give ``speed_mps`` as the segment's rough average speed. Without either, the
    count uses a default urban speed and should be treated as approximate.
    """
    left, right = detect_segment(video, start, end, both_sides=both_sides, fps=fps,
                                 detector=detector, speed_mps=speed_mps, trajectory=trajectory)
    cars = (track_parked(left, "left", lateral_m, min_frames)
            + track_parked(right, "right", lateral_m, min_frames))
    cars.sort(key=lambda c: c.along_m)
    return cars
