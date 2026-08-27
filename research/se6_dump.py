"""Dump every left-kerb detection per frame, with camera geometry and bearing.

The raw material for tracking: instead of collapsing detections immediately, keep
them all, tagged with which frame they came from, where the camera was, and the
bearing to the car. Association and triangulation happen downstream.
"""
import glob
import json
import math

import numpy as np
from PIL import Image

from carloc.appearance import classify_colour, dominant_rgb
from carloc.rfdetr_detect import COCO_VEHICLES

W, H = 1920, 1080
F_PX = 687.0
LATERAL_M = 7.0
MIN_BEARING_DEG = 20.0     # keep cars left of this; below it range is hopeless
MIN_BOX_H = 80


def frame_scale(lat):
    return 111_320*math.cos(math.radians(lat)), 110_540


def main():
    from rfdetr import RFDETRBase
    with open("reports/se6_track.json") as fh:
        track = json.load(fh)
    ts = np.array([r["t"] for r in track])
    # camera along-track arc length s for each fix
    mx, my = frame_scale(25.768)
    xs = np.array([r["lon"]*mx for r in track]); ys = np.array([r["lat"]*my for r in track])
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])

    model = RFDETRBase()
    files = sorted(glob.glob("reports/se6frames/f_*.jpg"))
    dets = []
    for i, path in enumerate(files):
        t = 420.0 + i/4.0
        k = int(np.clip(np.searchsorted(ts, t), 0, len(track)-1))
        fix = track[k]
        image = Image.open(path).convert("RGB"); arr = np.asarray(image)
        det = model.predict(image, threshold=0.45)
        cls = np.array(det.class_id); box = np.array(det.xyxy); conf = np.array(det.confidence)
        keep = np.isin(cls, list(COCO_VEHICLES))
        for (x1, y1, x2, y2), cid, sco in zip(box[keep], cls[keep], conf[keep]):
            if (y2-y1) < MIN_BOX_H:
                continue
            cx = (x1+x2)/2
            bearing = math.degrees(math.atan((cx - W/2)/F_PX))
            if bearing > -MIN_BEARING_DEG:
                continue
            crop = arr[int(max(y1, 0)):int(y2), int(max(x1, 0)):int(x2)]
            rgb = dominant_rgb(crop)
            dets.append({
                "frame": i, "t": round(t, 2), "s_cam": float(s[k]),
                "cam_lat": fix["lat"], "cam_lon": fix["lon"], "heading": fix["heading_deg"],
                "sig_along": fix["sigma_along_m"],
                "bearing": round(bearing, 2),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "cx": float(cx), "cy": float((y1+y2)/2), "bh": float(y2-y1), "bw": float(x2-x1),
                "conf": float(sco), "cls": COCO_VEHICLES.get(int(cid), "car"),
                "color": classify_colour(rgb), "rgb": [int(v) for v in rgb]})
        if i % 80 == 0:
            print(f"  frame {i}/{len(files)}  dets={len(dets)}", flush=True)
    json.dump({"dets": dets, "cam_s": s.tolist(), "lateral_m": LATERAL_M,
               "f_px": F_PX, "W": W}, open("reports/se6_dets.json", "w"))
    print(f"\n{len(dets)} left-kerb detections over {len(files)} frames")
    print(f"camera travelled {s[-1]:.1f} m")


if __name__ == "__main__":
    main()
