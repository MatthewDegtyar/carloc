"""Wynwood dense-parking run: detect both kerbs in one pass, track, place on NW 2nd Ave."""
import glob
import json
import math
from collections import Counter

import cv2
import numpy as np

from carloc.appearance import classify_colour, dominant_rgb
from carloc.rfdetr_detect import COCO_VEHICLES
from carloc.tracking import associate, slot, triangulate

W, H = 1920, 1080
F_PX = 687.0
LATERAL_M = 6.0
MIN_BEARING = 20.0
MIN_BOX_H = 90
LAT0, LAT1 = 25.79930, 25.80640
LON_AVE = -80.19890
MX = 111_320 * math.cos(math.radians(25.803))
MY = 110_540
D = (LAT1 - LAT0) * MY


def motion_from_frames(files):
    """Scene-motion cumulative from the extracted frames (fast, no seeking)."""
    mags = []
    pg0 = cv2.cvtColor(cv2.imread(files[0]), cv2.COLOR_BGR2GRAY)  # noqa
    pg = cv2.resize(pg, (640, 360))
    for path in files[1:]:
        g = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), (640, 360))
        p0 = cv2.goodFeaturesToTrack(pg, maxCorners=500, qualityLevel=0.01, minDistance=6, blockSize=7)
        v = 0.0
        if p0 is not None and len(p0) > 20:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, p0, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            if p1 is not None:
                s = st.ravel() == 1
                a, b = p0[s, 0], p1[s, 0]
                if len(a) > 20:
                    M, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if M is not None:
                        v = float(np.percentile(np.linalg.norm(b - ((a @ M[:, :2].T) + M[:, 2]), axis=1), 80))
        mags.append(v)
        pg = g
    cum = np.concatenate([[0], np.cumsum(np.convolve(mags, np.ones(5) / 5, mode="same"))])
    return cum / cum[-1] * D


def main():
    from rfdetr import RFDETRBase
    model = RFDETRBase()
    files = sorted(glob.glob("reports/wynf/f_*.jpg"))
    print(f"{len(files)} frames", flush=True)
    cum = motion_from_frames(files)

    left, right = [], []
    for i, path in enumerate(files):
        s_cam = float(cum[min(i, len(cum) - 1)])
        arr = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        res = model.predict(arr, threshold=0.45)
        cls = np.array(res.class_id)
        box = np.array(res.xyxy)
        for (x1, y1, x2, y2), cid in zip(box[np.isin(cls, list(COCO_VEHICLES))],
                                          cls[np.isin(cls, list(COCO_VEHICLES))], strict=False):
            if (y2 - y1) < MIN_BOX_H:
                continue
            cx = (x1 + x2) / 2
            bearing = math.degrees(math.atan((cx - W / 2) / F_PX))
            if abs(bearing) < MIN_BEARING:
                continue
            crop = arr[int(max(y1, 0)):int(y2), int(max(x1, 0)):int(x2)]
            d = {"frame": i, "s_cam": s_cam, "bearing": -abs(bearing),
                 "cx": float(cx), "cy": float((y1 + y2) / 2), "bh": float(y2 - y1),
                 "bw": float(x2 - x1), "color": classify_colour(dominant_rgb(crop)),
                 "cls": COCO_VEHICLES.get(int(cid), "car")}
            (left if bearing < 0 else right).append(d)
        if i % 150 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    def place(dets, sign):
        tracks = [t for t in associate(dets, W=W) if len(t.dets) >= 2]
        recs = []
        for t in tracks:
            S, sig, k = triangulate(t, LATERAL_M)
            if not math.isfinite(S):
                continue
            recs.append({"S": S, "sigma_S": max(sig, 1.0), "color": t.color, "cls": t.cls,
                         "ndet": k, "t_first": t.dets[0]["frame"], "t_last": t.dets[-1]["frame"]})
        cars = slot(recs, min_sep_m=4.5)
        out = []
        for c in cars:
            frac = max(0.0, min(1.0, c["S"] / D))
            out.append({"lat": LAT0 + frac * (LAT1 - LAT0), "lon": LON_AVE + sign * LATERAL_M / MX,
                        "color": c["color"], "cls": c["cls"], "ndet": c["ndet"],
                        "n_tracklets": c["n_tracklets"], "side": "west" if sign < 0 else "east"})
        return out

    cars = place(left, -1) + place(right, +1)
    nl = sum(1 for c in cars if c["side"] == "west")
    print(f"\nwest kerb: {nl} · east kerb: {len(cars)-nl} · total {len(cars)}")
    print("colours:", dict(Counter(c["color"] for c in cars)))
    with open("reports/wynwood_cars.json", "w") as fh:
        json.dump(cars, fh, indent=1)
    print("wrote reports/wynwood_cars.json")


if __name__ == "__main__":
    main()
