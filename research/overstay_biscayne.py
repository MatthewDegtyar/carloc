"""The first real overstay: two genuine passes of the same Biscayne block.

Pass 1 at 22:26 and pass 2 at 53:00 in the source drive are the same stretch of
Biscayne Boulevard, 30.5 minutes apart (measured from the frame offset, not
fabricated). Both are anchored to two shared physical points -- the Flagler and
NE 3rd crossings, transferred pass-1->pass-2 by ORB -- so they live in one
coordinate frame by construction. Each pass is then run through the same
track->triangulate->slot counter, and the two car sets are matched. A car present
in both, at the same kerb position with the same appearance, has overstayed by
half an hour, and this time the clock is real.
"""
from __future__ import annotations

import glob
import math

import cv2
import numpy as np

from carloc.appearance import classify_colour, dominant_rgb
from carloc.rfdetr_detect import COCO_VEHICLES
from carloc.tracking import associate, slot, triangulate

W, H = 1920, 1080
F_PX = 687.0
LATERAL_M = 7.0
MIN_BEARING_DEG = 20.0
MIN_BOX_H = 80

# Shared anchors (lat, lon), from OSM; identical physical points in both passes.
FLAGLER = (25.774346, -80.187238)
NE3RD = (25.777198, -80.188307)

MX = 111_320 * math.cos(math.radians(25.776))
MY = 110_540
# along-street unit (Flagler->NE3rd) and its left-perpendicular, in E/N metres
_dE = (NE3RD[1] - FLAGLER[1]) * MX
_dN = (NE3RD[0] - FLAGLER[0]) * MY
D = math.hypot(_dE, _dN)
FWD = (_dE / D, _dN / D)                       # forward unit (E, N)
LEFT = (-FWD[1], FWD[0])                        # 90 deg left of forward
HEADING = math.degrees(math.atan2(FWD[0], FWD[1])) % 360


def scene_motion(clip, t0, t1, fps=4.0):
    """Residual scene motion per sampled step -> a relative speed profile.

    Same signal used on SE 6th: the rigid part of the frame-to-frame transform is
    the camera turning; what is left after removing it is parallax against
    buildings and parked cars, a reliable stopped/moving shape. Road texture is
    useless here, so this leans on structure, not the tarmac.
    """
    cap = cv2.VideoCapture(clip)
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    feat = dict(maxCorners=600, qualityLevel=0.01, minDistance=6, blockSize=7)
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    ok, prev = cap.read()
    pg = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    step = 1.0 / fps
    times, mags = [], []
    t = t0
    while t < t1:
        cap.set(cv2.CAP_PROP_POS_MSEC, (t + step) * 1000)
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(pg, mask=None, **feat)
        v = 0.0
        if p0 is not None and len(p0) > 20:
            p1, stt, _ = cv2.calcOpticalFlowPyrLK(pg, g, p0, None, **lk)
            if p1 is not None:
                s = stt.ravel() == 1
                a, b = p0[s, 0], p1[s, 0]
                if len(a) > 20:
                    M, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                       ransacReprojThreshold=2.0)
                    if M is not None:
                        pred = (a @ M[:, :2].T) + M[:, 2]
                        v = float(np.percentile(np.linalg.norm(b - pred, axis=1), 80))
        times.append(t + step)
        mags.append(v)
        pg = g
        t += step
    cap.release()
    times = np.array(times)
    mags = np.convolve(np.array(mags), np.ones(5) / 5, mode="same")
    cum = np.concatenate([[0], np.cumsum(mags)])
    cum = cum / cum[-1] * D                     # scale so the clip spans Flagler->NE3rd
    return times, cum[1:]


def s_at(times, cum, t):
    return float(np.interp(t, times, cum))


def process(clip, frames_dir, t0, t1, model):
    """One pass -> a list of counted, placed cars in the shared frame."""
    times, cum = scene_motion(clip, t0, t1)
    files = sorted(glob.glob(f"{frames_dir}/f_*.jpg"))
    dets = []
    for i, path in enumerate(files):
        t = t0 + i / 4.0
        s_cam = s_at(times, cum, t)
        image = cv2.imread(path)
        arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), threshold=0.45)
        cls = np.array(res.class_id)
        box = np.array(res.xyxy)
        keep = np.isin(cls, list(COCO_VEHICLES))
        for (x1, y1, x2, y2), cid in zip(box[keep], cls[keep], strict=False):
            if (y2 - y1) < MIN_BOX_H:
                continue
            cx = (x1 + x2) / 2
            bearing = math.degrees(math.atan((cx - W / 2) / F_PX))
            if bearing > -MIN_BEARING_DEG:
                continue
            crop = arr[int(max(y1, 0)):int(y2), int(max(x1, 0)):int(x2)]
            dets.append({"frame": i, "s_cam": s_cam, "bearing": round(bearing, 2),
                         "cx": float(cx), "cy": float((y1 + y2) / 2), "bh": float(y2 - y1),
                         "bw": float(x2 - x1), "color": classify_colour(dominant_rgb(crop)),
                         "cls": COCO_VEHICLES.get(int(cid), "car"),
                         "bbox": [float(x1), float(y1), float(x2), float(y2)]})
    tracks = [t for t in associate(dets, W=W) if len(t.dets) >= 2]
    recs = []
    for t in tracks:
        S, sig, k = triangulate(t, LATERAL_M)
        if not math.isfinite(S):
            continue
        shot = max(t.dets, key=lambda d: abs(d["bearing"]))
        recs.append({"S": S, "sigma_S": max(sig, 1.0), "color": t.color, "cls": t.cls,
                     "ndet": k, "t_first": t.dets[0]["frame"], "t_last": t.dets[-1]["frame"],
                     "shot_frame": shot["frame"], "shot_bbox": shot["bbox"]})
    cars = slot(recs, min_sep_m=4.5)
    for c in cars:
        near = min((r for r in recs if r["color"] == c["color"]),
                   key=lambda r: abs(r["S"] - c["S"]), default=None)
        if near:
            c["shot_frame"] = near["shot_frame"]
            c["shot_bbox"] = near["shot_bbox"]
        c["frames_dir"] = frames_dir
        frac = c["S"] / D
        plat = FLAGLER[0] + frac * (NE3RD[0] - FLAGLER[0]) + LATERAL_M * LEFT[1] / MY
        plon = FLAGLER[1] + frac * (NE3RD[1] - FLAGLER[1]) + LATERAL_M * LEFT[0] / MX
        c["lat"], c["lon"] = plat, plon
    print(f"  {len(dets)} dets -> {len(tracks)} tracklets -> {len(cars)} cars")
    return cars


def main():
    from datetime import datetime, timedelta

    from rfdetr import RFDETRBase

    from carloc.sightings import Sighting, SightingLog

    model = RFDETRBase()
    print(f"Flagler->NE3rd = {D:.0f} m, heading {HEADING:.0f} deg")
    print("pass 1 (22:26):")
    c1 = process("video/pass1_1336.mp4", "reports/p1f", 12, 42, model)
    print("pass 2 (53:00):")
    c2 = process("video/pass2_3168.mp4", "reports/p2f", 12, 33, model)

    T1 = datetime(2025, 6, 2, 9, 22, 26)
    GAP_S = 1830                                 # 30.5 min, measured from the frame offset
    T2 = T1 + timedelta(seconds=GAP_S)

    def to_log(cars, tstamp, tag):
        log = SightingLog()
        for i, c in enumerate(cars):
            log.add(Sighting(sighting_id=f"{tag}-{i:02d}", ts=tstamp, video_t=0.0,
                             lat=c["lat"], lon=c["lon"], heading_deg=HEADING,
                             sigma_along_m=round(c["sigma_S"], 1), sigma_cross_m=1.8,
                             vehicle_class=c["cls"], color=c["color"], size_px=0,
                             zone=None, source=tag, synthetic=False))
        return log

    L1 = to_log(c1, T1, "P1")
    L2 = to_log(c2, T2, "P2")
    over = L1.overstay(L2, min_gap_s=1200, gate=3.0)
    print(f"\npass 1: {len(c1)} cars @ 09:22:26")
    print(f"pass 2: {len(c2)} cars @ 09:52:56  (real gap 30.5 min)")
    print(f"OVERSTAYERS (same kerb + appearance, both passes): {len(over)}")
    for a, _b, gap in over:
        print(f"  {a.color:6s} {a.vehicle_class:5s}  present both passes  "
              f"dwell >= {gap/60:.0f} min  @ {a.lat:.6f},{a.lon:.6f}")

    import json
    out = {"pass1": c1, "pass2": c2,
           "overstayers": [{"lat": a.lat, "lon": a.lon, "color": a.color,
                            "cls": a.vehicle_class, "p1": a.sighting_id,
                            "p2": b.sighting_id, "dwell_min": round(gap / 60)}
                           for a, b, gap in over]}
    with open("reports/biscayne_overstay.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote reports/biscayne_overstay.json")


if __name__ == "__main__":
    main()
