"""Track -> triangulate -> stitch the SE 6th detections into an atomic car count."""
import json
import math
from collections import Counter

import numpy as np

from carloc.tracking import associate, slot, triangulate

with open("reports/se6_dets.json") as fh:
    D = json.load(fh)
dets, W, L = D["dets"], D["W"], D["lateral_m"]

tracks = associate(dets, W=W)
print(f"{len(dets)} detections -> {len(tracks)} raw tracklets")
tracks = [t for t in tracks if len(t.dets) >= 2]        # drop single-frame flickers
print(f"{len(tracks)} tracklets with >=2 detections")

recs = []
for t in tracks:
    S, sig, n = triangulate(t, L)
    if not math.isfinite(S):
        continue
    recs.append({"S": S, "sigma_S": max(sig, 1.0), "color": t.color, "cls": t.cls,
                 "ndet": n, "t_first": t.dets[0]["t"], "t_last": t.dets[-1]["t"]})
print(f"{len(recs)} triangulated tracklets")

cars = slot(recs, min_sep_m=4.5)
print(f"{len(cars)} distinct parked cars after slotting (min 4.5 m apart)\n")

# place each car on the map: point on the camera path at arc-length S, offset L left
with open("reports/se6_track.json") as fh:
    track = json.load(fh)
mx = 111_320*math.cos(math.radians(25.768))
my = 110_540
xs = np.array([r["lon"]*mx for r in track])
ys = np.array([r["lat"]*my for r in track])
s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
hd = np.array([r["heading_deg"] for r in track])
sig_cam = np.array([r["sigma_along_m"] for r in track])

def place(S):
    k = int(np.clip(np.searchsorted(s, S) - 1, 0, len(track) - 2))
    f = (S - s[k]) / max(s[k + 1] - s[k], 1e-9)
    ex = xs[k] + f * (xs[k + 1] - xs[k])
    ny = ys[k] + f * (ys[k + 1] - ys[k])
    th = math.radians(hd[k])
    fe, fn = math.sin(th), math.cos(th)      # forward unit (E, N)
    le, ln = -fn, fe                          # 90 deg left of forward
    return (ex + L * le) / mx, (ny + L * ln) / my, sig_cam[k]

out = []
for i, c in enumerate(cars):
    lon, lat, scam = place(c["S"])
    out.append({"id": f"SE6-{i:03d}", "lat": lat, "lon": lon,
                "sigma_along_m": round(math.hypot(c["sigma_S"], scam), 1),
                "sigma_cross_m": 1.8, "color": c["color"], "vehicle_class": c["cls"],
                "ndet": c["ndet"], "n_tracklets": c["n_tracklets"],
                "video_t": round((c["t_first"]+c["t_last"])/2, 2),
                "heading_deg": round(float(hd[int(np.clip(
                    np.searchsorted(s, c["S"]) - 1, 0, len(track) - 2))]), 1)})
with open("reports/se6_cars_tracked.json", "w") as fh:
    json.dump(out, fh, indent=1)

print("colours:", dict(Counter(c["color"] for c in out)))
print("classes:", dict(Counter(c["vehicle_class"] for c in out)))
nd = np.array([c["ndet"] for c in out])
print(f"detections per car: median {int(np.median(nd))}, range {nd.min()}-{nd.max()}")
stit = sum(1 for c in cars if c["n_tracklets"] > 1)
print(f"cars rebuilt from >1 tracklet (occlusion-split): {stit}")
sa = np.array([c["sigma_along_m"] for c in out])
print(f"along-track sigma: median {np.median(sa):.1f} m (was ~11 m single-frame)")
print(f"\nOLD count (cluster-merge): 40   NEW count (track): {len(out)}")
