"""Place the parked cars this pass drives past, in lat/lon with an uncertainty.

Monocular range is not used and is not needed. The satellite survey already
MEASURED where a kerbside car sits -- 4.73 m from the street centreline, from 185
observations -- so a detection's lateral offset is known a priori. The only
unknown is how far ahead it is, and that follows from its bearing:

    along = lateral / tan(|bearing|),   bearing = atan((cx - W/2) / f)

with `f` self-calibrated from a turn the street grid fixes at 90 degrees. A car
at the left image edge is therefore about 5 m ahead, not at some depth the
network had to guess. This is the whole reason the demo survives a 640x360-class
camera: the hard quantity was measured from orbit, and the video only has to
supply a bearing.
"""
from __future__ import annotations

import glob
import json
import math

import numpy as np

W, H = 1920, 1080
F_PX = 687.0          # 229 px/rad at 640 wide, scaled to 1920
LATERAL_M = 7.0       # camera lane centre to parked-car centre, north kerb
MIN_BEARING_DEG = 22.0
MIN_BOX_H = 90
CLUSTER_M = 3.5

def main():
    from PIL import Image
    from rfdetr import RFDETRBase

    from carloc.rfdetr_detect import COCO_VEHICLES

    with open("reports/se6_track.json") as fh:
        track = json.load(fh)
    ts = np.array([r["t"] for r in track])
    model = RFDETRBase()
    files = sorted(glob.glob("reports/se6frames/f_*.jpg"))
    print(f"{len(files)} frames, {len(track)} fixes")

    mx = 111_320*math.cos(math.radians(25.768))
    my = 110_540
    raw = []
    for i, path in enumerate(files):
        t = 420.0 + i/4.0
        k = int(np.clip(np.searchsorted(ts, t), 0, len(track)-1))
        fix = track[k]
        image = Image.open(path).convert("RGB")
        det = model.predict(image, threshold=0.5)
        cls = np.array(det.class_id)
        box = np.array(det.xyxy)
        keep = np.isin(cls, list(COCO_VEHICLES))
        for (x1, y1, x2, y2) in box[keep]:
            if (y2-y1) < MIN_BOX_H:
                continue
            cx = (x1+x2)/2
            bearing = math.degrees(math.atan((cx - W/2)/F_PX))
            if bearing > -MIN_BEARING_DEG:      # keep the left kerb only
                continue
            along = LATERAL_M/math.tan(math.radians(-bearing))
            if along > 45:
                continue
            hd = math.radians(fix["heading_deg"])
            # forward along heading, then LATERAL_M to the left of it
            dn = along*math.cos(hd) + LATERAL_M*math.cos(hd - math.pi/2)
            de = along*math.sin(hd) + LATERAL_M*math.sin(hd - math.pi/2)
            raw.append({"t": t, "lat": fix["lat"] + dn/my, "lon": fix["lon"] + de/mx,
                        "sig_along": fix["sigma_along_m"], "sig_cross": fix["sigma_cross_m"],
                        "along_m": round(along,1), "bearing_deg": round(bearing,1)})
        if i % 80 == 0:
            print(f"  frame {i}/{len(files)}  raw={len(raw)}", flush=True)

    # one physical car appears in many frames; collapse to its best-known fix
    raw.sort(key=lambda r: r["sig_along"])
    cars = []
    for r in raw:
        q = np.array([r["lon"]*mx, r["lat"]*my])
        if any(np.linalg.norm(np.array([c["lon"]*mx, c["lat"]*my])-q) < CLUSTER_M for c in cars):
            continue
        cars.append(r)
    print(f"\n{len(raw)} kerbside detections -> {len(cars)} distinct parked cars")
    sa = np.array([c["sig_along"] for c in cars])
    print(f"along-track sigma of the placed cars: median {np.median(sa):.1f} m, "
          f"best {sa.min():.1f} m, worst {sa.max():.1f} m")
    with open("reports/se6_cars.csv","w") as fh:
        fh.write("t_s,latitude,longitude,sigma_along_m,sigma_cross_m,range_m,bearing_deg\n")
        for c in sorted(cars, key=lambda c: c["t"]):
            fh.write(f"{c['t']},{c['lat']:.7f},{c['lon']:.7f},{c['sig_along']},"
                     f"{c['sig_cross']},{c['along_m']},{c['bearing_deg']}\n")
    with open("reports/se6_cars.json","w") as fh:
        json.dump(cars, fh, indent=1)
    print("wrote reports/se6_cars.csv / .json")

if __name__ == "__main__":
    main()
