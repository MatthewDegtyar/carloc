"""Detect the parked cars on the SE 6th Street pass and capture appearance.

Same placement as before -- lateral offset is the MEASURED 4.73 m from the
satellite survey, only bearing comes from the video -- but each detection now
also records what the car *looks like*: class, dominant colour, apparent size.
That is the re-identification key. The whole premise of this system is that it
never reads a plate, so a car's identity across two passes has to be carried by
where it is plus what it looks like, and this is where "what it looks like" is
measured.
"""
from __future__ import annotations

import glob
import json
import math

import numpy as np

W, H = 1920, 1080
F_PX = 687.0
LATERAL_M = 7.0
MIN_BEARING_DEG = 22.0
MIN_BOX_H = 90
CLUSTER_M = 3.5

# Coarse named colours, in RGB. Deliberately few: a re-id key wants "silver vs
# black vs red", not a paint-chip match no two frames would agree on.
PALETTE = {
    "black": (25, 25, 28), "white": (232, 232, 232), "silver": (170, 172, 175),
    "grey": (110, 112, 115), "red": (150, 40, 40), "blue": (45, 65, 130),
    "green": (45, 95, 60), "tan": (170, 150, 115),
}


def name_colour(rgb):
    r, g, b = rgb
    best, bd = "grey", 1e9
    for name, (pr, pg, pb) in PALETTE.items():
        d = (r-pr)**2 + (g-pg)**2 + (b-pb)**2
        if d < bd:
            bd, best = d, name
    return best


def dominant_rgb(crop):
    """Median colour of the central body of the box, edges trimmed.

    The median rejects windscreen glare and the road showing under the car; the
    trim keeps the background at the box edges out of it.
    """
    h, w = crop.shape[:2]
    inner = crop[int(h*0.25):int(h*0.75), int(w*0.2):int(w*0.8)]
    if inner.size == 0:
        inner = crop
    flat = inner.reshape(-1, 3)
    return tuple(int(v) for v in np.median(flat, axis=0))


def main():
    from PIL import Image
    from rfdetr import RFDETRBase

    from carloc.rfdetr_detect import COCO_VEHICLES

    with open("reports/se6_track.json") as fh:
        track = json.load(fh)
    ts = np.array([r["t"] for r in track])
    model = RFDETRBase()
    files = sorted(glob.glob("reports/se6frames/f_*.jpg"))
    mx = 111_320*math.cos(math.radians(25.768))
    my = 110_540

    raw = []
    for i, path in enumerate(files):
        t = 420.0 + i/4.0
        k = int(np.clip(np.searchsorted(ts, t), 0, len(track)-1))
        fix = track[k]
        image = Image.open(path).convert("RGB")
        arr = np.asarray(image)
        det = model.predict(image, threshold=0.5)
        cls = np.array(det.class_id)
        box = np.array(det.xyxy)
        keep = np.isin(cls, list(COCO_VEHICLES))
        for (x1, y1, x2, y2), cid in zip(box[keep], cls[keep], strict=False):
            if (y2-y1) < MIN_BOX_H:
                continue
            cx = (x1+x2)/2
            bearing = math.degrees(math.atan((cx - W/2)/F_PX))
            if bearing > -MIN_BEARING_DEG:
                continue
            along = LATERAL_M/math.tan(math.radians(-bearing))
            if along > 45:
                continue
            crop = arr[int(max(y1, 0)):int(y2), int(max(x1, 0)):int(x2)]
            rgb = dominant_rgb(crop) if crop.size else (110, 112, 115)
            hd = math.radians(fix["heading_deg"])
            dn = along*math.cos(hd) + LATERAL_M*math.cos(hd - math.pi/2)
            de = along*math.sin(hd) + LATERAL_M*math.sin(hd - math.pi/2)
            raw.append({
                "video_t": round(t, 2),
                "lat": fix["lat"] + dn/my, "lon": fix["lon"] + de/mx,
                "sigma_along_m": fix["sigma_along_m"], "sigma_cross_m": fix["sigma_cross_m"],
                "heading_deg": fix["heading_deg"],
                "vehicle_class": COCO_VEHICLES.get(int(cid), "car"),
                "color_rgb": list(rgb), "color": name_colour(rgb),
                "size_px": int(y2-y1), "range_m": round(along, 1),
            })
        if i % 80 == 0:
            print(f"  frame {i}/{len(files)}  raw={len(raw)}", flush=True)

    raw.sort(key=lambda r: r["sigma_along_m"])       # keep each car's best fix
    cars = []
    for r in raw:
        q = np.array([r["lon"]*mx, r["lat"]*my])
        if any(np.hypot(c["lon"]*mx-q[0], c["lat"]*my-q[1]) < CLUSTER_M for c in cars):
            continue
        cars.append(r)
    print(f"\n{len(raw)} detections -> {len(cars)} distinct parked cars")
    from collections import Counter
    print("classes:", dict(Counter(c["vehicle_class"] for c in cars)))
    print("colours:", dict(Counter(c["color"] for c in cars)))
    with open("reports/se6_cars.json", "w") as fh:
        json.dump(cars, fh, indent=1)
    print("wrote reports/se6_cars.json")


if __name__ == "__main__":
    main()
