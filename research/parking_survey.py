"""Score every 6-second sample of the whole drive for on-street parking density.

Not localisation -- a coarse map of *when* the drive is beside parkable kerb, so
the pure side-street-parking segments can be found and localised. A vehicle counts
as a parked candidate when it sits well off-centre (either kerb) and low in the
frame (close), which is where a parallel-parked car projects; traffic ahead sits
central and higher and does not count.
"""
import glob
import json
import math

import numpy as np
from PIL import Image
from rfdetr import RFDETRBase

from carloc.rfdetr_detect import COCO_VEHICLES

W, H = 1280, 720
F = 458.0                       # 687 * 1280/1920
BEARING = 18.0                  # off-centre threshold (deg) for kerb-side
MIN_H = 55                      # box height: close enough to be kerbside


def main():
    model = RFDETRBase()
    files = sorted(glob.glob("reports/survey/s_*.jpg"))
    rows = []
    for i, path in enumerate(files):
        t = i * 6
        arr = np.asarray(Image.open(path).convert("RGB"))
        res = model.predict(arr, threshold=0.45)
        cls = np.array(res.class_id)
        box = np.array(res.xyxy)
        keep = np.isin(cls, list(COCO_VEHICLES))
        left = right = 0
        for (x1, y1, x2, y2) in box[keep]:
            if (y2 - y1) < MIN_H:
                continue
            if (y1 + y2) / 2 < H * 0.42:          # high in frame -> far traffic
                continue
            cx = (x1 + x2) / 2
            bearing = math.degrees(math.atan((cx - W / 2) / F))
            if bearing < -BEARING:
                left += 1
            elif bearing > BEARING:
                right += 1
        rows.append({"t": t, "left": left, "right": right, "kerb": left + right})
        if i % 100 == 0:
            print(f"  {i}/{len(files)}  t={t//60}:{t%60:02d}  kerb={left+right}", flush=True)
    json.dump(rows, open("reports/parking_survey.json", "w"))

    k = np.array([r["kerb"] for r in rows])
    sm = np.convolve(k, np.ones(3) / 3, mode="same")
    print(f"\n{len(rows)} samples. kerb-vehicle count: median {np.median(k):.0f}, max {k.max()}")
    # contiguous parking-rich segments (smoothed kerb >= 3 for >= 18 s)
    hot = sm >= 3.0
    segs = []
    i = 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j < len(hot) and hot[j]:
                j += 1
            if (j - i) * 6 >= 18:
                segs.append((rows[i]["t"], rows[j - 1]["t"], float(sm[i:j].mean())))
            i = j
        else:
            i += 1
    print(f"\n{len(segs)} parking-rich segments (kerb>=3 sustained >=18 s):")
    for a, b, m in sorted(segs, key=lambda s: -s[2]):
        print(f"  {a//60:02d}:{a%60:02d} - {b//60:02d}:{b%60:02d}  ({b-a+6:3d}s)  avg kerb {m:.1f}")
    json.dump(segs, open("reports/parking_segments.json", "w"))


if __name__ == "__main__":
    main()
