"""SE 6th Street reference sheet: each atomically-counted parked car, boxed at its
most-abeam frame, tracker-confirmed stationary. The clean-block analog of the
Biscayne detection sheets -- pure parallel side-of-street parking, the case the
method is built for."""
import glob
import json

import numpy as np
from PIL import Image, ImageDraw

from carloc.tracking import associate, slot, triangulate

W = 1920
LATERAL_M = 7.0

with open("reports/se6_dets.json") as fh:
    D = json.load(fh)
dets = D["dets"]

tracks = [t for t in associate(dets, W=W) if len(t.dets) >= 2]
recs = []
for t in tracks:
    S, sig, k = triangulate(t, LATERAL_M)
    if not np.isfinite(S):
        continue
    shot = max(t.dets, key=lambda d: abs(d["bearing"]))
    recs.append({"S": S, "sigma_S": max(sig, 1.0), "color": t.color, "cls": t.cls,
                 "ndet": k, "t_first": t.dets[0]["frame"], "t_last": t.dets[-1]["frame"],
                 "shot_frame": shot["frame"], "shot_bbox": shot["bbox"]})
cars = slot(recs, min_sep_m=4.5)
for c in cars:
    near = min((r for r in recs if r["color"] == c["color"]),
              key=lambda r: abs(r["S"] - c["S"]), default=None)
    c["shot_frame"] = near["shot_frame"] if near else None
    c["shot_bbox"] = near["shot_bbox"] if near else None
cars = [c for c in cars if c["shot_frame"] is not None]
cars.sort(key=lambda c: c["S"])
print(f"{len(cars)} parked cars")

files = sorted(glob.glob("reports/se6frames/f_*.jpg"))
cols, cw, ch = 4, 480, 270
rows = (len(cars) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cw + (cols + 1) * 6, rows * (ch + 26) + 46), (13, 17, 23))
dr = ImageDraw.Draw(sheet)
dr.text((10, 12), f"SE 6TH STREET, BRICKELL - pure side-of-street parking - {len(cars)} atomically-counted cars",
        fill=(230, 237, 243))
dr.text((10, 30), "each box = one physical car, tracked across frames, at its most-abeam moment - "
                  "green outline = occlusion-rebuilt from >=2 tracklets", fill=(140, 150, 160))
for i, c in enumerate(cars):
    im = Image.open(files[c["shot_frame"]]).convert("RGB")
    d = ImageDraw.Draw(im)
    x1, y1, x2, y2 = c["shot_bbox"]
    over = c["n_tracklets"] > 1
    edge = (99, 230, 196) if over else (255, 209, 102)
    d.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=(10, 14, 20), width=8)
    d.rectangle([x1, y1, x2, y2], outline=edge, width=6)
    cell = im.resize((cw, ch))
    cx = 6 + (i % cols) * (cw + 6)
    cy = 46 + (i // cols) * (ch + 26)
    sheet.paste(cell, (cx, cy))
    dr.text((cx + 4, cy + ch + 4),
            f"{c['color']} {c['cls']} - {c['ndet']} frames" + ("  [rebuilt]" if over else ""),
            fill=edge if over else (205, 214, 224))
sheet.save("reports/se6_reference.png")
print("wrote reports/se6_reference.png")
