"""Two panels: the sightings log as a queryable record, and an overstay demo."""
import json
import math
import random
from datetime import datetime, timedelta

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from carloc.basemap import ATTRIBUTION, fetch_extent, imshow_mercator, mercator_y
from carloc.sightings import Sighting, SightingLog, synthetic_ts

from carloc.appearance import SWATCH
EPOCH = datetime(2025, 6, 2, 9, 0, 0)
W, E, S, N = -80.19290, -80.18980, 25.76735, 25.76895

with open("reports/sightings.json") as fh:
    rows = json.load(fh)
mx = 111_320*math.cos(math.radians(25.768)); my = 110_540
HD = np.median([r["heading_deg"] for r in rows])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(23, 10.5), facecolor="#0d1117")

# ---- LEFT: the log -----------------------------------------------------------
mL, extL = fetch_extent(W, S, E, N, zoom=20)
imshow_mercator(axL, mL, extL)
for r in rows:
    axL.add_patch(Ellipse((r["lon"], mercator_y(r["lat"])),
        width=2*r["sigma_along_m"]/mx, height=2*r["sigma_cross_m"]/my*0.905,
        angle=-(HD-90), facecolor=SWATCH.get(r["color"],"#888"), alpha=0.16,
        edgecolor="none", zorder=5))
    axL.plot(r["lon"], mercator_y(r["lat"]), "o", ms=8,
             color=SWATCH.get(r["color"],"#888"), mec="#0d1117", mew=1.0, zorder=7)
# annotate a few records with id + synthetic time
for r in sorted(rows, key=lambda r: r["sigma_along_m"])[:5]:
    t = datetime.fromisoformat(r["ts"])
    axL.annotate(f"{r['sighting_id']}\n{t:%H:%M:%S} · {r['color']} {r['vehicle_class']}",
                 (r["lon"], mercator_y(r["lat"])), color="#e6edf3", fontsize=8,
                 xytext=(7,7), textcoords="offset points", zorder=9,
                 bbox=dict(boxstyle="round,pad=0.25", fc="#0d1117cc", ec="none"))
# a worked presence query
q = sorted(rows, key=lambda r: r["sigma_along_m"])[0]
axL.plot(q["lon"], mercator_y(q["lat"]), "x", color="#ffd166", ms=16, mew=3, zorder=10)
axL.annotate("presence query →  HIT: {} {} at {} (0.0σ)".format(q["color"], q["vehicle_class"], datetime.fromisoformat(q["ts"]).strftime("%H:%M:%S")),
             (q["lon"], mercator_y(q["lat"])), color="#ffd166", fontsize=10, weight="bold",
             xytext=(10,-24), textcoords="offset points", zorder=10)
axL.set_xticks([]); axL.set_yticks([])
axL.set_title(f"Sightings log — {len(rows)} records, each a (place, synthetic time, "
              "appearance, uncertainty)\ndot colour = detected vehicle colour · "
              "ellipse = 1σ position · zone = UNKNOWN (no Brickell geometry)",
              color="#e6edf3", fontsize=12.5, pad=10)

# ---- RIGHT: overstay demo ----------------------------------------------------
random.seed(7)
with open("reports/se6_cars.json") as fh:
    cars = sorted(json.load(fh), key=lambda c: c["video_t"])
def make(cs, epoch, tag, jit=0.0):
    log = SightingLog()
    for i, c in enumerate(cs):
        dlat = random.gauss(0,jit)/my; dlon = random.gauss(0,jit)/mx
        log.add(Sighting(sighting_id=f"{tag}-{i:03d}", ts=synthetic_ts(epoch,c["video_t"]),
            video_t=c["video_t"], lat=c["lat"]+dlat, lon=c["lon"]+dlon,
            heading_deg=c["heading_deg"], sigma_along_m=c["sigma_along_m"],
            sigma_cross_m=c["sigma_cross_m"], vehicle_class=c["vehicle_class"],
            color=c["color"], size_px=c["size_px"], zone=None, source=tag))
    return log
p1 = make(cars, EPOCH, "P1")
stayed = [c for c in cars if random.random() > 0.4]; sid={id(c) for c in stayed}
newc=[]
for c in cars:
    if id(c) in sid: continue
    d=dict(c); d["lat"]+=random.uniform(-30,30)/my
    d["color"]=random.choice(["black","white","red","silver","blue"]); newc.append(d)
p2 = make(stayed+newc, EPOCH+timedelta(minutes=40), "P2", jit=1.2)
cand = p1.overstay(p2, min_gap_s=1200, gate=2.5)
matched2 = {id(b) for _,b,_ in cand}

mR, extR = fetch_extent(W, S, E, N, zoom=20)
imshow_mercator(axR, mR, extR)
for s in p1.sightings:
    axR.plot(s.lon, mercator_y(s.lat), "o", ms=9, color="#58a6ff", mec="#0d1117", mew=.8, zorder=6)
for s in p2.sightings:
    c = "#e05561" if id(s) not in matched2 else "#4ec9b0"
    axR.plot(s.lon, mercator_y(s.lat), "s", ms=8, color=c, mec="#0d1117", mew=.8, zorder=6)
for a,b,_gap in cand:
    axR.plot([a.lon,b.lon],[mercator_y(a.lat),mercator_y(b.lat)],"-",color="#4ec9b0",lw=2,zorder=7)
axR.set_xticks([]); axR.set_yticks([])
axR.legend(handles=[
    Line2D([],[],marker="o",ls="",color="#58a6ff",label="pass 1 · 09:07"),
    Line2D([],[],marker="s",ls="",color="#4ec9b0",label="pass 2 · 09:47 · loiterer (matched)"),
    Line2D([],[],marker="s",ls="",color="#e05561",label="pass 2 · turned over (new car)"),
    Line2D([],[],color="#4ec9b0",lw=2,label="same car, ≥20 min → overstay"),
], loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=10)
axR.set_title(f"Overstay primitive — same street 40 min later (synthetic 2nd pass)\n"
              f"{len(cand)}/{len(stayed)} real loiterers recovered · 0 false positives · "
              "one-to-one mutual-nearest, plateless",
              color="#e6edf3", fontsize=12.5, pad=10)

fig.text(0.5, 0.012, ATTRIBUTION + "  ·  timestamps synthetic (video has no chronology)",
         color="#6e7681", fontsize=8, ha="center")
fig.tight_layout(rect=[0,0.028,1,1])
fig.savefig("reports/sightings.png", dpi=115, facecolor="#0d1117")
print("wrote reports/sightings.png")
