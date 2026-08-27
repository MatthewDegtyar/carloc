"""Snap ParkMobile zone anchors onto block faces, and map the result."""
import json, math
import numpy as np
from pathlib import Path
from carloc.blockface import split_into_faces
from carloc.parkmobile import fetch, to_geojson

CODES = [40701, 40703, 40711, 40712, 40713]
SNAP_M = 45.0
PALETTE = ["#4ec9b0", "#e0af68", "#7aa2f7", "#e05561", "#c678dd", "#56b6c2"]

zones = [z for z in (fetch(c) for c in CODES) if z and z.points]
COL = {z.signage_code: PALETTE[i % len(PALETTE)] for i, z in enumerate(zones)}
print(f"zones: {len(zones)}  ({', '.join(z.signage_code for z in zones)})")

ways = json.load(open("research/miami_streets_wide.json"))["elements"]
faces = split_into_faces(ways)
print(f"block faces: {len(faces)}")

centres = np.array([[f.centre[0], f.centre[1]] for f in faces])
lat0 = float(centres[:, 1].mean())
mx = 111_320 * math.cos(math.radians(lat0)); my = 110_540
C = np.column_stack([centres[:, 0] * mx, centres[:, 1] * my])

hits = {}
for z in zones:
    for lon, lat in z.points:
        d = np.linalg.norm(C - np.array([lon * mx, lat * my]), axis=1)
        i = int(np.argmin(d))
        if d[i] <= SNAP_M:
            hits.setdefault(i, set()).add(z.signage_code)

print(f"\nblock faces matched: {len(hits)}")
for z in zones:
    n = sum(1 for v in hits.values() if z.signage_code in v)
    print(f"  zone {z.signage_code}: {len(z.points):2d} anchors -> {n:2d} faces")
print(f"\nestimated spaces on matched faces: {sum(faces[i].spaces for i in hits)}")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(14, 11), facecolor="#0d1117"); ax.set_facecolor("#10141c")
for f in faces:
    ax.add_patch(Polygon(np.array(f.polygon), closed=True, facecolor="#232a34",
                         edgecolor="#39434f", lw=0.25, zorder=2))
for i, codes in hits.items():
    c = COL[sorted(codes)[0]]
    ax.add_patch(Polygon(np.array(faces[i].polygon), closed=True, facecolor=c,
                         edgecolor=c, lw=1.4, alpha=0.95, zorder=4))
for z in zones:
    P = np.array(z.points)
    ax.plot(P[:, 0], P[:, 1], "o", color=COL[z.signage_code], ms=5,
            mec="#0d1117", mew=0.6, zorder=6)

allp = np.vstack([np.array(z.points) for z in zones])
padx = np.ptp(allp[:, 0]) * 0.08; pady = np.ptp(allp[:, 1]) * 0.12
ax.set_xlim(allp[:, 0].min() - padx, allp[:, 0].max() + padx)
ax.set_ylim(allp[:, 1].min() - pady, allp[:, 1].max() + pady)
ax.set_aspect(1 / math.cos(math.radians(lat0)))
ax.tick_params(colors="#8b98a8", labelsize=8)
for s in ax.spines.values(): s.set_color("#39434f")
ax.set_xlabel("longitude", color="#dfe6ef"); ax.set_ylabel("latitude", color="#dfe6ef")
ax.set_title(f"ParkMobile on-street zones, Miami — {len(zones)} zones, "
             f"{len(hits)} block faces highlighted\n"
             f"dots = zone anchors from ParkMobile /api/locations  ·  "
             f"lit boxes = block faces within {SNAP_M:.0f} m  ·  grey = all other faces",
             color="#dfe6ef", fontsize=11, family="monospace")
ax.legend(handles=[Line2D([], [], color=COL[z.signage_code], lw=6,
                          label=f"zone {z.signage_code}  ({len(z.points)} anchors)")
                   for z in zones],
          facecolor="#161b22", edgecolor="#39434f", labelcolor="#dfe6ef",
          fontsize=9, loc="upper right")
fig.savefig("reports/miami_zones.png", dpi=130, facecolor="#0d1117", bbox_inches="tight")
Path("reports/miami_zones.geojson").write_text(json.dumps(to_geojson(zones)))
print("\nwrote reports/miami_zones.png and .geojson")
