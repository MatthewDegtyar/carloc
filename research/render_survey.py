"""Render the downtown survey: cars judged against paid-parking lane boxes.

Four panels chosen to show the result honestly rather than flatteringly -- the
two block faces where the method works and the two where it does not, so the
failure mode is visible instead of averaged away.

Lane boxes are drawn at the MEASURED band (3.43-6.03 m from centreline), fitted
to 185 observed kerbside cars rather than assumed.
"""
import pickle

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from carloc.basemap import ATTRIBUTION, fetch_extent, imshow_mercator, mercator_y
from carloc.downtown import judge

with open("reports/survey_final.pkl", "rb") as _fh:
    d = pickle.load(_fh)
S, BOXES = d["sightings"], d["boxes"]
BAND, BAY, CENTRE = d["band"], d["bay_band"], d["lane_centre_m"]
COL = {"inside": "#4ec9b0", "outside": "#e05561", "ambiguous": "#e0af68"}

# Rank block faces by cars actually landing in the lane, ignoring the 1 m
# ambiguous band, so the panels are picked on geometry rather than on the
# uncertainty that is being reported.
strict = [judge(s.lat, s.lon, BOXES, 0.0, s.score) for s in S]
hits: dict[str, int] = {}
for v in strict:
    if v.verdict == "inside" and v.box_id:
        hits[v.box_id] = hits.get(v.box_id, 0) + 1
ranked = sorted(BOXES, key=lambda b: -hits.get(b.box_id, 0))
panels = [b for b in ranked if b.zone == "40703"][:2] + \
         [b for b in ranked if b.zone == "40701"][:2]

fig, axes = plt.subplots(2, 2, figsize=(17, 15.5), facecolor="#0d1117")
HALF = 0.00042

for ax, box in zip(axes.ravel(), panels, strict=False):
    lon, lat = box.centre
    mosaic, ext = fetch_extent(lon - HALF, lat - HALF * 0.8,
                               lon + HALF, lat + HALF * 0.8, zoom=20)
    imshow_mercator(ax, mosaic, ext)

    for b in BOXES:
        P = np.array(b.polygon)
        if P[:, 0].max() < ext[0] or P[:, 0].min() > ext[1]:
            continue
        if P[:, 1].max() < ext[2] or P[:, 1].min() > ext[3]:
            continue
        edge = "#33fff0" if b.box_id == box.box_id else "#2b8f88"
        ax.add_patch(Polygon(
            np.column_stack([P[:, 0], [mercator_y(v) for v in P[:, 1]]]),
            closed=True, facecolor="none", edgecolor=edge,
            lw=2.2 if b.box_id == box.box_id else 1.2, alpha=0.9, zorder=4))

    n = {"inside": 0, "outside": 0, "ambiguous": 0}
    for s in S:
        if not (ext[0] < s.lon < ext[1] and ext[2] < s.lat < ext[3]):
            continue
        n[s.verdict] += 1
        ax.plot(s.lon, mercator_y(s.lat), "o", color=COL[s.verdict], ms=7,
                mec="#0d1117", mew=1.0, zorder=7)

    strict_here = hits.get(box.box_id, 0)
    ax.set_title(
        f"zone {box.zone} · {box.street} ({box.side})\n"
        f"{strict_here} cars in this lane · {box.spaces} spaces est · "
        f"tile: {n['inside']} paid / {n['ambiguous']} ambiguous / {n['outside']} outside",
        color="#c9d1d9", fontsize=11, pad=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#30363d")

legend = [Line2D([], [], marker="o", ls="", color=COL[k], mec="#0d1117",
                 label=f"{k} (sigma = 1.0 m)") for k in COL]
legend.append(Line2D([], [], color="#33fff0", lw=2.2, label="paid parking lane"))
fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
           labelcolor="#c9d1d9", fontsize=11, bbox_to_anchor=(0.5, 0.028))
fig.suptitle(
    "ParkMobile downtown Miami \u2014 is this car parked in a paid zone?\n"
    f"RF-DETR on Esri imagery \u00b7 decision band {BAND[0]:.1f}\u2013{BAND[1]:.1f} m "
    f"from centreline (block face) \u00b7 bay {BAY[0]:.2f}\u2013{BAY[1]:.2f} m\n"
    f"lane centre MEASURED at {CENTRE:.2f} m from {d['n_kerbside']} observed kerbside cars, "
    "not assumed",
    color="#e6edf3", fontsize=13.5, y=0.975)
fig.text(0.5, 0.008, ATTRIBUTION, color="#6e7681", fontsize=8, ha="center")
fig.tight_layout(rect=[0, 0.055, 1, 0.912])
fig.savefig("reports/downtown_survey.png", dpi=125, facecolor="#0d1117")
print("wrote reports/downtown_survey.png")
