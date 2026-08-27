"""One dashcam pass down SE 6th Street: place the car, then the cars it passes.

Route established from the 4K source, where street blades are legible: the
vehicle runs **east** on SE 6th Street (the colourful Brickell City Centre garage
is on the left, and BCC is south of SE 6th), crosses SE 1st Avenue, and turns
right onto southbound Brickell Avenue at the First Presbyterian Church.

Position comes from anchors plus a motion profile, not from odometry. Ground
optical flow does not work on this footage at any resolution -- Miami asphalt in
sun has a texture std of 2-6 grey levels and dense flow returns exactly 0.00 px.
What does work is *residual scene motion*: after the rigid (rotation) part of the
frame-to-frame transform is removed, what is left is dominated by parallax
against buildings and parked cars, which have plenty of texture. That gives a
reliable stopped/moving shape, and the anchors supply the scale.

So the along-track estimate is an interpolation between two known crossings,
weighted by observed motion rather than by time. Its uncertainty is honest: it
peaks between anchors and collapses at them.
"""
from __future__ import annotations

import json
import math

import numpy as np

HZ = 4.0
LANE_OFFSET_M = 4.73
"""Where a kerbside car sits from the centreline. MEASURED, not assumed -- the
median of 185 cars in the satellite survey (see FINDINGS section 9)."""

ANCHORS = [
    # (t seconds into the cut, lon, lat, sigma_s, label)
    (420.0, -80.192180, 25.768365, 4.0, "SE 6th St x SE 1st Ave"),
    (505.0, -80.190327, 25.767980, 3.0, "SE 6th St x Brickell Ave (turn)"),
]

def frame(lat): return 111_320*math.cos(math.radians(lat)), 110_540

def centreline():
    with open("research/brickell_streets.json") as fh:
        d = json.load(fh)
    pts = []
    for e in d["elements"]:
        if (e.get("tags") or {}).get("name") != "Southeast 6th Street":
            continue
        for p in e["geometry"]:
            pts.append((round(p["lon"], 7), round(p["lat"], 7)))
    return sorted(set(pts))                      # west -> east

def arc(pts):
    mx, my = frame(pts[0][1])
    s = [0.0]
    for a, b in zip(pts, pts[1:], strict=False):
        s.append(s[-1] + math.hypot((b[0]-a[0])*mx, (b[1]-a[1])*my))
    return np.array(s)

def at_s(pts, S, s):
    """Point and heading at arc length s."""
    i = int(np.clip(np.searchsorted(S, s) - 1, 0, len(pts)-2))
    f = (s - S[i]) / max(S[i+1]-S[i], 1e-9)
    lon = pts[i][0] + f*(pts[i+1][0]-pts[i][0])
    lat = pts[i][1] + f*(pts[i+1][1]-pts[i][1])
    mx, my = frame(lat)
    hd = math.degrees(math.atan2((pts[i+1][0]-pts[i][0])*mx,
                                 (pts[i+1][1]-pts[i][1])*my)) % 360
    return lon, lat, hd

def nearest_s(pts, S, lon, lat):
    mx, my = frame(lat)
    P = np.array([[p[0]*mx, p[1]*my] for p in pts])
    q = np.array([lon*mx, lat*my])
    return float(S[int(np.argmin(np.linalg.norm(P-q, axis=1)))])

def main():
    pts = centreline()
    S = arc(pts)
    m = np.load("reports/se6_motion.npz")
    mt, mag = m["t"], m["sm"]

    sA = nearest_s(pts, S, ANCHORS[0][1], ANCHORS[0][2])
    sB = nearest_s(pts, S, ANCHORS[1][1], ANCHORS[1][2])
    tA, tB = ANCHORS[0][0], ANCHORS[1][0]
    span = sB - sA
    print(f"anchors: {ANCHORS[0][4]} at t={tA}s  ->  {ANCHORS[1][4]} at t={tB}s")
    print(f"along-street distance {span:.1f} m over {tB-tA:.0f} s "
          f"(mean {span/(tB-tA):.2f} m/s)")

    # distance is distributed by observed motion, not uniformly in time
    win = (mt >= tA) & (mt <= tB)
    cum = np.cumsum(mag[win])
    cum = cum/cum[-1]
    twin = mt[win]

    ts = np.arange(tA, tB + 1e-9, 1.0/HZ)
    frac = np.interp(ts, twin, cum)
    s_of_t = sA + frac*span

    # Uncertainty. Cross-track is which lane the camera is in. Along-track is how
    # wrong the motion-weighted interpolation can be, largest mid-segment and
    # zero at the anchors, plus the anchors' own timing error carried as distance.
    speed = np.gradient(s_of_t, ts)
    # Along-track error accumulates only while the car is actually moving -- a
    # stopped period contributes no distance whatever the scale error is -- and
    # must vanish at both anchors, which are known crossings. So it is a fraction
    # of the distance to the nearer anchor, not a fraction of the whole span.
    SCALE_ERR = 0.25
    sig_along = np.sqrt((SCALE_ERR*np.minimum(s_of_t-sA, sB-s_of_t))**2
                        + (speed*np.interp(ts,[tA,tB],[ANCHORS[0][3],ANCHORS[1][3]]))**2)
    sig_cross = np.full_like(ts, 1.8)

    rows = []
    for t, s, sa, sc, v in zip(ts, s_of_t, sig_along, sig_cross, speed, strict=False):
        lon, lat, hd = at_s(pts, S, s)
        rows.append({"t": round(float(t),2), "lat": round(lat,7), "lon": round(lon,7),
                     "heading_deg": round(hd,1), "speed_ms": round(float(v),2),
                     "sigma_along_m": round(float(sa),1), "sigma_cross_m": round(float(sc),1)})
    with open("reports/se6_track.json","w") as fh:
        json.dump(rows, fh, indent=1)
    with open("reports/se6_track.csv","w") as fh:
        fh.write("t_s,latitude,longitude,heading_deg,speed_ms,sigma_along_m,sigma_cross_m\n")
        for r in rows:
            fh.write(f"{r['t']},{r['lat']},{r['lon']},{r['heading_deg']},"
                     f"{r['speed_ms']},{r['sigma_along_m']},{r['sigma_cross_m']}\n")
    sa = np.array([r["sigma_along_m"] for r in rows])
    print(f"\n{len(rows)} fixes at {HZ:.0f} Hz")
    print(f"along-track sigma: min {sa.min():.1f} m  median {np.median(sa):.1f} m  "
          f"max {sa.max():.1f} m (mid-segment)")
    print("cross-track sigma: 1.8 m (which lane)")
    print(f"speed: median {np.median(np.abs(speed)):.2f} m/s, max {np.abs(speed).max():.2f} m/s")
    print("wrote reports/se6_track.csv / .json")

if __name__ == "__main__":
    main()
