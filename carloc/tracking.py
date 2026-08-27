"""Count parked cars atomically by tracking them across frames, not merging points.

The naive pipeline placed every detection independently and then merged anything
within a few metres. It miscounts in both directions: one car seen across many
frames scatters along the street (its range from a single bearing is unstable at
shallow angles) and splits in two, while two genuinely adjacent cars collapse
into one. The fix is to stop treating detections as independent points.

A parked car is a fixed world point, and as the camera drives past, that car's
image position sweeps predictably -- leftward and downward, growing -- and its
bearing sweeps monotonically toward the frame edge. So:

1. **Associate** detections across frames into tracklets by that image motion plus
   appearance. One tracklet ideally is one physical car.
2. **Triangulate** each tracklet's along-street position from all its bearings at
   once. Each detection implies the car sits ``L / tan(bearing)`` ahead of the
   camera; near the frame edge (abeam) that is precise, far ahead it is nearly
   useless, so the estimates are combined weighted by ``sin^4(bearing)`` -- the
   camera-orientation variable doing the work the user asked it to do.
3. **Stitch** tracklets that an obstruction split: a tree or a passing car breaks
   one car into two tracklets whose triangulated positions and appearance agree,
   and those are re-merged. This is where "given obstructions" is handled.

The count is the number of stitched tracks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    color: str
    cls: str
    dets: list = field(default_factory=list)     # member detection dicts
    last_frame: int = -1
    cx: float = 0.0
    cy: float = 0.0
    bh: float = 0.0
    vcx: float = 0.0                              # image x-velocity (px/frame, negative)

    def predict_cx(self, frame: int) -> float:
        return self.cx + self.vcx * (frame - self.last_frame)


def associate(dets: list[dict], W: int, gate: float = 0.14,
              max_gap: int = 4) -> list[Track]:
    """Greedy per-frame association into tracklets.

    A left-kerb car moves left as the camera advances, so a detection that jumps
    appreciably rightward of a track's prediction is rejected outright; the rest
    are scored on predicted-image distance plus a size term, gated, and matched
    cheapest-first. Appearance must agree, which stops a black car inheriting the
    track of the silver one behind it.
    """
    by_frame: dict[int, list[dict]] = {}
    for d in dets:
        by_frame.setdefault(d["frame"], []).append(d)

    active: list[Track] = []
    done: list[Track] = []
    for frame in sorted(by_frame):
        df = by_frame[frame]
        pairs = []
        for ti, tr in enumerate(active):
            if frame - tr.last_frame > max_gap:
                continue
            pcx = tr.predict_cx(frame)
            for di, d in enumerate(df):
                if d["color"] != tr.color and d["cls"] != tr.cls:
                    continue
                if d["cx"] - tr.cx > 25:                 # a parked car never jumps right
                    continue
                cost = (abs(d["cx"] - pcx) / W
                        + 1.4 * abs(d["cy"] - tr.cy) / 1080
                        + 0.4 * abs(d["bh"] - tr.bh) / max(tr.bh, 1.0))
                if cost < gate:
                    pairs.append((cost, ti, di))
        pairs.sort(key=lambda p: p[0])
        used_t, used_d = set(), set()
        for _c, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr, d = active[ti], df[di]
            ncx = d["cx"]
            tr.vcx = 0.6 * tr.vcx + 0.4 * (ncx - tr.cx) if tr.dets else (ncx - tr.cx)
            tr.cx, tr.cy, tr.bh, tr.last_frame = ncx, d["cy"], d["bh"], frame
            tr.dets.append(d)
        for di, d in enumerate(df):
            if di in used_d:
                continue
            active.append(Track(color=d["color"], cls=d["cls"], dets=[d],
                                 last_frame=frame, cx=d["cx"], cy=d["cy"], bh=d["bh"]))
        still = [t for t in active if frame - t.last_frame <= max_gap]
        done += [t for t in active if frame - t.last_frame > max_gap]
        active = still
    return done + active


def triangulate(track: Track, lateral_m: float,
                floor_deg: float = 30.0) -> tuple[float, float, int]:
    """Along-street position of the car from its near-abeam bearings.

    Each detection implies S_i = s_cam_i + L / tan(|bearing|). Only bearings past
    ``floor_deg`` are used: below it the tangent is small and S_i swings by tens
    of metres for a pixel of noise, so a far, head-on glimpse contributes nothing
    but variance. A real parked car always sweeps past the floor as the camera
    reaches it; a car that never does (distant, or moving away) returns n=0 and is
    dropped, which is a useful filter in itself. Among the retained views S_i is
    combined weighted by sin^4(bearing) -- abeam dominates -- with one robust
    reweight, and the spread of those views is the 1-sigma.
    """
    S, wsum, scam = [], [], []
    for d in track.dets:
        b = math.radians(abs(d["bearing"]))
        if b < math.radians(floor_deg):
            continue
        S.append(d["s_cam"] + lateral_m / math.tan(b))
        wsum.append(math.sin(b) ** 4)
        scam.append(d["s_cam"])
    if len(S) < 1:
        return float("nan"), float("nan"), 0
    S = np.array(S)
    wsum = np.array(wsum)
    scam = np.array(scam)
    # Reject a vehicle moving with traffic. For a stationary car S_i is constant
    # as the camera advances; for one keeping pace ahead, S_i climbs with s_cam
    # (slope ~ +1), and for oncoming it falls. Only a near-flat slope is parked.
    if len(S) >= 3 and np.ptp(scam) > 10.0:
        slope = float(np.polyfit(scam, S, 1)[0])
        if abs(slope) > 0.75:
            return float("nan"), float("nan"), 0
    mu = float(np.average(S, weights=wsum))
    keep = np.ones(len(S), bool)
    for _ in range(2):
        resid = np.abs(S - mu)
        scale = 1.4826 * np.median(resid) + 1e-6
        keep = resid < 3 * scale
        if keep.sum() >= 1:
            mu = float(np.average(S[keep], weights=wsum[keep]))
    spread = (float(np.sqrt(np.average((S[keep] - mu) ** 2, weights=wsum[keep])))
              if keep.sum() > 1 else 3.0)
    n_eff = (wsum[keep].sum() ** 2) / np.sum(wsum[keep] ** 2)
    sigma = min(max(spread / math.sqrt(max(n_eff, 1.0)), 1.5), 10.0)
    return mu, sigma, int(keep.sum())


def slot(tracks: list[dict], min_sep_m: float = 4.5) -> list[dict]:
    """Collapse tracklets into physical cars with a minimum-separation prior.

    Union-find on "within N metres" chains: on a crawl the camera fragments one
    car into tracklets a few metres apart, and single-linkage then welds the whole
    block into one node. But two cars cannot occupy the same stretch of kerb, so
    the truth is a set of positions no two closer than about a parking space. This
    enforces that directly, as non-maximum suppression along the street: seed a
    slot with the best-supported tracklet, absorb every weaker tracklet within
    ``min_sep_m`` of an existing slot, and only start a new slot beyond that
    distance. Seeding by support (not by chaining) is what stops the runaway
    merge, and the min separation is what stops fragments becoming phantom cars.
    """
    slots: list[dict] = []
    for t in sorted(tracks, key=lambda t: -t["ndet"]):
        near = min(slots, key=lambda sl: abs(sl["S"] - t["S"]), default=None)
        if near is not None and abs(near["S"] - t["S"]) < min_sep_m:
            w0 = near["ndet"]
            w1 = t["ndet"]
            near["S"] = (near["S"] * w0 + t["S"] * w1) / (w0 + w1)
            near["sigma_S"] = min(near["sigma_S"], t["sigma_S"])
            near["ndet"] += t["ndet"]
            near["n_tracklets"] += 1
            near["t_first"] = min(near["t_first"], t["t_first"])
            near["t_last"] = max(near["t_last"], t["t_last"])
        else:
            slots.append({"S": t["S"], "sigma_S": max(t["sigma_S"], 1.0),
                          "color": t["color"], "cls": t["cls"], "ndet": t["ndet"],
                          "n_tracklets": 1, "t_first": t["t_first"], "t_last": t["t_last"]})
    slots.sort(key=lambda m: m["S"])
    return slots
