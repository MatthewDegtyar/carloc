"""Where a car must physically be to be parked in a paid zone.

The enforcement question is not "is this car near zone 40703". It is "is the
centre of this car inside the metered parking lane of a paid block face, by more
than the uncertainty on both". Those are different questions and only the second
survives an appeal.

So a zone box here is a **polygon over the parking lane itself**, and testing a
car against it returns one of three answers, never two:

    INSIDE      the fix is inside by more than its own uncertainty
    OUTSIDE     the fix is outside by more than its own uncertainty
    AMBIGUOUS   the fix straddles the boundary -- do not cite

The third is the whole point. A car 30 cm outside a lane whose position is known
to +/- 1 m has not been shown to be anywhere, and a system that rounds that to
OUTSIDE (free parker) or INSIDE (evader) is guessing with someone's money.

What the geometry rests on
--------------------------
Two inputs of very different quality.

**Which segments are paid** comes from ParkMobile and is solid: their zone
anchors sit a median of 0.69 m from the street centreline, so matching an anchor
to a street is unambiguous.

**Where the lane is** is assumed, and it is the weak link. The anchors are on the
centreline, so they say nothing about which side is metered or how far out the
bay sits. `LANE_INNER_M` and `LANE_OUTER_M` below are US design defaults, not
measurements of Miami, and they are the reason `ambiguous_band_m` exists and is
reported with every box.

**The right fix is to learn the lane from observed cars.** Detected, geolocated
parked cars *are* the parking lane -- fit the band to them and the assumption
disappears. `calibrate_from_observations()` does that; until it has been run on
real fixes, every box carries `calibrated=False` and should be treated as
provisional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

EARTH_R = 6_378_137.0

LANE_INNER_M = 3.3
"""Near edge of the parking lane, from the street centreline.

One travel lane out. US urban travel lanes run 3.0-3.6 m; 3.3 is mid-range. Too
small and the box eats moving traffic."""

LANE_OUTER_M = 5.9
"""Far edge: inner edge plus a 2.6 m parking bay (US standard 8-8.5 ft)."""

CORNER_CLEARANCE_M = 7.6
"""Trimmed from each end -- hydrants, daylighting, kerb return. 25 ft."""

ANCHOR_SNAP_M = 25.0
"""How close an anchor must be to a street segment to mark it paid.

Generous against the measured 0.69 m median so that a slightly-off anchor still
lands, tight enough not to claim the next street over."""


class Verdict(StrEnum):
    INSIDE = "inside"
    OUTSIDE = "outside"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ZoneBox:
    """One side of one block face, inside a paid zone."""

    box_id: str
    zone: str
    street: str
    side: str
    polygon: list[tuple[float, float]]
    """(lon, lat) corners, closed. The parking lane, not the street."""

    length_m: float
    width_m: float
    spaces: int
    calibrated: bool = False
    """False while the lane offset is assumed rather than fitted to observed cars."""

    @property
    def ambiguous_band_m(self) -> float:
        """How far in from the edge a fix must be before the verdict is safe.

        Half the lane width: a car centred in the bay is this far from either
        edge, so a fix with more uncertainty than this cannot be placed in or
        out of the lane at all.
        """
        return self.width_m / 2.0

    @property
    def centre(self) -> tuple[float, float]:
        pts = np.array(self.polygon[:4])
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _frame(lat0: float):
    mx = 111_320.0 * math.cos(math.radians(lat0))
    my = 110_540.0
    return mx, my


def _signed_distance_to_edges(point_m: np.ndarray, corners_m: np.ndarray) -> float:
    """Distance from a point to the polygon boundary, positive inside.

    The polygon is a rectangle, so this is the smallest distance to any of the
    four edges, signed by whether the point is within all of them.
    """
    inside = True
    best = float("inf")
    n = len(corners_m)
    for i in range(n):
        a, b = corners_m[i], corners_m[(i + 1) % n]
        edge = b - a
        length = np.linalg.norm(edge)
        if length < 1e-9:
            continue
        normal = np.array([-edge[1], edge[0]]) / length
        # Corners are wound consistently, so a consistent sign means inside.
        signed = float((point_m - a) @ normal)
        if signed < 0:
            inside = False
        t = np.clip(float((point_m - a) @ edge) / (length * length), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(point_m - (a + t * edge))))
    return best if inside else -best


def classify(box: ZoneBox, lat: float, lon: float, sigma_m: float) -> tuple[Verdict, float]:
    """Is a car at (lat, lon) +/- sigma parked inside this box?

    Returns the verdict and the signed margin in metres -- positive inside,
    negative outside. The margin is what to show an operator: "1.4 m inside" is
    an argument, "inside" is an assertion.
    """
    lat0 = box.polygon[0][1]
    mx, my = _frame(lat0)
    corners = np.array([[x * mx, y * my] for x, y in box.polygon[:4]])
    point = np.array([lon * mx, lat * my])
    margin = _signed_distance_to_edges(point, corners)

    if abs(margin) <= sigma_m:
        return Verdict.AMBIGUOUS, margin
    return (Verdict.INSIDE if margin > 0 else Verdict.OUTSIDE), margin


def paid_segments(ways: list[dict], zones, snap_m: float = ANCHOR_SNAP_M) -> dict:
    """Street segments carrying a paid zone -> the zone codes on them.

    Keyed by (way id, vertex index) so a long street can be paid on one stretch
    and free on another, which is how corridors actually work.
    """
    lat0 = 25.75
    mx, my = _frame(lat0)

    starts, ends, keys = [], [], []
    for way in ways:
        geometry = way.get("geometry") or []
        for index in range(len(geometry) - 1):
            a, b = geometry[index], geometry[index + 1]
            starts.append([a["lon"] * mx, a["lat"] * my])
            ends.append([b["lon"] * mx, b["lat"] * my])
            keys.append((way.get("id"), index))
    if not starts:
        return {}
    A = np.array(starts)
    B = np.array(ends)
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    L2[L2 < 1e-9] = 1e-9

    paid: dict = {}
    for zone in zones:
        for lon, lat in zone.points:
            p = np.array([lon * mx, lat * my])
            t = np.clip(np.einsum("ij,ij->i", p - A, AB) / L2, 0.0, 1.0)
            distance = np.linalg.norm(A + t[:, None] * AB - p, axis=1)
            i = int(np.argmin(distance))
            if distance[i] <= snap_m:
                paid.setdefault(keys[i], set()).add(zone.signage_code)
    return paid


def build(ways: list[dict], zones, inner_m: float = LANE_INNER_M,
          outer_m: float = LANE_OUTER_M, both_sides: bool = True) -> list[ZoneBox]:
    """Paid street segments -> parking-lane polygons.

    ``both_sides`` defaults True because the anchors cannot say which side is
    metered, and omitting a real side would let an evader off. The cost is the
    opposite error, so a box on a side with no parking is possible and is why
    every verdict carries a margin rather than a bare yes.
    """
    paid = paid_segments(ways, zones)
    by_way = {way.get("id"): way for way in ways}
    boxes: list[ZoneBox] = []

    for (way_id, index), codes in paid.items():
        way = by_way.get(way_id)
        if way is None:
            continue
        geometry = way.get("geometry") or []
        if index + 1 >= len(geometry):
            continue
        a, b = geometry[index], geometry[index + 1]
        lat0 = a["lat"]
        mx, my = _frame(lat0)
        p0 = np.array([a["lon"] * mx, a["lat"] * my])
        p1 = np.array([b["lon"] * mx, b["lat"] * my])
        along = p1 - p0
        length = float(np.linalg.norm(along))
        if length < 2 * CORNER_CLEARANCE_M + 4.0:
            continue
        along /= length
        normal = np.array([-along[1], along[0]])

        start = p0 + along * CORNER_CLEARANCE_M
        end = p1 - along * CORNER_CLEARANCE_M
        usable = float(np.linalg.norm(end - start))
        street = (way.get("tags") or {}).get("name") or f"way/{way_id}"
        zone = sorted(codes)[0]

        for side, sign in (("left", 1.0), ("right", -1.0)):
            if not both_sides and side == "right":
                continue
            corners_m = [
                start + normal * sign * inner_m,
                end + normal * sign * inner_m,
                end + normal * sign * outer_m,
                start + normal * sign * outer_m,
            ]
            polygon = [(float(c[0] / mx), float(c[1] / my)) for c in corners_m]
            polygon.append(polygon[0])
            boxes.append(ZoneBox(
                box_id=f"{way_id}:{index}:{side}",
                zone=zone, street=street, side=side, polygon=polygon,
                length_m=round(usable, 1), width_m=outer_m - inner_m,
                spaces=int(usable // 6.7),
            ))
    return boxes


def calibrate_from_observations(fixes: np.ndarray, ways: list[dict],
                                percentile: float = 90.0) -> tuple[float, float]:
    """Fit the lane band to where cars were actually observed.

    ``fixes`` is (N, 2) as (lon, lat) of parked-car positions. Returns
    (inner_m, outer_m) as perpendicular distances from the street centreline.

    This is the measurement that removes the guess. Until it is run, `build()`
    uses design defaults and every box says ``calibrated=False``.

    A percentile rather than min/max, because a handful of fixes will land in the
    travel lane or on the pavement and the band should not stretch to cover them.
    """
    lat0 = float(np.mean(fixes[:, 1]))
    mx, my = _frame(lat0)
    starts, ends = [], []
    for way in ways:
        geometry = way.get("geometry") or []
        for i in range(len(geometry) - 1):
            starts.append([geometry[i]["lon"] * mx, geometry[i]["lat"] * my])
            ends.append([geometry[i + 1]["lon"] * mx, geometry[i + 1]["lat"] * my])
    A = np.array(starts)
    B = np.array(ends)
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    L2[L2 < 1e-9] = 1e-9

    distances = []
    for lon, lat in fixes:
        p = np.array([lon * mx, lat * my])
        t = np.clip(np.einsum("ij,ij->i", p - A, AB) / L2, 0.0, 1.0)
        d = np.linalg.norm(A + t[:, None] * AB - p, axis=1)
        distances.append(float(d.min()))
    distances = np.array(distances)
    lower = float(np.percentile(distances, 100 - percentile))
    upper = float(np.percentile(distances, percentile))
    return lower, upper


def to_geojson(boxes: list[ZoneBox]) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[list(c) for c in b.polygon]]},
            "properties": {
                "box_id": b.box_id, "zone": b.zone, "street": b.street, "side": b.side,
                "length_m": b.length_m, "width_m": b.width_m, "est_spaces": b.spaces,
                "ambiguous_band_m": round(b.ambiguous_band_m, 2),
                "calibrated": b.calibrated,
                "source": "ParkMobile zone anchors + OSM centreline; lane offset ASSUMED",
            },
        } for b in boxes],
    }
