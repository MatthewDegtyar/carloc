"""Block faces: the geometric unit a parking zone actually occupies.

Miami publishes no on-street parking geometry -- not on the Parking Authority
site, not in the ParkMobile app (its zone box is a plain Google geocode; typing
40703 returns Costa Rica), and not in OpenStreetMap, where 1610 downtown street
ways carry only 8 parking tags between them.

What can be derived is the container. The MUTCD, which Florida adopts, says a
parking regulation runs to the next cross street unless a termination sign says
otherwise, and caps sign spacing at about 270 ft. So the natural unit of an
on-street zone is one side of one block, and a zone is one or more of those.

This builds that grid from the street network: every segment between two
intersections, offset to each kerb, as a rectangle with a length, an estimated
capacity and corner coordinates. It is a **scaffold, not an answer** -- it says
where parking can be, not where it is or what zone number it carries. Populating
it needs the Authority's own inventory, which under Florida's Chapter 119 is a
public record and can be requested.

Every number here is geometry plus a stated assumption. None of it is measured
from the street, and a lane that does not exist will still get a box.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

EARTH_R = 6_378_137.0

LANE_OFFSET_M = 4.6
"""Distance from street centreline to the middle of a kerbside parking lane.

Half a travel lane (3.6 m) plus half a parking lane (2.6 m) is 3.1 m for the
narrowest case; wider streets with two travel lanes per side push it past 7 m.
4.6 m is a mid-range figure for a downtown street with one travel lane each way.

It is the crudest assumption in the file. It shifts a box sideways, which matters
because a box misplaced by 2 m onto the travel lane will collect moving traffic
as if it were parked."""

LANE_WIDTH_M = 2.6
"""Kerbside parking lane width. US standard parallel bay is 8-8.5 ft."""

SPACE_LENGTH_M = 6.7
"""Length of one parallel space including manoeuvring room -- 22 ft, the common
US design figure. Marked bays are sometimes 20 ft, unmarked runs pack tighter."""

CORNER_CLEARANCE_M = 7.6
"""Unusable length at each end of a block face -- 25 ft.

Hydrant clearance, daylighting at the crossing, and the curve of the kerb return.
Ignoring it overstates capacity by two spaces on every block face, which across a
downtown grid is hundreds of spaces that do not exist."""

MIN_FACE_LENGTH_M = 20.0
"""Below this a face cannot hold a space after corner clearance."""


@dataclass
class BlockFace:
    """One side of one block: where a parking zone would sit."""

    face_id: str
    street: str
    side: str
    """`left` or `right` relative to the way's digitisation direction, which is
    arbitrary -- it is kept only so the two sides of a block stay distinguishable."""

    length_m: float
    centre: tuple[float, float]
    """(lon, lat) midpoint of the face."""

    polygon: list[tuple[float, float]]
    """Corners as (lon, lat), closed. This is the box to overlay."""

    spaces: int
    from_node: int = 0
    to_node: int = 0
    tags: dict = field(default_factory=dict)

    @property
    def usable_length_m(self) -> float:
        return max(0.0, self.length_m - 2 * CORNER_CLEARANCE_M)


def _local_metres(lon: float, lat: float, lat0: float) -> tuple[float, float]:
    """Equirectangular projection about a reference latitude.

    Good to well under a metre across a city centre, and it avoids a projection
    dependency for what is only ever used to offset and measure short segments.
    """
    x = math.radians(lon) * EARTH_R * math.cos(math.radians(lat0))
    y = math.radians(lat) * EARTH_R
    return x, y


def _to_wgs84(x: float, y: float, lat0: float) -> tuple[float, float]:
    lon = math.degrees(x / (EARTH_R * math.cos(math.radians(lat0))))
    lat = math.degrees(y / EARTH_R)
    return lon, lat


def intersections(ways: list[dict]) -> set[int]:
    """Nodes shared by more than one way, plus every way's endpoints.

    Endpoints count because a way can terminate at a dead end or at a data
    boundary, and a face has to stop there too.
    """
    counts: defaultdict[int, int] = defaultdict(int)
    for way in ways:
        nodes = way.get("nodes") or []
        for node in nodes:
            counts[node] += 1
        for end in (nodes[:1] + nodes[-1:]):
            counts[end] += 1
    return {node for node, count in counts.items() if count > 1}


PLAUSIBLE_HIGHWAY = {"residential", "unclassified", "living_street", "tertiary"}
"""Street classes that usually carry kerbside parking downtown.

Calibration, not taste. Assuming every block face of every street has parking
gives 39,946 spaces for downtown alone, against the ~11,800 the Parking Authority
reports for the **whole city** -- a 3.4x over-count, driven by arterials. Brickell
Avenue came out top of the list at 1,322 spaces; it is a six-lane arterial.

Secondary roads are admitted only at two lanes or fewer; primaries never. Both
directions of error remain: a two-lane secondary with a bus lane gets counted,
and a primary with a metered service lane does not."""

MAX_LANES_FOR_PARKING = 2


def plausible(way: dict) -> bool:
    """Could this street have kerbside parking at all?"""
    tags = way.get("tags", {})
    highway = tags.get("highway")
    if highway in PLAUSIBLE_HIGHWAY:
        return True
    if highway == "secondary":
        try:
            return int(str(tags.get("lanes", "9")).split(";")[0]) <= MAX_LANES_FOR_PARKING
        except ValueError:
            return False
    return False


def split_into_faces(ways: list[dict], lane_offset_m: float = LANE_OFFSET_M,
                     filter_plausible: bool = True) -> list[BlockFace]:
    """Street network -> one box per kerbside, per block."""
    if filter_plausible:
        ways = [w for w in ways if plausible(w)]
    junctions = intersections(ways)
    faces: list[BlockFace] = []

    for way in ways:
        geometry = way.get("geometry") or []
        nodes = way.get("nodes") or []
        if len(geometry) < 2 or len(geometry) != len(nodes):
            continue
        tags = way.get("tags", {})
        street = tags.get("name") or f"way/{way.get('id')}"
        lat0 = geometry[0]["lat"]

        # Cut the way wherever it meets another way.
        segment: list[int] = [0]
        for index in range(1, len(geometry)):
            segment.append(index)
            is_last = index == len(geometry) - 1
            if nodes[index] in junctions or is_last:
                if len(segment) >= 2:
                    faces.extend(_faces_for(
                        [geometry[i] for i in segment], street, way, lat0,
                        nodes[segment[0]], nodes[segment[-1]], lane_offset_m))
                segment = [index]
    return faces


def _faces_for(points: list[dict], street: str, way: dict, lat0: float,
               from_node: int, to_node: int, lane_offset_m: float) -> list[BlockFace]:
    metres = [_local_metres(p["lon"], p["lat"], lat0) for p in points]
    length = sum(math.dist(metres[i], metres[i + 1]) for i in range(len(metres) - 1))
    if length < MIN_FACE_LENGTH_M:
        return []

    start, end = metres[0], metres[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return []
    ux, uy = dx / norm, dy / norm
    # Left normal in a right-handed frame.
    nx, ny = -uy, ux

    out: list[BlockFace] = []
    for side, sign in (("left", 1.0), ("right", -1.0)):
        cx = (start[0] + end[0]) / 2 + sign * nx * lane_offset_m
        cy = (start[1] + end[1]) / 2 + sign * ny * lane_offset_m
        half_len = length / 2
        half_wid = LANE_WIDTH_M / 2
        corners = []
        for along, across in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)):
            px = cx + ux * along * half_len + nx * across * half_wid
            py = cy + uy * along * half_len + ny * across * half_wid
            corners.append(_to_wgs84(px, py, lat0))
        usable = max(0.0, length - 2 * CORNER_CLEARANCE_M)
        out.append(BlockFace(
            face_id=f"{way.get('id')}:{from_node}:{to_node}:{side}",
            street=street, side=side, length_m=round(length, 1),
            centre=_to_wgs84(cx, cy, lat0), polygon=corners,
            spaces=int(usable // SPACE_LENGTH_M),
            from_node=from_node, to_node=to_node,
            tags={k: v for k, v in way.get("tags", {}).items()
                  if k in ("highway", "oneway", "surface", "lanes")},
        ))
    return out


def to_geojson(faces: list[BlockFace]) -> dict:
    """GeoJSON, which Google My Maps imports directly as an overlay."""
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[list(c) for c in f.polygon]]},
            "properties": {
                "face_id": f.face_id, "street": f.street, "side": f.side,
                "length_m": f.length_m, "usable_m": round(f.usable_length_m, 1),
                "est_spaces": f.spaces,
                "lat": round(f.centre[1], 7), "lon": round(f.centre[0], 7),
                "zone": "", "source": "derived from OSM centreline; NOT surveyed",
            },
        } for f in faces],
    }


def write(faces: list[BlockFace], geojson_path: Path, csv_path: Path) -> None:
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    geojson_path.write_text(json.dumps(to_geojson(faces)))

    import csv as _csv

    with csv_path.open("w", newline="") as handle:
        writer = _csv.writer(handle)
        writer.writerow(["face_id", "street", "side", "lat", "lon",
                         "length_m", "usable_m", "est_spaces", "zone"])
        for f in faces:
            writer.writerow([f.face_id, f.street, f.side,
                             round(f.centre[1], 7), round(f.centre[0], 7),
                             f.length_m, round(f.usable_length_m, 1), f.spaces, ""])
