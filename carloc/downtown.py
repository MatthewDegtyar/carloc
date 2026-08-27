"""The downtown demo: are these cars parked in a paid zone or not?

Scoped deliberately. Five Miami zones were surveyed; only the downtown two are
usable, because `zonebox.character()` shows 40712 and 40713 are 57% and 95%
residential and therefore almost certainly permit parking -- where a resident and
a violator are visually identical and no camera can separate them. 40701 and
40703 are arterial and collector dominated, which is where metered kerbside
parking actually is.

It also runs on **satellite imagery rather than video**, which sidesteps the
blocker that has held up everything else: the Miami dashcam clip has no pose, and
without pose there are no world coordinates. Web Mercator tiles are georeferenced
by construction, so a car detected in a tile has a lat/lon immediately, with no
odometry, no intrinsics and no scale recovery.

What that costs is honesty about time. Satellite is one instant, months stale,
and says nothing about how long a car has been there. So this demonstrates
*"which cars are in a paid zone right now"*, which is the geometry half of
enforcement, and not *"has this car overstayed"*, which needs the video.

The detector is the one piece that must come from outside. COCO-trained YOLO is
useless from directly overhead -- measured on these exact tiles it returns
`clock`, `train`, `potted plant` and `person`, with two cars at the loosest
threshold. Overhead cars are rectangles and it has never seen one. An
aerial-trained model is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DOWNTOWN_ZONES = ("40701", "40703")
"""The zones this demo covers, and the only two of five that pass
`enforceable_plateless`."""


@dataclass
class Sighting:
    """One detected car, placed on the ground and judged against the zones."""

    lat: float
    lon: float
    score: float
    verdict: str
    margin_m: float
    zone: str | None
    box_id: str | None

    @property
    def paid_zone(self) -> bool:
        return self.verdict == "inside"


def pixel_to_lonlat(px: float, py: float, width: int, height: int,
                    extent: tuple[float, float, float, float], zoom: int
                    ) -> tuple[float, float]:
    """Mosaic pixel -> lon/lat.

    Done through tile coordinates rather than by interpolating latitude, because
    tiles are evenly spaced in Mercator northing and interpolating latitude
    directly walks off by metres toward the edges of the image.
    """
    from carloc.basemap import lonlat_to_tile, tile_to_lonlat

    west, east, south, north = extent
    x0, y0 = lonlat_to_tile(west, north, zoom)
    x1, y1 = lonlat_to_tile(east, south, zoom)
    return tile_to_lonlat(x0 + (px / width) * (x1 - x0),
                          y0 + (py / height) * (y1 - y0), zoom)


def judge(lat: float, lon: float, boxes, sigma_m: float,
          score: float = 1.0) -> Sighting:
    """Classify one car against every zone box, keeping the strongest claim.

    A car can only be inside one lane box, so the best verdict wins: an INSIDE
    anywhere beats AMBIGUOUS, which beats OUTSIDE. Ties break on margin. Reported
    OUTSIDE therefore means outside *all* of them, which is the claim an operator
    actually needs.
    """
    from carloc.zonebox import Verdict, classify

    rank = {Verdict.INSIDE: 2, Verdict.AMBIGUOUS: 1, Verdict.OUTSIDE: 0}
    best = (-1, -1e9, Verdict.OUTSIDE, -1e9, None)
    for box in boxes:
        verdict, margin = classify(box, lat, lon, sigma_m)
        key = (rank[verdict], margin)
        if key > (best[0], best[1]):
            best = (rank[verdict], margin, verdict, margin, box)
    _, _, verdict, margin, box = best
    return Sighting(lat=lat, lon=lon, score=score, verdict=str(verdict),
                    margin_m=round(float(margin), 2),
                    zone=box.zone if box else None,
                    box_id=box.box_id if box else None)


def survey(detector, boxes, zoom: int = 20, pad: float = 0.00040,
           sigma_m: float = 1.0, on_tile=None) -> list[Sighting]:
    """Detect cars over every zone box and judge each one.

    ``detector`` needs a single method, ``detect(image) -> [Detection]``, so a
    Roboflow model, a local YOLO, or anything else drops in unchanged.

    ``sigma_m`` is the position uncertainty of a satellite fix. Kept at 1 m and
    deliberately not smaller: the tile georeferencing is exact but the imagery
    itself carries orthorectification error, and a car's roof is not directly
    above its wheels. It feeds the AMBIGUOUS band, so understating it would
    manufacture confident verdicts.
    """
    from carloc.basemap import fetch_extent

    seen: set[tuple[int, int]] = set()
    sightings: list[Sighting] = []

    for box in boxes:
        lon, lat = box.centre
        # One tile mosaic per ~90 m cell, deduplicated, so overlapping boxes on
        # the same block are not fetched or detected twice.
        cell = (int(lon / pad), int(lat / pad))
        if cell in seen:
            continue
        seen.add(cell)

        mosaic, extent = fetch_extent(lon - pad, lat - pad * 0.8,
                                      lon + pad, lat + pad * 0.8, zoom=zoom)
        detections = detector.detect(mosaic)
        height, width = mosaic.shape[:2]
        for detection in detections:
            px, py = detection.centroid
            dlon, dlat = pixel_to_lonlat(float(px), float(py), width, height,
                                         extent, zoom)
            sightings.append(judge(dlat, dlon, boxes, sigma_m,
                                   score=float(getattr(detection, "score", 1.0))))
        if on_tile:
            on_tile(box, len(detections))
    return sightings


def summarise(sightings: list[Sighting]) -> dict:
    """Counts an operator would act on, with the refusals kept visible."""
    inside = [s for s in sightings if s.verdict == "inside"]
    outside = [s for s in sightings if s.verdict == "outside"]
    ambiguous = [s for s in sightings if s.verdict == "ambiguous"]
    by_zone: dict[str, int] = {}
    for s in inside:
        if s.zone:
            by_zone[s.zone] = by_zone.get(s.zone, 0) + 1
    return {
        "cars": len(sightings),
        "in_paid_zone": len(inside),
        "outside_any_zone": len(outside),
        "ambiguous": len(ambiguous),
        "ambiguous_share": len(ambiguous) / len(sightings) if sightings else 0.0,
        "by_zone": by_zone,
        "median_margin_inside": (float(np.median([s.margin_m for s in inside]))
                                 if inside else float("nan")),
    }


def occupancy(sightings: list[Sighting], boxes) -> list[dict]:
    """Per-box occupancy: cars found against spaces estimated.

    Over 100% is not impossible and is a useful smell test -- it means the space
    estimate is low, the lane box is catching a travel lane, or the detector is
    double-counting.
    """
    counts: dict[str, int] = {}
    for s in sightings:
        if s.verdict == "inside" and s.box_id:
            counts[s.box_id] = counts.get(s.box_id, 0) + 1
    rows = []
    for box in boxes:
        found = counts.get(box.box_id, 0)
        rows.append({
            "box_id": box.box_id, "zone": box.zone, "street": box.street,
            "side": box.side, "spaces": box.spaces, "cars": found,
            "occupancy": found / box.spaces if box.spaces else float("nan"),
            "lat": round(box.centre[1], 7), "lon": round(box.centre[0], 7),
        })
    rows.sort(key=lambda r: -r["cars"])
    return rows


def downtown_boxes(boxes):
    """Only the zones where a plateless verdict means anything."""
    return [b for b in boxes if b.zone in DOWNTOWN_ZONES]


def metres_per_pixel(lat: float, zoom: int) -> float:
    """Ground resolution, for sanity-checking that cars are big enough to find."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
