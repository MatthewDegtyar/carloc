"""A log of car sightings, and the query it exists to answer: was a car here?

This is the substrate for overstay. Overstay is *the same car, in the same
place, at two times far apart*, and none of that can be asked until sightings are
recorded in a form that supports the question. This module is that form.

Three design choices carry the weight:

* **No plates, ever.** The system's whole reason to exist is that it works
  without reading a plate, so a car's identity across sightings is carried by
  *where it is* plus *what it looks like* -- class and coarse colour -- not by an
  ID it was never allowed to read. Two sightings are "the same car" when they
  agree on both.

* **Time is synthetic and labelled as such.** This footage has no chronology, so
  each sighting's timestamp is a fixed epoch plus the moment in the video the car
  was seen. It is fabricated, consistently, and every record says so via
  ``synthetic=True``. The *mechanism* is real; only the clock is invented.

* **The presence query is a Mahalanobis gate, not a radius.** A fix here is a
  narrow cross-track sigma and a fat along-track one (see the SE 6th pass), so a
  circle of "close enough" is wrong in both directions -- too generous sideways,
  too mean lengthways. The query instead asks whether the point falls inside the
  sighting's own error ellipse, which is the honest form of "was a car *there*"
  when there is uncertainty.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

EARTH_MY = 110_540.0


def _mx(lat: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat))


@dataclass
class Sighting:
    """One observation that a vehicle was present at a place at a time."""

    sighting_id: str
    ts: datetime
    video_t: float
    lat: float
    lon: float
    heading_deg: float
    sigma_along_m: float
    sigma_cross_m: float
    vehicle_class: str
    color: str
    size_px: int
    zone: str | None
    source: str
    synthetic: bool = True
    color_rgb: tuple[int, int, int] = (110, 112, 115)

    def offset_m(self, lat: float, lon: float) -> tuple[float, float]:
        """(along, cross) metres from this sighting to a point, in its own frame.

        Rotated onto the vehicle heading so the two very different sigmas apply
        to the right axes.
        """
        dn = (lat - self.lat) * EARTH_MY
        de = (lon - self.lon) * _mx(self.lat)
        theta = math.radians(self.heading_deg)
        along = dn * math.cos(theta) + de * math.sin(theta)
        cross = -dn * math.sin(theta) + de * math.cos(theta)
        return along, cross

    def mahalanobis(self, lat: float, lon: float) -> float:
        """How many sigma a point is from this sighting, anisotropically."""
        along, cross = self.offset_m(lat, lon)
        return math.hypot(along / max(self.sigma_along_m, 1e-6),
                          cross / max(self.sigma_cross_m, 1e-6))

    def looks_like(self, other: Sighting) -> bool:
        """Same coarse appearance -- the plateless identity test."""
        return self.vehicle_class == other.vehicle_class and self.color == other.color


@dataclass
class SightingLog:
    """Append-only store of sightings, with spatial/temporal recall."""

    sightings: list[Sighting] = field(default_factory=list)

    def add(self, s: Sighting) -> None:
        self.sightings.append(s)

    def near(self, lat: float, lon: float, when: datetime | None = None,
             window_s: float | None = None, gate: float = 2.0
             ) -> list[tuple[Sighting, float]]:
        """Sightings whose error ellipse contains (lat, lon) within `gate` sigma.

        This is the presence query -- "was a car here?" -- and, with `when` and
        `window_s`, "was a car here around then?". Returns each match with its
        Mahalanobis distance, nearest first, so the caller sees not just whether
        a car was there but how squarely inside the uncertainty it sat.
        """
        out: list[tuple[Sighting, float]] = []
        for s in self.sightings:
            if (when is not None and window_s is not None
                    and abs((s.ts - when).total_seconds()) > window_s):
                continue
            d = s.mahalanobis(lat, lon)
            if d <= gate:
                out.append((s, d))
        out.sort(key=lambda pair: pair[1])
        return out

    def match(self, other: SightingLog, gate: float = 2.5,
              appearance: bool = True) -> list[tuple[Sighting, Sighting, float]]:
        """One-to-one correspondence between two passes, mutual-nearest.

        A car in pass A pairs with a car in pass B only when each is the other's
        closest admissible match -- symmetric Mahalanobis within `gate`, and the
        same coarse appearance if `appearance`. Mutual-nearest is what stops the
        combinatorial blow-up: on a dense block, all-pairs matching links every
        grey car to every other grey car, because the along-track sigma is metres
        wide and "grey car" is not a unique key. Forcing a one-to-one assignment
        is the honest ceiling on what a plateless descriptor can claim.
        """
        def cost(a: Sighting, b: Sighting) -> float:
            if appearance and not a.looks_like(b):
                return math.inf
            return max(a.mahalanobis(b.lat, b.lon), b.mahalanobis(a.lat, a.lon))

        best_b = {}
        for a in self.sightings:
            cands = [(cost(a, b), k) for k, b in enumerate(other.sightings)]
            c, k = min(cands, default=(math.inf, -1))
            if c <= gate:
                best_b[id(a)] = (k, c)
        best_a = {}
        for b in other.sightings:
            cands = [(cost(a, b), k) for k, a in enumerate(self.sightings)]
            c, k = min(cands, default=(math.inf, -1))
            if c <= gate:
                best_a[id(b)] = (k, c)

        pairs = []
        for ai, a in enumerate(self.sightings):
            if id(a) not in best_b:
                continue
            bk, c = best_b[id(a)]
            b = other.sightings[bk]
            if best_a.get(id(b), (-1,))[0] == ai:      # mutual
                gap = abs((a.ts - b.ts).total_seconds())
                pairs.append((a, b, gap))
        return pairs

    def overstay(self, other: SightingLog, min_gap_s: float, gate: float = 2.5
                 ) -> list[tuple[Sighting, Sighting, float]]:
        """Cars present in both passes and far apart in time -- the loiterers.

        Just ``match`` filtered to pairs whose synthetic times differ by more
        than `min_gap_s`. Comparing a pass against itself yields nothing, which is
        the correct answer for a single pass.
        """
        return [(a, b, g) for a, b, g in self.match(other, gate=gate)
                if g >= min_gap_s]

    # ---- persistence -------------------------------------------------------

    _COLS = ("sighting_id", "ts", "synthetic", "video_t", "latitude", "longitude",
             "heading_deg", "sigma_along_m", "sigma_cross_m", "vehicle_class",
             "color", "size_px", "zone", "source")

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self._COLS)
            for s in sorted(self.sightings, key=lambda s: s.ts):
                w.writerow([s.sighting_id, s.ts.isoformat(), s.synthetic, s.video_t,
                            f"{s.lat:.7f}", f"{s.lon:.7f}", s.heading_deg,
                            s.sigma_along_m, s.sigma_cross_m, s.vehicle_class,
                            s.color, s.size_px, s.zone or "UNKNOWN", s.source])

    def to_json(self, path: str) -> None:
        rows = []
        for s in self.sightings:
            d = asdict(s)
            d["ts"] = s.ts.isoformat()
            rows.append(d)
        with open(path, "w") as fh:
            json.dump(rows, fh, indent=1)


def synthetic_ts(epoch: datetime, video_t: float) -> datetime:
    """A fabricated wall-clock time: fixed epoch plus the video moment.

    Kept in one place so every sighting invents time the same way and the
    fabrication is obvious.
    """
    return epoch + timedelta(seconds=video_t)


def zone_for(lat: float, lon: float, anchors: list[tuple[float, float, str]],
             snap_m: float = 30.0) -> str | None:
    """Which ParkMobile zone a point sits in, by nearest on-street anchor.

    Returns None when no anchor is within `snap_m` -- which is the honest answer
    for the Brickell pass, where no zone geometry exists yet.
    """
    best, bz = snap_m, None
    for al, ao, z in anchors:
        d = math.hypot((al - lat) * EARTH_MY, (ao - lon) * _mx(lat))
        if d < best:
            best, bz = d, z
    return bz
