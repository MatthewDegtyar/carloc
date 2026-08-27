"""ParkMobile zone geometry, from the endpoints their own web app uses.

The zone data is reachable, which took three wrong turns to establish and is
worth recording so nobody repeats them:

1. The **search box** at `/search` is not a zone lookup. It passes your text to
   Google's geocoder, so `40703` returns a place in Costa Rica.
2. The **map viewport** endpoints -- `/api/zones/search` and
   `/api/zones/search/transient` -- return only bookable off-street garages and
   lots. Every `parkingType` value gives the same handful. On-street zones are
   simply not in them, in Miami or in South Beach.
3. The **zone-number flow** at `/zone/start` is the one that works. Typing a
   signage code resolves it, and the chain is:

       /api/proxy/parkmobileapi/zones/{signageCode}
           -> internalZoneCode, e.g. 40703 -> 97840703
       /api/locations?internalZoneCode={code}&supplierId={supplier}
           -> type ("OnStreet"), and a list of lat/lon points

`internalZoneCode` is the supplier prefix concatenated with the signage code:
`978` + `40703` = `97840703`, with `supplierId` 978040 for Miami.

**What `geometry` actually is.** A list of points, not a boundary. For zone 40703
it is 13 points spread over roughly 450 x 700 m, which is far too large to be one
block face -- these look like pay-station or meter positions inside a zone that
covers many blocks. So a zone is *not* the block-face unit the MUTCD default
implies; Miami numbers them much coarser. Treat the points as anchors and the
zone extent as unknown until the points are matched to block faces.

There is no bulk listing: `/api/locations` requires an `internalZoneCode`, and
supplier-only queries 404. Zones are therefore found by probing signage codes,
which is sparse -- 4 of 12 resolved around 40703.

Be considerate with the probe rate. This is a payment system's live API, the data
it returns is what is printed on street signs, and there is no reason to hammer
it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

BASE = "https://app.parkmobile.io/api"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
DEFAULT_DELAY_S = 0.6
"""Pause between requests. Deliberate: this is someone's production payment API."""


@dataclass
class Zone:
    signage_code: str
    internal_code: str
    supplier_id: str
    name: str
    zone_type: str
    points: list[tuple[float, float]] = field(default_factory=list)
    """(lon, lat) anchors inside the zone -- meters or pay stations, not a boundary."""

    restricted: bool = False

    @property
    def on_street(self) -> bool:
        return self.zone_type.lower() == "onstreet"

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        if not self.points:
            return None
        lons = [p[0] for p in self.points]
        lats = [p[1] for p in self.points]
        return min(lons), min(lats), max(lons), max(lats)

    @property
    def span_m(self) -> tuple[float, float]:
        """Rough extent, to show how far a zone is from being one block face."""
        import math

        b = self.bounds
        if b is None:
            return (0.0, 0.0)
        lat0 = (b[1] + b[3]) / 2
        return ((b[2] - b[0]) * 111_320 * math.cos(math.radians(lat0)),
                (b[3] - b[1]) * 110_540)


def _get(url: str, timeout: int = 20):
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Referer": "https://app.parkmobile.io/search",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def resolve(signage_code: int | str) -> tuple[str, str] | None:
    """Signage code -> (internalZoneCode, locationName), or None if unknown."""
    try:
        payload = _get(f"{BASE}/proxy/parkmobileapi/zones/{signage_code}")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    zones = (payload or {}).get("zones") or []
    if not zones:
        return None
    return zones[0].get("internalZoneCode"), zones[0].get("locationName")


def geometry(internal_code: str, supplier_id: str) -> list[dict] | None:
    try:
        return _get(f"{BASE}/locations?internalZoneCode={internal_code}"
                    f"&supplierId={supplier_id}")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def fetch(signage_code: int | str, supplier_id: str = "978040",
          delay: float = DEFAULT_DELAY_S) -> Zone | None:
    """Signage code -> a Zone with its anchor points, or None."""
    resolved = resolve(signage_code)
    if resolved is None:
        return None
    internal_code, name = resolved
    time.sleep(delay)

    records = geometry(internal_code, supplier_id) or []
    if not records:
        return Zone(str(signage_code), internal_code, supplier_id, name or "", "unknown")
    record = records[0]
    points = [
        (float(p["longitude"]), float(p["latitude"]))
        for p in (record.get("geometry") or [])
        if p.get("latitude") is not None and p.get("longitude") is not None
        # (0, 0) is null island, not a location. ParkMobile's own data carries
        # these -- zone 40708's single "coordinate" is exactly (0, 0) -- so a
        # consumer that trusts the field plots Miami parking in the Gulf of
        # Guinea. Dropped here rather than downstream, where it would blow out
        # every map extent it touched.
        if abs(float(p["latitude"])) > 1e-6 or abs(float(p["longitude"])) > 1e-6
    ]
    return Zone(
        signage_code=record.get("signageCode") or str(signage_code),
        internal_code=record.get("internalZoneCode") or internal_code,
        supplier_id=record.get("supplierId") or supplier_id,
        name=record.get("name") or name or "",
        zone_type=record.get("type") or "unknown",
        points=points,
        restricted=bool(record.get("isZoneRestricted")),
    )


def scan(codes, supplier_id: str = "978040", delay: float = DEFAULT_DELAY_S,
         on_hit=None) -> list[Zone]:
    """Walk a list of signage codes, keeping the ones that resolve."""
    found: list[Zone] = []
    for code in codes:
        zone = fetch(code, supplier_id=supplier_id, delay=delay)
        if zone is not None:
            found.append(zone)
            if on_hit:
                on_hit(zone)
        time.sleep(delay)
    return found


def to_geojson(zones: list[Zone]) -> dict:
    """Points as features. Not polygons -- the API gives anchors, not boundaries."""
    features = []
    for zone in zones:
        for index, (lon, lat) in enumerate(zone.points):
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "zone": zone.signage_code,
                    "internal_code": zone.internal_code,
                    "name": zone.name,
                    "zone_type": zone.zone_type,
                    "point_index": index,
                    "restricted": zone.restricted,
                    "source": "ParkMobile /api/locations",
                },
            })
    return {"type": "FeatureCollection", "features": features}
