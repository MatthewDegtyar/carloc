"""Local ENU <-> WGS84 conversion.

Kept deliberately small and explicit. The reason it exists as its own module is
the nuScenes gotcha: nuScenes poses are in a per-map local frame in metres with
no lat/lon anywhere. We pick a documented origin per map and convert with a
local tangent-plane transform. That is an assumption, not a measurement, and it
is stated as such wherever a lat/lon is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

WGS84_A = 6378137.0  # semi-major axis, metres
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass(frozen=True)
class GeoOrigin:
    """Origin of a local ENU frame, plus provenance for how we got it."""

    lat: float
    lon: float
    alt: float = 0.0
    name: str = ""
    provenance: str = "assumed"

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"latitude out of range: {self.lat}")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"longitude out of range: {self.lon}")

    def enu_to_wgs84(self, enu: np.ndarray) -> tuple[float, float, float]:
        """ENU metres -> (lat, lon, alt). Good to well under a metre at city scale."""
        east, north, up = (float(v) for v in np.asarray(enu, dtype=float).reshape(3))
        lat0 = math.radians(self.lat)
        sin_lat = math.sin(lat0)
        denom = math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        # Meridional and normal radii of curvature at the origin latitude.
        m_per_deg_lat = (WGS84_A * (1.0 - WGS84_E2) / denom**3) * math.pi / 180.0
        m_per_deg_lon = (WGS84_A / denom) * math.cos(lat0) * math.pi / 180.0
        return (self.lat + north / m_per_deg_lat, self.lon + east / m_per_deg_lon, self.alt + up)

    def wgs84_to_enu(self, lat: float, lon: float, alt: float = 0.0) -> np.ndarray:
        lat0 = math.radians(self.lat)
        sin_lat = math.sin(lat0)
        denom = math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        m_per_deg_lat = (WGS84_A * (1.0 - WGS84_E2) / denom**3) * math.pi / 180.0
        m_per_deg_lon = (WGS84_A / denom) * math.cos(lat0) * math.pi / 180.0
        return np.array(
            [(lon - self.lon) * m_per_deg_lon, (lat - self.lat) * m_per_deg_lat, alt - self.alt]
        )


# Documented origins for the nuScenes maps. These are the published map-region
# centres, not surveyed control points: nuScenes ships no georeference, so any
# lat/lon we emit is accurate *relative* to this assumed origin and no better
# than roughly city-block absolute. Never present these as a survey fix.
NUSCENES_ORIGINS = {
    "boston-seaport": GeoOrigin(
        42.336849, -71.05785, 0.0, "boston-seaport", "assumed map-region centre"
    ),
    "singapore-onenorth": GeoOrigin(
        1.2882100, 103.7891, 0.0, "singapore-onenorth", "assumed map-region centre"
    ),
    "singapore-queenstown": GeoOrigin(
        1.2993030, 103.7855, 0.0, "singapore-queenstown", "assumed map-region centre"
    ),
    "singapore-hollandvillage": GeoOrigin(
        1.3072920, 103.7930, 0.0, "singapore-hollandvillage", "assumed map-region centre"
    ),
}
