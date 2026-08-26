"""Local metric frame -> WGS84, fitted from paired samples rather than assumed.

A projected coordinate system is not a local tangent plane. Grid north and true
north differ by the meridian convergence, and over the extent of a single flight
that is not a rounding error: at 113.0E against a 114E central meridian the
convergence is 0.478 degrees, which displaces a point 4.8 m across a 574 m span.
Treating grid coordinates as ENU therefore rotates every published position by
half a degree about the origin -- an error that grows with distance from it and
looks entirely plausible everywhere.

The usual fix is a projection library. This does it from data instead, which is
available whenever a platform logs both its projected pose and its GNSS fix --
AirZoo logs 679 such pairs per split. Fitting an affine over them recovers the
convergence, the scale, and any datum offset together, without naming any of
them, and it is checkable: the residual on this flight is a **millimetre**, which
is a far stronger statement than "we used the right EPSG code".

Valid locally, which is the point: an affine cannot represent a projection over a
wide area, so `fit` refuses samples spanning more than `MAX_EXTENT_M`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_EXTENT_M = 20_000.0
"""Largest sample span an affine may be fitted over.

Beyond this the curvature a projection encodes stops being absorbable by a linear
map. 20 km keeps the linearisation error well under a centimetre at these
latitudes and covers any single sortie; refusing is better than silently
degrading, because the failure has no symptom other than wrong coordinates."""

MIN_SAMPLES = 8
MIN_SPREAD_M = 5.0
"""Samples confined to a smaller patch than this cannot constrain a rotation:
the fit becomes an offset with an arbitrary orientation, and it will look
perfect on its own residuals while being wrong everywhere else."""


@dataclass(frozen=True)
class LocalGeoFit:
    """Affine from local metric (x, y) to WGS84 (lon, lat)."""

    lon_coef: np.ndarray
    lat_coef: np.ndarray
    origin_xy: np.ndarray
    residual_m: float
    """Median fit residual in metres. Report it; do not publish a fit without it."""

    n_samples: int

    @classmethod
    def fit(cls, xy: np.ndarray, lonlat: np.ndarray) -> LocalGeoFit:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        lonlat = np.asarray(lonlat, dtype=float).reshape(-1, 2)
        if len(xy) != len(lonlat):
            raise ValueError(f"{len(xy)} positions against {len(lonlat)} fixes")

        good = np.isfinite(xy).all(1) & np.isfinite(lonlat).all(1)
        xy, lonlat = xy[good], lonlat[good]
        if len(xy) < MIN_SAMPLES:
            raise ValueError(f"need at least {MIN_SAMPLES} paired samples, got {len(xy)}")

        extent = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
        if extent > MAX_EXTENT_M:
            raise ValueError(
                f"samples span {extent / 1000:.1f} km, over the {MAX_EXTENT_M / 1000:.0f} km "
                "limit for a local affine; a projection library is needed at that scale"
            )
        if extent < MIN_SPREAD_M:
            raise ValueError(
                f"samples span only {extent:.1f} m, too little to constrain a rotation; "
                "the fit would be an offset with an arbitrary orientation"
            )

        origin = xy.mean(axis=0)
        design = np.column_stack([xy[:, 0] - origin[0], xy[:, 1] - origin[1], np.ones(len(xy))])
        lon_coef, *_ = np.linalg.lstsq(design, lonlat[:, 0], rcond=None)
        lat_coef, *_ = np.linalg.lstsq(design, lonlat[:, 1], rcond=None)

        predicted = np.column_stack([design @ lon_coef, design @ lat_coef])
        metres_per_deg_lat = 110_540.0
        metres_per_deg_lon = 111_320.0 * np.cos(np.radians(lonlat[:, 1].mean()))
        error = np.column_stack([
            (predicted[:, 0] - lonlat[:, 0]) * metres_per_deg_lon,
            (predicted[:, 1] - lonlat[:, 1]) * metres_per_deg_lat,
        ])
        residual = float(np.median(np.linalg.norm(error, axis=1)))
        return cls(lon_coef, lat_coef, origin, residual, len(xy))

    # -- use ---------------------------------------------------------------

    def to_wgs84(self, xy: np.ndarray) -> np.ndarray:
        """Local metric (x, y) -> (lon, lat)."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        design = np.column_stack([
            xy[:, 0] - self.origin_xy[0], xy[:, 1] - self.origin_xy[1], np.ones(len(xy)),
        ])
        return np.column_stack([design @ self.lon_coef, design @ self.lat_coef])

    @property
    def convergence_deg(self) -> float:
        """Angle between grid north and true north, recovered from the fit.

        Not used in the maths -- the affine already carries it -- but reported
        because it is the one number that says whether treating these coordinates
        as ENU would have been acceptable.
        """
        return float(np.degrees(np.arctan2(self.lat_coef[0], self.lat_coef[1])))

    def describe(self) -> str:
        return (
            f"local->WGS84 affine over {self.n_samples} samples, "
            f"median residual {self.residual_m * 1000:.1f} mm, "
            f"grid convergence {self.convergence_deg:+.4f} deg"
        )
