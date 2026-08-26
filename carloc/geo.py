"""Local metric frame to WGS84, fitted from paired samples and scored held-out.

A projected coordinate system is not a local tangent plane. Grid north and true
north differ by the meridian convergence, which at 113.0E against a 114E central
meridian is 0.478 degrees -- enough to displace a point about 5 m across a 600 m
flight. Treating grid coordinates as ENU rotates every published position about
the origin by half a degree, an error that grows with distance and looks
plausible everywhere.

The usual answer is a projection library. This fits an affine from the pose/GNSS
pairs the dataset already carries, which recovers convergence, scale and any
datum offset together without naming them -- and, unlike naming an EPSG code,
produces a **residual in metres** that can be reported.

Fitted on half the samples and scored on the other half. A fit scored on its own
training points cannot detect that it has absorbed a systematic error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_EXTENT_M = 20_000.0
"""Widest sample span an affine may cover before projection curvature stops
being absorbable. Refusing beats degrading silently: the failure has no symptom
other than wrong coordinates."""

MIN_SAMPLES = 8
MIN_SPREAD_M = 5.0
"""Samples in a smaller patch cannot constrain a rotation. The fit becomes an
offset with an arbitrary orientation and looks perfect on its own residuals."""

METRES_PER_DEG_LAT = 110_540.0


@dataclass(frozen=True)
class GeoFit:
    """Affine from local metric (x, y) to WGS84 (lon, lat)."""

    lon_coef: np.ndarray
    lat_coef: np.ndarray
    origin_xy: np.ndarray
    train_residual_m: float
    holdout_residual_m: float
    """Median error on samples the fit never saw. This is the number to quote."""

    n_train: int
    n_holdout: int

    @classmethod
    def fit(cls, xy: np.ndarray, lonlat: np.ndarray, holdout: float = 0.5,
            seed: int = 0) -> GeoFit:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        lonlat = np.asarray(lonlat, dtype=float).reshape(-1, 2)
        good = np.isfinite(xy).all(1) & np.isfinite(lonlat).all(1)
        xy, lonlat = xy[good], lonlat[good]
        if len(xy) < MIN_SAMPLES:
            raise ValueError(f"need {MIN_SAMPLES}+ paired samples, got {len(xy)}")

        extent = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
        if extent > MAX_EXTENT_M:
            raise ValueError(f"samples span {extent / 1000:.1f} km, past the local-affine limit")
        if extent < MIN_SPREAD_M:
            raise ValueError(f"samples span {extent:.1f} m, too little to fix a rotation")

        order = np.random.default_rng(seed).permutation(len(xy))
        cut = max(MIN_SAMPLES, int(len(xy) * (1.0 - holdout)))
        train, test = order[:cut], order[cut:]

        origin = xy[train].mean(axis=0)
        design = cls._design(xy[train], origin)
        lon_coef, *_ = np.linalg.lstsq(design, lonlat[train, 0], rcond=None)
        lat_coef, *_ = np.linalg.lstsq(design, lonlat[train, 1], rcond=None)

        fit = cls(lon_coef, lat_coef, origin, 0.0, 0.0, len(train), len(test))
        train_residual = fit._residual(xy[train], lonlat[train])
        holdout_residual = fit._residual(xy[test], lonlat[test]) if len(test) else float("nan")
        return cls(lon_coef, lat_coef, origin, train_residual, holdout_residual,
                   len(train), len(test))

    @staticmethod
    def _design(xy: np.ndarray, origin: np.ndarray) -> np.ndarray:
        return np.column_stack([xy[:, 0] - origin[0], xy[:, 1] - origin[1], np.ones(len(xy))])

    def _residual(self, xy: np.ndarray, lonlat: np.ndarray) -> float:
        if not len(xy):
            return float("nan")
        predicted = self.to_wgs84(xy)
        per_lon = 111_320.0 * np.cos(np.radians(lonlat[:, 1].mean()))
        error = np.column_stack([(predicted[:, 0] - lonlat[:, 0]) * per_lon,
                                 (predicted[:, 1] - lonlat[:, 1]) * METRES_PER_DEG_LAT])
        return float(np.median(np.linalg.norm(error, axis=1)))

    # -- use ---------------------------------------------------------------

    def to_wgs84(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        design = self._design(xy, self.origin_xy)
        return np.column_stack([design @ self.lon_coef, design @ self.lat_coef])

    @property
    def convergence_deg(self) -> float:
        """Angle between grid north and true north, recovered from the fit.

        Not used in the arithmetic -- the affine already carries it -- but it is
        the number that says whether treating these as ENU would have been safe.
        """
        return float(np.degrees(np.arctan2(self.lat_coef[0], self.lat_coef[1])))

    def describe(self) -> str:
        return (f"WGS84 affine: {self.n_train} train / {self.n_holdout} held out, "
                f"held-out residual {self.holdout_residual_m * 100:.1f} cm, "
                f"grid convergence {self.convergence_deg:+.4f} deg")
