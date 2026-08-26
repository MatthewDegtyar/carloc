"""Digital surface model: a georeferenced height field, and rays cast against it.

A DSM is the 2.5-D generalisation of the flat ground plane. It holds one height
per ground cell, which is enough to range against real terrain -- slopes, embankments,
building roofs -- without the flat-earth assumption that a plane forces.

"2.5-D" is the honest description and the limitation worth stating: one height per
cell means no overhangs, no bridge decks with road underneath, and no building
facades. A ray that should pass under a bridge instead stops on top of it. For
downward-looking aerial work that is nearly always the right trade; for anything
looking sideways it is not.

Georeferencing comes from an ESRI world file (``.tfw``) rather than an embedded
GeoTIFF header, so this needs no GDAL. A world file is six numbers defining the
affine from pixel to world, and for a north-up raster -- which these are -- the
rotation terms are zero and it reduces to a scale and an offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NODATA_BELOW = -1000.0
"""Heights below this are treated as absent.

Photogrammetric DSMs mark unreconstructed cells with a sentinel -- these use
-9999 -- and water is the usual cause: it has no stable texture to match, so
lakes and rivers come back as holes. A sentinel silently averaged into a height
would put the surface a kilometre underground, so it is masked to NaN on load and
every consumer has to handle absence explicitly."""


@dataclass(frozen=True)
class GeoTransform:
    """Pixel-to-world affine, north-up. From an ESRI world file."""

    pixel_width: float
    pixel_height: float
    """Negative for a north-up raster: row index grows as northing falls."""

    x_origin: float
    y_origin: float
    """World coordinate of the *centre* of the top-left pixel, per the .tfw spec."""

    @classmethod
    def from_tfw(cls, path: str | Path) -> GeoTransform:
        values = [float(line) for line in Path(path).read_text().split()]
        if len(values) != 6:
            raise ValueError(f"a world file has 6 numbers, {path} has {len(values)}")
        pixel_width, rot_y, rot_x, pixel_height, x_origin, y_origin = values
        if abs(rot_x) > 1e-12 or abs(rot_y) > 1e-12:
            raise ValueError(
                f"{path} carries rotation terms ({rot_x}, {rot_y}); only north-up "
                "rasters are supported, and silently ignoring them would misplace "
                "every height by a bearing-dependent amount"
            )
        return cls(pixel_width, pixel_height, x_origin, y_origin)

    def world_to_pixel(self, x, y):
        """World coordinates -> fractional (column, row)."""
        return (np.asarray(x, float) - self.x_origin) / self.pixel_width, \
               (np.asarray(y, float) - self.y_origin) / self.pixel_height


class DigitalSurfaceModel:
    """A georeferenced height raster with ray intersection."""

    def __init__(self, heights: np.ndarray, transform: GeoTransform, crs: str = "") -> None:
        self.heights = np.asarray(heights, dtype=np.float64)
        if self.heights.ndim != 2:
            raise ValueError(f"a DSM is a single band, got shape {self.heights.shape}")
        self.heights[self.heights < NODATA_BELOW] = np.nan
        self.transform = transform
        self.crs = crs

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, tif_path: str | Path, tfw_path: str | Path | None = None,
             prj_path: str | Path | None = None) -> DigitalSurfaceModel:
        from PIL import Image

        tif_path = Path(tif_path)
        tfw_path = Path(tfw_path) if tfw_path else tif_path.with_suffix(".tfw")
        prj_path = Path(prj_path) if prj_path else tif_path.with_suffix(".prj")

        # A city-scale DSM at half-metre spacing exceeds Pillow's decompression-bomb
        # guard, which exists for untrusted uploads and is not meaningful here.
        previous, Image.MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS, None
        try:
            heights = np.array(Image.open(tif_path))
        finally:
            Image.MAX_IMAGE_PIXELS = previous

        crs = prj_path.read_text().strip() if prj_path.exists() else ""
        return cls(heights, GeoTransform.from_tfw(tfw_path), crs=crs)

    # -- queries -----------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.heights.shape

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) in world units."""
        rows, cols = self.heights.shape
        xs = (self.transform.x_origin, self.transform.x_origin + cols * self.transform.pixel_width)
        ys = (self.transform.y_origin, self.transform.y_origin + rows * self.transform.pixel_height)
        return min(xs), min(ys), max(xs), max(ys)

    def height_at(self, x, y) -> np.ndarray:
        """Surface height at world (x, y). NaN outside the raster or over nodata.

        Nearest-neighbour rather than bilinear: interpolating across the edge of a
        nodata hole would blend a real height with a NaN and quietly produce one or
        the other depending on the interpolation order. At half-metre spacing the
        difference is well inside the surface's own accuracy.
        """
        col, row = self.transform.world_to_pixel(x, y)
        rows, cols = self.heights.shape
        ci = np.clip(np.round(col).astype(int), 0, cols - 1)
        ri = np.clip(np.round(row).astype(int), 0, rows - 1)
        inside = (col >= -0.5) & (col <= cols - 0.5) & (row >= -0.5) & (row <= rows - 0.5)
        # np.where rather than masked assignment so a scalar query stays scalar --
        # the ray-cast bisection queries one point at a time.
        return np.where(inside, self.heights[ri, ci], np.nan)

    def raycast(self, origin: np.ndarray, direction: np.ndarray, max_distance: float = 2000.0,
                step: float = 1.0, refine: int = 12) -> float:
        """Distance along ``direction`` to the first surface crossing, or NaN.

        March at a fixed step until the ray is below the surface, then bisect. The
        march is what makes this robust on a height field: a closed-form solution
        exists only for a plane, and any single-shot approximation walks straight
        through a ridge that happens to sit between two samples.

        Step size trades accuracy for cost, but only up to a point -- the bisection
        recovers sub-step precision, so the step really controls whether a thin
        feature is *missed*, not how precisely a hit is located.
        """
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return float("nan")
        direction = direction / norm

        distances = np.arange(step, max_distance + step, step)
        points = origin[None, :] + distances[:, None] * direction[None, :]
        ground = self.height_at(points[:, 0], points[:, 1])
        below = points[:, 2] < ground          # NaN compares False: holes are not hits
        hits = np.flatnonzero(below)
        if not len(hits):
            return float("nan")

        # Bisect between the last point above the surface and the first below it.
        index = int(hits[0])
        near = distances[index] - step
        far = float(distances[index])
        for _ in range(refine):
            middle = 0.5 * (near + far)
            point = origin + middle * direction
            height = float(self.height_at(point[0], point[1]))
            if np.isnan(height) or point[2] >= height:
                near = middle
            else:
                far = middle
        return float(0.5 * (near + far))

    def translated(self, offset: np.ndarray) -> DigitalSurfaceModel:
        """A copy shifted by ``-offset``, for use in a local frame.

        A session that subtracts a local origin from its poses -- as this one does,
        because projected coordinates near 3.1e6 waste precision in covariance
        arithmetic -- moves the cameras but not the terrain. Rays then start far
        outside the raster and every one of them misses, which surfaces as a
        ranger that refuses everything rather than as an obvious crash. The frame
        convention belongs to whoever set the origin, so the loader hands out a
        surface already in it.
        """
        offset = np.asarray(offset, dtype=float).reshape(3)
        moved = GeoTransform(
            pixel_width=self.transform.pixel_width,
            pixel_height=self.transform.pixel_height,
            x_origin=self.transform.x_origin - offset[0],
            y_origin=self.transform.y_origin - offset[1],
        )
        shifted = DigitalSurfaceModel.__new__(DigitalSurfaceModel)
        shifted.heights = self.heights - offset[2]
        shifted.transform = moved
        shifted.crs = f"{self.crs} (translated to a local frame)" if self.crs else ""
        return shifted

    def coverage(self) -> float:
        """Fraction of cells carrying a real height. The rest is water and shadow."""
        return float(np.isfinite(self.heights).mean())
