"""The ground, as a height raster, and rays cast against it.

Georeferencing comes from an ESRI world file (`.tfw`) rather than an embedded
GeoTIFF header, so no GDAL is needed: six numbers defining a pixel-to-world
affine, which for a north-up raster reduces to a scale and an offset.

It is 2.5-D -- one height per cell. No overhangs, no bridge decks with road
underneath, no building facades. A ray that should pass under a bridge stops on
top of it. For a camera looking down that is nearly always the right trade, and
it fails predictably on building silhouettes rather than randomly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NODATA_BELOW = -1000.0
"""Heights below this are absent, not low.

Photogrammetric surfaces mark unreconstructed cells with a sentinel -- these use
-9999 -- and water is the usual cause, having no stable texture to match. Left
unmasked, a sentinel averaged into a height puts the ground a kilometre down."""


@dataclass(frozen=True)
class GeoTransform:
    """Pixel-to-world affine, north-up, from a world file."""

    pixel_width: float
    pixel_height: float
    x_origin: float
    y_origin: float

    @classmethod
    def from_tfw(cls, path: str | Path) -> GeoTransform:
        values = [float(v) for v in Path(path).read_text().split()]
        if len(values) != 6:
            raise ValueError(f"a world file has 6 numbers, {path} has {len(values)}")
        pixel_width, rot_y, rot_x, pixel_height, x_origin, y_origin = values
        if abs(rot_x) > 1e-12 or abs(rot_y) > 1e-12:
            raise ValueError(
                f"{path} carries rotation terms; only north-up rasters are handled, "
                "and ignoring them would misplace every height by a bearing-dependent "
                "amount rather than failing"
            )
        return cls(pixel_width, pixel_height, x_origin, y_origin)


class Terrain:
    """A georeferenced height field with ray intersection."""

    def __init__(self, heights: np.ndarray, transform: GeoTransform, crs: str = "") -> None:
        self.heights = np.asarray(heights, dtype=np.float64)
        if self.heights.ndim != 2:
            raise ValueError(f"expected a single band, got shape {self.heights.shape}")
        self.heights[self.heights < NODATA_BELOW] = np.nan
        self.transform = transform
        self.crs = crs

    @classmethod
    def load(cls, tif: str | Path) -> Terrain:
        from PIL import Image

        tif = Path(tif)
        # A city-scale raster at half-metre spacing trips Pillow's
        # decompression-bomb guard, which is for untrusted uploads.
        previous, Image.MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS, None
        try:
            heights = np.array(Image.open(tif))
        finally:
            Image.MAX_IMAGE_PIXELS = previous
        prj = tif.with_suffix(".prj")
        return cls(heights, GeoTransform.from_tfw(tif.with_suffix(".tfw")),
                   crs=prj.read_text().strip() if prj.exists() else "")

    # -- queries -----------------------------------------------------------

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        rows, cols = self.heights.shape
        xs = (self.transform.x_origin,
              self.transform.x_origin + cols * self.transform.pixel_width)
        ys = (self.transform.y_origin,
              self.transform.y_origin + rows * self.transform.pixel_height)
        return min(xs), min(ys), max(xs), max(ys)

    def height_at(self, x, y) -> np.ndarray:
        """Height at world (x, y). NaN outside the raster or over a hole.

        Nearest-neighbour, not bilinear: interpolating across the edge of a hole
        blends a real height with a NaN, and at half-metre spacing the difference
        is well inside the surface's own accuracy anyway.
        """
        col = (np.asarray(x, float) - self.transform.x_origin) / self.transform.pixel_width
        row = (np.asarray(y, float) - self.transform.y_origin) / self.transform.pixel_height
        rows, cols = self.heights.shape
        ci = np.clip(np.round(col).astype(int), 0, cols - 1)
        ri = np.clip(np.round(row).astype(int), 0, rows - 1)
        inside = (col >= -0.5) & (col <= cols - 0.5) & (row >= -0.5) & (row <= rows - 0.5)
        # np.where keeps a scalar query scalar, which the ray-cast bisection needs.
        return np.where(inside, self.heights[ri, ci], np.nan)

    def raycast(self, origin: np.ndarray, direction: np.ndarray,
                max_distance: float = 2000.0, step: float = 1.0, refine: int = 12) -> float:
        """Distance to the first surface crossing, or NaN.

        March at a fixed step, then bisect. A closed form exists only for a
        plane; on a height field any single-shot approximation walks through a
        ridge that happens to fall between samples. The bisection recovers
        sub-step precision, so the step controls what is *missed*, not how
        precisely a hit is placed.
        """
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            return float("nan")
        direction = direction / norm

        distances = np.arange(step, max_distance + step, step)
        points = origin[None, :] + distances[:, None] * direction[None, :]
        ground = self.height_at(points[:, 0], points[:, 1])
        below = points[:, 2] < ground        # NaN compares False: holes are not hits
        hits = np.flatnonzero(below)
        if not len(hits):
            return float("nan")

        index = int(hits[0])
        near, far = distances[index] - step, float(distances[index])
        for _ in range(refine):
            middle = 0.5 * (near + far)
            point = origin + middle * direction
            height = float(self.height_at(point[0], point[1]))
            if np.isnan(height) or point[2] >= height:
                near = middle
            else:
                far = middle
        return float(0.5 * (near + far))

    def translated(self, offset: np.ndarray) -> Terrain:
        """A copy shifted by ``-offset``, for a session with a local origin.

        Subtracting an origin from the poses moves the cameras and not the
        ground, so rays start outside the raster and every one misses -- which
        shows up as a ranger that refuses everything rather than as a crash.
        """
        offset = np.asarray(offset, dtype=float).reshape(3)
        moved = GeoTransform(self.transform.pixel_width, self.transform.pixel_height,
                             self.transform.x_origin - offset[0],
                             self.transform.y_origin - offset[1])
        shifted = Terrain.__new__(Terrain)
        shifted.heights = self.heights - offset[2]
        shifted.transform = moved
        shifted.crs = self.crs
        return shifted

    def coverage(self) -> float:
        return float(np.isfinite(self.heights).mean())
