"""Georeferenced satellite tiles, so an overlay can be checked rather than eyeballed.

A screenshot of a map has no defined extent, so anything drawn on top of it is
aligned by eye and proves nothing. Web Mercator tiles do have an exact extent --
a tile at (z, x, y) covers a known lon/lat box by definition -- so stitching
tiles gives a canvas whose corners are known to the metre, and a polygon drawn on
it lands where its coordinates say.

That is the difference between "these boxes look about right" and "this box is
here".

Imagery is Esri World Imagery, which is publicly served for viewing and must be
attributed on anything shown. Tiles are cached on disk so re-rendering does not
re-fetch, which is both faster and politer.
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import numpy as np

TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
ATTRIBUTION = "Imagery: Esri, Maxar, Earthstar Geographics"
TILE_PX = 256
CACHE = Path(".cache/tiles")
USER_AGENT = "carloc/0.1 (parking research; contact via repo)"


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    """Fractional tile coordinates. Web Mercator, the standard slippy-map scheme."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x: float, y: float, z: int) -> tuple[float, float]:
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def _fetch(z: int, x: int, y: int) -> np.ndarray | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{z}_{x}_{y}.jpg"
    if not path.exists():
        url = TILE_URL.format(z=z, x=x, y=y)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                path.write_bytes(response.read())
        except Exception:
            return None
    from PIL import Image

    try:
        with Image.open(path) as image:
            return np.array(image.convert("RGB"))
    except Exception:
        path.unlink(missing_ok=True)
        return None


def fetch_extent(west: float, south: float, east: float, north: float,
                 zoom: int = 19) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Stitch tiles covering a lon/lat box.

    Returns the mosaic and its **actual** extent, which is the union of whole
    tiles and therefore slightly larger than requested. Returning the real extent
    rather than the asked-for one is the whole point: draw with the former and
    everything lines up, draw with the latter and the overlay is subtly wrong.
    """
    x0f, y0f = lonlat_to_tile(west, north, zoom)
    x1f, y1f = lonlat_to_tile(east, south, zoom)
    x0, x1 = int(math.floor(x0f)), int(math.floor(x1f))
    y0, y1 = int(math.floor(y0f)), int(math.floor(y1f))

    columns, rows = x1 - x0 + 1, y1 - y0 + 1
    if columns * rows > 400:
        raise ValueError(f"{columns * rows} tiles requested; lower the zoom or "
                         "shrink the extent")

    mosaic = np.zeros((rows * TILE_PX, columns * TILE_PX, 3), dtype=np.uint8)
    for row in range(rows):
        for column in range(columns):
            tile = _fetch(zoom, x0 + column, y0 + row)
            if tile is not None:
                mosaic[row * TILE_PX:(row + 1) * TILE_PX,
                       column * TILE_PX:(column + 1) * TILE_PX] = tile[:, :, :3]

    west_actual, north_actual = tile_to_lonlat(x0, y0, zoom)
    east_actual, south_actual = tile_to_lonlat(x1 + 1, y1 + 1, zoom)
    return mosaic, (west_actual, east_actual, south_actual, north_actual)


def mercator_y(lat: float) -> float:
    """Web Mercator northing, normalised. Needed because tiles are equally spaced
    in *this*, not in latitude, so plotting against latitude directly bends the
    overlay away from the imagery as you move north."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    return math.degrees(math.asinh(math.tan(math.radians(lat))))


def imshow_mercator(ax, mosaic: np.ndarray, extent):
    """Draw a mosaic with a Mercator-correct y axis.

    Everything plotted afterwards must pass latitudes through `mercator_y`.
    """
    west, east, south, north = extent
    ax.imshow(mosaic, extent=[west, east, mercator_y(south), mercator_y(north)],
              origin="upper", interpolation="bilinear", zorder=1, aspect="auto")
    ax.set_xlim(west, east)
    ax.set_ylim(mercator_y(south), mercator_y(north))
