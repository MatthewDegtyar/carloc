"""Appearance attributes sampled from imagery.

The pipeline's class posterior says *what* an object is. An operator asking for
"the white pickup" needs *which one*, and that is appearance, not class. This adds
the minimum needed to make that query answerable, and is honest about how coarse
it is.

Colour is sampled from the upper-middle of the box, not the whole of it. A vehicle
box contains road, shadow under the chassis, wheels and window glass; the body
panels sit in the upper middle. Sampling everything measures the tarmac as much as
the car.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fractions of the box used for sampling: horizontally centred, upper-middle
# vertically. Avoids wheels and under-chassis shadow at the bottom and roof
# glare at the very top.
BAND = (0.15, 0.55, 0.25, 0.75)  # y0, y1, x0, x1

MIN_PATCH_PX = 25
"""Fewest sampled pixels that may produce a colour at all.

Clipping the sample band to the image can leave a sliver: a box that has walked
off the edge of the frame clamps to a one-pixel strip, and the median of one
pixel is still a number, so it comes back as a confident tone. Tone feeds the
queries directly, so that one pixel could put a track in the "light vehicles"
answer. Below this floor the honest return is None -- unknown, and excluded --
rather than a colour with no support behind it."""

DARK = 0.30
LIGHT = 0.48
CHROMATIC = 0.25  # saturation above which a colour name means something


@dataclass(frozen=True)
class Appearance:
    value: float
    """Perceived lightness, 0-1."""

    saturation: float
    """0 = neutral (white/grey/black), 1 = vivid."""

    hue_deg: float
    rgb: tuple[float, float, float]
    n_pixels: int

    @property
    def tone(self) -> str:
        """Coarse tone label. Deliberately coarse -- three buckets it can defend."""
        if self.saturation >= CHROMATIC:
            return "coloured"
        if self.value <= DARK:
            return "dark"
        if self.value >= LIGHT:
            return "light"
        return "mid"

    @property
    def label(self) -> str:
        if self.saturation >= CHROMATIC:
            return _hue_name(self.hue_deg)
        return {"dark": "black/dark", "light": "white/light", "mid": "grey"}[self.tone]

    @property
    def confident(self) -> bool:
        """Enough pixels, and not sitting on a bucket boundary."""
        if self.n_pixels < 200:
            return False
        return not (DARK < self.value < DARK + 0.06 or LIGHT - 0.06 < self.value < LIGHT)


def _hue_name(hue: float) -> str:
    for lo, hi, name in (
        (0, 20, "red"), (20, 45, "orange"), (45, 70, "yellow"), (70, 160, "green"),
        (160, 200, "cyan"), (200, 260, "blue"), (260, 320, "purple"), (320, 360, "red"),
    ):
        if lo <= hue < hi:
            return name
    return "coloured"


def sample_appearance(image: np.ndarray, bbox) -> Appearance | None:
    """Median colour over the body-panel band of a detection box."""
    if image is None:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return None

    y0 = int(np.clip(y1 + bh * BAND[0], 0, height - 1))
    y3 = int(np.clip(y1 + bh * BAND[1], 0, height))
    x0 = int(np.clip(x1 + bw * BAND[2], 0, width - 1))
    x3 = int(np.clip(x1 + bw * BAND[3], 0, width))
    if y3 <= y0 or x3 <= x0:
        return None

    patch = np.asarray(image[y0:y3, x0:x3], dtype=float).reshape(-1, 3)
    if patch.shape[0] < MIN_PATCH_PX:
        return None

    # Median, not mean: a specular highlight or a dark window would drag a mean
    # across a bucket boundary on its own.
    r, g, b = np.median(patch, axis=0)
    mx, mn = max(r, g, b), min(r, g, b)
    saturation = float((mx - mn) / mx) if mx > 0 else 0.0

    if mx == mn:
        hue = 0.0
    elif mx == r:
        hue = (60 * ((g - b) / (mx - mn)) + 360) % 360
    elif mx == g:
        hue = 60 * ((b - r) / (mx - mn)) + 120
    else:
        hue = 60 * ((r - g) / (mx - mn)) + 240

    return Appearance(value=float(np.mean([r, g, b]) / 255.0), saturation=saturation,
                      hue_deg=float(hue), rgb=(float(r), float(g), float(b)),
                      n_pixels=int(patch.shape[0]))


class AppearanceMemory:
    """Accumulates appearance per track across frames.

    A single frame's sample is noisy -- sun, shadow, a passing reflection. Tracks
    persist, so the running median over a track's life is far steadier than any
    one look at it, and it is what a query should be answered from.
    """

    def __init__(self) -> None:
        self._values: dict[int, list[Appearance]] = {}

    def observe(self, track_id: int, appearance: Appearance | None) -> None:
        if appearance is not None:
            self._values.setdefault(track_id, []).append(appearance)

    def get(self, track_id: int) -> Appearance | None:
        seen = self._values.get(track_id)
        if not seen:
            return None
        return Appearance(
            value=float(np.median([a.value for a in seen])),
            saturation=float(np.median([a.saturation for a in seen])),
            hue_deg=float(np.median([a.hue_deg for a in seen])),
            rgb=tuple(np.median([a.rgb for a in seen], axis=0)),
            n_pixels=int(np.sum([a.n_pixels for a in seen])),
        )

    def n_observations(self, track_id: int) -> int:
        return len(self._values.get(track_id, []))
