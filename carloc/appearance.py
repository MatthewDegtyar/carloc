"""Name a vehicle's colour the way a person would, not by RGB nearest-neighbour.

The first attempt matched each car's median RGB to the closest of eight fixed
colours. It failed in a specific, instructive way: a dark neutral car in shade
has a median like (54, 67, 75) -- a hair of blue-green from the sky and the
foliage above it -- and in plain RGB distance that sits closer to a saturated
"green" anchor than to "black". Nine of forty cars came back green, and none of
them were.

The fix is to separate *how colourful* a car is from *what colour*. Almost every
car on a street is achromatic -- black, grey, silver, white -- and those are told
apart by brightness, not hue. Only a genuinely saturated car has a colour name at
all. So: gate on saturation first, and read hue only once it is high enough to
mean something. A faint tint no longer outvotes brightness.
"""

from __future__ import annotations

import colorsys

# Saturation below this is treated as achromatic. Set above the tinted-neutral
# band (the mislabelled cars topped out at s=0.30) and below a real colour: the
# one true blue car in the SE 6th pass sits at s=0.36.
ACHROMATIC_S = 0.33


def classify_colour(rgb: tuple[int, int, int]) -> str:
    """A coarse, human colour name for a vehicle's dominant RGB.

    Coarse on purpose: a re-identification key wants "silver vs black vs red",
    agreed on by two passes under different light, not a paint-chip match no two
    frames would ever reproduce.
    """
    r, g, b = (c / 255.0 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue = h * 360.0

    if s < ACHROMATIC_S:
        # Neutral: brightness alone. These cut points are eyeballed against the
        # SE 6th cars -- dark bodies land near v=0.1-0.2, mid greys near 0.45,
        # a silver flank near 0.6, white above.
        if v < 0.25:
            return "black"
        if v < 0.52:
            return "grey"
        if v < 0.75:
            return "silver"
        return "white"

    # Saturated enough to have a real colour; read it off the hue wheel.
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 45:
        return "tan" if v < 0.6 else "orange"
    if hue < 65:
        return "tan"
    if hue < 170:
        return "green"
    if hue < 260:
        return "blue"
    return "purple"


# Display swatch per name, brightened so a pin reads on dark satellite imagery.
SWATCH = {
    "black": "#3a3a40", "grey": "#8b8f98", "silver": "#c3c6cc", "white": "#eef0f2",
    "red": "#e8574d", "orange": "#e08a3c", "tan": "#cbb083", "green": "#54c47d",
    "blue": "#5f84e0", "purple": "#a87ad8",
}


def dominant_rgb(crop) -> tuple[int, int, int]:
    """Median colour of the box's central body, edges and glare trimmed.

    The median rejects windscreen glare and the strip of road showing under the
    car; the central crop keeps the background at the box edges out of the vote.
    """
    import numpy as np

    if crop is None or getattr(crop, "size", 0) == 0:
        return (110, 112, 115)
    h, w = crop.shape[:2]
    inner = crop[int(h * 0.25):int(h * 0.75), int(w * 0.2):int(w * 0.8)]
    if inner.size == 0:
        inner = crop
    return tuple(int(v) for v in np.median(inner.reshape(-1, 3), axis=0))
