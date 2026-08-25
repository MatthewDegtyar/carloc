"""Range from apparent object size.

If you know roughly how tall a car is, its height in pixels tells you roughly how
far away it is: ``R = f * H_real / h_pixels``. That is a weak estimate -- real
cars vary, boxes are noisy, and a truncated box is badly wrong -- but "weak" is
not "useless". A prior good to +/- 35% is enough to place a new track within a
factor of two instead of at an arbitrary default, and that is the difference
between an association gate that admits the right detection and one that admits
every detection in the frame.

It is produced as a ``range_prior``, never as a ``range``. Fusing a class-size
assumption as a measurement on every frame would let the filter converge
confidently onto the assumption itself.
"""

from __future__ import annotations

import numpy as np

from geoloc_agent.contracts import Detection, Intrinsics, RangeMeas, RangeMethod

# Typical real-world height in metres, and the fractional spread within the class.
# Heights are used rather than widths because height is invariant to viewing
# aspect -- a car seen end-on is a third as wide but exactly as tall.
CLASS_HEIGHTS: dict[str, tuple[float, float]] = {
    "car": (1.55, 0.18),
    "truck": (3.20, 0.45),
    "bus": (3.30, 0.25),
    "trailer": (3.50, 0.35),
    "pedestrian": (1.70, 0.12),
    "person": (1.70, 0.12),
    "bicycle": (1.20, 0.25),
    "motorcycle": (1.40, 0.25),
    "construction": (2.50, 0.50),
}

MIN_PIXELS = 6.0
BBOX_SIGMA_PX = 6.0
MAX_RANGE_M = 200.0

HULL_SPREAD = 0.25
"""A 2-D box is the convex hull of a 3-D object, not a measurement of its height.

Its near-top corner projects higher than the object's centroid-depth height, so
the box is systematically taller than ``f*H/R`` predicts and the implied range
comes out short. Measured against nuScenes truth this runs about 0.80x for cars
and 0.87x for pedestrians, varying with viewing aspect. It is a modelling error,
not noise, and the honest response for a *prior* is to carry a spread wide enough
to contain it -- not to fit a per-class correction factor to one detector on one
scene, which would not survive contact with a real detector."""


def range_prior_from_size(
    detection: Detection,
    intrinsics: Intrinsics,
    class_heights: dict[str, tuple[float, float]] | None = None,
    touching_edge_tolerance: float = 2.0,
) -> RangeMeas:
    """Coarse range from bounding-box height, with an honest sigma.

    Returns an invalid measurement rather than a guess when the assumption does
    not hold: an unknown class, a box too small to measure, or a box touching the
    image edge (which is truncated, so its height is a lower bound and the range
    would be over-estimated).
    """
    table = class_heights or CLASS_HEIGHTS
    entry = table.get(detection.cls)
    if entry is None:
        return RangeMeas.invalid(RangeMethod.MONO_DEPTH, f"no size prior for '{detection.cls}'")

    height_m, spread = entry
    h_px = detection.height
    if h_px < MIN_PIXELS:
        return RangeMeas.invalid(RangeMethod.MONO_DEPTH, f"box only {h_px:.1f} px tall")

    x1, y1, x2, y2 = detection.bbox
    if (
        y1 <= touching_edge_tolerance
        or y2 >= intrinsics.height - touching_edge_tolerance
    ):
        return RangeMeas.invalid(RangeMethod.MONO_DEPTH, "box is vertically truncated")

    value = float(intrinsics.fy * height_m / h_px)
    if not np.isfinite(value) or value <= 0 or value > MAX_RANGE_M:
        return RangeMeas.invalid(RangeMethod.MONO_DEPTH, f"implied range {value:.0f} m implausible")

    # Two independent error sources, both multiplicative in range: how much the
    # real object's height varies within its class, and how well the box height
    # is measured. Combined in quadrature.
    relative = float(np.sqrt(spread**2 + (BBOX_SIGMA_PX / h_px) ** 2 + HULL_SPREAD**2))
    return RangeMeas(
        value=value,
        sigma=max(value * relative, 0.5),
        method=RangeMethod.MONO_DEPTH,
        valid=True,
    )
