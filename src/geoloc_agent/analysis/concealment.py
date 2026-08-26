"""Which objects could conceal a person, and what ground they hide.

The question an operator actually asks is not "what is in this scene" but "where
could someone be that I cannot see". That is a geometry question and the pipeline
already has the inputs: a bearing, a range, and an apparent size.

Physical extent is derived from the detection box and the estimated range --
`w = w_px * R / fx` -- so this uses only what the pipeline produces. It does not
read object dimensions from the dataset annotations, which would make the demo
look better and mean nothing.

Everything here is a *lower bound on visibility*, not a claim about occupancy: it
says where a person could be hidden, not that anyone is there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geoloc_agent.range.size_prior import CLASS_HEIGHTS

# A person to hide. Crouching is the harder case and therefore the interesting
# one -- an object that conceals a standing adult is obvious.
PERSON_CROUCH_M = 1.10
PERSON_STAND_M = 1.75

MIN_WIDTH_M = 1.50
"""An object must be substantially wider than a person to hide one.

Calibrated against derived widths pooled over four nuScenes scenes (n=311
pedestrians, n=960 cars), using true range so the gate is not tuned around
ranging error:

    pedestrian   p10 0.72   median 0.98   p90 1.22 m
    car          p10 3.29   median 4.89   p90 5.79 m

A person's derived width sits right on 1.0 m -- the projected hull is ~1.45x the
true shoulder width, because limbs and box padding are included. So a 1.0 m gate
admits 47% of pedestrians as "concealment", which is not an answer an operator
can use: you cannot meaningfully hide behind someone your own size. At 1.5 m the
figure is 1.3%, while every car and truck still passes. The gap between the two
populations is wide enough that the exact threshold inside it barely matters."""

HULL_MARGIN = 1.15
"""Guard against the 2-D box being the hull of a 3-D object.

A bounding box stands taller than the object inside it. Measured against nuScenes
annotated heights at true range, the ratio of derived to true height is stable
across classes:

    pedestrian  1.10      car  1.15      truck  1.14

A standing person is 1.75 m and a sedan roof is about 1.5 m, so a 15% bias alone
flips ordinary cars from "cannot conceal" to "can", which would make the query
match essentially everything and mean nothing. Requiring a margin above the person
height means an object only qualifies when it clears the threshold by more than the
measurement error can account for.

The cost is the other direction: something genuinely 1.8 m tall is reported as not
concealing a standing person. For this query that is the right way to be wrong --
it under-claims rather than over-claims."""


CLASS_CONSISTENCY = 1.30
"""How far a derived height may exceed what its class allows before the whole
derivation is treated as untrustworthy rather than as a finding.

The pipeline knows two independent things about a track: what it is (the class
posterior) and how big it appears (box plus range). When those disagree badly,
one of them is wrong -- usually the box, adopted from a neighbouring object, or a
range that has drifted. A pedestrian track deriving 3.4 m tall is not a very tall
pedestrian; it is a broken pairing. The honest output there is no answer, not a
confident claim about concealment built on the broken half.

The bound is the class's typical height, widened by its own spread and by
HULL_MARGIN, then by this factor. Anything past it is reported as inconsistent."""


@dataclass(frozen=True)
class Concealment:
    width_m: float
    height_m: float
    range_m: float
    hides_crouching: bool
    hides_standing: bool
    truncated: bool
    """The box runs into the frame edge, so the derived size is a lower bound."""

    indeterminate: bool
    """Wide enough to matter, but truncated such that height cannot be measured.

    Distinct from "too small". For this query the two must not be collapsed: an
    object whose size is unknown is not an object known to be harmless, and the
    near, large, partly-out-of-frame vehicle is exactly the case an operator
    would notice being left out."""

    shadow_area_m2: float
    """Ground area behind the object that the camera cannot see, out to the
    distance a person could stand behind it and remain hidden."""

    reason: str

    @property
    def level(self) -> str:
        if self.hides_standing:
            return "full"
        if self.indeterminate:
            # Ranks above "partial" deliberately. A truncated box that clears the
            # crouch bar has a true height that may well clear the standing bar
            # too -- the frame edge is what stopped the measurement, not the
            # object. Reporting it as merely partial states more than is known.
            return "unknown"
        if self.hides_crouching:
            return "partial"
        return "none"


def physical_extent(bbox, range_m: float, fx: float, fy: float) -> tuple[float, float]:
    """Apparent size plus range -> physical size. The pinhole relation, inverted."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (x2 - x1) * range_m / fx, (y2 - y1) * range_m / fy


def _inconsistent_with_class(height_m: float, cls: str | None) -> bool:
    """Is this derived height impossible for something of that class?"""
    entry = CLASS_HEIGHTS.get(cls or "")
    if entry is None:
        return False
    typical, spread = entry
    return height_m > typical * (1.0 + spread) * HULL_MARGIN * CLASS_CONSISTENCY


EDGE_PX = 2.0
"""How close to the frame edge counts as running into it."""


def _touches_edge(bbox, image_size) -> bool:
    if image_size is None:
        return False
    width, height = image_size
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (x1 <= EDGE_PX or y1 <= EDGE_PX
            or x2 >= width - 1 - EDGE_PX or y2 >= height - 1 - EDGE_PX)


def assess(bbox, range_m: float, fx: float, fy: float, depth_m: float = 2.0,
           cls: str | None = None, image_size: tuple[float, float] | None = None) -> Concealment:
    """Could a person hide behind this, and how much ground does it shadow?

    `depth_m` is how far behind the object a person could stand and still be
    occluded. Kept modest and explicit rather than projected to infinity: the
    occlusion cone technically extends forever, but a person 40 m behind a car is
    not meaningfully "hidden behind the car".

    `cls` is the track's top class, used only to detect that the size derivation
    contradicts it -- see CLASS_CONSISTENCY. It is never used to *grant*
    concealment, so the geometry still carries the decision for anything the
    class table does not cover.

    `image_size` lets a box that runs into the frame edge be recognised as
    truncated. Its derived extent is then a lower bound rather than a
    measurement, and failing to clear a threshold says nothing.
    """
    width, height = physical_extent(bbox, range_m, fx, fy)
    truncated = _touches_edge(bbox, image_size)
    if _inconsistent_with_class(height, cls):
        return Concealment(
            width_m=width, height_m=height, range_m=range_m,
            hides_crouching=False, hides_standing=False, truncated=truncated,
            indeterminate=False, shadow_area_m2=0.0,
            reason=f"{width:.1f}x{height:.1f} m is not a {cls} -- size and class "
                   f"disagree, no claim made",
        )
    wide_enough = width >= MIN_WIDTH_M
    crouch = wide_enough and height >= PERSON_CROUCH_M * HULL_MARGIN
    stand = wide_enough and height >= PERSON_STAND_M * HULL_MARGIN

    # A truncated box under-measures whatever the frame cut off, so a height that
    # fails the bar is not evidence the object is short. Width survives a bottom
    # or top clip, so it still carries the "bigger than a person" test.
    indeterminate = truncated and wide_enough and not stand

    # The occlusion shadow widens with distance from the camera: an object of
    # width w at range R shadows a region that grows as (R + d) / R.
    if (crouch or indeterminate) and range_m > 0:
        far = width * (range_m + depth_m) / range_m
        shadow = 0.5 * (width + far) * depth_m
    else:
        shadow = 0.0

    if stand:
        reason = f"{width:.1f}x{height:.1f} m -- conceals a standing person"
    elif indeterminate:
        reason = f"{width:.1f} m wide, cut off by frame edge -- height not measurable"
    elif crouch:
        reason = f"{width:.1f}x{height:.1f} m -- conceals a crouching person only"
    else:
        reason = f"{width:.1f}x{height:.1f} m -- too small to conceal a person"

    return Concealment(width_m=width, height_m=height, range_m=range_m,
                       hides_crouching=crouch, hides_standing=stand,
                       truncated=truncated, indeterminate=indeterminate,
                       shadow_area_m2=shadow, reason=reason)


def shadow_polygon(origin: np.ndarray, centre: np.ndarray, width_m: float,
                   depth_m: float = 2.0) -> np.ndarray:
    """Ground-plane quad the camera cannot see behind an object.

    Built in the camera's frame of reference: the occluded region is bounded by
    the two sight-lines grazing the object's edges, extended `depth_m` further.
    Returned as world-frame XY for drawing on the map.
    """
    origin = np.asarray(origin, dtype=float)[:2]
    centre = np.asarray(centre, dtype=float)[:2]
    los = centre - origin
    distance = float(np.linalg.norm(los))
    if distance < 1e-6 or width_m <= 0:
        return np.zeros((0, 2))
    direction = los / distance
    across = np.array([-direction[1], direction[0]])

    half = width_m / 2.0
    far_half = half * (distance + depth_m) / distance
    near = centre
    far = origin + direction * (distance + depth_m)
    return np.array([
        near + across * half, near - across * half,
        far - across * far_half, far + across * far_half,
    ])
