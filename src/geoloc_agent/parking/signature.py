"""A vehicle's identity signature, without reading its plate.

Enforcement needs to answer one question: is the car in this bay now the same car
that was here on the last pass? That is *verification* against a single
candidate, not identification against a database, and it is a far easier problem
than general re-identification -- which is why it can be done at all from cues
this coarse.

None of these cues identify a vehicle on its own. A silver sedan is common. A
silver sedan of that exact length, sitting at that exact offset in the bay, is
not. The cues are combined as independent likelihood ratios in `verify.py`, and
the false-match rate that results is measured rather than assumed.

Nothing here reads a licence plate, and that is a product decision as much as a
technical one: it removes the strongest privacy objection a city council will
raise, and it sidesteps plate-recognition error entirely.

Cue strength, weakest to strongest:

``cls``        the detector's class. A handful of buckets; weak alone.
``tone``       body colour, median over the whole observation. Coarse but stable.
``length_m``   physical size, from geolocation. A real measurement in metres, not
               pixels, so it is comparable between passes taken from different
               distances and angles -- which pixel measurements are not.
``bay_pose``   offset and heading within the bay. The strongest cue, and the
               digital equivalent of a chalk mark: a parked car does not move, so
               this is reproduced exactly on a second pass and essentially never
               by a different car taking the space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class VehicleSignature:
    """What is known about one parked vehicle, from one pass."""

    bay_id: int | None
    cls: str
    tone: str
    value: float
    """Body lightness 0-1, median over every look at this vehicle."""

    hue_deg: float
    saturation: float
    length_m: float
    width_m: float
    """Physical extent in metres, from box plus range. Median over the pass."""

    offset_along_m: float
    offset_across_m: float
    """Position within the bay, along and across its axis. The chalk mark."""

    heading_deg: float
    n_looks: int
    position_sigma_m: float
    """Uncertainty on the fix these offsets came from. A signature built on a
    fix the geometry does not trust must not be used to cite anyone."""

    instance_id: str | None = None
    """Ground-truth identity, populated only in eval. Never read by the matcher."""

    extras: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Enough evidence, and a fix tight enough to place a car within a bay.

        Bays are about 2.5 m wide, so a metre of position sigma already spans a
        large fraction of one. Below this bar the vehicle is reported as present
        but never matched, which is the difference between a defensible citation
        and a refunded one.
        """
        return self.n_looks >= 3 and self.position_sigma_m <= 1.0


def summarise(looks: list[dict]) -> VehicleSignature | None:
    """Reduce many observations of one vehicle into one signature.

    Medians throughout, not means: a single frame catching sun, a partial
    occlusion, or a box that clipped the frame edge would otherwise drag an
    average across a decision boundary. The same reasoning as the appearance
    memory in `analysis/appearance.py`, for the same reason.
    """
    if not looks:
        return None
    usable = [look for look in looks if np.isfinite(look.get("length_m", np.nan))]
    if len(usable) < 2:
        return None

    def median_of(key: str, default: float = float("nan")) -> float:
        values = [look[key] for look in usable if np.isfinite(look.get(key, np.nan))]
        return float(np.median(values)) if values else default

    classes = [look["cls"] for look in usable]
    tones = [look["tone"] for look in usable if look.get("tone")]
    return VehicleSignature(
        bay_id=usable[-1].get("bay_id"),
        cls=max(set(classes), key=classes.count),
        tone=max(set(tones), key=tones.count) if tones else "unknown",
        value=median_of("value"),
        hue_deg=median_of("hue_deg"),
        saturation=median_of("saturation"),
        length_m=median_of("length_m"),
        width_m=median_of("width_m"),
        offset_along_m=median_of("offset_along_m", 0.0),
        offset_across_m=median_of("offset_across_m", 0.0),
        heading_deg=median_of("heading_deg", 0.0),
        n_looks=len(usable),
        position_sigma_m=median_of("position_sigma_m", float("inf")),
        instance_id=usable[-1].get("instance_id"),
    )
