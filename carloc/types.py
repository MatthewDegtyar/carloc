"""The data the pipeline passes around: a parked car and a placed car.

Kept deliberately small. A ``ParkedCar`` is one physical car, counted
once, positioned *relative* to the
segment (metres along the street) — this is what the vision alone can give you. A
``GeolocatedCar`` is a ParkedCar after a :class:`~carloc.trajectory.Trajectory`
has turned that relative position into an absolute lat/lon and a timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ParkedCar:
    """One physical parked car, position relative to the segment start.

    ``along_m`` is distance along the street from where the segment began;
    ``abeam_t`` is the video time the camera was closest to abreast of it (its
    best-localised moment), which is what a Trajectory keys on to place it.
    """

    along_m: float
    side: str                # "left" | "right", relative to travel
    abeam_t: float           # video time of the most-abeam detection, seconds
    first_t: float
    last_t: float
    sigma_along_m: float
    vehicle_class: str
    color: str
    n_detections: int
    n_tracklets: int         # >1 means rebuilt from occlusion-split pieces
    confidence: float        # 0..1, how sure this is a real parked car


@dataclass
class GeolocatedCar:
    """A parked car in absolute coordinates, with a timestamp."""

    lat: float
    lon: float
    timestamp: datetime | None
    side: str
    vehicle_class: str
    color: str
    sigma_along_m: float
    sigma_cross_m: float
    confidence: float        # 0..1, carried from the ParkedCar
    source_t: float          # video time it was placed from
