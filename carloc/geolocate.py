"""Turn relative parked cars + a trajectory into absolute, timestamped cars.

A :class:`~carloc.types.ParkedCar` knows *when* the camera was abreast of it
(``abeam_t``) and which side of the street it sat on. A
:class:`~carloc.trajectory.Trajectory` knows *where* the camera was at any time.
Put them together: place the car at the camera's position when it was abeam,
offset sideways to the kerb by the (measured) lateral distance, and stamp it with
the trajectory's clock. That's the whole step — and it's why GPS "just works":
GPS is exactly a time→position function.
"""

from __future__ import annotations

import math

from carloc.types import GeolocatedCar, ParkedCar

_EARTH = 6_378_137.0


def geolocate(cars: list[ParkedCar], trajectory, lateral_m: float = 7.0,
              sigma_cross_m: float = 1.8) -> list[GeolocatedCar]:
    """Place each parked car in absolute lat/lon with a timestamp.

    ``lateral_m`` is the distance from the camera's lane to the kerb (the
    measured Miami value is ~4.7 m centreline-to-car; ~7 m works from a middle
    lane). ``sigma_cross_m`` is the across-street position uncertainty.
    """
    out: list[GeolocatedCar] = []
    for c in cars:
        lat, lon, heading = trajectory.position_at(c.abeam_t)
        my = (math.pi / 180) * _EARTH
        mx = my * math.cos(math.radians(lat))
        th = math.radians(heading)
        # forward unit (E, N); left is 90 deg CCW of it
        fe, fn = math.sin(th), math.cos(th)
        le, ln = -fn, fe
        sign = -1.0 if c.side == "left" else 1.0
        out.append(GeolocatedCar(
            lat=lat + sign * lateral_m * ln / my,
            lon=lon + sign * lateral_m * le / mx,
            timestamp=trajectory.timestamp_at(c.abeam_t),
            side=c.side, vehicle_class=c.vehicle_class, color=c.color,
            sigma_along_m=c.sigma_along_m, sigma_cross_m=sigma_cross_m,
            source_t=c.abeam_t))
    return out
