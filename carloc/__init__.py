"""carloc — geolocated parked cars, with timestamps, from dashcam video.

The whole package in four names:

    import carloc

    cars   = carloc.count_parked("drive.mp4", start=420, end=505)   # video -> parked cars
    trip   = carloc.Trajectory.from_gps(gps_points)                  # you choose how
    placed = carloc.geolocate(cars, trip)                            # -> lat/lon + time

    for c in placed:
        print(c.timestamp, c.lat, c.lon, c.color, c.vehicle_class)

`count_parked` does the vision (detect → track → count) and returns cars
positioned *along the street*. `Trajectory` is the one piece you choose — GPS,
map anchors, or a straight run — and `geolocate` combines the two into absolute,
timestamped positions. Swap the detector by passing `detector=` to
`count_parked`; bring your own camera model by subclassing `Trajectory`.

Optional extras live in submodules: `carloc.parkmobile` (paid-zone lookup),
`carloc.zonebox` (in-zone verdict), `carloc.sightings` (overstay matching),
`carloc.export` (KML/CSV).
"""

from __future__ import annotations

from carloc.geolocate import geolocate
from carloc.trajectory import Trajectory
from carloc.types import GeolocatedCar, ParkedCar
from carloc.video import count_parked, extract_frames, track_parked

__version__ = "0.1.0"

__all__ = [
    "count_parked",
    "track_parked",
    "extract_frames",
    "Trajectory",
    "geolocate",
    "ParkedCar",
    "GeolocatedCar",
    "__version__",
]
