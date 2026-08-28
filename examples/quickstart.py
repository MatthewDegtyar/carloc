"""carloc quickstart — video to geolocated, timestamped parked cars.

    uv run python examples/quickstart.py drive.mp4 420 505

The three steps are: count the parked cars in a segment, describe how the camera
moved (here: two known map anchors — swap in `Trajectory.from_gps(...)` if you
have a GPS track), then combine them into absolute positions with timestamps.
"""

import sys
from datetime import datetime

import carloc


def main(video, start, end):
    # 1. vision: video segment -> parked cars, positioned along the street
    cars = carloc.count_parked(video, start=float(start), end=float(end))
    print(f"{len(cars)} parked cars found\n")

    # 2. how the camera moved — YOU choose. Two options shown:
    #
    #    (a) GPS track (best): real position + real clock, any street.
    # trip = carloc.Trajectory.from_gps(
    #     [(t_datetime, lat, lon), ...])
    #
    #    (b) map anchors: a couple of known (video_seconds, lat, lon) fixes,
    #        e.g. intersections the drive crossed. Positions between are
    #        interpolated; pass an epoch to get wall-clock timestamps.
    trip = carloc.Trajectory.from_anchors(
        [(float(start), 25.774346, -80.187238),    # start of segment @ intersection A
         (float(end),   25.777198, -80.188307)],   # end of segment  @ intersection B
        epoch=datetime(2025, 6, 2, 9, 22, 26))

    # 3. combine: relative cars + trajectory -> absolute lat/lon + timestamp
    placed = carloc.geolocate(cars, trip, lateral_m=7.0)

    for c in placed:
        stamp = c.timestamp.strftime("%H:%M:%S") if c.timestamp else "  --  "
        print(f"{stamp}  {c.lat:.6f}, {c.lon:.6f}  {c.side:>5} kerb  "
              f"{c.color} {c.vehicle_class}  (±{c.sigma_along_m:.0f} m along)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
