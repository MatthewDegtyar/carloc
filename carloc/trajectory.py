"""How the camera moved over time — the piece you choose.

Everything upstream (detect, track) is fixed; *this* is where you decide how the
video gets turned into geography. A ``Trajectory`` answers one question —
"where was the camera, and when, at video-time ``t``?" — and the rest of the
package doesn't care how you answer it:

* :meth:`Trajectory.from_gps` — you have a GPS track. The best case: exact
  position and real wall-clock time, any street, no anchors needed.
* :meth:`Trajectory.from_anchors` — you have a few known (time, lat, lon) fixes
  (e.g. read off street-name signs, or map intersections the drive crossed);
  positions between them are interpolated.
* :meth:`Trajectory.straight` — a straight segment from a start point, heading and
  speed. Good for a single known block.

Bring your own by subclassing and implementing :meth:`position_at`.
"""

from __future__ import annotations

import bisect
import math
from datetime import datetime, timedelta

_EARTH = 6_378_137.0


def _heading(lat1, lon1, lat2, lon2):
    mx = math.cos(math.radians((lat1 + lat2) / 2))
    return math.degrees(math.atan2((lon2 - lon1) * mx, lat2 - lat1)) % 360


class Trajectory:
    """Camera position as a function of video time. Subclass and override
    :meth:`position_at` (and optionally :meth:`timestamp_at`) for a custom source."""

    def position_at(self, t: float) -> tuple[float, float, float]:
        """Return (lat, lon, heading_deg) of the camera at video time ``t``."""
        raise NotImplementedError

    def timestamp_at(self, t: float) -> datetime | None:
        """Absolute wall-clock time at video time ``t`` (None if unknown)."""
        return None

    # ---- constructors ------------------------------------------------------

    @classmethod
    def from_gps(cls, points, epoch: datetime | None = None) -> Trajectory:
        """Build from a GPS track.

        ``points`` is a sequence of ``(t, lat, lon)`` where ``t`` is either video
        seconds (floats) or ``datetime`` objects. With datetimes, timestamps are
        real; with floats, pass ``epoch`` to get wall-clock time.
        """
        pts = sorted(points, key=lambda p: p[0])
        if pts and isinstance(pts[0][0], datetime):
            t0 = pts[0][0]
            epoch = epoch or t0
            secs = [(p[0] - t0).total_seconds() for p in pts]
        else:
            secs = [float(p[0]) for p in pts]
        lats = [float(p[1]) for p in pts]
        lons = [float(p[2]) for p in pts]
        return _Interpolated(secs, lats, lons, epoch)

    @classmethod
    def from_anchors(cls, anchors, epoch: datetime | None = None) -> Trajectory:
        """Build from known ``(t_seconds, lat, lon)`` fixes; interpolate between.

        Fewer points than GPS — e.g. two street-sign crossings bounding a block.
        Positions between anchors are linear in time; supply a speed profile by
        adding more anchors where the speed changed.
        """
        pts = sorted(anchors, key=lambda p: p[0])
        return _Interpolated([float(p[0]) for p in pts],
                             [float(p[1]) for p in pts],
                             [float(p[2]) for p in pts], epoch)

    @classmethod
    def straight(cls, lat0: float, lon0: float, heading_deg: float,
                 speed_mps: float, t0: float = 0.0,
                 epoch: datetime | None = None) -> Trajectory:
        """A straight run from ``(lat0, lon0)`` at a constant heading and speed."""
        return _Straight(lat0, lon0, heading_deg, speed_mps, t0, epoch)


class _Interpolated(Trajectory):
    def __init__(self, secs, lats, lons, epoch):
        self.secs, self.lats, self.lons, self.epoch = secs, lats, lons, epoch

    def position_at(self, t):
        s = self.secs
        if t <= s[0]:
            i = 0
        elif t >= s[-1]:
            i = len(s) - 2
        else:
            i = bisect.bisect_right(s, t) - 1
        i = max(0, min(i, len(s) - 2))
        span = s[i + 1] - s[i] or 1e-9
        f = (t - s[i]) / span
        lat = self.lats[i] + f * (self.lats[i + 1] - self.lats[i])
        lon = self.lons[i] + f * (self.lons[i + 1] - self.lons[i])
        hd = _heading(self.lats[i], self.lons[i], self.lats[i + 1], self.lons[i + 1])
        return lat, lon, hd

    def timestamp_at(self, t):
        return self.epoch + timedelta(seconds=t - self.secs[0]) if self.epoch else None


class _Straight(Trajectory):
    def __init__(self, lat0, lon0, heading_deg, speed, t0, epoch):
        self.lat0, self.lon0, self.hd = lat0, lon0, heading_deg
        self.speed, self.t0, self.epoch = speed, t0, epoch

    def position_at(self, t):
        d = self.speed * (t - self.t0)
        th = math.radians(self.hd)
        my = (math.pi / 180) * _EARTH
        mx = my * math.cos(math.radians(self.lat0))
        lat = self.lat0 + (d * math.cos(th)) / my
        lon = self.lon0 + (d * math.sin(th)) / mx
        return lat, lon, self.hd

    def timestamp_at(self, t):
        return self.epoch + timedelta(seconds=t - self.t0) if self.epoch else None
