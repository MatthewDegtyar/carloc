"""Cursor-on-Target event generation and delivery to a local TAK server.

CoT has a genuinely good property for this pipeline: its ``<point>`` element
carries ``ce`` and ``le`` -- circular and linear error -- as first-class required
attributes. The format refuses to let you publish a position without stating how
well you know it, which is the same rule this codebase applies everywhere else.
So the track covariance maps onto the wire format directly rather than being
dropped at the boundary.

Uses only the standard library. No dependency on the Lattice tree.
"""

from __future__ import annotations

import socket
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from geoloc_agent.contracts import TrackState
from geoloc_agent.geo import GeoOrigin

# CoT type strings (MIL-STD-2525-ish). a-u-G = atom, unknown affiliation, ground.
COT_TYPES = {
    "car": "a-u-G-E-V-C",
    "truck": "a-u-G-E-V-C",
    "pedestrian": "a-u-G-U-C-I",
    "person": "a-u-G-U-C-I",
    "unknown": "a-u-G",
    "clutter": "a-u-G",
}

DEFAULT_STALE_S = 60.0


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def track_to_cot(
    track: TrackState,
    origin: GeoOrigin,
    stale_s: float = DEFAULT_STALE_S,
    now: datetime | None = None,
    uid_prefix: str = "geoloc",
    origin_note: str = "",
) -> ET.Element:
    """One TrackState -> one CoT event element.

    ``ce`` is the 1-sigma horizontal error and ``le`` the vertical. CoT defines
    these as 95%-ish circular error in some dialects; we publish CEP50 for ``ce``
    and state the convention in the remarks rather than leaving a consumer to
    guess which one they are looking at.
    """
    now = now or datetime.now(timezone.utc)
    lat, lon, alt = origin.enu_to_wgs84(track.mean)
    cls, confidence = track.top_class

    event = ET.Element(
        "event",
        {
            "version": "2.0",
            "uid": f"{uid_prefix}-{track.track_id}",
            "type": COT_TYPES.get(cls, COT_TYPES["unknown"]),
            "how": "m-g",  # machine-generated, GPS-derived
            "time": _iso(now),
            "start": _iso(now),
            "stale": _iso(now + timedelta(seconds=stale_s)),
        },
    )
    ET.SubElement(
        event,
        "point",
        {
            "lat": f"{lat:.7f}",
            "lon": f"{lon:.7f}",
            "hae": f"{alt:.2f}",
            "ce": f"{track.cep50:.2f}",
            "le": f"{float(track.sigma_xyz[2]):.2f}",
        },
    )
    detail = ET.SubElement(event, "detail")
    ET.SubElement(detail, "contact", {"callsign": f"{cls.upper()}-{track.track_id}"})
    ET.SubElement(
        detail,
        "track",
        {"course": "0.0", "speed": "0.0"},  # static object model
    )
    ET.SubElement(detail, "precisionlocation", {"geopointsrc": "CALC", "altsrc": "CALC"})

    remarks = [
        f"class={cls} p={confidence:.2f} entropy={track.class_entropy:.2f}",
        f"sigma_h={track.sigma_horizontal:.2f}m cep50={track.cep50:.2f}m (ce is CEP50)",
        f"n_obs={track.n_obs} age={track.age:.1f}s status={track.status.value}",
    ]
    if track.degenerate:
        # A consumer must be able to see this without reading the covariance.
        remarks.append(f"DEGENERATE GEOMETRY: {track.degeneracy_reason or 'low parallax'}")
    if origin_note:
        remarks.append(origin_note)
    ET.SubElement(detail, "remarks").text = " | ".join(remarks)
    return event


def cot_xml(event: ET.Element) -> bytes:
    return b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + ET.tostring(
        event, encoding="utf-8"
    )


class CotPublisher:
    """Sends CoT events to a TAK server over UDP or TCP.

    Defaults to UDP because that is what a local TAK server and ATAK multicast
    setup expect, and because a dropped position update on a 1 Hz feed is not
    worth blocking the pipeline over.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8087,
        protocol: str = "udp",
        origin: GeoOrigin | None = None,
        stale_s: float = DEFAULT_STALE_S,
        uid_prefix: str | None = None,
    ) -> None:
        if protocol not in ("udp", "tcp"):
            raise ValueError("protocol must be 'udp' or 'tcp'")
        self.host = host
        self.port = port
        self.protocol = protocol
        self.origin = origin
        self.stale_s = stale_s
        self.uid_prefix = uid_prefix or f"geoloc-{uuid.uuid4().hex[:6]}"
        self._socket: socket.socket | None = None
        self.sent = 0

    def __enter__(self) -> CotPublisher:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def connect(self) -> None:
        if self.protocol == "udp":
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self.host, self.port))

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def publish(self, track: TrackState, now: datetime | None = None) -> bytes:
        """Serialise and send one track. Returns the bytes that went on the wire."""
        if self.origin is None:
            raise ValueError(
                "CotPublisher needs a GeoOrigin: CoT is lat/lon and the pipeline is "
                "metric-local, so the conversion assumption must be supplied explicitly"
            )
        note = f"origin={self.origin.name} ({self.origin.provenance})"
        event = track_to_cot(
            track, self.origin, stale_s=self.stale_s, now=now,
            uid_prefix=self.uid_prefix, origin_note=note,
        )
        payload = cot_xml(event)
        if self._socket is None:
            self.connect()
        if self.protocol == "udp":
            self._socket.sendto(payload, (self.host, self.port))
        else:
            self._socket.sendall(payload)
        self.sent += 1
        return payload

    def publish_all(self, tracks: list[TrackState], now: datetime | None = None) -> int:
        for track in tracks:
            self.publish(track, now=now)
        return len(tracks)
