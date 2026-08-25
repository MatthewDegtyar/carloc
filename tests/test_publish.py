"""Phase 5: CoT output, and the licensing separation between the two publishers."""

import socket
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
import pytest

from geoloc_agent.contracts import TrackState, TrackStatus
from geoloc_agent.geo import NUSCENES_ORIGINS, GeoOrigin
from geoloc_agent.publish.cot import CotPublisher, track_to_cot
from geoloc_agent.publish.cot.event import cot_xml

ORIGIN = NUSCENES_ORIGINS["boston-seaport"]
NOW = datetime(2026, 8, 25, 12, 0, 0, tzinfo=timezone.utc)


def make_track(track_id=1, sigma=2.0, degenerate=False, reason="", cls="car"):
    return TrackState(
        track_id=track_id,
        mean=np.array([100.0, 250.0, 3.0]),
        cov=np.diag([sigma**2 / 2, sigma**2 / 2, 1.0]),
        class_posterior={cls: 0.9, "clutter": 0.1},
        n_obs=25,
        age=2.5,
        status=TrackStatus.CONFIRMED,
        degenerate=degenerate,
        degeneracy_reason=reason,
    )


def test_cot_event_has_the_required_structure():
    event = track_to_cot(make_track(), ORIGIN, now=NOW)
    assert event.tag == "event"
    assert event.get("version") == "2.0"
    assert event.get("uid").endswith("-1")
    assert event.get("how") == "m-g"
    assert event.find("point") is not None
    assert event.find("detail/contact") is not None


def test_cot_point_carries_error_not_just_position():
    """CoT requires ce/le. That matches this codebase's no-bare-estimates rule."""
    event = track_to_cot(make_track(sigma=4.0), ORIGIN, now=NOW)
    point = event.find("point")
    assert float(point.get("ce")) > 0
    assert float(point.get("le")) > 0
    # ce must reflect the actual covariance, not a constant.
    wider = track_to_cot(make_track(sigma=20.0), ORIGIN, now=NOW).find("point")
    assert float(wider.get("ce")) > float(point.get("ce"))


def test_ce_convention_is_stated_because_cot_dialects_disagree():
    event = track_to_cot(make_track(), ORIGIN, now=NOW)
    assert "ce is CEP50" in event.find("detail/remarks").text


def test_position_converts_through_the_documented_origin():
    event = track_to_cot(make_track(), ORIGIN, now=NOW)
    point = event.find("point")
    lat, lon, _ = ORIGIN.enu_to_wgs84(np.array([100.0, 250.0, 3.0]))
    assert float(point.get("lat")) == pytest.approx(lat, abs=1e-6)
    assert float(point.get("lon")) == pytest.approx(lon, abs=1e-6)
    assert 42.0 < float(point.get("lat")) < 43.0  # Boston


def test_degenerate_geometry_is_visible_without_reading_the_covariance():
    event = track_to_cot(
        make_track(degenerate=True, reason="parallax 0.3 deg"), ORIGIN, now=NOW
    )
    remarks = event.find("detail/remarks").text
    assert "DEGENERATE GEOMETRY" in remarks
    assert "parallax 0.3 deg" in remarks


def test_stale_time_is_after_start_time():
    event = track_to_cot(make_track(), ORIGIN, now=NOW, stale_s=30.0)
    assert event.get("stale") > event.get("start")


def test_class_maps_to_a_cot_type():
    car = track_to_cot(make_track(cls="car"), ORIGIN, now=NOW).get("type")
    person = track_to_cot(make_track(cls="pedestrian"), ORIGIN, now=NOW).get("type")
    assert car.startswith("a-u-G")
    assert person.startswith("a-u-G")
    assert car != person


def test_xml_serialises_and_reparses():
    payload = cot_xml(track_to_cot(make_track(), ORIGIN, now=NOW))
    assert payload.startswith(b"<?xml")
    root = ET.fromstring(payload.decode().split("?>", 1)[1])
    assert root.tag == "event"


def test_publisher_sends_over_udp():
    """Round-trip against a real local socket."""
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    port = receiver.getsockname()[1]
    try:
        with CotPublisher(host="127.0.0.1", port=port, origin=ORIGIN) as publisher:
            publisher.publish(make_track(), now=NOW)
        payload, _ = receiver.recvfrom(65535)
        assert b"<event" in payload
        assert publisher.sent == 1
    finally:
        receiver.close()


def test_publisher_refuses_to_guess_an_origin():
    """Metric-local to lat/lon is an assumption; it must be supplied, not invented."""
    publisher = CotPublisher(origin=None)
    with pytest.raises(ValueError, match="GeoOrigin"):
        publisher.publish(make_track())


def test_publisher_records_the_origin_assumption_in_every_event():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    port = receiver.getsockname()[1]
    try:
        with CotPublisher(host="127.0.0.1", port=port, origin=ORIGIN) as publisher:
            payload = publisher.publish(make_track(), now=NOW)
        assert b"origin=boston-seaport" in payload
        assert b"assumed map-region centre" in payload
    finally:
        receiver.close()


def test_rejects_unknown_protocol():
    with pytest.raises(ValueError, match="udp.*tcp"):
        CotPublisher(protocol="carrier-pigeon")


# --- licensing -------------------------------------------------------------


def test_cot_tree_does_not_import_the_lattice_tree():
    """The licensing constraint, enforced by a test rather than a comment.

    The Lattice SDK licence permits use only for building against Lattice, so
    the CoT path must not depend on that tree even transitively.
    """
    import pathlib

    cot_dir = pathlib.Path("src/geoloc_agent/publish/cot")
    for path in cot_dir.rglob("*.py"):
        source = path.read_text()
        assert "publish.lattice" not in source, f"{path} imports the Lattice tree"
        assert "import anduril" not in source, f"{path} imports the Lattice SDK"


def test_importing_the_pipeline_never_pulls_in_the_lattice_sdk():
    """Import of the SDK is local to the function that needs it, by design."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable, "-c",
            "import geoloc_agent, geoloc_agent.pipeline, geoloc_agent.publish.cot, sys; "
            "print('anduril' in sys.modules)",
        ],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "False"


def test_lattice_publisher_reports_unavailable_without_credentials(monkeypatch):
    from geoloc_agent.publish.lattice.client import LatticeConfig, LatticePublisher

    monkeypatch.delenv("LATTICE_URL", raising=False)
    monkeypatch.delenv("LATTICE_TOKEN", raising=False)
    publisher = LatticePublisher(LatticeConfig(), origin=ORIGIN)
    assert publisher.available is False
    assert "not configured" in publisher.config.describe()


def test_lattice_entity_carries_the_covariance_and_the_degeneracy_flag():
    from geoloc_agent.publish.lattice.client import LatticeConfig, LatticePublisher

    publisher = LatticePublisher(
        LatticeConfig(base_url="https://example.invalid", token="x"), origin=ORIGIN
    )
    entity = publisher.entity_for(make_track(sigma=30.0, degenerate=True, reason="low parallax"))
    ellipse = entity["location"]["error_ellipse"]
    assert ellipse["semi_major_axis_meters"] > 0
    assert entity["_diagnostics"]["degenerate_geometry"] is True
    assert "boston-seaport" in entity["_diagnostics"]["origin_assumption"]


def test_lattice_publisher_refuses_to_guess_an_origin():
    from geoloc_agent.publish.lattice.client import LatticeConfig, LatticePublisher

    publisher = LatticePublisher(LatticeConfig(base_url="u", token="t"), origin=None)
    with pytest.raises(ValueError, match="GeoOrigin"):
        publisher.entity_for(make_track())


def test_geo_round_trip_is_accurate_at_city_scale():
    origin = GeoOrigin(42.336849, -71.05785)
    for point in ([0, 0, 0], [1000, -500, 20], [-2500, 3000, -5]):
        lat, lon, alt = origin.enu_to_wgs84(np.array(point, dtype=float))
        back = origin.wgs84_to_enu(lat, lon, alt)
        assert np.allclose(back, point, atol=0.05)
