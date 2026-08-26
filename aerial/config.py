"""The aerial configuration, in one place, with the reason for each number.

Everything here is a consequence of one change: the camera now looks down from
135 m instead of forward from a car. Nothing in `src/geoloc_agent` needed
modifying to support it -- the envelope was always the knob, this is just the
first time it has been turned that far.
"""

from __future__ import annotations

from geoloc_agent.envelope import OperatingEnvelope
from geoloc_agent.fuse.tracker import TrackerConfig

AERIAL_ENVELOPE = OperatingEnvelope(max_range_m=450.0, min_range_m=100.0)
"""100-450 m, against 2-50 m for the ground-vehicle case.

Set by the flight, not chosen: median height above ground is 135 m and the gimbal
sits at 45 degrees, so the nearest ground in view is about 140 m away and the far
edge of frame reaches roughly 400 m. A 50 m envelope here is not conservative, it
is wrong -- it made the association gate so tight that 1800 detections produced
1499 tracks, one per detection, and only 11 survived to three observations.
Declaring the real envelope took that to 152 without touching the tracker."""

DETECTOR_TILE = 320
"""Tile size for `TiledDetector`.

A car at 45 degrees from 135 m is about 21 px, and letterboxing a 1008 px frame
into a 640 px network leaves it at 13 px. Full-frame detection finds **zero**
objects. A 320 px tile upscaled to the network input doubles apparent size and
finds roughly 20 per frame. 640 px tiles find 2.

Cost is 72 ms per frame against a 100 ms fast-loop budget -- affordable, but the
headroom is gone, which is worth knowing before adding anything else to the loop."""

DETECTOR_THRESHOLD = 0.15
"""Low, deliberately, because the size gate does the rejecting.

COCO weights on 45-degree aerial imagery score real cars and swimming pools in
the same band, so no threshold separates them -- raising it loses cars before it
loses pools. Physical size separates them completely, so the threshold is set to
recall and the geometry filters."""

GATE_CHI2 = 25.0
"""Association gate, widened from the 9.21 (2 dof, 99%) used on the ground.

The platform covers ~10 m between frames while the objects are static and small,
so a track's predicted reprojection moves much further per frame than in the
driving case. At 9.21 the gate rejects correct associations: tracks surviving to
three observations go 152 -> 232 -> 279 as the gate opens to 25 and then 60.

Held at 25 rather than 60 because the wider gate is buying persistence by
accepting associations it should not: with objects this small and this dense, a
gate that never rejects will happily merge two adjacent parked cars, and track
purity is not measurable on this dataset to catch it. 25 is the point where
fragmentation improves without the gate becoming decorative."""


def aerial_tracker_config(gate_chi2: float = GATE_CHI2) -> TrackerConfig:
    return TrackerConfig(
        prior_range=AERIAL_ENVELOPE.prior_range,
        init_range_sigma=AERIAL_ENVELOPE.init_range_sigma,
        max_range=AERIAL_ENVELOPE.track_max_range,
        gate_chi2=gate_chi2,
        # Static objects: process noise here is pure covariance inflation.
        process_noise_per_s=0.0,
    )


def build_detector(session, model_path: str = "models/yolo11n.mlpackage",
                   tile: int = DETECTOR_TILE, threshold: float = DETECTOR_THRESHOLD,
                   size_gate: bool = True):
    """Tiled YOLO, optionally audited by the geometry.

    ``band=None`` because the horizon-band restriction that keeps tiling cheap for
    a forward-looking camera is meaningless here -- objects are spread across the
    whole frame, not concentrated near a horizon.
    """
    from geoloc_agent.detect.coreml import CoreMLDetector
    from geoloc_agent.detect.size_gate import SizeGatedDetector
    from geoloc_agent.detect.tiled import TiledDetector

    base = CoreMLDetector(model_path, score_threshold=threshold)
    base.warmup()
    tiled = TiledDetector(base, tile=tile, band=None)
    if not size_gate:
        return tiled
    return SizeGatedDetector(tiled, build_ranger(session))


def build_ranger(session, step: float = 1.0):
    """Ranger against the session's surface model, in the session's own frame.

    ``local_dsm``, never ``dsm``: the session subtracts a local origin from its
    poses, which moves the cameras and not the terrain.
    """
    from geoloc_agent.range.ground_plane import DsmRanger

    return DsmRanger(session.local_dsm, step=step,
                     max_range_m=AERIAL_ENVELOPE.track_max_range)
