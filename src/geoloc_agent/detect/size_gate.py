"""Reject detections whose physical size contradicts their class.

A detector returns a class and a box. A ranger turns that box into metres. When
the two disagree -- a "truck" twenty-three metres tall -- one of them is wrong,
and it is almost never the geometry.

This matters most exactly where a detector is weakest. COCO-trained weights on
45-degree aerial imagery are out of distribution: on AirZoo they call swimming
pools buses and apartment blocks trucks, with scores in the same range as the
real cars, so no confidence threshold separates them. Physical size does, and
cleanly -- measured over 784 detections, cars come back at 2.8 x 2.6 m while
"trucks" come back at 21.2 x 23.0 m.

The gate is deliberately loose. It is not trying to decide whether something is
a hatchback or an estate; it is trying to notice that it is a building. Anything
inside an order of magnitude of its class passes.

This is only available because ranging happens early enough to inform
perception. In the ground-vehicle pipeline range arrives from triangulation
after several frames, far too late to filter a detection. Ranging from a surface
model resolves on the first frame, so the check is affordable per detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geoloc_agent.contracts import Detection, Frame, Observation
from geoloc_agent.detect.base import Detector
from geoloc_agent.range.base import Ranger
from geoloc_agent.range.size_prior import CLASS_HEIGHTS

HEIGHT_TOLERANCE = 2.5
"""How many times its typical height an object may derive before it is rejected.

Generous on purpose. A 2-D box is the convex hull of a 3-D object seen at an
angle, so derived height runs about 1.15x true even when everything is correct,
and an oblique view of a car inflates it further. The failures this catches are
not marginal -- they are 8x and worse -- so a tight bound would buy nothing and
cost real detections."""

WIDTH_TOLERANCE = 4.0
"""Width is judged against height, scaled: a vehicle is longer than it is tall,
and viewing aspect swings apparent width far more than height. Buildings still
fail it by a wide margin."""


@dataclass(frozen=True)
class SizeVerdict:
    detection: Detection
    range_m: float
    width_m: float
    height_m: float
    passed: bool
    reason: str


def physical_size(detection: Detection, range_m: float, fx: float,
                  fy: float) -> tuple[float, float]:
    """Apparent box plus range -> metres. The pinhole relation, inverted."""
    x1, y1, x2, y2 = (float(v) for v in detection.bbox)
    return (x2 - x1) * range_m / fx, (y2 - y1) * range_m / fy


def judge(detection: Detection, range_m: float, fx: float, fy: float) -> SizeVerdict:
    """Is this detection's physical size possible for the class it claims?"""
    width, height = physical_size(detection, range_m, fx, fy)
    entry = CLASS_HEIGHTS.get(detection.cls)
    if entry is None:
        return SizeVerdict(detection, range_m, width, height, True,
                           f"{detection.cls} has no size prior; not judged")
    typical = entry[0]
    if height > typical * HEIGHT_TOLERANCE:
        return SizeVerdict(detection, range_m, width, height, False,
                           f"{height:.1f} m tall is not a {detection.cls} "
                           f"(typical {typical:.1f} m)")
    if width > typical * HEIGHT_TOLERANCE * WIDTH_TOLERANCE:
        return SizeVerdict(detection, range_m, width, height, False,
                           f"{width:.1f} m wide is not a {detection.cls}")
    return SizeVerdict(detection, range_m, width, height, True,
                       f"{width:.1f}x{height:.1f} m is consistent with {detection.cls}")


class SizeGatedDetector(Detector):
    """Wraps a detector, dropping anything the geometry says cannot be that class."""

    def __init__(self, base: Detector, ranger: Ranger, bearing_sigma: float = 2e-3) -> None:
        self.base = base
        self.ranger = ranger
        self.bearing_sigma = float(bearing_sigma)
        self.name = f"size-gated({getattr(base, 'name', 'detector')})"
        self.last_verdicts: list[SizeVerdict] = []
        """Every judgement from the most recent frame, rejections included.

        Kept because a filter that silently discards is untrustworthy: the
        rejected boxes and the reason are what let you tell a working gate from
        one that is quietly eating real detections."""

    def warmup(self) -> None:
        self.base.warmup()

    def detect(self, frame: Frame) -> list[Detection]:
        from geoloc_agent.geometry import bearing_from_pixel

        verdicts: list[SizeVerdict] = []
        kept: list[Detection] = []
        for detection in self.base.detect(frame):
            centre_x, centre_y = detection.centroid
            bearing = bearing_from_pixel(centre_x, centre_y, frame.intrinsics, frame.pose)
            measurement = self.ranger.range_for(
                Observation(t=frame.timestamp, frame_id=frame.frame_id, origin=frame.pose.t,
                            bearing=bearing, bearing_sigma=self.bearing_sigma,
                            cls=detection.cls, score=detection.score),
                [],
            )
            if not measurement.valid:
                # No range means no judgement. Keeping it is the conservative
                # choice: the gate exists to remove things it can prove wrong,
                # not to remove everything it cannot check.
                verdicts.append(SizeVerdict(detection, float("nan"), float("nan"),
                                            float("nan"), True,
                                            f"unranged, not judged: {measurement.reason}"))
                kept.append(detection)
                continue
            verdict = judge(detection, measurement.value, frame.intrinsics.fx,
                            frame.intrinsics.fy)
            verdicts.append(verdict)
            if verdict.passed:
                kept.append(detection)
        self.last_verdicts = verdicts
        return kept

    @property
    def rejection_rate(self) -> float:
        if not self.last_verdicts:
            return 0.0
        return float(np.mean([not v.passed for v in self.last_verdicts]))
