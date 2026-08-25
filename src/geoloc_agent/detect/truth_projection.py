"""A detector that projects ground-truth 3-D boxes.

Named for exactly what it is: an oracle, not a perception model. It exists so
geometry and filter error can be measured on real imagery without detector error
mixed in. Every number it produces is an upper bound on what a real detector
would give you, and any report using it has to say so.

It is deliberately NOT wired into `NuScenesSession.scripted_detections()`, which
still returns None. A session that silently handed out ground truth as if it were
perception would make every downstream metric quietly meaningless.
"""

from __future__ import annotations

import numpy as np

from geoloc_agent.contracts import Detection, Frame
from geoloc_agent.detect.base import Detector
from geoloc_agent.envelope import DEFAULT_ENVELOPE
from geoloc_agent.io.base import TruthObject

MIN_BOX_PX = 14.0
MIN_DEPTH_M = 1.0


class TruthProjectionDetector(Detector):
    """Projects oriented 3-D truth boxes into the image as 2-D detections."""

    name = "truth-projection"

    def __init__(
        self,
        truth: dict[str, TruthObject],
        max_range_m: float = DEFAULT_ENVELOPE.detector_max_range,
        min_box_px: float = MIN_BOX_PX,
        classes: tuple[str, ...] | None = ("car", "pedestrian", "truck", "bus"),
        score: float = 0.95,
    ) -> None:
        self.truth = truth
        self.max_range_m = max_range_m
        self.min_box_px = min_box_px
        self.classes = classes
        self.score = score

    def detect(self, frame: Frame) -> list[Detection]:
        detections: list[Detection] = []
        for obj in self.truth.values():
            if self.classes is not None and obj.cls not in self.classes:
                continue
            detection = self._project(obj, frame)
            if detection is not None:
                detections.append(detection)
        return detections

    def _project(self, obj: TruthObject, frame: Frame) -> Detection | None:
        corners = obj.corners(frame.frame_id)
        cam = (frame.pose.R.T @ (corners - frame.pose.t).T).T
        # Every corner must be in front: a box straddling the image plane
        # projects to a hull spanning the whole frame.
        if np.any(cam[:, 2] < MIN_DEPTH_M):
            return None
        depth = float(np.mean(cam[:, 2]))
        if depth > self.max_range_m:
            return None

        uv = (frame.intrinsics.K @ cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        x1, y1 = uv.min(axis=0)
        x2, y2 = uv.max(axis=0)

        # Reject boxes essentially outside the sensor before clipping, so a
        # sliver of a truck behind the camera does not become a detection.
        intr = frame.intrinsics
        if x2 < 0 or y2 < 0 or x1 > intr.width or y1 > intr.height:
            return None
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if not intr.contains(cx, cy):
            return None

        x1, x2 = np.clip([x1, x2], 0.0, intr.width - 1.0)
        y1, y2 = np.clip([y1, y2], 0.0, intr.height - 1.0)
        if (x2 - x1) < self.min_box_px or (y2 - y1) < self.min_box_px:
            return None

        return Detection(
            bbox=np.array([x1, y1, x2, y2]),
            cls=obj.cls,
            score=self.score,
            frame_id=frame.frame_id,
        )
