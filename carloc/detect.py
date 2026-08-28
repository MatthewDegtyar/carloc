"""Vehicle detection: tiled YOLO, because full-frame finds nothing.

A car seen from 130 m at 30 degrees is roughly 20 px across, and letterboxing a
1008 px frame into a 640 px network leaves it at about 13. Measured on this
footage, full-frame detection returns **zero** objects. Running the same network
over native-resolution tiles keeps the object at its original size relative to
the network input, and it starts finding things.

That is the whole trick, and it costs a linear factor in tiles per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TILE = 320
"""Tile side in source pixels. Upscaled to the network input, this roughly
doubles apparent object size. 640 px tiles find an order of magnitude fewer
objects here; 320 is the point where cars become reliably detectable."""

OVERLAP = 0.25
SCORE_THRESHOLD = 0.20
NMS_IOU = 0.55
VEHICLES = {"car", "truck", "bus", "motorcycle"}


@dataclass(frozen=True)
class Detection:
    bbox: np.ndarray
    cls: str
    score: float

    @property
    def centroid(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    @property
    def bottom_centre(self) -> np.ndarray:
        """Where the object meets the ground.

        For a near-nadir view the centroid is close enough, but as the camera
        tilts the box top is the roof and its centre floats above the ground the
        ray should hit. The bottom edge is the better ground contact point.
        """
        x1, _, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, y2])


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def _suppress(detections: list[Detection], threshold: float) -> list[Detection]:
    """Overlapping tiles see the same car twice; keep the most confident copy."""
    if not detections:
        return []
    order = sorted(detections, key=lambda d: -d.score)
    kept: list[Detection] = []
    for candidate in order:
        if kept:
            boxes = np.array([k.bbox for k in kept])
            if _iou(candidate.bbox, boxes).max() > threshold:
                continue
        kept.append(candidate)
    return kept
