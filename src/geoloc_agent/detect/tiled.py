"""Tiled inference, for objects too small to survive the downscale.

The problem this solves is arithmetic, not modelling. A detector with a 640x640
input sees a 1920x1440 frame downscaled 3x. A person at 150 m is 17 px tall in
that frame and arrives at the network as under 6 px. No amount of ranging
cleverness recovers an object that was never detected.

Running the same detector over overlapping native-resolution crops removes the
downscale entirely: inside a 640x640 crop of the original frame, that person is
still 17 px. Cost is one inference per tile, so this is a latency-for-range
trade and it is stated as such rather than hidden.

``band`` is the part that makes it affordable. Distant objects are near the
horizon by construction -- something 150 m away on the ground plane images within
a few degrees of it -- so tiling only a horizontal band around the horizon buys
nearly all of the range at a fraction of the tiles. The full-frame pass still
runs, so near objects are unaffected.
"""

from __future__ import annotations

import numpy as np

from geoloc_agent.contracts import Detection, Frame
from geoloc_agent.detect.base import Detector


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def merge_detections(detections: list[Detection], iou_threshold: float = 0.55) -> list[Detection]:
    """Greedy NMS across tiles.

    Overlap between tiles is deliberate -- it stops an object being cut in half by
    a seam -- so the same object is detected more than once by construction and
    has to be merged back. Matching ignores class: the same car found in two tiles
    can come back "car" in one and "truck" in the other, and keeping both would
    put two tracks on one object.
    """
    out: list[Detection] = []
    for detection in sorted(detections, key=lambda d: -d.score):
        if all(iou(detection.bbox, kept.bbox) < iou_threshold for kept in out):
            out.append(detection)
    return out


class TiledDetector(Detector):
    """Runs a base detector over the full frame plus overlapping native-res tiles."""

    def __init__(
        self,
        base: Detector,
        tile: int = 640,
        overlap: float = 0.25,
        band: tuple[float, float] | None = (0.30, 0.75),
        include_full_frame: bool = True,
        iou_threshold: float = 0.55,
    ) -> None:
        self.base = base
        self.tile = int(tile)
        self.overlap = float(overlap)
        self.band = band
        self.include_full_frame = include_full_frame
        self.iou_threshold = iou_threshold
        self.name = f"tiled({getattr(base, 'name', 'detector')})"

    def warmup(self) -> None:
        self.base.warmup()

    def tiles_for(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        """Tile origins covering the band, clamped to the image."""
        size = min(self.tile, width, height)
        step = max(1, int(size * (1.0 - self.overlap)))

        if self.band is None:
            y_lo, y_hi = 0, height
        else:
            y_lo = int(self.band[0] * height)
            y_hi = int(self.band[1] * height)
            # Always give the band at least one full tile of vertical room.
            if y_hi - y_lo < size:
                centre = (y_lo + y_hi) // 2
                y_lo = max(0, min(centre - size // 2, height - size))
                y_hi = y_lo + size

        def starts(lo: int, hi: int) -> list[int]:
            if hi - lo <= size:
                return [max(0, min(lo, height - size))]
            values = list(range(lo, hi - size + 1, step))
            if values[-1] != hi - size:
                values.append(hi - size)
            return values

        boxes = []
        for y in starts(y_lo, y_hi):
            for x in starts(0, width) if width > size else [0]:
                boxes.append((int(x), int(y), int(x + size), int(y + size)))
        return boxes

    def detect(self, frame: Frame) -> list[Detection]:
        if frame.image is None:
            raise ValueError("TiledDetector needs pixels")
        height, width = frame.image.shape[:2]
        detections: list[Detection] = []

        if self.include_full_frame:
            detections.extend(self.base.detect(frame))

        for x1, y1, x2, y2 in self.tiles_for(width, height):
            crop = frame.image[y1:y2, x1:x2]
            tile_frame = Frame(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                intrinsics=frame.intrinsics,
                pose=frame.pose,
                image=crop,
                source=frame.source,
                is_keyframe=frame.is_keyframe,
            )
            for detection in self.base.detect(tile_frame):
                # Boxes come back in tile coordinates; shift them into the frame.
                # Everything downstream projects through the FULL-frame
                # intrinsics, so a box left in tile coordinates lands somewhere
                # plausible and completely wrong.
                bbox = detection.bbox + np.array([x1, y1, x1, y1], dtype=float)
                bbox[0] = np.clip(bbox[0], 0, width - 1)
                bbox[2] = np.clip(bbox[2], 0, width - 1)
                bbox[1] = np.clip(bbox[1], 0, height - 1)
                bbox[3] = np.clip(bbox[3], 0, height - 1)
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                detections.append(
                    Detection(
                        bbox=bbox, cls=detection.cls, score=detection.score,
                        frame_id=frame.frame_id,
                    )
                )

        return merge_detections(detections, self.iou_threshold)
