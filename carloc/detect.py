"""Vehicle detection: RF-DETR over native-resolution tiles.

Two facts drive this module. RF-DETR (Apache-2.0, no key) reads straight-down and
oblique imagery where a COCO-trained YOLO cannot — on one downtown tile YOLO
returned 5 detections (all "clock"), RF-DETR returned 31 cars. And tiling is
still required: a car ~20-33 px across shrinks below what any network resolves
once a full frame is scaled to its input, so full-frame inference finds almost
nothing while native-resolution tiles keep the car at trainable size.

`Detection` + `_suppress` are the shared primitives (a box, and NMS to dedupe the
same car seen in overlapping tiles); `RFDETRDetector` is the tiled detector built
on them. `COCO_VEHICLES` maps the 91-class ids RF-DETR returns to names.
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
THRESHOLD = 0.30
"""Detection confidence floor. At 0.15 a tile drags in a tail of junk classes;
0.30 keeps the cars and almost nothing else — precision over recall for a demo."""
NMS_IOU = 0.55
VEHICLES = {"car", "truck", "bus", "motorcycle"}
COCO_VEHICLES = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
"""COCO 91-class ids (not the 80-class ordering) — RF-DETR returns ids, and the
wrong table silently relabels everything."""


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


class RFDETRDetector:
    """RF-DETR behind a simple ``detect(image) -> [Detection]`` contract, tiled."""

    def __init__(self, tile: int = TILE, overlap: float = OVERLAP,
                 threshold: float = THRESHOLD, resolution: int | None = None) -> None:
        self.tile = tile
        self.overlap = overlap
        self.threshold = threshold
        self.resolution = resolution
        self._model = None
        self.last_labels: dict[str, int] = {}
        """Every class id returned, vehicle or not — the first question when a
        detector finds nothing is whether it found something else."""

    def load(self) -> None:
        if self._model is not None:
            return
        import warnings

        warnings.filterwarnings("ignore")
        from rfdetr import RFDETRBase

        self._model = (RFDETRBase() if self.resolution is None
                       else RFDETRBase(resolution=self.resolution))

    def tiles_for(self, width: int, height: int):
        size = min(self.tile, width, height)
        step = max(1, int(size * (1.0 - self.overlap)))
        xs = list(range(0, max(width - size, 0) + 1, step))
        ys = list(range(0, max(height - size, 0) + 1, step))
        if xs and xs[-1] + size < width:
            xs.append(width - size)
        if ys and ys[-1] + size < height:
            ys.append(height - size)
        return [(x, y, size, size) for y in ys for x in xs]

    def detect(self, image: np.ndarray) -> list[Detection]:
        from PIL import Image

        self.load()
        self.last_labels = {}
        height, width = image.shape[:2]
        found: list[Detection] = []
        for x, y, tw, th in self.tiles_for(width, height):
            crop = image[y:y + th, x:x + tw]
            result = self._model.predict(Image.fromarray(crop), threshold=self.threshold)
            boxes = np.asarray(result.xyxy) if len(result) else np.empty((0, 4))
            ids = np.asarray(result.class_id) if len(result) else np.empty((0,))
            scores = (np.asarray(result.confidence)
                      if getattr(result, "confidence", None) is not None
                      else np.ones(len(boxes)))
            for box, class_id, score in zip(boxes, ids, scores, strict=False):
                name = COCO_VEHICLES.get(int(class_id), f"id{int(class_id)}")
                self.last_labels[name] = self.last_labels.get(name, 0) + 1
                if int(class_id) not in COCO_VEHICLES:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                found.append(Detection(
                    bbox=np.array([x1 + x, y1 + y, x2 + x, y2 + y]),
                    cls=name, score=float(score)))
        return _suppress(found, NMS_IOU)
