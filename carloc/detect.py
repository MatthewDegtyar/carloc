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
from pathlib import Path

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


class CoreMLDetector:
    """YOLO exported to CoreML, run over overlapping native-resolution tiles."""

    def __init__(self, model_path: str | Path = "models/yolo11n.mlpackage",
                 tile: int = TILE, overlap: float = OVERLAP,
                 score_threshold: float = SCORE_THRESHOLD) -> None:
        self.model_path = Path(model_path)
        self.tile = tile
        self.overlap = overlap
        self.score_threshold = score_threshold
        self._model = None
        self._labels: list[str] = []
        self._input_size = 640

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"no model at {self.model_path}. Export one with:\n"
                "  YOLO('yolo11n.pt').export(format='coreml', nms=True, imgsz=640)"
            )
        import coremltools as ct

        self._model = ct.models.MLModel(str(self.model_path))
        spec = self._model.get_spec()
        self._input_name = spec.description.input[0].name
        image_type = spec.description.input[0].type.imageType
        self._input_size = int(image_type.width)
        meta = self._model.user_defined_metadata
        names = meta.get("names") or meta.get("classes") or ""
        if names.startswith("{"):
            import ast

            self._labels = [v for _, v in sorted(ast.literal_eval(names).items())]
        elif names:
            self._labels = [n.strip() for n in names.split(",")]

    def tiles_for(self, width: int, height: int):
        size = min(self.tile, width, height)
        step = max(1, int(size * (1.0 - self.overlap)))
        xs = list(range(0, max(width - size, 0) + 1, step))
        ys = list(range(0, max(height - size, 0) + 1, step))
        if xs[-1] + size < width:
            xs.append(width - size)
        if ys[-1] + size < height:
            ys.append(height - size)
        return [(x, y, size, size) for y in ys for x in xs]

    def _run(self, crop: np.ndarray, offset_x: int, offset_y: int,
             scale: float) -> list[Detection]:
        from PIL import Image

        image = Image.fromarray(crop).resize((self._input_size, self._input_size),
                                             Image.BILINEAR)
        out = self._model.predict({self._input_name: image})
        boxes = np.asarray(out.get("coordinates", np.empty((0, 4))))
        confidence = np.asarray(out.get("confidence", np.empty((0, 1))))
        if not len(boxes):
            return []

        found: list[Detection] = []
        side = crop.shape[0]
        for box, scores in zip(boxes, confidence, strict=False):
            best = int(np.argmax(scores))
            score = float(scores[best])
            if score < self.score_threshold:
                continue
            label = self._labels[best] if best < len(self._labels) else str(best)
            if label not in VEHICLES:
                continue
            # CoreML YOLO emits normalised centre-width-height.
            cx, cy, bw, bh = (float(v) for v in box)
            x1 = (cx - bw / 2) * side + offset_x
            y1 = (cy - bh / 2) * side + offset_y
            x2 = (cx + bw / 2) * side + offset_x
            y2 = (cy + bh / 2) * side + offset_y
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            found.append(Detection(np.array([x1, y1, x2, y2]) * scale, label, score))
        return found

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.load()
        height, width = image.shape[:2]
        found: list[Detection] = []
        for x, y, tw, th in self.tiles_for(width, height):
            crop = image[y:y + th, x:x + tw]
            found.extend(self._run(crop, x, y, scale=1.0))
        return _suppress(found, NMS_IOU)
