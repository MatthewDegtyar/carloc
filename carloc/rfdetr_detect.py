"""RF-DETR: the detector that can actually read straight-down imagery.

Roboflow's RF-DETR, Apache-2.0, weights fetched on first use, no API key. It
replaces the CoreML YOLO path for satellite work for one measured reason -- on
the same downtown tile, at the same tile size:

    YOLO11n     5 detections, all `clock`, 0 vehicles
    RF-DETR    35 detections, 31 cars

That is not a threshold difference. A COCO-trained CNN has never seen a car from
above and reads the rectangle as furniture; the DETR backbone generalises to the
viewpoint well enough to be useful. Worth knowing before assuming any
COCO-pretrained model is interchangeable with another.

Tiling is still required and for the unchanged reason: full-frame inference on a
1024 px mosaic finds **zero** objects, because a 33 px car shrinks below what the
network can resolve once the image is scaled to its input. At 320 px tiles the
car arrives near the size the model was trained on.
"""

from __future__ import annotations

import numpy as np

from carloc.detect import NMS_IOU, Detection, _suppress

COCO_VEHICLES = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
"""COCO 91-class ids. RF-DETR returns ids, not names, and the 91-class ordering
is not the 80-class one -- reading it with the wrong table silently relabels
everything."""

TILE = 320
OVERLAP = 0.25
THRESHOLD = 0.30
"""0.30 rather than lower. At 0.15 the same tile yields 93 vehicles but drags in
a tail of junk classes; at 0.30 it is 31 cars and almost nothing else. For a
demo that has to be trusted, precision beats recall."""


class RFDETRDetector:
    """RF-DETR behind the pipeline's `detect(image) -> [Detection]` contract."""

    def __init__(self, tile: int = TILE, overlap: float = OVERLAP,
                 threshold: float = THRESHOLD, resolution: int | None = None) -> None:
        self.tile = tile
        self.overlap = overlap
        self.threshold = threshold
        self.resolution = resolution
        self._model = None
        self.last_labels: dict[str, int] = {}
        """Every class id returned, vehicle or not. Kept because the first
        question when a detector finds nothing is whether it found something
        else, which is exactly how the YOLO failure was diagnosed."""

    def load(self) -> None:
        if self._model is not None:
            return
        import warnings

        warnings.filterwarnings("ignore")
        from rfdetr import RFDETRBase

        self._model = RFDETRBase() if self.resolution is None else \
            RFDETRBase(resolution=self.resolution)

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
            scores = (np.asarray(result.confidence) if getattr(result, "confidence", None)
                      is not None else np.ones(len(boxes)))
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
