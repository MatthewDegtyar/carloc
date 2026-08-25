"""CoreML detector: YOLO11n on the Apple Neural Engine.

Imports are lazy so that importing ``geoloc_agent`` never pulls in coremltools,
and so the whole pipeline keeps running on machines without it.

Scope note, from the build plan: the computer vision here is commodity and is
treated as such. There is no detector tuning and no fine-tuning. COCO has no
infrastructure classes, so this covers cars and people, which is what the
geometry work needs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from geoloc_agent.contracts import Detection, Frame
from geoloc_agent.detect.base import Detector

# COCO classes that map onto something this pipeline cares about.
COCO_KEEP = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class CoreMLDetector(Detector):
    """YOLO11n exported via coremltools, running on the ANE.

    Export (done once, outside this file)::

        from ultralytics import YOLO
        YOLO("yolo11n.pt").export(format="coreml", nms=True, imgsz=640)
    """

    name = "coreml-yolo11n"

    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        compute_units: str = "ALL",
    ) -> None:
        self.model_path = Path(model_path)
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.compute_units = compute_units
        self._model = None

    def warmup(self) -> None:
        """Load and run once. The first inference pays compilation cost.

        Benchmarking without this measures model compilation, not inference, and
        overstates per-frame latency by an order of magnitude.
        """
        self._ensure_model()
        blank = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self._predict(blank)

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"no CoreML model at {self.model_path}. Export one with:\n"
                f"  from ultralytics import YOLO\n"
                f"  YOLO('yolo11n.pt').export(format='coreml', nms=True, "
                f"imgsz={self.input_size})\n"
                f"Note: the Torch->CoreML converter needs numpy<2."
            )
        try:
            import coremltools as ct
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "coremltools is required for CoreMLDetector. Install with "
                "`uv sync --extra coreml`; the rest of the pipeline runs without it."
            ) from exc
        unit_name = "CPU_AND_NE" if self.compute_units == "ANE" else "ALL"
        units = getattr(ct.ComputeUnit, unit_name)
        self._model = ct.models.MLModel(str(self.model_path), compute_units=units)
        return self._model

    def _letterbox(self, image: np.ndarray):
        """Resize preserving aspect ratio, padding to a square.

        The model was trained on letterboxed input. Squashing 1600x900 into
        640x640 stretches everything vertically by 1.78x, which the network has
        never seen. Coordinates still map back correctly either way -- the squash
        is a pure per-axis scale -- so this is about detection quality, not
        geometry, but object *height* is exactly what the size prior reads.

        Returns the padded image and the (scale, pad_x, pad_y) needed to invert it.
        """
        from PIL import Image

        height, width = image.shape[:2]
        size = self.input_size
        scale = min(size / width, size / height)
        new_w, new_h = int(round(width * scale)), int(round(height * scale))
        resized = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))  # YOLO's pad grey
        pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))
        return canvas, scale, float(pad_x), float(pad_y)

    def _predict(self, image: np.ndarray):
        model = self._ensure_model()
        canvas, scale, pad_x, pad_y = self._letterbox(image)
        raw = model.predict(
            {
                "image": canvas,
                "iouThreshold": self.iou_threshold,
                "confidenceThreshold": self.score_threshold,
            }
        )
        return raw, scale, pad_x, pad_y

    def detect(self, frame: Frame) -> list[Detection]:
        if frame.image is None:
            raise ValueError(
                "CoreMLDetector needs pixels. Load the session with load_images=True, or "
                "use StubDetector for geometry-only sessions."
            )
        height, width = frame.image.shape[:2]
        raw, scale, pad_x, pad_y = self._predict(frame.image)
        return self._to_detections(raw, frame, width, height, scale, pad_x, pad_y)

    def _to_detections(
        self,
        raw: dict,
        frame: Frame,
        width: int,
        height: int,
        scale: float = 1.0,
        pad_x: float = 0.0,
        pad_y: float = 0.0,
    ) -> list[Detection]:
        """Ultralytics CoreML NMS export emits `confidence` (N,C) and `coordinates` (N,4).

        Coordinates are normalised centre-form (cx, cy, w, h) against the padded
        network input. They are mapped back to the ORIGINAL frame -- undoing the
        letterbox pad and scale -- because every bearing downstream is taken
        through the original intrinsics. A box left in network coordinates yields
        a plausible-looking and entirely wrong map.
        """
        confidence = np.asarray(raw.get("confidence", np.empty((0, 0))))
        coordinates = np.asarray(raw.get("coordinates", np.empty((0, 4))))
        size = float(self.input_size)
        detections: list[Detection] = []
        for i in range(coordinates.shape[0]):
            scores = confidence[i]
            class_index = int(np.argmax(scores))
            score = float(scores[class_index])
            if score < self.score_threshold or class_index not in COCO_KEEP:
                continue
            cx, cy, bw, bh = (float(v) * size for v in coordinates[i])
            x1 = (cx - bw / 2 - pad_x) / scale
            y1 = (cy - bh / 2 - pad_y) / scale
            x2 = (cx + bw / 2 - pad_x) / scale
            y2 = (cy + bh / 2 - pad_y) / scale
            x1, x2 = np.clip([x1, x2], 0.0, width - 1.0)
            y1, y2 = np.clip([y1, y2], 0.0, height - 1.0)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    bbox=np.array([x1, y1, x2, y2]),
                    cls=COCO_KEEP[class_index],
                    score=min(score, 1.0),
                    frame_id=frame.frame_id,
                )
            )
        return detections


# --- benchmarking ------------------------------------------------------------

# The three-loop budget from the design rules. Perception must fit in the frame
# interval; ranging and decision are allowed to be slower because nothing blocks
# on them.
BUDGET_MS = {"perception": 100.0, "ranging": 500.0, "decision": 5000.0}


@dataclass
class StageTiming:
    stage: str
    samples_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.samples_ms)) if self.samples_ms else float("nan")

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.samples_ms, 95)) if self.samples_ms else float("nan")

    @property
    def within_budget(self) -> bool:
        budget = BUDGET_MS.get(self.stage)
        return budget is not None and np.isfinite(self.p95_ms) and self.p95_ms <= budget


def benchmark_stage(fn, iterations: int = 50, stage: str = "perception") -> StageTiming:
    """Time a callable. p95 is what matters, not the mean.

    A pipeline that meets its budget on average and misses it one frame in ten
    drops one frame in ten, so the tail is the number that decides whether the
    loop holds.
    """
    timing = StageTiming(stage=stage)
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        timing.samples_ms.append((time.perf_counter() - started) * 1000.0)
    return timing
