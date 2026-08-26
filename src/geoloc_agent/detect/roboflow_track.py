"""Image-plane association from `roboflow/trackers`, feeding the world-frame filter.

Two trackers run, and they are answering different questions. Roboflow's trackers
(SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU, McByte) associate in pixels: is this
the same *box* as last frame, judged on overlap and box motion. `fuse/tracker.py`
associates in metres: is this the same *place*, judged on a Mahalanobis distance
against a 3-D covariance.

Neither subsumes the other, and each fails where the other holds up.

Image-plane association breaks when the camera moves, because overlap between
consecutive frames stops being a proxy for identity -- on the aerial flight the
platform covers about ten metres per frame. World-frame association breaks before
a track has localised, when its range prior is metres wide and the gate is
correspondingly loose or, tuned tighter, rejects correct matches. That is exactly
what produced one track per detection on the first aerial run.

So this wrapper runs the image-plane tracker first and stamps its id onto each
detection as ``track_hint``. The filter treats that as independent evidence and
widens its gate for hinted pairs -- see ``TrackerConfig.hint_gate_scale`` -- but
never lets it override the geometry. An upstream identity swap stays recoverable.

Requires the ``tracking`` extra: ``uv sync --extra tracking``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from geoloc_agent.contracts import Detection, Frame
from geoloc_agent.detect.base import Detector

TRACKERS = ("sort", "bytetrack", "ocsort", "botsort", "cbiou")
"""Those usable here."""

MIN_RECOVERY_IOU = 0.30
"""Overlap needed to say a returned box came from a given detection."""

USES_FRAME = frozenset({"botsort"})
"""Trackers that actually read the image. Only BoT-SORT does, for its camera
motion compensation; passing a frame to the others earns a warning per call."""
"""McByte is excluded: it needs SAM and Cutie, which is a torch-and-segmentation
dependency an association hint does not justify."""


def _build(name: str, **kwargs: Any):
    import trackers as rf

    table = {
        "sort": rf.SORTTracker,
        "bytetrack": rf.ByteTrackTracker,
        "ocsort": rf.OCSORTTracker,
        "botsort": rf.BoTSORTTracker,
        "cbiou": rf.CBIoUTracker,
    }
    if name not in table:
        raise ValueError(f"unknown tracker {name!r}; expected one of {sorted(table)}")
    return table[name](**kwargs)


def to_supervision(detections: list[Detection], classes: list[str]):
    """Our detections -> ``supervision.Detections``, the format they speak.

    Class names are mapped to indices because supervision carries ``class_id`` as
    an integer array. The mapping is built per call from the classes actually
    present, and the inverse is returned so nothing depends on a global registry.
    """
    import supervision as sv

    if not detections:
        return sv.Detections.empty()
    lookup = {name: index for index, name in enumerate(classes)}
    return sv.Detections(
        xyxy=np.array([d.bbox for d in detections], dtype=float),
        confidence=np.array([d.score for d in detections], dtype=float),
        class_id=np.array([lookup[d.cls] for d in detections], dtype=int),
    )


class RoboflowTrackedDetector(Detector):
    """Wraps a detector, stamping an image-plane track id onto each detection."""

    def __init__(self, base: Detector, tracker: str = "bytetrack",
                 per_class: bool = True, **tracker_kwargs: Any) -> None:
        self.base = base
        self.tracker_name = tracker
        self.per_class = per_class
        self._tracker_kwargs = tracker_kwargs
        self._trackers: dict[str, Any] = {}
        self._global = None
        self.name = f"{tracker}({getattr(base, 'name', 'detector')})"
        self.unhinted = 0
        """Detections the tracker declined to give an id this frame.

        Kept visible because it is the honest measure of whether the hint is
        doing anything: most of these trackers withhold an id until a track has
        survived ``minimum_consecutive_frames``, so an early frame legitimately
        returns none, and a run where that never falls is a run where the hint
        is not helping."""

    def warmup(self) -> None:
        self.base.warmup()

    def _tracker_for(self, cls: str):
        """One tracker per class, or one for everything.

        Per class by default because these trackers associate on overlap and box
        motion without regard to label, so a car and a truck whose boxes overlap
        are candidates for the same id. Keeping them separate costs nothing --
        the trackers are cheap and stateless between classes -- and removes a
        failure the geometry would then have to undo.
        """
        if not self.per_class:
            if self._global is None:
                self._global = _build(self.tracker_name, **self._tracker_kwargs)
            return self._global
        if cls not in self._trackers:
            self._trackers[cls] = _build(self.tracker_name, **self._tracker_kwargs)
        return self._trackers[cls]

    def detect(self, frame: Frame) -> list[Detection]:
        detections = self.base.detect(frame)
        if not detections:
            # Still step every tracker, or their internal age and miss counts
            # drift out of step with the video and tracks are reaped late.
            for tracker in (*self._trackers.values(), *( [self._global] if self._global else [] )):
                image = frame.image if self.tracker_name in USES_FRAME else None
                tracker.update(to_supervision([], []), image)
            self.unhinted = 0
            return []

        groups: dict[str, list[Detection]] = {}
        for detection in detections:
            groups.setdefault(detection.cls if self.per_class else "", []).append(detection)

        out: list[Detection] = []
        unhinted = 0
        for cls, group in groups.items():
            tracker = self._tracker_for(cls)
            classes = sorted({d.cls for d in group})
            image = frame.image if self.tracker_name in USES_FRAME else None
            tracked = tracker.update(to_supervision(group, classes), image)
            ids = getattr(tracked, "tracker_id", None)
            boxes = np.asarray(tracked.xyxy, dtype=float) if len(tracked) else np.empty((0, 4))

            for detection in group:
                hint = _match_id(detection, boxes, ids)
                if hint is None:
                    unhinted += 1
                out.append(
                    Detection(bbox=detection.bbox, cls=detection.cls, score=detection.score,
                              frame_id=detection.frame_id, track_hint=hint)
                )
        self.unhinted = unhinted
        return out


def _match_id(detection: Detection, boxes: np.ndarray, ids) -> int | None:
    """Recover which returned box corresponds to which input detection.

    These trackers return a filtered, reordered set of *filter-predicted* boxes,
    not the inputs, so neither index order nor exact geometry recovers the
    correspondence -- both were tried and both silently mislabel. Overlap does:
    a returned box is a smoothed version of the detection it belongs to, so it
    overlaps that one far more than any other.
    """
    if ids is None or not len(boxes):
        return None
    box = np.asarray(detection.bbox, dtype=float)
    x1 = np.maximum(boxes[:, 0], box[0])
    y1 = np.maximum(boxes[:, 1], box[1])
    x2 = np.minimum(boxes[:, 2], box[2])
    y2 = np.minimum(boxes[:, 3], box[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    iou = inter / np.maximum(area_a + area_b - inter, 1e-9)
    best = int(np.argmax(iou))
    if iou[best] < MIN_RECOVERY_IOU:
        return None
    identifier = ids[best]
    if identifier is None:
        return None
    identifier = int(identifier)
    # supervision uses -1 for "no id assigned". These trackers withhold an id
    # until a track has survived `minimum_consecutive_frames`, so early frames
    # legitimately return it; treating it as a real id would collapse every
    # unconfirmed detection onto one shared identity.
    return None if identifier < 0 else identifier
