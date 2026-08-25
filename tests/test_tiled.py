"""Tiled inference: the fix for objects too small to survive the downscale."""

import numpy as np
import pytest

from geoloc_agent.contracts import Detection, Frame, Intrinsics, Pose
from geoloc_agent.detect.base import Detector
from geoloc_agent.detect.tiled import TiledDetector, iou, merge_detections

INTR = Intrinsics(fx=1500.0, fy=1500.0, cx=960.0, cy=720.0, width=1920, height=1440)


def frame_with(image):
    return Frame(
        frame_id=3, timestamp=0.0, intrinsics=INTR,
        pose=Pose(R=np.eye(3), t=np.zeros(3)), image=image,
    )


class FixedDetector(Detector):
    """Returns one box at a fixed spot in whatever image it is handed."""

    name = "fixed"

    def __init__(self, offset=(10.0, 20.0), size=30.0):
        self.offset = offset
        self.size = size
        self.calls = 0
        self.shapes = []

    def detect(self, frame):
        self.calls += 1
        self.shapes.append(frame.image.shape[:2])
        x, y = self.offset
        return [
            Detection(
                bbox=np.array([x, y, x + self.size, y + self.size]),
                cls="car", score=0.9, frame_id=frame.frame_id,
            )
        ]


def test_tiles_cover_the_band_and_stay_inside_the_image():
    tiled = TiledDetector(FixedDetector(), tile=640, overlap=0.25, band=(0.35, 0.80))
    boxes = tiled.tiles_for(1920, 1440)
    assert boxes
    for x1, y1, x2, y2 in boxes:
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1440
        assert (x2 - x1) == (y2 - y1) == 640


def test_band_restricts_tiles_to_the_horizon():
    """Distant objects image near the horizon; tiling the sky and road is waste."""
    detector = FixedDetector()
    banded = TiledDetector(detector, tile=640, band=(0.35, 0.80)).tiles_for(1920, 1440)
    everywhere = TiledDetector(detector, tile=640, band=None).tiles_for(1920, 1440)
    assert len(banded) < len(everywhere)
    for _, y1, _, y2 in banded:
        assert y1 >= 0.30 * 1440
        assert y2 <= 0.85 * 1440


def test_tiles_overlap_so_objects_are_not_cut_by_seams():
    boxes = TiledDetector(FixedDetector(), tile=640, overlap=0.25).tiles_for(1920, 1440)
    xs = sorted({b[0] for b in boxes})
    assert len(xs) > 1
    assert xs[1] - xs[0] < 640, "consecutive tiles must overlap"


def test_detections_are_returned_in_full_frame_coordinates():
    """The bug this guards: a box left in tile coordinates lands plausibly and wrong.

    Every bearing downstream projects through the FULL-frame intrinsics, so an
    unshifted box produces a confident, entirely incorrect map.
    """
    base = FixedDetector(offset=(10.0, 20.0), size=30.0)
    tiled = TiledDetector(base, tile=640, overlap=0.25, band=(0.35, 0.80),
                          include_full_frame=False, iou_threshold=0.99)
    image = np.zeros((1440, 1920, 3), dtype=np.uint8)
    detections = tiled.detect(frame_with(image))

    expected = {
        (x1 + 10.0, y1 + 20.0) for x1, y1, _, _ in tiled.tiles_for(1920, 1440)
    }
    got = {(float(d.bbox[0]), float(d.bbox[1])) for d in detections}
    assert got == expected
    # And nothing may sit at the raw tile-local origin unless a tile starts there.
    assert all(d.bbox[0] >= 10.0 and d.bbox[1] >= 20.0 for d in detections)


def test_base_detector_sees_native_resolution_crops():
    """The entire point: crops are NOT downscaled, so small objects stay large."""
    base = FixedDetector()
    tiled = TiledDetector(base, tile=640, band=(0.35, 0.80), include_full_frame=False)
    tiled.detect(frame_with(np.zeros((1440, 1920, 3), dtype=np.uint8)))
    assert base.shapes, "base detector was never called"
    assert all(shape == (640, 640) for shape in base.shapes)


def test_full_frame_pass_is_included_by_default():
    base = FixedDetector()
    tiled = TiledDetector(base, tile=640, band=(0.35, 0.80))
    n_tiles = len(tiled.tiles_for(1920, 1440))
    tiled.detect(frame_with(np.zeros((1440, 1920, 3), dtype=np.uint8)))
    assert base.calls == n_tiles + 1
    assert base.shapes[0] == (1440, 1920), "the first pass must be the whole frame"


def test_iou_and_nms_merge_duplicates_across_seams():
    assert iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == pytest.approx(1.0)
    assert iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) == 0.0

    near = [
        Detection(bbox=np.array([100.0, 100.0, 160.0, 160.0]), cls="car", score=0.9, frame_id=0),
        Detection(bbox=np.array([102.0, 101.0, 162.0, 161.0]), cls="car", score=0.8, frame_id=0),
        Detection(bbox=np.array([500.0, 500.0, 560.0, 560.0]), cls="car", score=0.7, frame_id=0),
    ]
    merged = merge_detections(near, iou_threshold=0.55)
    assert len(merged) == 2
    assert merged[0].score == 0.9  # the more confident duplicate survives


def test_merge_ignores_class_so_one_object_yields_one_track():
    """The same car can come back 'car' in one tile and 'truck' in another."""
    same_object = [
        Detection(bbox=np.array([100.0, 100.0, 160.0, 160.0]), cls="car", score=0.9, frame_id=0),
        Detection(bbox=np.array([101.0, 100.0, 161.0, 160.0]), cls="truck", score=0.6, frame_id=0),
    ]
    assert len(merge_detections(same_object, iou_threshold=0.55)) == 1


def test_tiling_recovers_objects_the_downscale_destroys():
    """Arithmetic check, no model needed: a crop preserves what a resize removes."""
    downscale = 640 / INTR.width
    person_px_at_150m = INTR.fy * 1.7 / 150.0
    assert person_px_at_150m == pytest.approx(17.0, abs=1.0)
    # Whole-frame inference sees it at a third of that -- far below any detector.
    assert person_px_at_150m * downscale < 6.0
    # A native-resolution crop preserves the full height.
    assert person_px_at_150m > 15.0


def test_tiled_detector_requires_pixels():
    tiled = TiledDetector(FixedDetector())
    with pytest.raises(ValueError, match="pixels"):
        tiled.detect(frame_with(None))
