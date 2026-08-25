"""Real-data path: nuScenes tables, size priors, and the truth-projection oracle.

The nuScenes tests skip when the dataset is absent -- it is a 4 GB download and
`sessions/` is gitignored -- so this file must stay green in a clean checkout.
Everything that does NOT need the data is tested unconditionally, because that
is where the convention bugs live.
"""

from pathlib import Path

import numpy as np
import pytest

from geoloc_agent.contracts import Detection, Frame, Intrinsics, Pose, RangeMethod
from geoloc_agent.detect.truth_projection import TruthProjectionDetector
from geoloc_agent.io.base import TruthObject
from geoloc_agent.io.synthetic import look_along
from geoloc_agent.range.size_prior import CLASS_HEIGHTS, range_prior_from_size

DATAROOT = Path("sessions/nuscenes")
HAS_NUSCENES = (DATAROOT / "v1.0-mini" / "sample.json").exists()
needs_data = pytest.mark.skipif(not HAS_NUSCENES, reason="nuScenes v1.0-mini not downloaded")

INTR = Intrinsics(fx=1266.4, fy=1266.4, cx=800.0, cy=450.0, width=1600, height=900)


def frame_at(position=(0.0, 0.0, 1.5), facing=(0.0, 1.0, 0.0), frame_id=0):
    return Frame(
        frame_id=frame_id, timestamp=0.0, intrinsics=INTR,
        pose=Pose(R=look_along(np.array(facing, dtype=float)), t=np.array(position, dtype=float)),
    )


# --- size prior --------------------------------------------------------------


def test_size_prior_recovers_range_from_a_synthetic_box():
    """R = f*H/h. Construct the box from that and check it inverts."""
    height_m = CLASS_HEIGHTS["car"][0]
    for true_range in (10.0, 25.0, 60.0):
        h_px = INTR.fy * height_m / true_range
        det = Detection(
            bbox=np.array([700.0, 450.0 - h_px / 2, 760.0, 450.0 + h_px / 2]),
            cls="car", score=0.9, frame_id=0,
        )
        prior = range_prior_from_size(det, INTR)
        assert prior.valid
        assert prior.value == pytest.approx(true_range, rel=1e-6)
        assert prior.method is RangeMethod.MONO_DEPTH


def test_size_prior_sigma_is_relative_to_range():
    """Doubling the range must roughly double the absolute uncertainty."""
    height_m = CLASS_HEIGHTS["car"][0]
    sigmas = []
    for true_range in (20.0, 40.0):
        h_px = INTR.fy * height_m / true_range
        det = Detection(
            bbox=np.array([700.0, 450.0 - h_px / 2, 760.0, 450.0 + h_px / 2]),
            cls="car", score=0.9, frame_id=0,
        )
        sigmas.append(range_prior_from_size(det, INTR).sigma)
    assert sigmas[1] == pytest.approx(2 * sigmas[0], rel=0.15)


def test_size_prior_declines_rather_than_guessing():
    small = Detection(bbox=np.array([10.0, 400.0, 24.0, 404.0]), cls="car", score=0.9, frame_id=0)
    assert not range_prior_from_size(small, INTR).valid

    unknown = Detection(
        bbox=np.array([700.0, 400.0, 760.0, 500.0]), cls="bollard", score=0.9, frame_id=0
    )
    prior = range_prior_from_size(unknown, INTR)
    assert not prior.valid
    assert "no size prior" in prior.reason


def test_truncated_boxes_are_rejected_because_their_height_is_a_lower_bound():
    """A box clipped by the image edge is shorter than the object, so f*H/h lies."""
    touching_bottom = Detection(
        bbox=np.array([700.0, 600.0, 800.0, INTR.height - 1.0]), cls="car", score=0.9, frame_id=0
    )
    prior = range_prior_from_size(touching_bottom, INTR)
    assert not prior.valid
    assert "truncated" in prior.reason


def test_size_prior_is_never_offered_as_a_measurement():
    """A prior initialises and gates; it must not be fused every frame."""
    from geoloc_agent.contracts import Observation

    obs = Observation(
        t=0.0, frame_id=0, origin=np.zeros(3), bearing=np.array([0.0, 1.0, 0.0]),
        bearing_sigma=1e-3,
        range_prior=range_prior_from_size(
            Detection(bbox=np.array([700.0, 400.0, 760.0, 500.0]), cls="car",
                      score=0.9, frame_id=0),
            INTR,
        ),
    )
    assert obs.range_prior is not None and obs.range_prior.valid
    assert obs.range is None
    assert not obs.has_range  # the filter must not treat it as a range update


def test_size_prior_seeds_the_track_near_the_right_range():
    from geoloc_agent.contracts import Observation
    from geoloc_agent.fuse.ekf import initial_state

    prior = range_prior_from_size(
        Detection(
            bbox=np.array([700.0, 450.0 - 39.0, 760.0, 450.0 + 39.0]),
            cls="car", score=0.9, frame_id=0,
        ),
        INTR,
    )
    obs = Observation(
        t=0.0, frame_id=0, origin=np.zeros(3), bearing=np.array([0.0, 1.0, 0.0]),
        bearing_sigma=1e-3, range_prior=prior,
    )
    mean, _ = initial_state(obs, prior_range=50.0, range_sigma=60.0)
    # Seeded from the prior (~50 m for a 78 px car), not the 50 m default -- and
    # crucially not with the default's 60 m sigma.
    assert np.linalg.norm(mean) == pytest.approx(prior.value, rel=1e-6)


# --- truth projection --------------------------------------------------------


def test_truth_projection_boxes_an_object_in_front():
    obj = TruthObject("a", np.array([0.0, 25.0, 0.8]), cls="car", size=(1.9, 4.5, 1.6))
    detections = TruthProjectionDetector({"a": obj}).detect(frame_at())
    assert len(detections) == 1
    assert detections[0].cls == "car"
    # Centred horizontally, since the object sits on the optical axis.
    assert detections[0].centroid[0] == pytest.approx(INTR.cx, abs=2.0)


def test_oriented_boxes_are_narrower_than_axis_aligned_ones():
    """A car rotated end-on is a third as wide; ignoring rotation inflates it."""
    end_on = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    side = TruthObject("a", np.array([0.0, 25.0, 0.8]), cls="car", size=(1.9, 4.5, 1.6))
    rotated = TruthObject(
        "a", np.array([0.0, 25.0, 0.8]), cls="car", size=(1.9, 4.5, 1.6), rotation=end_on
    )
    w_side = TruthProjectionDetector({"a": side}).detect(frame_at())[0].width
    w_rot = TruthProjectionDetector({"a": rotated}).detect(frame_at())[0].width
    assert w_rot > w_side  # end-on presents the 4.5 m length across the view
    assert w_rot / w_side == pytest.approx(4.5 / 1.9, rel=0.15)


def test_objects_behind_or_beyond_range_are_not_detected():
    behind = TruthObject("b", np.array([0.0, -20.0, 0.8]), cls="car", size=(1.9, 4.5, 1.6))
    far = TruthObject("f", np.array([0.0, 300.0, 0.8]), cls="car", size=(1.9, 4.5, 1.6))
    detector = TruthProjectionDetector({"b": behind, "f": far}, max_range_m=60.0)
    assert detector.detect(frame_at()) == []


def test_class_filter_is_respected():
    objs = {
        "c": TruthObject("c", np.array([0.0, 25.0, 0.8]), cls="car", size=(1.9, 4.5, 1.6)),
        "b": TruthObject("b", np.array([4.0, 25.0, 0.6]), cls="barrier", size=(0.4, 2.0, 1.0)),
    }
    detections = TruthProjectionDetector(objs, classes=("car",)).detect(frame_at())
    assert [d.cls for d in detections] == ["car"]


def test_truth_projection_is_not_wired_in_as_scripted_detections():
    """A session must never hand out ground truth as if it were perception."""
    from geoloc_agent.io.nuscenes import NuScenesSession

    assert NuScenesSession.scripted_detections(object()) is None


# --- nuScenes (needs the download) ------------------------------------------


@needs_data
def test_tables_load_and_derive_the_devkit_fields():
    from geoloc_agent.io.nuscenes_tables import NuScenesTables

    tables = NuScenesTables(DATAROOT, "v1.0-mini")
    assert len(tables.scene) == 10
    sample = tables.sample[0]
    # The three fields the devkit derives rather than stores.
    assert "CAM_FRONT" in sample["data"]
    assert sample["anns"]
    annotation = tables.get("sample_annotation", sample["anns"][0])
    assert "." in annotation["category_name"]


@needs_data
def test_session_produces_valid_posed_frames():
    from geoloc_agent.io.nuscenes import NuScenesSession

    session = NuScenesSession(dataroot=DATAROOT, scene="scene-0655", version="v1.0-mini")
    frames = list(session.frames())
    assert len(frames) > 20
    assert all(f.is_keyframe for f in frames)
    for frame in frames[:5]:
        assert np.allclose(frame.pose.R.T @ frame.pose.R, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(frame.pose.R), 1.0, atol=1e-9)
    # CAM_FRONT is 1600x900 with a known focal length.
    assert (frames[0].intrinsics.width, frames[0].intrinsics.height) == (1600, 900)
    # Intrinsics are calibrated per vehicle, so this varies by a percent or two
    # between scenes; pin the plausible range rather than one scene's value.
    assert 1200.0 < frames[0].intrinsics.fx < 1300.0
    # Timestamps monotonic and roughly 2 Hz keyframes.
    times = [f.timestamp for f in frames]
    assert times == sorted(times)
    assert 0.3 < np.median(np.diff(times)) < 0.7


@needs_data
def test_camera_is_level_and_faces_forward():
    """Catches a transposed extrinsic, which produces a plausible but wrong map."""
    from geoloc_agent.io.nuscenes import NuScenesSession

    session = NuScenesSession(dataroot=DATAROOT, scene="scene-0655", version="v1.0-mini")
    frames = list(session.frames())
    for frame in frames[:10]:
        # The camera's -y axis is up; for a road vehicle it must point skyward.
        assert (frame.pose.R @ [0, -1, 0])[2] > 0.95
        # And its optical axis must be near-horizontal.
        assert abs((frame.pose.R @ [0, 0, 1])[2]) < 0.15
        # Camera sits at a plausible height above the map ground plane.
        assert 0.5 < frame.pose.t[2] < 3.0


@needs_data
def test_geolocation_accuracy_on_real_data():
    """Phase 2 acceptance: same fuse/ code, scored against real 3-D annotations."""
    import collections

    from geoloc_agent.fuse.tracker import TrackerConfig
    from geoloc_agent.io.nuscenes import NuScenesSession
    from geoloc_agent.pipeline import run_pipeline

    session = NuScenesSession(dataroot=DATAROOT, scene="scene-0655", version="v1.0-mini")
    truth = session.truth()
    result = run_pipeline(
        session,
        detector=TruthProjectionDetector(truth, max_range_m=60.0),
        bearing_sigma_px=4.0,
        use_size_prior=True,
        tracker_config=TrackerConfig(process_noise_per_s=0.0),
    )
    assert len(result.all_tracks) > 40
    # Most tracks die before the run ends as objects leave the field of view;
    # scoring only survivors would measure a handful of them.
    assert len(result.completed_tracks) > len(result.final_tracks)

    last_frame = result.frames[-1].frame_id
    errors = []
    for track in result.all_tracks:
        record = result.track_records.get(track.track_id)
        if not record or not record.truth_ids or track.truth_id not in truth:
            continue
        counts = collections.Counter(record.truth_ids)
        pure = counts.most_common(1)[0][1] == len(record.truth_ids)
        if pure and track.n_obs >= 3 and not track.degenerate:
            errors.append(float(np.linalg.norm(track.mean - truth[track.truth_id].at(last_frame))))

    assert len(errors) >= 20, f"only {len(errors)} scoreable tracks"
    assert float(np.median(errors)) < 3.0, f"median {np.median(errors):.2f} m"


@needs_data
def test_wgs84_conversion_lands_in_boston():
    from geoloc_agent.io.nuscenes import NuScenesSession

    session = NuScenesSession(dataroot=DATAROOT, scene="scene-0655", version="v1.0-mini")
    frame = next(iter(session.frames()))
    lat, lon, _ = session.to_wgs84(frame.pose.t)
    assert 42.0 < lat < 42.7, lat
    assert -71.3 < lon < -70.8, lon
    # And the assumption is stated, not implied.
    assert "assumed origin" in session.georeference_note
    assert "no georeference" in session.georeference_note
