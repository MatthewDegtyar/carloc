"""Session loaders: the synthetic generator, and the two real-data loaders.

nuScenes and Stray Scanner cannot be exercised without their data, so what is
tested here is everything that does not need it: the interface contract, the
quaternion and axis conventions, and the georeference honesty. Those are also
where the bugs actually live -- a transposed convention is far more likely than
a CSV parse failure, and far harder to notice later.
"""

import numpy as np
import pytest

from geoloc_agent.contracts import Pose
from geoloc_agent.geo import NUSCENES_ORIGINS
from geoloc_agent.io.base import Session, TruthObject
from geoloc_agent.io.nuscenes import NuScenesSession, _simplify_class, quaternion_to_matrix
from geoloc_agent.io.stray_scanner import (
    ARKIT_TO_OPENCV,
    StrayScannerSession,
    quaternion_xyzw_to_matrix,
)
from geoloc_agent.io.synthetic import SyntheticScenario, SyntheticSession, look_along

# --- synthetic ---------------------------------------------------------------


def test_synthetic_session_satisfies_the_interface():
    session = SyntheticSession(SyntheticScenario(n_frames=10))
    assert isinstance(session, Session)
    frames = list(session.frames())
    assert len(frames) == 10
    assert all(isinstance(f.pose, Pose) for f in frames)
    assert session.truth()
    assert session.scripted_detections() is not None


def test_timestamps_are_monotonic():
    frames = list(SyntheticSession(SyntheticScenario(n_frames=20, rate_hz=10.0)).frames())
    times = [f.timestamp for f in frames]
    assert times == sorted(times)
    assert times[1] - times[0] == pytest.approx(0.1)


def test_look_along_builds_a_right_handed_camera():
    for direction in ([0, 1, 0], [1, 0, 0], [0.5, 0.5, 0.0], [1, 2, 0]):
        R = look_along(np.array(direction, dtype=float))
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
        # +z of the camera is the view direction.
        expected = np.array(direction, dtype=float)
        assert np.allclose(R @ [0, 0, 1], expected / np.linalg.norm(expected), atol=1e-12)
        # +y of the camera points down in the world.
        assert (R @ [0, 1, 0])[2] < 1e-9


def test_look_along_survives_a_straight_down_view():
    R = look_along(np.array([0.0, 0.0, -1.0]))
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_paths_produce_the_geometry_they_claim():
    lateral = list(SyntheticSession(SyntheticScenario(path="lateral", n_frames=20)).frames())
    straight = list(SyntheticSession(SyntheticScenario(path="straight", n_frames=20)).frames())
    facing = lateral[0].pose.R @ [0, 0, 1]
    lateral_motion = lateral[-1].pose.t - lateral[0].pose.t
    straight_motion = straight[-1].pose.t - straight[0].pose.t
    # Lateral motion is perpendicular to the view; straight motion is along it.
    assert abs(float(lateral_motion @ facing)) < 1e-6
    assert float(straight_motion @ facing) == pytest.approx(np.linalg.norm(straight_motion))


def test_arc_starts_at_the_requested_pose():
    frames = list(
        SyntheticSession(SyntheticScenario(path="arc", n_frames=20, heading_deg=90.0)).frames()
    )
    assert np.allclose(frames[0].pose.t[:2], [0.0, 0.0], atol=1e-9)
    assert np.allclose(frames[0].pose.R @ [0, 0, 1], [0.0, 1.0, 0.0], atol=1e-9)
    # It must actually turn.
    assert not np.allclose(frames[-1].pose.R, frames[0].pose.R)


def test_scenario_validates_its_inputs():
    with pytest.raises(ValueError, match="unknown path"):
        SyntheticScenario(path="teleport")
    with pytest.raises(ValueError, match="at least 2 frames"):
        SyntheticScenario(n_frames=1)


def test_detections_project_onto_their_objects():
    session = SyntheticSession(SyntheticScenario(path="lateral", n_frames=5))
    frame = next(iter(session.frames()))
    for detection in session.scripted_detections()[frame.frame_id]:
        assert frame.intrinsics.contains(*detection.centroid)


def test_truth_object_supports_moving_objects():
    obj = TruthObject("a", np.array([1.0, 2.0, 3.0]))
    assert np.allclose(obj.at(0), [1, 2, 3])
    obj.positions[5] = np.array([9.0, 9.0, 9.0])
    assert np.allclose(obj.at(5), [9, 9, 9])
    assert np.allclose(obj.at(6), [1, 2, 3])


# --- nuScenes ----------------------------------------------------------------


def test_nuscenes_quaternion_is_wxyz_and_orthonormal():
    assert np.allclose(quaternion_to_matrix(np.array([1.0, 0, 0, 0])), np.eye(3))
    # 90 degrees about z: +x maps to +y.
    R = quaternion_to_matrix(np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]))
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_nuscenes_quaternion_rejects_zero():
    with pytest.raises(ValueError, match="zero quaternion"):
        quaternion_to_matrix(np.zeros(4))


def test_nuscenes_class_names_are_simplified():
    assert _simplify_class("vehicle.car") == "car"
    assert _simplify_class("human.pedestrian.adult") == "pedestrian"
    assert _simplify_class("movable_object.barrier") == "barrier"


def test_nuscenes_map_origins_are_documented_as_assumptions():
    """The gotcha the plan calls out: no lat/lon exists in nuScenes."""
    assert "boston-seaport" in NUSCENES_ORIGINS
    for origin in NUSCENES_ORIGINS.values():
        assert origin.provenance, "an origin without provenance is a fabricated datum"
        assert "assumed" in origin.provenance


def test_nuscenes_session_needs_the_devkit_and_says_so():
    with pytest.raises((ImportError, FileNotFoundError, Exception)) as excinfo:
        NuScenesSession(dataroot="/nonexistent/nuscenes", scene=0)
    message = str(excinfo.value).lower()
    assert "nuscenes" in message or "not" in message


def test_nuscenes_session_declares_no_scripted_detections():
    """Real imagery means a real detector; there is no shortcut."""
    assert NuScenesSession.scripted_detections(object()) is None


# --- Stray Scanner -----------------------------------------------------------


def test_arkit_to_opencv_is_a_180_degree_flip_about_x():
    assert np.allclose(ARKIT_TO_OPENCV @ ARKIT_TO_OPENCV, np.eye(3))
    assert np.linalg.det(ARKIT_TO_OPENCV) == pytest.approx(1.0)
    # ARKit looks down -z with +y up; OpenCV looks down +z with +y down.
    assert np.allclose(ARKIT_TO_OPENCV @ [0, 0, -1], [0, 0, 1])
    assert np.allclose(ARKIT_TO_OPENCV @ [0, 1, 0], [0, -1, 0])


def test_stray_quaternion_is_xyzw():
    assert np.allclose(quaternion_xyzw_to_matrix(0, 0, 0, 1), np.eye(3))
    R = quaternion_xyzw_to_matrix(0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_stray_session_fails_clearly_when_the_capture_is_missing():
    with pytest.raises(FileNotFoundError, match="Stray Scanner session"):
        StrayScannerSession("/nonexistent/capture")


def test_stray_loader_parses_a_synthetic_capture(tmp_path):
    """Build a capture on disk so the loader is exercised without a real phone."""
    (tmp_path / "camera_matrix.csv").write_text("1000,0,960\n0,1000,720\n0,0,1\n")
    rows = ["timestamp,frame,x,y,z,qx,qy,qz,qw"]
    for i in range(5):
        rows.append(f"{100.0 + i * 0.1},{i},{i * 0.5},1.5,0.0,0,0,0,1")
    (tmp_path / "odometry.csv").write_text("\n".join(rows) + "\n")

    session = StrayScannerSession(tmp_path)
    frames = list(session.frames())
    assert len(frames) == 5
    assert frames[0].timestamp == pytest.approx(0.0)
    assert frames[-1].timestamp == pytest.approx(0.4)
    assert session.intrinsics.fx == 1000
    for frame in frames:
        assert np.allclose(frame.pose.R.T @ frame.pose.R, np.eye(3), atol=1e-9)
    # ARKit y-up maps to our z-up: the phone's y=1.5 becomes world z=1.5.
    assert frames[0].pose.t[2] == pytest.approx(1.5)


def test_stray_session_is_honest_about_having_no_ground_truth():
    """A phone capture cannot score geolocation error, and must not pretend to."""
    import pathlib

    source = pathlib.Path("src/geoloc_agent/io/stray_scanner.py").read_text()
    assert "no ground truth" in source
    assert "arbitrary origin" in source


def test_stray_georeference_note_states_the_assumption():
    from geoloc_agent.geo import GeoOrigin

    class _Stub(StrayScannerSession):
        def __init__(self, origin):
            self._origin = origin

    assert "no lat/lon" in _Stub(None).georeference_note
    assert "assumed origin" in _Stub(GeoOrigin(42.0, -71.0)).georeference_note


def _write_capture(root, n=40, lateral=True, quat=(0.0, 0.0, 0.0, 1.0)):
    """A minimal Stray Scanner capture on disk, in ARKit conventions."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "camera_matrix.csv").write_text("1000,0,960\n0,1000,720\n0,0,1\n")
    rows = ["timestamp,frame,x,y,z,qx,qy,qz,qw"]
    for i in range(n):
        # ARKit is y-up; a camera at identity looks down -z, so lateral motion is
        # along x and forward motion is along -z.
        x, z = (i * 0.05, 0.0) if lateral else (0.0, -i * 0.05)
        rows.append(f"{100.0 + i / 60:.6f},{i},{x:.4f},1.50,{z:.4f}," + ",".join(map(str, quat)))
    (root / "odometry.csv").write_text("\n".join(rows) + "\n")
    return root


def test_stray_loader_applies_frame_stride(tmp_path):
    """ARKit runs at 60 Hz; consecutive frames add no baseline but cost a full pass."""
    from geoloc_agent.io.stray_scanner import StrayScannerSession

    capture = _write_capture(tmp_path / "cap", n=60)
    assert len(list(StrayScannerSession(capture).frames())) == 60
    strided = list(StrayScannerSession(capture, frame_stride=10).frames())
    assert len(strided) == 6
    assert [f.frame_id for f in strided] == [0, 10, 20, 30, 40, 50]


def test_stray_loader_refuses_images_when_the_video_is_missing(tmp_path):
    """Silence here would pair poses with no imagery and look like a working run."""
    from geoloc_agent.io.stray_scanner import StrayScannerSession

    capture = _write_capture(tmp_path / "cap", n=5)
    with pytest.raises(FileNotFoundError, match="rgb.mp4"):
        StrayScannerSession(capture, load_images=True)


def test_stray_capture_maps_arkit_axes_into_enu(tmp_path):
    """The highest-risk conversion in the loader, pinned.

    ARKit: +y up, camera looks down -z. Ours: +z up, camera looks down +z.
    Getting this wrong yields a self-consistent, entirely wrong map.
    """
    from geoloc_agent.io.stray_scanner import StrayScannerSession

    capture = _write_capture(tmp_path / "cap", n=3)
    frames = list(StrayScannerSession(capture).frames())
    pose = frames[0].pose
    # ARKit y=1.5 (up) must become world z=1.5.
    assert pose.t[2] == pytest.approx(1.5)
    # The camera's -y axis (its "up") must point at the world sky.
    assert (pose.R @ [0, -1, 0])[2] == pytest.approx(1.0, abs=1e-9)
    # Its optical axis must be horizontal for a level phone.
    assert abs((pose.R @ [0, 0, 1])[2]) < 1e-9


def test_validator_passes_a_good_capture_and_flags_forward_motion(tmp_path):
    """The validator must actually catch the mistake it exists to catch."""
    import subprocess
    import sys

    def run(capture):
        return subprocess.run(
            [sys.executable, "scripts/validate_capture.py", str(capture)],
            capture_output=True, text=True,
        )

    good = run(_write_capture(tmp_path / "lateral", n=40, lateral=True))
    assert good.returncode == 0, good.stdout
    assert "across the view" in good.stdout

    # Walking straight at the subject: the degenerate case. Must be reported as a
    # failure AND reflected in the exit code, or it is useless in a script.
    forward = run(_write_capture(tmp_path / "forward", n=40, lateral=False))
    assert forward.returncode != 0, forward.stdout
    assert "0% of motion is perpendicular" in forward.stdout
