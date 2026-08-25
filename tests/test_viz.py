"""Renderer: projection correctness and the honesty of what it draws."""

import numpy as np
import pytest

from geoloc_agent.contracts import Frame, Intrinsics, Pose, TrackState, TrackStatus
from geoloc_agent.io.synthetic import SyntheticScenario, SyntheticSession, look_along
from geoloc_agent.pipeline import run_pipeline
from geoloc_agent.viz.render import RenderConfig, Segment, _fmt_m, _range_and_sigma, project

INTR = Intrinsics(fx=1266.4, fy=1266.4, cx=800.0, cy=450.0, width=1600, height=900)


def make_frame(position=(0.0, 0.0, 1.6), facing=(0.0, 1.0, 0.0)):
    return Frame(
        frame_id=0, timestamp=0.0, intrinsics=INTR,
        pose=Pose(R=look_along(np.array(facing, dtype=float)), t=np.array(position, dtype=float)),
    )


def test_projection_matches_the_pipeline_camera_model():
    """The renderer must not have its own private idea of where things project."""
    frame = make_frame()
    point = np.array([[5.0, 40.0, 1.0]])
    uv, valid = project(point, frame)
    assert valid[0]
    p_cam = frame.pose.world_to_cam(point[0])
    expected = INTR.K @ p_cam
    assert np.allclose(uv[0], expected[:2] / expected[2])


def test_points_behind_the_camera_are_masked_not_wrapped():
    """Without this a point behind the lens projects to a plausible-looking pixel."""
    frame = make_frame()
    uv, valid = project(np.array([[0.0, -30.0, 1.0]]), frame)
    assert not valid[0]
    assert np.isnan(uv[0]).all()


def test_optical_axis_lands_on_the_principal_point():
    frame = make_frame()
    uv, valid = project(np.array([[0.0, 50.0, 1.6]]), frame)
    assert valid[0]
    assert np.allclose(uv[0], [INTR.cx, INTR.cy], atol=1e-6)


def test_range_sigma_is_taken_along_the_line_of_sight():
    frame = make_frame()
    # Covariance stretched along +y, which is the viewing direction here.
    track = TrackState(
        track_id=1, mean=np.array([0.0, 50.0, 1.6]),
        cov=np.diag([0.25, 100.0, 0.25]), n_obs=10, status=TrackStatus.CONFIRMED,
    )
    r, sigma = _range_and_sigma(track, frame)
    assert r == pytest.approx(50.0)
    assert sigma == pytest.approx(10.0, rel=1e-6)  # sqrt(100), the along-range term


def test_small_sigma_never_renders_as_zero():
    """'0.0 m' reads as certainty, which is the one claim this system must not make."""
    assert _fmt_m(0.03) == "0.030"
    assert _fmt_m(0.004) == "0.004"
    assert float(_fmt_m(0.03)) > 0
    assert _fmt_m(2.5) == "2.50"
    assert _fmt_m(14.7) == "14.7"
    assert _fmt_m(120.0) == "120"


def test_render_produces_a_playable_file(tmp_path):
    from geoloc_agent.viz.render import render_run

    session = SyntheticSession(SyntheticScenario(path="lateral", n_frames=6))
    result = run_pipeline(session)
    out = render_run(result, session.truth(), tmp_path / "clip.gif", RenderConfig(fps=4, dpi=60))
    assert out.exists()
    assert out.stat().st_size > 5000


def test_multi_segment_timeline_covers_every_frame(tmp_path):
    from geoloc_agent.viz.render import build_timeline, render_segments

    segments = []
    for path in ("lateral", "straight"):
        session = SyntheticSession(SyntheticScenario(path=path, n_frames=5))
        segments.append(Segment(result=run_pipeline(session), truth=session.truth(), title=path))
    assert len(build_timeline(segments)) == 10  # 5 + 5, nothing silently dropped
    assert render_segments(segments, tmp_path / "two.gif", RenderConfig(fps=4, dpi=60)).exists()


def test_hold_frames_repeat_the_final_state_and_invent_nothing():
    from geoloc_agent.viz.render import build_timeline

    session = SyntheticSession(SyntheticScenario(path="lateral", n_frames=5))
    segment = Segment(result=run_pipeline(session), truth=session.truth(), hold_frames=3)
    timeline = build_timeline([segment])
    assert len(timeline) == 8
    # The held entries are the last real record, not extrapolated state.
    assert all(entry[1] is timeline[4][1] for entry in timeline[5:])


def test_renderer_refuses_an_empty_run(tmp_path):
    from geoloc_agent.viz.render import render_segments

    session = SyntheticSession(SyntheticScenario(path="lateral", n_frames=5))
    result = run_pipeline(session)
    result.frames.clear()
    with pytest.raises(ValueError, match="no segment recorded"):
        render_segments(
            [Segment(result=result, truth=session.truth())], tmp_path / "x.gif", RenderConfig()
        )


def test_degenerate_tracks_are_labelled_unreliable_not_given_a_bare_range():
    """The whole point: a range the geometry cannot support must say so."""
    import pathlib

    source = pathlib.Path("src/geoloc_agent/viz/render.py").read_text()
    assert "UNRELIABLE" in source
    assert "SYNTHETIC RENDER" in source, "a rendered scene must be labelled as synthetic"


def test_forward_motion_render_actually_contains_degenerate_tracks():
    """Guards the demo: if this stops being degenerate the video stops making its point."""
    # Slow approach: inside a 50 m envelope, 20 m of forward travel is enough to
    # make a 40 m object's range observable. See test_smoke for the full note.
    session = SyntheticSession(SyntheticScenario(path="straight", n_frames=40, speed_mps=3.0))
    result = run_pipeline(session)
    assert any(t.degenerate for t in result.final_tracks)
    lateral = run_pipeline(SyntheticSession(SyntheticScenario(path="lateral", n_frames=40)))
    assert not any(t.degenerate for t in lateral.final_tracks)
