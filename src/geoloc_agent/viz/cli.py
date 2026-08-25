"""`geoloc-render`: render a pipeline run to video.

Default output is a two-part clip contrasting the two geometry regimes, because
the well-conditioned result only means something next to the degenerate one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from geoloc_agent.fuse.tracker import TrackerConfig
from geoloc_agent.io.synthetic import SyntheticScenario, SyntheticSession
from geoloc_agent.noise import NoiseModel
from geoloc_agent.pipeline import run_pipeline
from geoloc_agent.viz.render import RenderConfig, Segment, render_segments

BEARING_SIGMA_PX = 2.0
BEARING_SIGMA_RAD = BEARING_SIGMA_PX / 1266.4


def _run(path: str, frames: int, seed: int, arc_radius: float = 40.0):
    session = SyntheticSession(
        SyntheticScenario(path=path, n_frames=frames, arc_radius_m=arc_radius, name=path)
    )
    result = run_pipeline(
        session,
        noise=NoiseModel(bearing_sigma=BEARING_SIGMA_RAD),
        seed=seed,
        bearing_sigma_px=BEARING_SIGMA_PX,
    )
    return result, session.truth()


def _nuscenes_segment(dataroot: str, scene: str, fps: int) -> Segment:
    """A real nuScenes scene, scored end to end.

    Keyframes only. Camera sweeps run at 12 Hz and would make a smoother clip,
    but they carry no annotations -- the truth-projection detector would have to
    reuse the last keyframe's boxes, which smears every moving object. A 2 Hz
    clip that is right beats a 12 Hz one that is interpolated.
    """
    from geoloc_agent.detect.truth_projection import TruthProjectionDetector
    from geoloc_agent.io.nuscenes import NuScenesSession

    session = NuScenesSession(
        dataroot=dataroot, scene=scene, version="v1.0-mini", load_images=True
    )
    truth = session.truth()
    result = run_pipeline(
        session,
        detector=TruthProjectionDetector(truth, max_range_m=60.0),
        bearing_sigma_px=4.0,
        use_size_prior=True,
        tracker_config=TrackerConfig(process_noise_per_s=0.0),
    )
    return Segment(
        result=result,
        truth=truth,
        hold_frames=fps * 2,
        map_half_span=55.0,
        title=f"REAL DATA — nuScenes {scene} ({session.map_name}), CAM_FRONT",
        banner=(
            "REAL IMAGERY  |  real ego pose  |  detector = ground-truth projection "
            "(an oracle, not a perception model)"
        ),
        caption=(
            "Every box carries a range and its 1-sigma. Green is a fix worth acting on,\n"
            "red is geometry that cannot support one. Faint wireframes are the 3-D truth boxes."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geoloc-render", description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "nuscenes"), default="synthetic")
    parser.add_argument("--dataroot", default="sessions/nuscenes")
    parser.add_argument("--scene", default="scene-0655")
    parser.add_argument("--out", type=Path, default=Path("reports/geoloc_demo.mp4"))
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument(
        "--path", choices=("lateral", "straight", "arc", "both"), default="both",
        help="'both' renders the good and degenerate cases as one two-part clip.",
    )
    args = parser.parse_args(argv)

    config = RenderConfig(fps=args.fps, dpi=args.dpi)
    segments = []

    if args.source == "nuscenes":
        segments.append(_nuscenes_segment(args.dataroot, args.scene, args.fps))
        for segment in segments:
            print(f"[render] {args.scene}: {len(segment.result.all_tracks)} tracks")
        out = render_segments(segments, args.out, config)
        print(f"\nwrote {out}")
        return 0

    if args.path in ("lateral", "both"):
        result, truth = _run("lateral", args.frames, args.seed)
        segments.append(
            Segment(
                result=result, truth=truth, hold_frames=args.fps,
                title="1/2  GOOD GEOMETRY — camera strafes across its view direction",
                caption=(
                    "Perpendicular baseline grows every frame, so range becomes observable\n"
                    "and the error ellipses collapse."
                ),
            )
        )

    if args.path in ("straight", "both"):
        result, truth = _run("straight", args.frames, args.seed)
        segments.append(
            Segment(
                result=result, truth=truth, hold_frames=args.fps * 2,
                title="2/2  DEGENERATE GEOMETRY — camera drives straight ahead",
                caption=(
                    "Same objects, same detector, same filter. Motion is along the line of "
                    "sight, so there is\nalmost no perpendicular baseline. Range is barely "
                    "observable, and the system says so."
                ),
            )
        )

    if args.path == "arc":
        result, truth = _run("arc", args.frames, args.seed)
        segments.append(
            Segment(
                result=result, truth=truth, hold_frames=args.fps,
                title="ARC — turning camera sweeps bearing continuously",
                caption="Parallax accumulates through the turn.",
            )
        )

    for segment in segments:
        tracked = len(segment.result.final_tracks)
        print(f"[render] {segment.title.split('—')[0].strip()}: {tracked} tracks")

    out = render_segments(segments, args.out, config)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
