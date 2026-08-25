"""`geoloc-render`: render a pipeline run to video.

Default output is a two-part clip contrasting the two geometry regimes, because
the well-conditioned result only means something next to the degenerate one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geoloc-render", description=__doc__)
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
