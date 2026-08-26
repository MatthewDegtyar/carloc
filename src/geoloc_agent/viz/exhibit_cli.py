"""`geoloc-exhibit`: query-driven analysis of a nuScenes scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from geoloc_agent.analysis.queries import BY_NAME, QUERIES
from geoloc_agent.detect.truth_projection import TruthProjectionDetector
from geoloc_agent.fuse.tracker import TrackerConfig
from geoloc_agent.io.nuscenes import NuScenesSession
from geoloc_agent.pipeline import run_pipeline
from geoloc_agent.viz.exhibition import render_exhibition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geoloc-exhibit", description=__doc__)
    parser.add_argument("--dataroot", default="sessions/nuscenes")
    # scene-0103 carries every query: 116 tracks over 40 frames, no frame
    # empty, and 41 pedestrians so "people" is not a two-box section.
    parser.add_argument("--scene", default="scene-0103")
    parser.add_argument("--out", type=Path, default=Path("reports/exhibition.mp4"))
    parser.add_argument("--query", action="append", choices=sorted(BY_NAME))
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-range", type=float, default=45.0)
    parser.add_argument("--detector", choices=("truth", "yolo"), default="truth")
    args = parser.parse_args(argv)

    session = NuScenesSession(dataroot=args.dataroot, scene=args.scene,
                              version="v1.0-mini", load_images=True, min_visibility=3)
    truth = session.truth()

    if args.detector == "yolo":
        from geoloc_agent.detect.coreml import CoreMLDetector

        detector = CoreMLDetector("models/yolo11n.mlpackage", score_threshold=0.35)
        detector.warmup()
        sigma_px = 6.0
    else:
        detector = TruthProjectionDetector(truth, max_range_m=args.max_range)
        sigma_px = 4.0

    print(f"[exhibit] {args.scene}: running perception + filter once ...", flush=True)
    result = run_pipeline(session, detector=detector, bearing_sigma_px=sigma_px,
                          use_size_prior=True,
                          tracker_config=TrackerConfig(process_noise_per_s=0.0))
    print(f"[exhibit] {len(result.all_tracks)} tracks over {result.n_frames} frames")

    queries = [BY_NAME[n] for n in args.query] if args.query else QUERIES
    out = render_exhibition(result, args.out, scene=args.scene, queries=queries, fps=args.fps)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
