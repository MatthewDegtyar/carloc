"""`geoloc-parking`: build signatures on a street of parked cars, then verify."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from geoloc_agent.analysis.appearance import sample_appearance
from geoloc_agent.analysis.concealment import physical_extent
from geoloc_agent.detect.truth_projection import TruthProjectionDetector
from geoloc_agent.io.nuscenes import NuScenesSession
from geoloc_agent.parking.signature import summarise
from geoloc_agent.parking.verify import CITE_LOG_ODDS, Population, compare

VEHICLES = ("car", "truck", "bus")


def gather(scene: str, dataroot: str, want_frames: bool):
    session = NuScenesSession(dataroot=dataroot, scene=scene, version="v1.0-mini",
                              load_images=True, min_visibility=3)
    truth = session.truth()
    detector = TruthProjectionDetector(truth, max_range_m=45.0, classes=VEHICLES)
    looks: dict[str, list] = defaultdict(list)
    per_frame = []

    for frame in session.frames():
        if frame.image is None:
            continue
        intr = frame.intrinsics
        entries = []
        for obj in truth.values():
            if obj.cls not in VEHICLES:
                continue
            detection = detector._project(obj, frame)
            if detection is None:
                continue
            distance = float(np.linalg.norm(obj.at(frame.frame_id) - frame.pose.t))
            width, height = physical_extent(detection.bbox, distance, intr.fx, intr.fy)
            appearance = sample_appearance(frame.image, detection.bbox)
            if appearance is None:
                continue
            position = obj.at(frame.frame_id)
            rotation = obj.rotation_at(frame.frame_id)
            key = f"{scene}/{obj.obj_id}"
            looks[key].append(dict(
                cls=obj.cls, tone=appearance.tone, value=appearance.value,
                hue_deg=appearance.hue_deg, saturation=appearance.saturation,
                length_m=width, width_m=height,
                offset_along_m=float(position[0]), offset_across_m=float(position[1]),
                heading_deg=float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
                position_sigma_m=0.5, instance_id=obj.obj_id, bay_id=None, R=distance))
            entries.append((detection.bbox, appearance.tone, width, len(looks[key])))
        if want_frames:
            per_frame.append((frame, entries))
    return looks, per_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geoloc-parking", description=__doc__)
    parser.add_argument("--dataroot", default="sessions/nuscenes")
    parser.add_argument("--scene", default="scene-0655")
    parser.add_argument("--also", nargs="*",
                        default=["scene-0916", "scene-0061", "scene-0103"],
                        help="extra scenes, used only to widen the population")
    parser.add_argument("--out", type=Path, default=Path("reports/parking_verify.mp4"))
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args(argv)

    print(f"[parking] {args.scene}: building signatures ...", flush=True)
    looks, per_frame = gather(args.scene, args.dataroot, want_frames=True)
    for extra in args.also:
        more, _ = gather(extra, args.dataroot, want_frames=False)
        looks.update(more)

    # Same vehicle, two viewpoints: split each vehicle's looks in half.
    pairs = []
    for key, entries in looks.items():
        if len(entries) < 8:
            continue
        half = len(entries) // 2
        before, after = summarise(entries[:half]), summarise(entries[half:])
        if before and after and before.usable and after.usable:
            pairs.append((key, before, after))

    singles = [s for s in (summarise(v) for v in looks.values()) if s and s.usable]
    population = Population.fit(singles)
    print(f"[parking] {len(singles)} vehicles, {len(pairs)} with two usable viewpoints")

    def appearance_only(a, b) -> float:
        verdict = compare(a, b, population)
        return (verdict.log_odds - verdict.contributions.get("bay_offset", 0.0)
                - verdict.contributions.get("heading", 0.0))

    same = np.array([appearance_only(a, b) for _, a, b in pairs])
    diff = []
    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            diff.append(appearance_only(singles[i], singles[j]))
    diff = np.array(diff)

    print(f"[parking] same-vehicle  n={len(same)}  median {np.median(same):+.2f}  "
          f"cleared {int((same >= CITE_LOG_ODDS).sum())}")
    print(f"[parking] different     n={len(diff)}  p95 {np.percentile(diff, 95):+.2f}  "
          f"cleared {int((diff >= CITE_LOG_ODDS).sum())}")

    from geoloc_agent.parking.render import render

    out = render(None, per_frame, same, diff, CITE_LOG_ODDS, args.out,
                 scene=args.scene, fps=args.fps)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
