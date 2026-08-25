"""What range can this camera actually work at?

    uv run python scripts/range_envelope.py --fx 1500
    uv run python scripts/range_envelope.py --capture sessions/my_capture

Two limits decide it, and they bind in the opposite order to intuition.

**Geometry** is the easy one: range accuracy is bought with sideways walk, via
`sigma_R = sqrt(2) R^2 sigma_theta / B`. Walk further across the scene, get a
better fix, at any range.

**Detection** is the hard one. An object's apparent size falls as 1/R, and below
roughly 20 px a small detector stops finding it. That is what actually sets the
ceiling.

The recall numbers below were MEASURED, not assumed: YOLO11n against nuScenes
annotations filtered to >=60% visibility, pooled over four boston-seaport scenes.
They are indexed by apparent pixel height, which is what transfers between
cameras -- range depends on focal length, pixels on target do not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Measured per-frame recall by ground-truth box height, YOLO11n @ score 0.25.
# (px_height, full_frame_recall, tiled_recall)
RECALL_CURVE = [
    (15, 0.03, 0.09),
    (25, 0.17, 0.30),
    (35, 0.34, 0.41),
    (45, 0.40, 0.50),
    (55, 0.44, 0.55),
    (65, 0.60, 0.67),
    (75, 0.61, 0.63),
    (90, 0.82, 0.84),
]

OBJECT_HEIGHTS = {"person": 1.7, "car": 1.55}


def recall_at(px: float, tiled: bool) -> float:
    xs = [c[0] for c in RECALL_CURVE]
    ys = [c[2] if tiled else c[1] for c in RECALL_CURVE]
    return float(np.interp(px, xs, ys))


def p_track(per_frame: float, n_frames: int, need: int = 3) -> float:
    """Probability of at least `need` detections across `n_frames`.

    This is the number that matters, and it is why a low per-frame recall is not
    fatal. A track needs a handful of hits, not a hit every frame: at 30% recall
    over 40 frames, three or more detections is essentially certain.

    OPTIMISTIC, and knowingly so: it assumes detections are independent between
    frames. They are not. A detector that misses an object because of its pose,
    lighting or occlusion tends to keep missing it while those persist, so real
    failures are correlated and the true figure sits below this. Treat it as an
    upper bound on track formation, not a prediction.
    """
    from math import comb

    p = float(np.clip(per_frame, 1e-9, 1 - 1e-9))
    below = sum(comb(n_frames, k) * p**k * (1 - p) ** (n_frames - k) for k in range(need))
    return float(1.0 - below)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fx", type=float, default=None, help="focal length in pixels")
    parser.add_argument("--capture", type=Path, default=None, help="read fx from a capture")
    parser.add_argument("--frames", type=int, default=40, help="frames the object stays in view")
    parser.add_argument("--object", choices=sorted(OBJECT_HEIGHTS), default="person")
    parser.add_argument("--bearing-sigma-px", type=float, default=2.5)
    args = parser.parse_args(argv)

    fx = args.fx
    if args.capture:
        from geoloc_agent.io.stray_scanner import StrayScannerSession

        fx = StrayScannerSession(args.capture, max_frames=2).intrinsics.fx
        print(f"read fx = {fx:.1f} px from {args.capture}")
    if fx is None:
        fx = 1500.0
        print("no --fx or --capture given; assuming fx = 1500 px (iPhone wide, 1920x1440)")

    height = OBJECT_HEIGHTS[args.object]
    sigma_theta = args.bearing_sigma_px / fx
    print(f"\nobject: {args.object} ({height} m tall)   sigma_theta = {sigma_theta:.2e} rad")
    print(f"assuming it stays in view for {args.frames} frames\n")

    print("DETECTION LIMIT  (per-frame recall, and the chance a track forms at all)")
    header = f"{'range':>7} {'px tall':>8} {'full-frame':>22} {'tiled':>22}"
    print(header)
    print(f"{'':7} {'':8} {'recall  P(track)':>22} {'recall  P(track)':>22}")
    for R in (25, 50, 75, 100, 125, 150, 200):
        px = fx * height / R
        full, tile = recall_at(px, False), recall_at(px, True)
        print(
            f"{R:5d} m {px:7.0f} {full * 100:9.0f}% {p_track(full, args.frames) * 100:9.0f}%"
            f" {tile * 100:12.0f}% {p_track(tile, args.frames) * 100:9.0f}%"
        )

    print("\nGEOMETRY LIMIT  (1-sigma range error for a given sideways walk)")
    walks = (2, 5, 10, 20)
    print(f"{'range':>7}" + "".join(f"{w:>10} m walk" for w in walks))
    for R in (25, 50, 75, 100, 125, 150, 200):
        cells = ""
        for B in walks:
            sigma = np.sqrt(2) * R**2 * sigma_theta / B
            cells += f"{sigma:9.1f} m ({sigma / R * 100:3.0f}%)".rjust(12)
        print(f"{R:5d} m " + cells)

    print(
        "\nDetection sets the ceiling; geometry sets the precision below it.\n"
        "P(track) stays high well past the range where per-frame recall collapses,\n"
        "because a track needs three detections rather than one per frame. It\n"
        "assumes those detections are independent, which they are not -- misses\n"
        "are correlated by pose and occlusion -- so read it as an upper bound."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
