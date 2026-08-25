"""Check a Stray Scanner capture before trusting anything it produces.

    uv run python scripts/validate_capture.py sessions/my_capture

Reports what is actually in the capture and checks the conventions that, when
wrong, produce a self-consistent and completely incorrect map. It does not assume
the export format -- it reports what it finds, because the format varies across
app versions and this loader has never met real data.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def check(label: str, passed: bool | None, detail: str = "") -> bool:
    tag = OK if passed else (WARN if passed is None else BAD)
    print(f"[{tag}] {label}" + (f"  --  {detail}" if detail else ""))
    return bool(passed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args(argv)
    path = args.path

    print(f"\n=== capture: {path} ===\n")
    if not path.is_dir():
        print(f"[{BAD}] not a directory")
        return 1

    print("--- files present ---")
    for name in ("camera_matrix.csv", "odometry.csv", "rgb.mp4"):
        found = (path / name).exists()
        check(name, found or None if name == "rgb.mp4" else found)
    for name in ("depth", "confidence"):
        d = path / name
        n = len(list(d.glob("*"))) if d.is_dir() else 0
        check(f"{name}/ (LiDAR, optional)", None if n == 0 else True, f"{n} files")

    print("\n--- loader ---")
    from geoloc_agent.io.stray_scanner import StrayScannerSession

    try:
        session = StrayScannerSession(path, max_frames=args.frames)
    except Exception as exc:  # noqa: BLE001 - this script exists to report failures
        check("StrayScannerSession loads", False, f"{type(exc).__name__}: {exc}")
        return 1
    frames = list(session.frames())
    check("StrayScannerSession loads", True, f"{len(frames)} frames read")
    if not frames:
        return 1

    intr = session.intrinsics
    print(
        f"        intrinsics fx={intr.fx:.1f} fy={intr.fy:.1f} "
        f"cx={intr.cx:.1f} cy={intr.cy:.1f}  {intr.width}x{intr.height}"
    )
    check(
        "principal point near image centre",
        abs(intr.cx - intr.width / 2) < intr.width * 0.15
        and abs(intr.cy - intr.height / 2) < intr.height * 0.15,
        "if this fails, camera_matrix.csv and the video resolution disagree",
    )

    print("\n--- pose conventions (the ones that fail silently) ---")
    ok = True
    ok &= check(
        "rotations orthonormal, right-handed",
        all(
            np.allclose(f.pose.R.T @ f.pose.R, np.eye(3), atol=1e-6)
            and np.isclose(np.linalg.det(f.pose.R), 1.0, atol=1e-6)
            for f in frames
        ),
    )
    times = np.array([f.timestamp for f in frames])
    ok &= check("timestamps monotonic", bool(np.all(np.diff(times) >= 0)))
    if len(times) > 2:
        rate = 1.0 / max(np.median(np.diff(times)), 1e-9)
        check("frame rate plausible", 5 < rate < 130, f"{rate:.1f} Hz")

    ups = np.array([f.pose.R @ [0, -1, 0] for f in frames])
    median_up = float(np.median(ups[:, 2]))
    ok &= check(
        "camera up-axis points skyward",
        median_up > 0.5,
        f"median z of camera -y is {median_up:+.3f}; a negative value means the "
        f"ARKit->ENU axis map is upside down",
    )

    heights = np.array([f.pose.t[2] for f in frames])
    check(
        "camera height varies little (held steady)",
        None if float(np.ptp(heights)) > 1.0 else True,
        f"z range {float(np.ptp(heights)):.2f} m",
    )

    print("\n--- motion (this decides whether geolocation is possible at all) ---")
    positions = np.array([f.pose.t for f in frames])
    total = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    net = float(np.linalg.norm(positions[-1] - positions[0]))
    print(f"        path length {total:.2f} m, net displacement {net:.2f} m")
    ok &= check("camera actually moved", total > 1.0, "a static camera cannot triangulate")

    forward = np.array([f.pose.R @ [0, 0, 1] for f in frames])
    step = np.diff(positions, axis=0)
    mid_forward = forward[:-1]
    along = np.abs(np.sum(step * mid_forward, axis=1))
    along_vec = mid_forward * np.sum(step * mid_forward, axis=1)[:, None]
    across = np.linalg.norm(step - along_vec, axis=1)
    ratio = float(np.sum(across) / max(np.sum(along) + np.sum(across), 1e-9))
    # Counts toward the exit code. This is the single thing most likely to make a
    # capture worthless, and it is invisible until you score the result.
    ok &= check(
        "motion is across the view, not along it",
        ratio > 0.35,
        f"{ratio * 100:.0f}% of motion is perpendicular to the look direction. "
        f"Walking straight at a subject gives no parallax and range stays unobservable.",
    )

    print("\n--- imagery ---")
    if (path / "rgb.mp4").exists():
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height,nb_frames,r_frame_rate", "-of", "csv=p=0",
                 str(path / "rgb.mp4")],
                capture_output=True, text=True, check=True,
            )
            print(f"        rgb.mp4: {probe.stdout.strip()}")
            width_height = probe.stdout.split(",")[:2]
            check(
                "video resolution matches intrinsics",
                [str(intr.width), str(intr.height)] == width_height,
                f"intrinsics say {intr.width}x{intr.height}, video says "
                f"{'x'.join(width_height)} -- a mismatch scales every bearing",
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            check("ffprobe", None, str(exc)[:80])

    print("\n--- georeference ---")
    print("        " + session.georeference_note.replace("\n", "\n        "))

    print()
    if ok:
        print("Conventions look right. Geolocation will still be RELATIVE unless you")
        print("supply an origin and a heading offset -- ARKit has no idea where north is.")
    else:
        print("Fix the failures above before trusting any coordinate this produces.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
