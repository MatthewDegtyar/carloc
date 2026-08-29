"""`carloc` — plateless parking analysis from dashcam video.

Three commands that each run end to end on their own:

    carloc count  VIDEO --start S --end E   detect + track + atomically count parked cars
    carloc zone   CODE                      fetch a ParkMobile on-street zone's geometry
    carloc survey VIDEO                      score on-street parking density across a video

`count` is the core vision pipeline (carloc/pipeline.py). Absolute lat/lon needs
per-street anchoring (see the README); by default it reports each car's position
*along the street* with its uncertainty, which is what the count and the overstay
matching actually depend on.
"""

from __future__ import annotations

import argparse
import csv
import sys


def _cmd_count(args) -> int:
    from carloc.video import count_parked

    print("  detecting… (this runs the model over the segment)", file=sys.stderr)
    cars = count_parked(args.video, args.start, args.end, lateral_m=args.lateral,
                        both_sides=not args.one_side, fps=args.fps,
                        speed_mps=args.speed, min_frames=args.min_frames)
    left = sum(1 for c in cars if c.side == "left")
    rebuilt = sum(1 for c in cars if c.n_tracklets > 1)
    print(f"\n{len(cars)} parked cars  ({left} left kerb, {len(cars)-left} right)  "
          f"· {rebuilt} rebuilt from occlusion-split tracklets")
    from collections import Counter
    print("colours:", dict(Counter(c.color for c in cars)),
          " classes:", dict(Counter(c.vehicle_class for c in cars)))
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["along_m", "sigma_along_m", "side", "class", "color", "confidence",
                        "abeam_t", "first_t", "last_t", "n_detections", "n_tracklets"])
            for c in cars:
                w.writerow([c.along_m, c.sigma_along_m, c.side, c.vehicle_class, c.color,
                            c.confidence, c.abeam_t, c.first_t, c.last_t,
                            c.n_detections, c.n_tracklets])
        print(f"wrote {args.out}")
    return 0
    return 0


def _cmd_zone(args) -> int:
    from carloc.parkmobile import fetch

    zone = fetch(args.code, supplier_id=args.supplier)
    if zone is None:
        print(f"zone {args.code}: not found / no on-street geometry", file=sys.stderr)
        return 1
    print(f"zone {zone.signage_code}  internal {zone.internal_code}  type {zone.zone_type}")
    print(f"  {len(zone.points)} anchor points"
          + (f"  ·  {zone.name}" if zone.name else ""))
    b = zone.bounds
    if b:
        print(f"  bounds lat {b[1]:.6f}..{b[3]:.6f}  lon {b[0]:.6f}..{b[2]:.6f}")
    if args.kml:
        from carloc.export import points_to_kml
        points_to_kml(
            [{"name": f"{zone.signage_code} #{i}", "lon": lo, "lat": la, "props": {}}
             for i, (lo, la) in enumerate(zone.points)],
            args.kml, name=f"ParkMobile zone {zone.signage_code}")
        print(f"wrote {args.kml}")
    return 0


def _cmd_survey(args) -> int:
    import glob
    import math
    import subprocess
    import tempfile

    import numpy as np
    from PIL import Image
    from rfdetr import RFDETRBase

    from carloc.rfdetr_detect import COCO_VEHICLES

    Wp, Hp, F = 1280, 720, 458.0
    model = RFDETRBase()
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", args.video, "-vf",
                        f"fps=1/{args.step},scale={Wp}:{Hp}", "-pix_fmt", "yuvj420p",
                        f"{tmp}/s_%05d.jpg"], check=True)
        files = sorted(glob.glob(f"{tmp}/s_*.jpg"))
        rows = []
        for i, path in enumerate(files):
            arr = np.asarray(Image.open(path).convert("RGB"))
            res = model.predict(arr, threshold=0.45)
            cls = np.array(res.class_id)
            box = np.array(res.xyxy)
            kerb = 0
            for (x1, y1, x2, y2) in box[np.isin(cls, list(COCO_VEHICLES))]:
                if (y2 - y1) < 55 or (y1 + y2) / 2 < Hp * 0.42:
                    continue
                if abs(math.degrees(math.atan(((x1 + x2) / 2 - Wp / 2) / F))) > 18:
                    kerb += 1
            rows.append((i * args.step, kerb))
    kerbs = np.array([r[1] for r in rows])
    print(f"{len(rows)} samples every {args.step}s · median kerb vehicles {np.median(kerbs):.0f}"
          f" · max {kerbs.max()}")
    print("parking-rich windows (kerb ≥ 3):")
    for t, k in rows:
        if k >= 3:
            print(f"  {t//60:02d}:{t%60:02d}  ({k})")
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "kerb_vehicles"])
            w.writerows(rows)
        print(f"wrote {args.out}")
    return 0


def _cmd_confidence(args) -> int:
    from carloc.confidence import sweep
    from carloc.video import detect_segment

    print("  detecting once…", file=sys.stderr)
    left, right = detect_segment(args.video, args.start, args.end,
                                 both_sides=not args.one_side, fps=args.fps,
                                 speed_mps=args.speed)
    thresholds = tuple(range(args.min, args.max + 1))
    rows = sweep(left, right, thresholds=thresholds, lateral_m=args.lateral,
                 n_boot=args.boot)
    print(f"\nparked-car count vs frame-confidence  ({args.boot} bootstraps, 90% CI)")
    print(f"{'min_frames':>10}  {'count':>6}  {'90% CI':>12}")
    for r in rows:
        print(f"{r.min_frames:>10}  {r.count:>6}  {f'[{r.ci_lo}, {r.ci_hi}]':>12}")
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["min_frames", "count", "ci_lo", "ci_hi", "boot_median"])
            for r in rows:
                w.writerow([r.min_frames, r.count, r.ci_lo, r.ci_hi, r.boot_median])
        print(f"wrote {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="carloc", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("count", help="detect + track + count parked cars in a video segment")
    c.add_argument("video")
    c.add_argument("--start", type=float, required=True, help="segment start (seconds)")
    c.add_argument("--end", type=float, required=True, help="segment end (seconds)")
    c.add_argument("--lateral", type=float, default=7.0, help="metres from camera lane to kerb")
    c.add_argument("--one-side", action="store_true", help="left kerb only (default: both)")
    c.add_argument("--fps", type=float, default=4.0)
    c.add_argument("--min-frames", type=int, default=2, dest="min_frames",
                   help="frame-confidence: fewest frames a car must be tracked across")
    c.add_argument("--speed", type=float, default=7.0, dest="speed",
                   help="avg speed m/s for metric scale (or use a GPS trajectory in code)")
    c.add_argument("--out", help="write per-car CSV to this path")
    c.set_defaults(func=_cmd_count)

    z = sub.add_parser("zone", help="fetch a ParkMobile on-street zone by signage code")
    z.add_argument("code")
    z.add_argument("--supplier", default="978040", help="supplierId (default Miami)")
    z.add_argument("--kml", help="write the zone anchors to this KML path")
    z.set_defaults(func=_cmd_zone)

    s = sub.add_parser("survey", help="score on-street parking density across a whole video")
    s.add_argument("video")
    s.add_argument("--step", type=int, default=6, help="sample every N seconds")
    s.add_argument("--out", help="write the timeline CSV to this path")
    s.set_defaults(func=_cmd_survey)

    cf = sub.add_parser("confidence",
                        help="count vs frame-confidence threshold, with bootstrap CIs")
    cf.add_argument("video")
    cf.add_argument("--start", type=float, required=True)
    cf.add_argument("--end", type=float, required=True)
    cf.add_argument("--min", type=int, default=2, help="lowest min_frames threshold")
    cf.add_argument("--max", type=int, default=6, help="highest min_frames threshold")
    cf.add_argument("--boot", type=int, default=60, help="bootstrap resamples")
    cf.add_argument("--lateral", type=float, default=7.0)
    cf.add_argument("--speed", type=float, default=7.0, help="avg speed m/s (metric scale)")
    cf.add_argument("--one-side", action="store_true")
    cf.add_argument("--fps", type=float, default=4.0)
    cf.add_argument("--out", help="write the CI table CSV to this path")
    cf.set_defaults(func=_cmd_confidence)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
