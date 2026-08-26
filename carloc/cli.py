"""`carloc`: every car in the scene, once, with a lat/lon."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from carloc.dedupe import MIN_FIXES, audit, cluster_fixes, sweep
from carloc.detect import CoreMLDetector
from carloc.geo import GeoFit
from carloc.locate import locate
from carloc.session import Session


def collect(session: Session, frames, detector, terrain) -> tuple[list[dict], dict]:
    """Detect in every frame and put each detection on the ground."""
    records: list[dict] = []
    stats = {"detections": 0, "located": 0, "refused": 0}
    reasons: dict[str, int] = {}

    for frame in frames:
        image = frame.image()
        if image is None:
            continue
        for detection in detector.detect(image):
            stats["detections"] += 1
            u, v = detection.bottom_centre
            fix = locate(frame, terrain, float(u), float(v))
            if not fix.valid:
                stats["refused"] += 1
                key = fix.reason.split("--")[0].strip()[:44]
                reasons[key] = reasons.get(key, 0) + 1
                continue
            stats["located"] += 1
            records.append({
                "xy": fix.xy, "sigma": fix.sigma_m, "frame_id": frame.frame_id,
                "score": detection.score, "cls": detection.cls,
                "range_m": fix.range_m, "depression": fix.depression_deg,
            })
    stats["reasons"] = reasons
    return records, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="carloc", description=__doc__)
    parser.add_argument("--root", default="sessions/airzoo")
    parser.add_argument("--site", default="jiaxiao")
    parser.add_argument("--split", default="12-14")
    parser.add_argument("--radius", type=float, default=2.5,
                        help="merge radius in metres")
    parser.add_argument("--limit", type=int, default=0, help="cap frames, 0 = all on disk")
    parser.add_argument("--out", type=Path, default=Path("reports/cars.csv"))
    parser.add_argument("--map", type=Path, default=Path("reports/cars.png"))
    args = parser.parse_args(argv)

    session = Session(root=args.root, site=args.site, split=args.split)
    frames = session.available()
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        print("no imagery on disk for that split", file=sys.stderr)
        return 1

    print(f"[carloc] {args.site}/{args.split}: {len(frames)} frames "
          f"({frames[0].frame_id}-{frames[-1].frame_id})")
    print(f"[carloc] {session.axis_note}")

    fit = GeoFit.fit(*session.gnss_pairs())
    print(f"[carloc] {fit.describe()}")

    terrain = session.local_terrain
    detector = CoreMLDetector()
    detector.load()

    records, stats = collect(session, frames, detector, terrain)
    print(f"[carloc] {stats['detections']} detections, {stats['located']} located, "
          f"{stats['refused']} refused by geometry")
    for reason, count in sorted(stats["reasons"].items(), key=lambda kv: -kv[1])[:3]:
        print(f"           {count:5d}  {reason}")
    if not records:
        print("nothing located", file=sys.stderr)
        return 1

    print("\n[dedupe] count against merge radius — a real population plateaus")
    print(f"  {'radius':>7} {'cars':>6} {'dropped':>8} {'suspect':>8} {'near':>5} {'consist':>8}")
    for row in sweep(records):
        print(f"  {row['radius_m']:6.1f}m {row['reported']:6d} {row['singletons']:8d} "
              f"{row['suspect']:8d} {row['near_pairs']:5d} {row['median_consistency']:8.2f}")

    clusters = cluster_fixes(records, radius_m=args.radius)
    kept = [c for c in clusters if c.n >= MIN_FIXES]
    report = audit(clusters, radius_m=args.radius)
    print(f"\n[carloc] at {args.radius:.1f} m: {len(kept)} cars "
          f"({report['singletons']} dropped below {MIN_FIXES} fixes, "
          f"{report['suspect']} flagged, {report['near_pairs']} near pairs)")

    kept.sort(key=lambda c: -c.n)
    rows = []
    for index, cluster in enumerate(kept, start=1):
        lon, lat = fit.to_wgs84(cluster.centre)[0]
        rows.append({
            "id": index, "lat": round(float(lat), 7), "lon": round(float(lon), 7),
            "class": cluster.top_class, "confidence": round(cluster.confidence, 3),
            "frames": cluster.n,
            "sigma_m": round(cluster.position_sigma_m, 2),
            "spread_m": round(cluster.spread_m, 2),
            "consistency": round(cluster.consistency, 2),
            "trusted": "yes" if cluster.consistency <= 2.0 else "no",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")

    print(f"\n{'id':>3} {'lat':>11} {'lon':>12} {'cls':>6} {'n':>3} {'+/-':>6} {'trust':>6}")
    for row in rows[:12]:
        print(f"{row['id']:3d} {row['lat']:11.7f} {row['lon']:12.7f} {row['class']:>6} "
              f"{row['frames']:3d} {row['sigma_m']:5.2f}m {row['trusted']:>6}")

    centre = np.mean([c.centre for c in kept], axis=0)
    lon, lat = fit.to_wgs84(centre)[0]
    print(f"\n[verify] scene centre: {lat:.6f}, {lon:.6f}")
    print(f"[verify] https://www.google.com/maps/@{lat:.6f},{lon:.6f},250m/data=!3m1!1e3")

    try:
        from carloc.plot import draw

        draw(kept, fit, frames, args.map)
        print(f"wrote {args.map}")
    except ImportError:
        print("(matplotlib not installed; skipping map)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
