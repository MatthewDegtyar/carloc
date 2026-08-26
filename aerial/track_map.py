"""Tracked objects on a lat/lon map, with uncertainty measured from their history.

The whole stack in one pass:

    YOLO + tiling          find something
    size gate              reject what the geometry says cannot be that class
    roboflow tracker       say which detections are the same object over time
    DSM ray-cast           turn each of those boxes into a ground position
    track_geo              turn the scatter of those positions into a covariance
    geo_fit                put it on the globe

The interesting number is the last one before the map: a *measured* uncertainty
rather than a predicted one, obtained from repeated observation of the same
object and nothing else. No ground truth is involved anywhere.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import build_detector, build_ranger  # noqa: E402

from geoloc_agent.analysis.track_geo import estimate_from_fixes  # noqa: E402
from geoloc_agent.contracts import Observation  # noqa: E402
from geoloc_agent.geo_fit import LocalGeoFit  # noqa: E402
from geoloc_agent.geometry import bearing_from_pixel  # noqa: E402
from geoloc_agent.io.airzoo import AirZooSession  # noqa: E402

BOTSORT = {
    "minimum_consecutive_frames": 2,
    "track_activation_threshold": 0.10,
    "high_conf_det_threshold": 0.2,
    "enable_cmc": True,
}


def collect(session, frames, tracker: str = "botsort"):
    """Run perception + tracking, and ground every hinted detection."""
    from geoloc_agent.detect.roboflow_track import RoboflowTrackedDetector

    base = build_detector(session)
    ranger = build_ranger(session)
    detector = RoboflowTrackedDetector(base, tracker=tracker,
                                       **(BOTSORT if tracker == "botsort" else {}))
    detector.warmup()

    fixes: dict[int, list] = defaultdict(list)
    classes: dict[int, list[str]] = defaultdict(list)
    total = hinted = 0
    for frame in frames:
        for detection in detector.detect(frame):
            total += 1
            if detection.track_hint is None:
                continue
            hinted += 1
            u, v = detection.centroid
            bearing = bearing_from_pixel(u, v, frame.intrinsics, frame.pose)
            measurement = ranger.range_for(
                Observation(t=frame.timestamp, frame_id=frame.frame_id, origin=frame.pose.t,
                            bearing=bearing, bearing_sigma=2e-3, cls=detection.cls,
                            score=detection.score), [])
            if not measurement.valid:
                continue
            point = frame.pose.t + bearing * measurement.value
            fixes[detection.track_hint].append((point[:2], measurement.sigma, frame.pose.t))
            classes[detection.track_hint].append(detection.cls)
    return fixes, classes, total, hinted


GROUND_ASSOC_M = 6.0
"""How close two ground fixes must be to be called the same object.

Set from the geometry rather than tuned: the analytic single-fix sigma here is
about 3 m, so a 2-sigma radius is 6 m. Parked cars sit roughly 3 m apart across a
bay and 5-6 m along a row, so this is genuinely near the resolution limit -- it
will merge adjacent cars in a dense row, which is the honest failure and is why
the consistency ratio is reported alongside every estimate rather than being
trusted silently."""


def collect_by_ground(session, frames):
    """Associate in ground coordinates instead of in the image.

    The alternative to an image-plane tracker, and the reason it works here: a
    static object projects to the *same ground point* from every viewpoint, so
    camera motion -- which destroys IoU association entirely at 48.7 px of
    inter-frame displacement -- does not affect it at all. Association becomes
    nearest-neighbour clustering in metres.

    Deliberately the simplest thing that can work, single-linkage against a fixed
    radius, so that any difference against the image-plane result is attributable
    to the *frame* the association happens in rather than to a cleverer algorithm.
    """
    base = build_detector(session)
    ranger = build_ranger(session)
    base.warmup()

    fixes: dict[int, list] = defaultdict(list)
    classes: dict[int, list[str]] = defaultdict(list)
    centres: list[np.ndarray] = []
    total = grounded = 0

    for frame in frames:
        for detection in base.detect(frame):
            total += 1
            u, v = detection.centroid
            bearing = bearing_from_pixel(u, v, frame.intrinsics, frame.pose)
            measurement = ranger.range_for(
                Observation(t=frame.timestamp, frame_id=frame.frame_id, origin=frame.pose.t,
                            bearing=bearing, bearing_sigma=2e-3, cls=detection.cls,
                            score=detection.score), [])
            if not measurement.valid:
                continue
            grounded += 1
            point = (frame.pose.t + bearing * measurement.value)[:2]

            best, best_distance = None, GROUND_ASSOC_M
            for index, centre in enumerate(centres):
                distance = float(np.linalg.norm(centre - point))
                if distance < best_distance:
                    best, best_distance = index, distance
            if best is None:
                best = len(centres)
                centres.append(point.copy())
            fixes[best].append((point, measurement.sigma, frame.pose.t))
            classes[best].append(detection.cls)
            # Running centroid, so a cluster tracks its members rather than
            # anchoring on whichever fix happened to arrive first.
            centres[best] = np.mean([r[0] for r in fixes[best]], axis=0)

    return fixes, classes, total, grounded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="track-map", description=__doc__)
    parser.add_argument("--site", default="guangchang")
    parser.add_argument("--split", default="12-14")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--tracker", default="botsort")
    parser.add_argument("--assoc", choices=("image", "ground"), default="ground",
                        help="associate detections in the image plane (roboflow) "
                             "or in ground coordinates (geometry)")
    parser.add_argument("--out", type=Path, default=Path("aerial/reports/track_map.png"))
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")

    session = AirZooSession(site=args.site, split=args.split, load_images=True,
                            max_frames=args.frames)
    frames = [f for f in session.frames() if f.image is not None]
    print(f"[track-map] {args.site}/{args.split}: {len(frames)} frames with imagery")

    geo = LocalGeoFit.fit(
        np.array([[p.centre[0] - session.frame_origin[0],
                   p.centre[1] - session.frame_origin[1]] for p in session.poses]),
        np.array([[p.longitude, p.latitude] for p in session.poses]),
    )
    print(f"[track-map] {geo.describe()}")

    if args.assoc == "image":
        fixes, classes, total, kept = collect(session, frames, args.tracker)
        print(f"[track-map] image-plane assoc ({args.tracker}): {total} detections, "
              f"{kept} carried a track id, {len(fixes)} distinct tracks")
    else:
        fixes, classes, total, kept = collect_by_ground(session, frames)
        print(f"[track-map] ground assoc: {total} detections, {kept} grounded, "
              f"{len(fixes)} distinct tracks")

    estimates = []
    for track_id, records in fixes.items():
        estimate = estimate_from_fixes(
            track_id,
            np.array([r[0] for r in records]),
            np.array([r[1] for r in records]),
            np.array([r[2] for r in records]),
        )
        if estimate is not None:
            estimates.append((estimate, max(set(classes[track_id]),
                                            key=classes[track_id].count)))
    print(f"[track-map] {len(estimates)} tracks with enough history to estimate a covariance")
    if not estimates:
        print("[track-map] nothing to map: no track was seen often enough")
        return 1

    print()
    print(f"{'id':>5} {'class':10s} {'n':>3} {'span':>7} {'emp sigma':>10} "
          f"{'analytic':>9} {'ratio':>6}  {'lat':>10} {'lon':>11}")
    ratios = []
    for estimate, cls in sorted(estimates, key=lambda e: -e[0].n_fixes)[:20]:
        lon, lat = geo.to_wgs84(estimate.mean_xy)[0]
        ratios.append(estimate.consistency)
        print(f"{estimate.track_id:5d} {cls:10s} {estimate.n_fixes:3d} "
              f"{estimate.span_m:6.1f}m {estimate.empirical_sigma_m:9.2f}m "
              f"{estimate.analytic_sigma_m:8.2f}m {estimate.consistency:6.2f}  "
              f"{lat:10.6f} {lon:11.6f}")

    ratios = np.array([r for r in ratios if np.isfinite(r)])
    if len(ratios):
        print()
        median = float(np.median(ratios))
        if median > 5:
            note = ("far beyond optimistic -- at this size the scatter is not "
                    "ranging error, it is the track associating different objects")
        elif median > 2:
            note = "optimistic -- reported sigma too small"
        elif median < 0.5:
            note = "pessimistic -- sigma too large"
        else:
            note = "consistent"
        print(f"[calibration] median empirical/analytic = {median:.2f}")
        print(f"              {note}")

    _render(estimates, geo, session, frames, args.out)
    print(f"\nwrote {args.out}")
    return 0


def _render(estimates, geo, session, frames, out_path: Path) -> None:
    """Top-down map in WGS84, coloured by whether the uncertainty is believable.

    Ellipses are drawn at 3 sigma. At 1 sigma they are sub-metre against a
    400 m scene and render as dots, which would hide the one thing the map is
    for; the multiplier is stated on the figure rather than left to be assumed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SIGMA = 3.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 9.5), facecolor="#0d1117")
    ax.set_facecolor("#10141c")

    track = np.array([f.pose.t[:2] for f in frames])
    track_ll = geo.to_wgs84(track)
    ax.plot(track_ll[:, 0], track_ll[:, 1], "-", color="#8b98a8", lw=1.2, alpha=0.55,
            label="flight path", zorder=2)

    def colour_for(ratio: float) -> str:
        # Colour encodes trust in the uncertainty, not the class -- nearly
        # everything here is a car, so class would carry no information.
        if not np.isfinite(ratio):
            return "#8b98a8"
        if ratio > 3.0:
            return "#e05561"
        if ratio > 1.6:
            return "#e0af68"
        return "#4ec9b0"

    for estimate, _cls in estimates:
        colour = colour_for(estimate.consistency)
        centre = geo.to_wgs84(estimate.mean_xy)[0]
        ring = geo.to_wgs84(estimate.ellipse(sigma=SIGMA))
        ax.fill(ring[:, 0], ring[:, 1], color=colour, alpha=0.22, zorder=3)
        ax.plot(ring[:, 0], ring[:, 1], color=colour, lw=1.1, alpha=0.9, zorder=4)
        ax.plot(centre[0], centre[1], "o", color=colour, ms=2.5, zorder=5)

    points = np.array([geo.to_wgs84(e.mean_xy)[0] for e, _ in estimates])
    pad_x = max(np.ptp(points[:, 0]) * 0.10, 1e-4)
    pad_y = max(np.ptp(points[:, 1]) * 0.10, 1e-4)
    ax.set_xlim(points[:, 0].min() - pad_x, points[:, 0].max() + pad_x)
    ax.set_ylim(points[:, 1].min() - pad_y, points[:, 1].max() + pad_y)

    mean_lat = float(points[:, 1].mean())
    metres_per_deg_lon = 111_320.0 * np.cos(np.radians(mean_lat))
    bar_m = 50.0
    x0 = ax.get_xlim()[0] + pad_x * 0.4
    y0 = ax.get_ylim()[0] + pad_y * 0.5
    ax.plot([x0, x0 + bar_m / metres_per_deg_lon], [y0, y0], "-", color="#dfe6ef",
            lw=2.5, zorder=6)
    ax.text(x0, y0 + pad_y * 0.10, f"{bar_m:.0f} m", color="#dfe6ef", fontsize=8,
            family="monospace", zorder=6)

    ratios = np.array([e.consistency for e, _ in estimates if np.isfinite(e.consistency)])
    ax.set_xlabel("longitude", color="#dfe6ef")
    ax.set_ylabel("latitude", color="#dfe6ef")
    ax.tick_params(colors="#8b98a8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#39434f")
    ax.set_title(
        f"tracked objects in WGS84 — ellipse = {SIGMA:.0f} sigma, MEASURED from "
        f"each object's own history\n"
        f"{session.site}/{session.split} · {len(estimates)} tracks · "
        f"median empirical/analytic {np.median(ratios):.2f} · "
        f"{geo.describe()}",
        color="#dfe6ef", fontsize=9.5, family="monospace",
    )
    from matplotlib.lines import Line2D

    ax.legend(handles=[
        Line2D([], [], color="#8b98a8", lw=1.2, label="flight path"),
        Line2D([], [], color="#4ec9b0", lw=6, alpha=0.6, label="scatter matches prediction"),
        Line2D([], [], color="#e0af68", lw=6, alpha=0.6, label="1.6-3x  optimistic"),
        Line2D([], [], color="#e05561", lw=6, alpha=0.6, label=">3x  distrust this fix"),
    ], facecolor="#161b22", edgecolor="#39434f", labelcolor="#dfe6ef", fontsize=8,
        loc="upper right")
    ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    fig.text(0.01, 0.012,
             "uncertainty is MEASURED from repeated observation of the same object — "
             "no ground truth used. It captures precision, not bias: an error common "
             "to every look moves all fixes together and leaves the scatter untouched.",
             color="#8b98a8", fontsize=7.6, family="monospace")
    fig.savefig(out_path, dpi=130, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
