"""Render a pipeline run to video.

What is real and what is drawn, stated plainly, because this is a synthetic
session and a video is exactly the kind of artefact that gets mistaken for a
camera feed:

* **Synthesised:** every pixel. There is no imagery in a `SyntheticSession` --
  the ground grid, horizon and object bodies are drawn from the true geometry
  through the real camera matrix. This is a visualisation of a simulation.
* **Real pipeline output:** the detection boxes (projected by the same code the
  tracker consumes), the class and confidence from the class posterior, the
  range and its sigma from the filter's covariance projected onto the line of
  sight, and the top-down error ellipses. None of it is re-derived for display.

The range label is the point of the whole exercise, so it is never shown as a
bare number: it is always `R +/- sigma`, and a track whose geometry cannot
support a range says so instead of printing a confident figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geoloc_agent.contracts import Frame, Intrinsics, TrackState
from geoloc_agent.io.base import TruthObject
from geoloc_agent.pipeline import PipelineResult

# Dark palette: high contrast for boxes and text over a rendered scene.
SKY = "#10141c"
GROUND = "#1b2028"
GRID = "#39434f"
HORIZON = "#3d4757"
GOOD = "#4ec9b0"
WARN = "#e2b93d"
BAD = "#e05561"
TRUTH = "#7f8ea3"
TEXT = "#dfe6ef"
MUTED = "#8b98a8"

CLASS_COLORS = {"car": "#4ea1e0", "pedestrian": "#c678dd", "truck": "#e0a04e"}


def project(points: np.ndarray, frame: Frame, min_z: float = 0.35):
    """World points -> pixel coords. Returns (uv, valid) with behind-camera masked."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    cam = (frame.pose.R.T @ (points - frame.pose.t).T).T
    valid = cam[:, 2] > min_z
    uv = np.full((len(points), 2), np.nan)
    safe = cam[valid]
    if len(safe):
        projected = (frame.intrinsics.K @ safe.T).T
        uv[valid] = projected[:, :2] / projected[:, 2:3]
    return uv, valid


def _segment(ax, a: np.ndarray, b: np.ndarray, frame: Frame, **kwargs) -> None:
    """Draw a world-space segment, subdivided so it clips sanely at the horizon."""
    steps = 24
    ts = np.linspace(0.0, 1.0, steps)
    points = a[None, :] + (b - a)[None, :] * ts[:, None]
    uv, valid = project(points, frame)
    if valid.sum() < 2:
        return
    run: list[np.ndarray] = []
    for point, ok in zip(uv, valid, strict=True):
        if ok:
            run.append(point)
        elif len(run) > 1:
            arr = np.array(run)
            ax.plot(arr[:, 0], arr[:, 1], **kwargs)
            run = []
        else:
            run = []
    if len(run) > 1:
        arr = np.array(run)
        ax.plot(arr[:, 0], arr[:, 1], **kwargs)


def draw_scene(ax, frame: Frame, truth: dict[str, TruthObject]) -> None:
    """The camera image if there is one, otherwise a render of the true geometry."""
    from matplotlib.patches import Rectangle

    intr: Intrinsics = frame.intrinsics

    if frame.image is not None:
        # Real imagery. The 3-D truth boxes are still drawn, faintly, because
        # they are what the truth-projection detector is boxing -- showing them
        # keeps the provenance of every 2-D box visible.
        ax.imshow(frame.image, extent=(0, intr.width, intr.height, 0), zorder=0,
                  interpolation="bilinear")
        for obj in truth.values():
            _draw_box(ax, obj, frame, alpha=0.30)
        return

    ax.add_patch(Rectangle((0, 0), intr.width, intr.height, color=SKY, zorder=0))

    # Horizon: where the ground plane images. Project a very distant ground point.
    far, _ = project(np.array([[0.0, 0.0, 0.0]]) + frame.pose.t + (frame.pose.R @ [0, 0, 1]) * 5000,
                     frame)
    horizon_v = float(far[0, 1]) if np.isfinite(far[0, 1]) else intr.height * 0.5
    ax.add_patch(
        Rectangle((0, horizon_v), intr.width, intr.height - horizon_v, color=GROUND, zorder=0)
    )
    ax.axhline(horizon_v, color=HORIZON, lw=1.0, zorder=1)

    # Ground grid in world coordinates, centred on the camera.
    cx, cy = frame.pose.t[0], frame.pose.t[1]
    for offset in range(-40, 41, 10):
        _segment(ax, np.array([cx + offset, cy - 20.0, 0.0]),
                 np.array([cx + offset, cy + 140.0, 0.0]),
                 frame, color=GRID, lw=0.7, zorder=1)
    for depth in range(-20, 141, 10):
        _segment(ax, np.array([cx - 40.0, cy + depth, 0.0]),
                 np.array([cx + 40.0, cy + depth, 0.0]),
                 frame, color=GRID, lw=0.7, zorder=1)

    for obj in truth.values():
        _draw_box(ax, obj, frame)


def _draw_box(ax, obj: TruthObject, frame: Frame, alpha: float = 0.85) -> None:
    """Project the object's oriented 3-D box. Real vehicles are not axis-aligned."""
    from matplotlib.patches import Polygon

    uv, valid = project(obj.corners(frame.frame_id), frame)
    if valid.sum() < 8:
        return
    color = CLASS_COLORS.get(obj.cls, MUTED)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    top = [uv[i] for i in (4, 5, 6, 7)]
    ax.add_patch(Polygon(top, closed=True, color=color, alpha=0.18 * alpha, zorder=2))
    for i, j in edges:
        ax.plot(uv[[i, j], 0], uv[[i, j], 1], color=color, lw=1.1, alpha=alpha, zorder=3)


def _fmt_m(value: float) -> str:
    """Metres at a precision that does not throw away the number.

    A 0.03 m sigma printed as "0.0 m" reads as zero uncertainty, which is the
    one thing this pipeline must never claim.
    """
    if not np.isfinite(value):
        return "inf"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _range_and_sigma(track: TrackState, frame: Frame) -> tuple[float, float]:
    """Range to the track, and the sigma along that line of sight."""
    delta = track.mean - frame.pose.t
    r = float(np.linalg.norm(delta))
    if r < 1e-6:
        return r, float("inf")
    direction = delta / r
    return r, float(np.sqrt(max(direction @ track.cov @ direction, 0.0)))


def _match_track(detection, tracks: list[TrackState], frame: Frame):
    """Pair a detection with the track whose estimate reprojects nearest to it."""
    best, best_d = None, 120.0
    for track in tracks:
        uv, valid = project(track.mean[None, :], frame)
        if not valid[0]:
            continue
        distance = float(np.hypot(*(uv[0] - detection.centroid)))
        if distance < best_d:
            best, best_d = track, distance
    return best, best_d


def draw_overlays(ax, record, tracks: list[TrackState], max_labels: int = 8) -> None:
    """Detection boxes with class, range and uncertainty.

    One label per TRACK, not per detection. Several detections can reproject
    nearest to the same track, and labelling each of them prints the same range
    two or three times in different places -- which reads as three objects at
    that range rather than one.
    """
    from matplotlib.patches import Rectangle

    frame = record.frame

    # Best (nearest-reprojecting) detection for each track.
    best: dict[int, tuple[float, object]] = {}
    unmatched = []
    for detection in record.detections:
        track, distance = _match_track(detection, tracks, frame)
        if track is None:
            unmatched.append(detection)
            continue
        if track.track_id not in best or distance < best[track.track_id][0]:
            if track.track_id in best:
                unmatched.append(best[track.track_id][1])
            best[track.track_id] = (distance, detection)
        else:
            unmatched.append(detection)

    for detection in unmatched:
        x1, y1, x2, y2 = detection.bbox
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=MUTED,
                      lw=0.9, alpha=0.5, zorder=4)
        )

    by_track = {tid: t for t in tracks for tid in [t.track_id]}
    rows = []
    for track_id, (_, detection) in best.items():
        track = by_track[track_id]
        r, sigma = _range_and_sigma(track, frame)
        rows.append((r, track, detection, sigma))
    rows.sort(key=lambda row: row[0])  # nearest first: they matter most

    placed: list[tuple[float, float, float, float]] = []
    for index, (r, track, detection, sigma) in enumerate(rows):
        x1, y1, x2, y2 = detection.bbox
        cls, confidence = track.top_class
        if sigma > 0.5 * r:
            # The uncertainty is the same size as the estimate. Printing "50 m"
            # here would be reporting the birth PRIOR as if it were a fix; the
            # track has not yet earned a range at all.
            color = BAD
            label = f"{cls} {confidence:.2f}\nRANGE INDETERMINATE"
        elif track.degenerate:
            color = BAD
            # Never print a confident range the geometry cannot support.
            label = f"{cls} {confidence:.2f}\n{r:.0f} m +/- {_fmt_m(sigma)} m  UNRELIABLE"
        else:
            color = GOOD if sigma < 2.0 else WARN
            label = f"{cls} {confidence:.2f}\n{r:.1f} m +/- {_fmt_m(sigma)} m"

        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, lw=2.0, zorder=5)
        )
        if index >= max_labels:
            continue  # box only: past this the labels obscure the scene

        # Lift the label until it clears every label already placed. Overlapping
        # range readouts are worse than useless -- they are misattributable.
        #
        # The collision box has to be sized in DATA units, not pixels: the text is
        # drawn in points, so its extent depends on the axes scale. At 8.5 pt
        # monospace a character is ~0.6 em wide and a line ~1.35 em tall.
        units_per_inch = frame.intrinsics.width / 10.0
        char_w = 0.60 * 8.5 / 72.0 * units_per_inch
        line_h = 1.35 * 8.5 / 72.0 * units_per_inch
        text_rows = label.split("\n")
        width = max(len(row) for row in text_rows) * char_w + 6.0
        height = len(text_rows) * line_h + 6.0
        lx, ly = float(x1), float(y1) - 6.0
        for _ in range(20):
            box = (lx, ly - height, lx + width, ly)
            if not any(
                box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3]
                for q in placed
            ):
                break
            ly -= height + 3.0
        placed.append((lx, ly - height, lx + width, ly))

        ax.plot([x1 + 3, lx + 6], [y1, ly + 1], color=color, lw=0.7, alpha=0.6, zorder=5)
        ax.text(
            lx, ly, label, color="#0d1117", fontsize=8.5, va="bottom", ha="left",
            family="monospace", zorder=6,
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.93, "pad": 2.5},
        )


def draw_map(
    ax, record, truth: dict[str, TruthObject], trail: np.ndarray,
    xlim: tuple | None = None, ylim: tuple | None = None,
) -> None:
    """Top-down view: truth, estimates, and 1-sigma error ellipses."""
    from matplotlib.patches import Ellipse

    frame = record.frame
    ax.set_facecolor(SKY)

    for obj in truth.values():
        position = obj.at(record.frame_id)
        if xlim is not None and not (xlim[0] <= position[0] <= xlim[1]):
            continue
        if ylim is not None and not (ylim[0] <= position[1] <= ylim[1]):
            continue
        ax.plot(position[0], position[1], "x", color=TRUTH, ms=7, mew=1.4, alpha=0.8, zorder=3)

    for track in record.track_states:
        color = BAD if track.degenerate else (GOOD if track.sigma_horizontal < 2.0 else WARN)
        values, vectors = np.linalg.eigh(track.cov[:2, :2])
        values = np.clip(values, 1e-9, None)
        order = values.argsort()[::-1]
        values, vectors = values[order], vectors[:, order]
        angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
        ax.add_patch(
            Ellipse(
                (track.mean[0], track.mean[1]),
                width=2 * np.sqrt(values[0]), height=2 * np.sqrt(values[1]), angle=angle,
                facecolor=color, alpha=0.16, edgecolor=color, lw=1.3, zorder=2,
            )
        )
        ax.plot(track.mean[0], track.mean[1], "o", color=color, ms=4, zorder=4)
        # At map scale a well-conditioned ellipse is smaller than a pixel, so the
        # number is printed too. The ellipse shows shape; the text shows size.
        ax.annotate(
            f"{_fmt_m(track.sigma_horizontal)} m",
            (track.mean[0], track.mean[1]), textcoords="offset points", xytext=(7, -3),
            color=color, fontsize=7.0, family="monospace", zorder=5,
        )

    if len(trail) > 1:
        ax.plot(trail[:, 0], trail[:, 1], "-", color=MUTED, lw=1.0, alpha=0.7, zorder=3)
    facing = frame.pose.R @ [0, 0, 1]
    ax.plot(frame.pose.t[0], frame.pose.t[1], "^", color=TEXT, ms=9, zorder=5)
    ax.arrow(
        frame.pose.t[0], frame.pose.t[1], facing[0] * 8, facing[1] * 8,
        head_width=2.0, color=TEXT, alpha=0.8, zorder=5,
    )


@dataclass
class RenderConfig:
    fps: int = 10
    dpi: int = 110
    figsize: tuple[float, float] = (15.5, 6.0)
    title: str = ""
    subtitle: str = ""


@dataclass
class Segment:
    """One labelled run inside a multi-part video."""

    result: PipelineResult
    truth: dict
    title: str = ""
    caption: str = ""
    hold_frames: int = 0
    """Extra frames held on the final state, so a viewer can read the result."""
    map_half_span: float | None = None
    """If set, the top-down panel follows the camera with this half-width instead
    of covering the whole scene. A driving run spans hundreds of metres; a fixed
    extent shrinks every error ellipse to a dot."""

    banner: str = ""
    """Provenance line drawn over the camera panel. Defaults to stating whether
    the pixels are real or synthesised, which a viewer must never have to guess."""


def build_timeline(segments: list[Segment]) -> list[tuple]:
    """Flatten segments into one frame timeline of (segment, record, trail, xlim, ylim).

    Separate from rendering so the frame accounting can be asserted directly. GIF
    encoders de-duplicate identical consecutive frames, so counting frames in the
    output file silently under-reports held frames.
    """
    timeline: list[tuple] = []
    for segment in segments:
        records = [r for r in segment.result.frames if r.frame is not None]
        if not records:
            continue
        positions = np.array([r.frame.pose.t for r in records])
        truth_xy = (
            np.array([o.position[:2] for o in segment.truth.values()])
            if segment.truth
            else np.zeros((0, 2))
        )
        all_xy = np.vstack([positions[:, :2], truth_xy]) if len(truth_xy) else positions[:, :2]
        # Square the extent: the map is equal-aspect, so unequal limits letterbox
        # it into a sliver and the error ellipses become unreadable.
        pad = 12.0
        cx = 0.5 * (all_xy[:, 0].min() + all_xy[:, 0].max())
        cy = 0.5 * (all_xy[:, 1].min() + all_xy[:, 1].max())
        half = 0.5 * max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) + pad
        xlim, ylim = (cx - half, cx + half), (cy - half, cy + half)

        # Defaults bind the loop variables explicitly: a bare closure over them
        # would make every segment use the last segment's extent.
        def limits_for(record, span=segment.map_half_span, fallback=(xlim, ylim)):
            if span is None:
                return fallback
            # Centre ahead of the camera, since that is where the objects are.
            eye = record.frame.pose.t
            ahead = eye + (record.frame.pose.R @ [0, 0, 1]) * span * 0.45
            return (
                (float(ahead[0] - span), float(ahead[0] + span)),
                (float(ahead[1] - span), float(ahead[1] + span)),
            )

        for index, record in enumerate(records):
            lo, hi = limits_for(record)
            timeline.append((segment, record, positions[: index + 1], lo, hi))
        # Hold on the final state so a viewer can read the converged result.
        lo, hi = limits_for(records[-1])
        for _ in range(segment.hold_frames):
            timeline.append((segment, records[-1], positions, lo, hi))
    if not timeline:
        raise ValueError("nothing to render: no segment recorded any frames")
    return timeline


def render_segments(
    segments: list[Segment], out_path: str | Path, config: RenderConfig | None = None
) -> Path:
    """Render several runs into one video, each with its own caption.

    Separate videos would make the comparison the viewer's job. The whole point
    of the degenerate-geometry result is that it sits next to the well-conditioned
    one, so they belong in the same file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    config = config or RenderConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timeline = build_timeline(segments)
    intr = timeline[0][1].frame.intrinsics

    fig = plt.figure(figsize=config.figsize, facecolor="#0d1117")
    grid = fig.add_gridspec(1, 2, width_ratios=[10.0, 5.0], wspace=0.06,
                            left=0.010, right=0.990, top=0.845, bottom=0.105)
    cam_ax = fig.add_subplot(grid[0, 0])
    map_ax = fig.add_subplot(grid[0, 1])

    title = fig.text(0.010, 0.955, "", color=TEXT, fontsize=13, family="monospace", ha="left")
    caption = fig.text(
        0.010, 0.912, "", color=MUTED, fontsize=8.5, family="monospace",
        ha="left", va="top", linespacing=1.45,
    )
    clock = fig.text(0.990, 0.955, "", color=MUTED, fontsize=9, family="monospace", ha="right")
    footer = fig.text(0.010, 0.020, "", color=MUTED, fontsize=8.5, family="monospace", ha="left")

    def draw(index: int):
        segment, record, trail, xlim, ylim = timeline[index]
        for ax in (cam_ax, map_ax):
            ax.clear()

        cam_ax.set_xlim(0, intr.width)
        cam_ax.set_ylim(intr.height, 0)  # image coords: v grows downward
        # box_aspect rather than aspect="equal": this fills the panel at the
        # sensor's real 16:9 instead of letterboxing it inside a square axes.
        cam_ax.set_box_aspect(intr.height / intr.width)
        cam_ax.set_xticks([])
        cam_ax.set_yticks([])
        for spine in cam_ax.spines.values():
            spine.set_color(GRID)

        draw_scene(cam_ax, record.frame, segment.truth)
        draw_overlays(cam_ax, record, record.track_states)
        banner = (
            segment.banner
            if segment.banner
            else (
                "REAL IMAGERY  |  boxes, class and range are live pipeline output"
                if record.frame.image is not None
                else "SYNTHETIC RENDER  |  boxes, class and range are live pipeline output"
            )
        )
        cam_ax.text(
            10, 26, banner, color=TEXT, fontsize=8, family="monospace", va="top", zorder=7,
            bbox={"facecolor": "#0d1117", "edgecolor": "none", "alpha": 0.55, "pad": 2.0},
        )

        map_ax.set_xlim(*xlim)
        map_ax.set_ylim(*ylim)
        map_ax.set_box_aspect(1.0)
        map_ax.tick_params(colors=MUTED, labelsize=7)
        for spine in map_ax.spines.values():
            spine.set_color(GRID)
        draw_map(map_ax, record, segment.truth, trail, xlim, ylim)
        map_ax.set_title(
            "top-down  |  x truth   o estimate   ellipse = 1 sigma",
            color=MUTED, fontsize=8.5, family="monospace",
        )

        title.set_text(segment.title or f"geoloc-agent — {segment.result.session_name}")
        caption.set_text(segment.caption)
        clock.set_text(
            f"t={record.t:5.2f}s   frame {record.frame_id:3d}   "
            f"{len(record.detections)} det   {len(record.track_states)} trk"
        )
        lines = []
        for track in sorted(record.track_states, key=lambda t: t.track_id)[:4]:
            r, sigma = _range_and_sigma(track, record.frame)
            cls, confidence = track.top_class
            flag = "  << RANGE UNRELIABLE" if track.degenerate else ""
            lines.append(
                f"trk{track.track_id:02d} {cls:<10s} p={confidence:.2f}  "
                f"range {r:6.1f} m +/- {_fmt_m(sigma):>6s} m  (n={track.n_obs:3d}){flag}"
            )
        tail = "\n" + "   ".join(lines[2:4]) if lines[2:4] else ""
        footer.set_text("   ".join(lines[:2]) + tail)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(timeline), interval=1000 / config.fps)
    if out_path.suffix.lower() == ".gif":
        anim.save(str(out_path), writer="pillow", fps=config.fps, dpi=config.dpi)
    else:
        writer = animation.FFMpegWriter(
            fps=config.fps, bitrate=4600, metadata={"title": config.title or "geoloc-agent"}
        )
        anim.save(str(out_path), writer=writer, dpi=config.dpi,
                  savefig_kwargs={"facecolor": "#0d1117"})
    plt.close(fig)
    return out_path


def render_run(
    result: PipelineResult,
    truth: dict[str, TruthObject],
    out_path: str | Path,
    config: RenderConfig | None = None,
) -> Path:
    """Render a single run. Thin wrapper over `render_segments`."""
    config = config or RenderConfig()
    return render_segments(
        [Segment(result=result, truth=truth, title=config.title, caption=config.subtitle)],
        out_path,
        config,
    )
