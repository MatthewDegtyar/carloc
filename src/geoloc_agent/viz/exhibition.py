"""Query-driven exhibition render.

One pass of perception and filtering, then the same tracks interrogated several
ways. That ordering is the point of the demo: nothing is re-detected between
queries, only re-selected. Show "light vehicles", then "dark vehicles", and the
boxes that move are evidence that the system is holding attributes about persistent
objects rather than re-reading the picture.

Matched tracks are drawn in the query's colour with their rationale; unmatched ones
stay visible but dimmed. Hiding them entirely would make the selection look more
decisive than it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from geoloc_agent.analysis.appearance import AppearanceMemory, sample_appearance
from geoloc_agent.analysis.concealment import assess, shadow_polygon
from geoloc_agent.analysis.queries import QUERIES, Candidate, Query
from geoloc_agent.pipeline import PipelineResult
from geoloc_agent.viz.render import (
    GRID,
    MUTED,
    SKY,
    TEXT,
    _range_and_sigma,
    resolve_matches,
)


@dataclass
class ExhibitFrame:
    record: object
    candidates: dict[int, Candidate] = field(default_factory=dict)
    boxes: dict[int, object] = field(default_factory=dict)
    """track_id -> the one detection box that track is drawn and measured from."""


def build_candidates(result: PipelineResult, intrinsics) -> tuple[list[ExhibitFrame], dict]:
    """One pass over the run: sample appearance, size every object, once."""
    memory = AppearanceMemory()
    frames: list[ExhibitFrame] = []

    # Resolved once and reused: the pairing must be identical in the appearance
    # pass, the sizing pass and the draw, or a track is coloured from one box and
    # measured from another.
    pairing: dict[int, dict[int, object]] = {}
    for record in result.frames:
        if record.frame is None:
            continue
        best, _ = resolve_matches(record.detections, record.track_states, record.frame)
        pairing[record.frame_id] = {tid: det for tid, (_, det) in best.items()}
        for track_id, detection in pairing[record.frame_id].items():
            memory.observe(track_id, sample_appearance(record.frame.image, detection.bbox))
        frames.append(ExhibitFrame(record=record))

    # Appearance is resolved once, over each track's whole life, then applied to
    # every frame. A per-frame colour would flicker between buckets under changing
    # light and make the selection look arbitrary.
    for ef in frames:
        record = ef.record
        by_track = {t.track_id: t for t in record.track_states}
        ef.boxes = pairing[record.frame_id]
        for track_id, detection in ef.boxes.items():
            track = by_track.get(track_id)
            if track is None:
                continue
            r, _ = _range_and_sigma(track, record.frame)
            ef.candidates[track_id] = Candidate(
                track=track,
                appearance=memory.get(track_id),
                concealment=assess(detection.bbox, r, intrinsics.fx, intrinsics.fy,
                                   cls=track.top_class[0],
                                   image_size=(intrinsics.width, intrinsics.height)),
                range_m=r,
            )
    return frames, {"memory": memory}


def _draw_query_frame(cam_ax, map_ax, ef: ExhibitFrame, query: Query, truth, xlim, ylim,
                      trail: np.ndarray) -> int:
    from matplotlib.patches import Ellipse, Polygon, Rectangle

    from geoloc_agent.viz.render import draw_scene

    record = ef.record
    frame = record.frame
    intr = frame.intrinsics

    draw_scene(cam_ax, frame, {})     # imagery only; truth boxes would clutter it
    cam_ax.set_xlim(0, intr.width)
    cam_ax.set_ylim(intr.height, 0)
    cam_ax.set_box_aspect(intr.height / intr.width)
    cam_ax.set_xticks([])
    cam_ax.set_yticks([])
    for spine in cam_ax.spines.values():
        spine.set_color(GRID)

    # Count unique TRACKS, not boxes: several detections can reproject onto the
    # same track, which is how "9 of 8 match" happens.
    matched_ids: set[int] = set()
    placed: list[tuple] = []
    rows = []
    for track_id, detection in ef.boxes.items():
        cand = ef.candidates.get(track_id)
        if cand is None:
            continue
        hit, why = query.match(cand)
        rows.append((cand.range_m, detection, track_id, cand, hit, why))
    rows.sort(key=lambda r: -r[0])   # far to near, so near labels land on top

    for _, detection, track_id, cand, hit, why in rows:
        x1, y1, x2, y2 = detection.bbox
        if not hit:
            cam_ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=MUTED, lw=0.8, alpha=0.35, zorder=4))
            continue
        matched_ids.add(track_id)
        colour = query.colour_of(cand) if query.colour_of else query.colour
        cam_ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=colour, lw=2.4, zorder=6))
        label = f"{why}"
        units = intr.width / 10.0
        char_w, line_h = 0.60 * 8.0 / 72.0 * units, 1.35 * 8.0 / 72.0 * units
        wpx = len(label) * char_w + 6
        hpx = line_h + 6
        # Keep the label inside the image. A box near the right edge would
        # otherwise run its label out over the map panel.
        lx = float(np.clip(x1, 0.0, max(0.0, intr.width - wpx)))
        ly = float(y1) - 6
        for _ in range(16):
            box = (lx, ly - hpx, lx + wpx, ly)
            if not any(box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3]
                       for q in placed):
                break
            ly -= hpx + 3
        placed.append((lx, ly - hpx, lx + wpx, ly))
        cam_ax.text(lx, ly, label, color="#0d1117", fontsize=8, va="bottom",
                    family="monospace", zorder=7,
                    bbox={"facecolor": colour, "edgecolor": "none",
                          "alpha": 0.93, "pad": 2.0})

    # --- map -------------------------------------------------------------
    map_ax.set_facecolor(SKY)
    map_ax.set_xlim(*xlim)
    map_ax.set_ylim(*ylim)
    map_ax.set_box_aspect(1.0)
    map_ax.tick_params(colors=MUTED, labelsize=7)
    for spine in map_ax.spines.values():
        spine.set_color(GRID)

    for track in record.track_states:
        cand = ef.candidates.get(track.track_id)
        if cand is None:
            continue
        hit, _ = query.match(cand)
        if not hit:
            colour = MUTED
        else:
            colour = query.colour_of(cand) if query.colour_of else query.colour
        alpha = 0.9 if hit else 0.25

        if hit and query.name == "concealment" and cand.concealment:
            poly = shadow_polygon(frame.pose.t, track.mean, cand.concealment.width_m)
            if len(poly):
                map_ax.add_patch(Polygon(poly, closed=True, facecolor=query.colour,
                                         alpha=0.20, edgecolor="none", zorder=2))

        values, vectors = np.linalg.eigh(track.cov[:2, :2])
        values = np.clip(values, 1e-9, None)
        order = values.argsort()[::-1]
        values, vectors = values[order], vectors[:, order]
        angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
        map_ax.add_patch(Ellipse((track.mean[0], track.mean[1]),
                                 2 * np.sqrt(values[0]), 2 * np.sqrt(values[1]), angle=angle,
                                 facecolor=colour, alpha=0.14 * (1 if hit else 0.5),
                                 edgecolor=colour, lw=1.2 if hit else 0.6, zorder=3))
        map_ax.plot(track.mean[0], track.mean[1], "o", color=colour,
                    ms=5 if hit else 3, alpha=alpha, zorder=4)

    if len(trail) > 1:
        map_ax.plot(trail[:, 0], trail[:, 1], "-", color=MUTED, lw=1.0, alpha=0.6, zorder=3)
    facing = frame.pose.R @ [0, 0, 1]
    map_ax.plot(frame.pose.t[0], frame.pose.t[1], "^", color=TEXT, ms=9, zorder=5)
    map_ax.arrow(frame.pose.t[0], frame.pose.t[1], facing[0] * 6, facing[1] * 6,
                 head_width=1.6, color=TEXT, alpha=0.8, zorder=5)
    return len(matched_ids)


def render_exhibition(result: PipelineResult, out_path: str | Path, scene: str,
                      queries: list[Query] | None = None, fps: int = 6,
                      hold: int = 8, dpi: int = 110) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    queries = queries or QUERIES
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = [r for r in result.frames if r.frame is not None]
    if not records:
        raise ValueError("nothing to render")
    intr = records[0].frame.intrinsics
    frames, _ = build_candidates(result, intr)

    # Drop leading and trailing frames with nothing in them. A scene that ends
    # with the ego driving into an empty stretch would otherwise play the same
    # blank frames at the tail of all seven sections. Interior gaps are kept --
    # a moment where the system genuinely sees nothing is part of the record.
    occupied = [i for i, ef in enumerate(frames) if ef.candidates]
    if occupied:
        frames = frames[occupied[0]: occupied[-1] + 1]

    # Indexed by position in `frames`, so it must be rebuilt after the trim
    # above or the ego trail lags the frame being drawn.
    positions = np.array([ef.record.frame.pose.t for ef in frames])
    span = 45.0
    timeline = []
    for query in queries:
        for i, ef in enumerate(frames):
            timeline.append((query, ef, i))
        for _ in range(hold):
            timeline.append((query, frames[-1], len(frames) - 1))

    fig = plt.figure(figsize=(15.5, 6.2), facecolor="#0d1117")
    grid = fig.add_gridspec(1, 2, width_ratios=[10.0, 5.0], wspace=0.06,
                            left=0.010, right=0.990, top=0.820, bottom=0.075)
    cam_ax = fig.add_subplot(grid[0, 0])
    map_ax = fig.add_subplot(grid[0, 1])
    title = fig.text(0.010, 0.960, "", color=TEXT, fontsize=14, family="monospace", ha="left")
    caption = fig.text(0.010, 0.915, "", color=MUTED, fontsize=8.5, family="monospace",
                       ha="left", va="top", linespacing=1.45)
    footer = fig.text(0.010, 0.020, "", color=MUTED, fontsize=8, family="monospace", ha="left")

    def draw(index: int):
        query, ef, i = timeline[index]
        cam_ax.clear()
        map_ax.clear()
        eye = ef.record.frame.pose.t
        ahead = eye + (ef.record.frame.pose.R @ [0, 0, 1]) * span * 0.45
        xlim = (ahead[0] - span, ahead[0] + span)
        ylim = (ahead[1] - span, ahead[1] + span)
        n = _draw_query_frame(cam_ax, map_ax, ef, query, {}, xlim, ylim, positions[: i + 1])

        title.set_text(f'QUERY:  "{query.name}"     {n} of '
                       f'{len(ef.candidates)} tracks in view match')
        caption.set_text(_wrap(query.caption, 140))
        cam_ax.text(10, 26,
                    f"nuScenes {scene} — real imagery, real ego pose  |  "
                    f"frame {ef.record.frame_id}",
                    color=TEXT, fontsize=8, family="monospace", va="top", zorder=8,
                    bbox={"facecolor": "#0d1117", "edgecolor": "none", "alpha": 0.55, "pad": 2})
        map_ax.set_title("top-down  |  ellipse = 1 sigma", color=MUTED, fontsize=8.5,
                         family="monospace")
        footer.set_text("one detection pass, one filter pass — only the QUERY changes "
                        "between sections; nothing is re-detected")
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(timeline), interval=1000 / fps)
    writer = animation.FFMpegWriter(fps=fps, bitrate=5200)
    anim.save(str(out_path), writer=writer, dpi=dpi, savefig_kwargs={"facecolor": "#0d1117"})
    plt.close(fig)
    return out_path


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width)[:3])
