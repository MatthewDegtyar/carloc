"""Video of the plateless verification: signatures forming, then the verdict.

Built to show the negative result rather than hide it. The first act looks like a
working product -- cars found, signatures accumulating, everything green. The
second act puts the same-vehicle and different-vehicle score distributions side
by side and they overlap, which is why the recognition rate is 5.7% and why this
is not yet sellable.

That ordering is deliberate. A demo that stops after the first act is the one
that gets built by mistake.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BG = "#0d1117"
PANEL = "#10141c"
GRID = "#39434f"
TEXT = "#dfe6ef"
MUTED = "#8b98a8"
GOOD = "#4ec9b0"
WARN = "#e0af68"
BAD = "#e05561"
SAME = "#4ec9b0"
DIFF = "#e05561"

TONE_COLOUR = {"light": "#e8e8e8", "dark": "#7aa2f7", "mid": "#c0c8d4",
               "coloured": "#e0af68", "unknown": "#8b98a8"}


def render(frames, per_frame, same_scores, diff_scores, threshold,
           out_path: str | Path, scene: str, fps: int = 6, hold: int = 26,
           dpi: int = 110) -> Path:
    """`per_frame` is a list of (frame, [(bbox, tone, length_m, n_looks), ...])."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Rectangle

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    same = np.asarray(same_scores, dtype=float)
    diff = np.asarray(diff_scores, dtype=float)
    n_drive = len(per_frame)
    total = n_drive + hold

    fig = plt.figure(figsize=(15.5, 6.4), facecolor=BG)
    grid = fig.add_gridspec(1, 2, width_ratios=[10.0, 6.4], wspace=0.14,
                            left=0.010, right=0.988, top=0.815, bottom=0.095)
    cam_ax = fig.add_subplot(grid[0, 0])
    side_ax = fig.add_subplot(grid[0, 1])

    title = fig.text(0.010, 0.955, "", color=TEXT, fontsize=14, family="monospace")
    caption = fig.text(0.010, 0.905, "", color=MUTED, fontsize=8.5, family="monospace",
                       va="top", linespacing=1.5)
    footer = fig.text(0.010, 0.022, "", color=MUTED, fontsize=8, family="monospace")

    def draw(index: int):
        cam_ax.clear()
        side_ax.clear()

        phase_two = index >= n_drive
        i = min(index, n_drive - 1)
        frame, entries = per_frame[i]
        intr = frame.intrinsics

        cam_ax.imshow(frame.image)
        cam_ax.set_xlim(0, intr.width)
        cam_ax.set_ylim(intr.height, 0)
        cam_ax.set_xticks([])
        cam_ax.set_yticks([])
        for spine in cam_ax.spines.values():
            spine.set_color(GRID)

        placed = []
        for bbox, tone, length_m, n_looks in entries:
            colour = TONE_COLOUR.get(tone, MUTED)
            x1, y1, x2, y2 = bbox
            solid = n_looks >= 3
            cam_ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=colour, lw=2.0 if solid else 1.0,
                                       alpha=1.0 if solid else 0.45, zorder=5))
            if not solid or (x2 - x1) < 55:
                continue
            label = f"{tone} {length_m:.1f}m"
            lx, ly = float(x1), float(y1) - 5
            for _ in range(20):
                box = (lx, ly - 13, lx + len(label) * 6.4, ly)
                if not any(box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3]
                           for q in placed):
                    break
                ly -= 15
            placed.append((lx, ly - 13, lx + len(label) * 6.4, ly))
            cam_ax.text(lx, ly, label, color="#0d1117", fontsize=7.5, va="bottom",
                        family="monospace", zorder=6,
                        bbox={"facecolor": colour, "edgecolor": "none", "alpha": 0.93,
                              "pad": 1.6})

        cam_ax.text(10, 24, f"nuScenes {scene} — real imagery, real ego pose  |  "
                            f"frame {frame.frame_id}",
                    color=TEXT, fontsize=8, family="monospace", va="top", zorder=8,
                    bbox={"facecolor": BG, "edgecolor": "none", "alpha": 0.6, "pad": 2})

        side_ax.set_facecolor(PANEL)
        side_ax.tick_params(colors=MUTED, labelsize=8)
        for spine in side_ax.spines.values():
            spine.set_color(GRID)

        if not phase_two:
            seen = sum(1 for _, t, _, n in entries if n >= 3)
            side_ax.barh([0], [seen], color=GOOD, alpha=0.8, height=0.55)
            side_ax.text(seen + 0.15, 0, f"  {seen}", color=TEXT, fontsize=11,
                         family="monospace", va="center")
            side_ax.set_xlim(0, max(12, seen + 3))
            # Bar pinned low so the explanatory text above it has clear space.
            side_ax.set_ylim(-0.8, 4.0)
            side_ax.set_yticks([])
            side_ax.set_xlabel("vehicles with a usable signature", color=MUTED, fontsize=9)
            side_ax.text(0.5, 0.80, "building signatures", transform=side_ax.transAxes,
                         color=TEXT, fontsize=13, family="monospace", ha="center")
            side_ax.text(0.5, 0.60,
                         "colour · metric size · class · bay pose\n"
                         "median over every look, never one frame",
                         transform=side_ax.transAxes, color=MUTED, fontsize=9,
                         family="monospace", ha="center", linespacing=1.6)
            title.set_text('PLATELESS VERIFICATION — "is this the same car as last pass?"')
            caption.set_text(
                "No licence plate is read. Each parked vehicle gets a signature from cues that\n"
                "are individually weak: body colour, physical size in metres from geolocation,\n"
                "class, and where it sits in its bay.")
            footer.set_text("act 1 of 2 — this is the part that looks like it works")
        else:
            bins = np.linspace(min(diff.min(), same.min()) - 0.5,
                               max(diff.max(), same.max()) + 0.5, 26)
            # Density, not counts: there are 6441 different-vehicle pairs against
            # 53 same-vehicle ones, so raw counts would render the thing being
            # argued about as a flat line. The claim is about the shapes.
            side_ax.hist(diff, bins=bins, color=DIFF, alpha=0.55, density=True,
                         label=f"different vehicles (n={len(diff)})")
            side_ax.hist(same, bins=bins, color=SAME, alpha=0.6, density=True,
                         label=f"same vehicle, two viewpoints (n={len(same)})")
            side_ax.axvline(threshold, color=TEXT, lw=1.8, ls="--")
            side_ax.text(threshold + 0.12, side_ax.get_ylim()[1] * 0.92,
                         f"cite bar {threshold:.1f}\n(99:1)", color=TEXT, fontsize=8,
                         family="monospace", va="top")
            side_ax.set_xlabel("log-odds that it is the same vehicle", color=MUTED, fontsize=9)
            side_ax.set_ylabel("density", color=MUTED, fontsize=9)
            side_ax.legend(facecolor="#161b22", edgecolor=GRID, labelcolor=TEXT, fontsize=8,
                           loc="upper left")
            recognised = int((same >= threshold).sum())
            false_matches = int((diff >= threshold).sum())
            title.set_text("VERDICT — the distributions overlap")
            caption.set_text(
                f"Same vehicle recognised {recognised}/{len(same)} ({recognised / len(same):.0%}). "
                f"False matches {false_matches}/{len(diff)}.\n"
                f"Zero false matches was bought by a matcher that almost never matches: "
                f"same-vehicle median\n{np.median(same):+.2f} against different-vehicle p95 "
                f"{np.percentile(diff, 95):+.2f}. Appearance alone does not carry this.")
            footer.set_text(
                "the bay-offset cue closes it, and needs <=29 cm relative precision — "
                "absolute geolocation gives 50 cm, differential in-image measurement can give 5")
        return []

    animation.FuncAnimation(fig, draw, frames=total, interval=1000 / fps).save(
        str(out_path), writer=animation.FFMpegWriter(fps=fps, bitrate=5000), dpi=dpi,
        savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    return out_path
