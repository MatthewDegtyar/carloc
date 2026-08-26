"""Top-down map of the located cars, in WGS84."""

from __future__ import annotations

from pathlib import Path

import numpy as np

BG = "#0d1117"
PANEL = "#10141c"
GRID = "#39434f"
TEXT = "#dfe6ef"
MUTED = "#8b98a8"


def draw(clusters, fit, frames, out_path: str | Path, sigma: float = 3.0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 9.5), facecolor=BG)
    ax.set_facecolor(PANEL)

    path = fit.to_wgs84(np.array([f.centre[:2] for f in frames]))
    ax.plot(path[:, 0], path[:, 1], "-", color=MUTED, lw=1.1, alpha=0.55, zorder=2)

    points = []
    for cluster in clusters:
        trusted = np.isfinite(cluster.consistency) and cluster.consistency <= 2.0
        colour = "#4ec9b0" if trusted else "#e0af68"
        centre = fit.to_wgs84(cluster.centre)[0]
        points.append(centre)
        # Ellipse from the cluster's own scatter, drawn at `sigma` for visibility.
        spread = max(cluster.position_sigma_m, 0.05) * sigma
        per_lat = 110_540.0
        per_lon = 111_320.0 * np.cos(np.radians(centre[1]))
        angle = np.linspace(0, 2 * np.pi, 48)
        ring = np.column_stack([centre[0] + spread * np.cos(angle) / per_lon,
                                centre[1] + spread * np.sin(angle) / per_lat])
        ax.fill(ring[:, 0], ring[:, 1], color=colour, alpha=0.22, zorder=3)
        ax.plot(ring[:, 0], ring[:, 1], color=colour, lw=0.9, alpha=0.85, zorder=4)
        ax.plot(centre[0], centre[1], "o", color=colour, ms=3, zorder=5)

    points = np.array(points)
    pad_x = max(np.ptp(points[:, 0]) * 0.12, 2e-4)
    pad_y = max(np.ptp(points[:, 1]) * 0.12, 2e-4)
    ax.set_xlim(points[:, 0].min() - pad_x, points[:, 0].max() + pad_x)
    ax.set_ylim(points[:, 1].min() - pad_y, points[:, 1].max() + pad_y)

    mean_lat = float(points[:, 1].mean())
    per_lon = 111_320.0 * np.cos(np.radians(mean_lat))
    x0 = ax.get_xlim()[0] + pad_x * 0.35
    y0 = ax.get_ylim()[0] + pad_y * 0.45
    ax.plot([x0, x0 + 25.0 / per_lon], [y0, y0], "-", color=TEXT, lw=2.5, zorder=6)
    ax.text(x0, y0 + pad_y * 0.10, "25 m", color=TEXT, fontsize=8,
            family="monospace", zorder=6)

    ax.set_xlabel("longitude", color=TEXT)
    ax.set_ylabel("latitude", color=TEXT)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    ax.set_title(f"{len(clusters)} cars, deduplicated, WGS84 — "
                 f"ellipse = {sigma:.0f} sigma from each car's own fixes\n"
                 f"{fit.describe()}",
                 color=TEXT, fontsize=10, family="monospace")
    ax.legend(handles=[
        Line2D([], [], color=MUTED, lw=1.1, label="flight path"),
        Line2D([], [], color="#4ec9b0", lw=6, alpha=0.6, label="scatter matches prediction"),
        Line2D([], [], color="#e0af68", lw=6, alpha=0.6, label="scatter too large — may be 2 cars"),
    ], facecolor="#161b22", edgecolor=GRID, labelcolor=TEXT, fontsize=8, loc="upper right")
    fig.savefig(out_path, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out_path
