"""Report and plot generation.

The report is written to be read by someone deciding whether to trust the
system, so it leads with what is broken. Degenerate geometry and the
heading-bias case are not buried in an appendix -- they are the two things a
reader most needs to know before believing any coordinate this pipeline emits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geoloc_agent.eval.metrics import aggregate
from geoloc_agent.eval.runner import SweepResult

CHI2_DOF = 3


def _fmt(value: float | None, spec: str = ".3f", dash: str = "--") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return dash
    return format(value, spec)


def _plots_available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def plot_sweep(sweep: SweepResult, out_dir: Path) -> list[Path]:
    """Error-vs-noise and NEES-vs-noise for one sweep axis."""
    if not _plots_available():
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parameter = next(iter(sweep.points[0].overrides), "value") if sweep.points else "value"
    xs, median, p90, rmse, nees, lower, upper = [], [], [], [], [], [], []
    for point in sweep.points:
        xs.append(float(point.overrides.get(parameter, np.nan)))
        median.append(point.summary.get("median_good", np.nan))
        p90.append(point.summary.get("p90_good", np.nan))
        rmse.append(point.summary.get("rmse_good", np.nan))
        nees.append(point.summary.get("nees_good_mean", np.nan))
        lower.append(point.summary.get("nees_lower", np.nan))
        upper.append(point.summary.get("nees_upper", np.nan))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(xs, median, "o-", label="median")
    ax.plot(xs, p90, "s--", label="p90")
    ax.plot(xs, rmse, "^:", label="RMSE")
    ax.set_xlabel(parameter)
    ax.set_ylabel("geolocation error (m)")
    ax.set_title(f"{sweep.name}: error vs {parameter} ({sweep.scenario}, good geometry)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    error_path = out_dir / f"error_vs_{sweep.name}.png"
    fig.savefig(error_path, dpi=130)
    plt.close(fig)
    paths.append(error_path)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(xs, nees, "o-", color="tab:red", label="mean NEES")
    finite = [v for v in lower + upper if np.isfinite(v)]
    if finite:
        ax.fill_between(
            xs, lower, upper, alpha=0.25, color="tab:green",
            label="95% chi-square band (calibrated)",
        )
    ax.axhline(CHI2_DOF, color="k", ls="--", lw=1, label=f"nominal = {CHI2_DOF}")
    ax.set_xlabel(parameter)
    ax.set_ylabel("mean NEES")
    ax.set_yscale("log")
    ax.set_title(f"{sweep.name}: filter calibration vs {parameter}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    nees_path = out_dir / f"nees_vs_{sweep.name}.png"
    fig.savefig(nees_path, dpi=130)
    plt.close(fig)
    paths.append(nees_path)
    return paths


def plot_geometry_split(scenario_runs: dict, out_dir: Path) -> Path | None:
    """Good vs degenerate geometry, side by side. The headline picture."""
    if not _plots_available():
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, good, degenerate = [], [], []
    for name, runs in scenario_runs.items():
        summary = aggregate(runs)
        names.append(name)
        good.append(summary.get("median_good", np.nan))
        degenerate.append(summary.get("median_degenerate", np.nan))
    if not names:
        return None

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(names)), 4.2))
    ax.bar(x - 0.2, good, 0.4, label="good geometry", color="tab:green")
    ax.bar(x + 0.2, degenerate, 0.4, label="degenerate geometry", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("median geolocation error (m)")
    ax.set_yscale("log")
    ax.set_title("Geolocation error by geometry class")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "geometry_split.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def build_report(
    scenario_runs: dict,
    sweeps: list[SweepResult],
    out_path: str | Path,
    plots_dir: str | Path | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(plots_dir) if plots_dir else out_path.parent / "plots"

    summaries = {name: aggregate(runs) for name, runs in scenario_runs.items()}
    split_plot = plot_geometry_split(scenario_runs, plots_dir)
    sweep_plots = {sweep.name: plot_sweep(sweep, plots_dir) for sweep in sweeps}

    lines: list[str] = []
    add = lines.append

    add("# Geolocation evaluation")
    add("")
    add(
        "Generated by `geoloc-eval`. Every number below comes from synthetic "
        "sessions with injected noise, scored against exact ground truth."
    )
    add("")

    add("## Read this first")
    add("")
    add(
        "Two results matter more than the accuracy table, because both are cases "
        "where the system is wrong:"
    )
    add("")
    degenerate_scenarios = [
        n for n, s in summaries.items() if s.get("n_degenerate", 0) > s.get("n_good", 0)
    ]
    if degenerate_scenarios:
        worst = summaries[degenerate_scenarios[0]]
        add(
            f"1. **Degenerate geometry is a real limit, not a tuning problem.** Under "
            f"forward motion the perpendicular baseline goes to zero and range becomes "
            f"unobservable. In `{degenerate_scenarios[0]}`, "
            f"{worst.get('n_degenerate', 0)} of "
            f"{worst.get('n_tracks', 0)} tracks fall in this class, with median error "
            f"{_fmt(worst.get('median_degenerate'))} m against "
            f"{_fmt(summaries.get('clean_lateral', {}).get('median_good'))} m for "
            f"well-conditioned geometry. The system detects the condition with recall "
            f"{_fmt(worst.get('degenerate_recall'), '.2f')} and reports the affected "
            f"tracks with inflated covariance rather than suppressing them."
        )
    bias = summaries.get("heading_bias")
    if bias:
        add("")
        add(
            f"2. **A constant heading bias defeats the covariance.** With a 2-degree yaw "
            f"error the median error is {_fmt(bias.get('median_good'))} m while mean NEES "
            f"reaches {_fmt(bias.get('nees_good_mean'), '.0f')} against a nominal 3. The "
            f"filter is not merely wrong, it is *confidently* wrong, because a constant "
            f"bias is unobservable to a filter that models only zero-mean noise. No amount "
            f"of covariance bookkeeping fixes this; it needs an independent heading "
            f"reference or an observability-aware calibration step."
        )
    add("")

    add("## Scenarios")
    add("")
    add(
        "Accuracy is reported over **confirmed** tracks only. Tentative tracks are "
        "internal filter state that never reaches an operator. Errors are split by "
        "geometry class derived from *ground truth* rather than from the system's own "
        "degeneracy flag, so the flag can be scored rather than assumed."
    )
    add("")
    header = (
        "| scenario | tracks | good / degen | median good (m) | p90 good (m) | RMSE good (m) "
        "| median degen (m) | mean NEES | calibrated | gross err | purity |"
    )
    add(header)
    add("|" + "---|" * 11)
    for name, s in summaries.items():
        add(
            f"| `{name}` | {s.get('n_tracks', 0)} | {s.get('n_good', 0)} / "
            f"{s.get('n_degenerate', 0)} | {_fmt(s.get('median_good'))} | "
            f"{_fmt(s.get('p90_good'))} | {_fmt(s.get('rmse_good'))} | "
            f"{_fmt(s.get('median_degenerate'))} | {_fmt(s.get('nees_good_mean'), '.2f')} | "
            f"{'yes' if s.get('nees_calibrated') else 'no'} | "
            f"{_fmt(s.get('gross_error_rate'), '.3f')} | {_fmt(s.get('track_purity'), '.3f')} |"
        )
    add("")
    add(
        "`RMSE good` sits well above `median good` throughout. That gap is the point: "
        "the error distribution is heavy-tailed, so RMSE alone would misrepresent both "
        "typical accuracy and worst-case behaviour. `gross err` is the fraction of "
        "confirmed tracks beyond 5 m."
    )
    add("")

    if split_plot:
        add(f"![Geometry split]({split_plot.relative_to(out_path.parent)})")
        add("")

    add("## Degenerate-geometry detection")
    add("")
    add(
        "Scored as a detector. Ground truth for 'degenerate' is the fractional range "
        "uncertainty the geometry actually supports, `sqrt(2) R sigma_theta / B_perp`, "
        "computed from the true object position and the real camera centres."
    )
    add("")
    add("| scenario | recall | precision | truly degenerate |")
    add("|---|---|---|---|")
    for name, s in summaries.items():
        add(
            f"| `{name}` | {_fmt(s.get('degenerate_recall'), '.2f')} | "
            f"{_fmt(s.get('degenerate_precision'), '.2f')} | {s.get('n_degenerate', 0)} |"
        )
    add("")

    add("## Filter calibration (NEES)")
    add("")
    add(
        "NEES is the normalised estimation error squared: "
        "`e^T P^-1 e` for error `e` and reported covariance `P`. For an honest 3-D filter "
        "it is chi-square with 3 degrees of freedom, so the mean should be 3. Above the "
        "band means overconfident -- the system claims more precision than it has, which "
        "is the dangerous direction. Below means it is discarding information."
    )
    add("")
    add("| scenario | mean NEES | median | 95% band | verdict |")
    add("|---|---|---|---|---|")
    for name, s in summaries.items():
        if "nees_good_mean" not in s:
            continue
        low, high = s.get("nees_lower"), s.get("nees_upper")
        mean = s["nees_good_mean"]
        verdict = (
            "calibrated" if s.get("nees_calibrated")
            else ("overconfident" if mean > (high or 0) else "conservative")
        )
        add(
            f"| `{name}` | {_fmt(mean, '.2f')} | {_fmt(s.get('nees_median'), '.2f')} | "
            f"[{_fmt(low, '.2f')}, {_fmt(high, '.2f')}] | {verdict} |"
        )
    add("")

    if sweeps:
        add("## Error vs injected noise")
        add("")
        for sweep in sweeps:
            parameter = (
                next(iter(sweep.points[0].overrides), "value") if sweep.points else "value"
            )
            add(f"### `{sweep.name}` — {parameter}")
            add("")
            add(
                f"| {parameter} | median good (m) | p90 (m) | RMSE (m) "
                "| mean NEES | calibrated | tracks |"
            )
            add("|---|---|---|---|---|---|---|")
            for point in sweep.points:
                s = point.summary
                add(
                    f"| {point.overrides.get(parameter)} | {_fmt(s.get('median_good'))} | "
                    f"{_fmt(s.get('p90_good'))} | {_fmt(s.get('rmse_good'))} | "
                    f"{_fmt(s.get('nees_good_mean'), '.2f')} | "
                    f"{'yes' if s.get('nees_calibrated') else 'no'} | {s.get('n_good', 0)} |"
                )
            add("")
            for plot in sweep_plots.get(sweep.name, []):
                add(f"![{plot.stem}]({plot.relative_to(out_path.parent)})")
            add("")

    add("## Limitations")
    add("")
    add(
        "- **Bounding-box centroid drift.** Bearings are taken through the centroid of a "
        "2-D box, which is not the centroid of the 3-D object and moves with viewing "
        "aspect. This is a bias, not noise, so it does not average out over a track and "
        "is not represented in the covariance."
    )
    add(
        "- **Degenerate geometry under forward motion.** The common vehicle case is the "
        "bad case. It is detected and reported, not solved."
    )
    add(
        "- **Constant heading bias is invisible to the filter.** See above. It needs an "
        "external reference."
    )
    add(
        "- **Linearised covariance is local.** For short tracks on small baselines the "
        "covariance is computed about an estimate that may itself be badly wrong, so it "
        "can be small and wrong together. Mitigated by refusing to confirm tracks below "
        "a minimum observation count and perpendicular baseline; not eliminated."
    )
    add(
        "- **Synthetic data only.** These numbers come from a geometric simulator with a "
        "perfect detector. They isolate geometry and filter error, which is what they are "
        "for, but they are an upper bound on real-world performance."
    )
    add(
        "- **nuScenes poses are map-local, not WGS84.** When that loader is used, any "
        "lat/lon emitted is relative to an assumed map-region origin and is accurate "
        "relatively, not absolutely."
    )
    add("")

    out_path.write_text("\n".join(lines))
    return out_path
