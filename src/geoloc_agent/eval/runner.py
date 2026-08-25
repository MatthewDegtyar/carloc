"""Scenario and sweep execution."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from geoloc_agent.eval.metrics import ScenarioMetrics, aggregate, score_result
from geoloc_agent.eval.scenario import Scenario, Sweep
from geoloc_agent.pipeline import run_pipeline
from geoloc_agent.range.triangulation import TriangulationRanger


@dataclass
class SweepPoint:
    label: str
    overrides: dict
    runs: list[ScenarioMetrics] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


@dataclass
class SweepResult:
    name: str
    scenario: str
    points: list[SweepPoint] = field(default_factory=list)


def run_scenario(
    scenario: Scenario, noise_overrides: dict | None = None, seeds: int | None = None
) -> list[ScenarioMetrics]:
    """Run one scenario across its seeds and score every run.

    A fresh session is built per seed so that the seed drives noise only; the
    underlying geometry is identical across seeds, which is what makes the spread
    across seeds interpretable as noise sensitivity rather than scene variation.
    """
    noise = scenario.noise_model(noise_overrides)
    tracker_config = scenario.tracker_config()
    count = seeds if seeds is not None else scenario.seeds
    results: list[ScenarioMetrics] = []
    for seed in range(count):
        session = scenario.build_session()
        result = run_pipeline(
            session,
            tracker_config=tracker_config,
            noise=noise,
            seed=seed,
            bearing_sigma_px=scenario.bearing_sigma_px,
            ranger=TriangulationRanger() if scenario.use_ranger else None,
            range_every_n=scenario.range_every_n,
        )
        results.append(score_result(result, scenario=scenario.name, seed=seed))
    return results


def run_sweep(sweep: Sweep, scenarios: dict[str, Scenario]) -> SweepResult:
    if sweep.scenario not in scenarios:
        raise KeyError(f"sweep '{sweep.name}' references unknown scenario '{sweep.scenario}'")
    scenario = scenarios[sweep.scenario]
    out = SweepResult(name=sweep.name, scenario=sweep.scenario)
    for label, overrides in sweep.points():
        runs = run_scenario(scenario, noise_overrides=overrides, seeds=sweep.seeds)
        point = SweepPoint(label=label, overrides=overrides, runs=runs)
        point.summary = aggregate(runs)
        out.points.append(point)
    return out


def write_runs_csv(runs: list[ScenarioMetrics], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_row() for r in runs]
    if not rows:
        path.write_text("")
        return path
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_sweep_csv(sweeps: list[SweepResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sweep in sweeps:
        for point in sweep.points:
            row = {"sweep": sweep.name, "scenario": sweep.scenario, "point": point.label}
            row.update({f"noise_{k}": v for k, v in point.overrides.items()})
            row.update(point.summary)
            rows.append(row)
    if not rows:
        path.write_text("")
        return path
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
