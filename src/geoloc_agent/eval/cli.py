"""`geoloc-eval`: one command, one report.

    geoloc-eval                       # scenarios + sweeps -> reports/eval.md
    geoloc-eval --quick               # fewer seeds, no sweeps
    geoloc-eval --scenario clean_lateral
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from geoloc_agent.eval.metrics import aggregate
from geoloc_agent.eval.report import build_report
from geoloc_agent.eval.runner import run_scenario, run_sweep, write_runs_csv, write_sweep_csv
from geoloc_agent.eval.scenario import Scenario, Sweep

DEFAULT_SCENARIOS = Path("configs/scenarios/synthetic.yaml")
DEFAULT_SWEEPS = Path("configs/sweeps/noise.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geoloc-eval", description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--sweeps", type=Path, default=DEFAULT_SWEEPS)
    parser.add_argument("--out", type=Path, default=Path("reports/eval.md"))
    parser.add_argument("--scenario", action="append", help="run only these scenarios")
    parser.add_argument("--sweep", action="append", help="run only these sweeps")
    parser.add_argument("--seeds", type=int, default=None, help="override seed count")
    parser.add_argument("--quick", action="store_true", help="few seeds, skip sweeps")
    parser.add_argument("--no-sweeps", action="store_true")
    args = parser.parse_args(argv)

    scenarios = {s.name: s for s in Scenario.load(args.scenarios)}
    if args.scenario:
        missing = set(args.scenario) - set(scenarios)
        if missing:
            parser.error(f"unknown scenario(s): {sorted(missing)}")
        scenarios = {k: v for k, v in scenarios.items() if k in args.scenario}

    seeds = 3 if args.quick else args.seeds
    started = time.time()

    scenario_runs = {}
    for name, scenario in scenarios.items():
        print(f"[scenario] {name} ...", end="", flush=True)
        runs = run_scenario(scenario, seeds=seeds)
        scenario_runs[name] = runs
        summary = aggregate(runs)
        median = summary.get("median_good")
        print(
            f" {summary['n_tracks']} tracks, median(good)="
            f"{median if median is None else round(median, 3)} m"
        )

    sweeps = []
    if not (args.quick or args.no_sweeps):
        specs = Sweep.load(args.sweeps)
        if args.sweep:
            specs = [s for s in specs if s.name in args.sweep]
        for spec in specs:
            print(f"[sweep] {spec.name} ({len(spec.points())} points) ...", end="", flush=True)
            sweeps.append(run_sweep(spec, scenarios if not args.scenario else scenarios))
            print(" done")

    out_dir = args.out.parent
    all_runs = [run for runs in scenario_runs.values() for run in runs]
    write_runs_csv(all_runs, out_dir / "runs.csv")
    if sweeps:
        write_sweep_csv(sweeps, out_dir / "sweeps.csv")
    report = build_report(scenario_runs, sweeps, args.out, plots_dir=out_dir / "plots")

    print(f"\nwrote {report}  ({time.time() - started:.1f}s)")
    print(f"      {out_dir / 'runs.csv'}")
    if sweeps:
        print(f"      {out_dir / 'sweeps.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
