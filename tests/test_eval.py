"""Phase 3: the eval harness itself, plus the pinned regression bounds.

The bounds are deliberately wide. Their job is to catch a regression that
changes behaviour by a factor, not to freeze the third decimal place of a
stochastic pipeline.
"""

import numpy as np
import pytest

from geoloc_agent.eval.metrics import aggregate, score_result
from geoloc_agent.eval.report import build_report
from geoloc_agent.eval.runner import run_scenario, run_sweep, write_runs_csv, write_sweep_csv
from geoloc_agent.eval.scenario import Scenario, Sweep
from geoloc_agent.io.synthetic import SyntheticScenario, SyntheticSession
from geoloc_agent.noise import NoiseModel
from geoloc_agent.pipeline import run_pipeline

SCENARIOS = "configs/scenarios/synthetic.yaml"
SWEEPS = "configs/sweeps/noise.yaml"


@pytest.fixture(scope="module")
def scenarios():
    return {s.name: s for s in Scenario.load(SCENARIOS)}


def test_scenarios_and_sweeps_load_and_cross_reference(scenarios):
    sweeps = Sweep.load(SWEEPS)
    assert scenarios and sweeps
    for sweep in sweeps:
        assert sweep.scenario in scenarios, f"{sweep.name} references a missing scenario"
        assert sweep.points()


def test_unknown_keys_are_rejected_rather_than_ignored():
    """A typo in a YAML key must fail loudly, not silently change nothing."""
    with pytest.raises(ValueError, match="unknown scenario keys"):
        Scenario.from_dict({"name": "x", "bearing_sigma": 0.1})
    with pytest.raises(ValueError, match="unknown noise parameters"):
        NoiseModel.from_dict({"bearing_sgima": 0.1})
    with pytest.raises(ValueError, match="unknown tracker keys"):
        Scenario(name="x", tracker={"gate_chi_2": 5.0}).tracker_config()


def test_scoring_produces_every_declared_metric():
    result = run_pipeline(SyntheticSession(SyntheticScenario(path="lateral", n_frames=40)))
    metrics = score_result(result, scenario="t", seed=0)
    for field in (
        "rmse_good", "median_good", "nees_good_mean", "track_purity",
        "fragmentation", "time_to_converge", "degenerate_recall",
    ):
        assert hasattr(metrics, field)
    assert metrics.n_tracks > 0


# --- pinned regression bounds ------------------------------------------------


def test_regression_clean_lateral_accuracy(scenarios):
    summary = aggregate(run_scenario(scenarios["clean_lateral"], seeds=8))
    assert summary["n_good"] >= 8
    assert summary["median_good"] < 0.5, summary["median_good"]
    assert summary["p90_good"] < 3.0, summary["p90_good"]
    assert summary["gross_error_rate"] < 0.10, summary["gross_error_rate"]
    assert summary["track_purity"] > 0.95, summary["track_purity"]


def test_regression_clean_lateral_is_not_wildly_overconfident(scenarios):
    """NEES bound. Nominal is 3; this catches an order-of-magnitude regression."""
    summary = aggregate(run_scenario(scenarios["clean_lateral"], seeds=8))
    assert 1.0 < summary["nees_good_mean"] < 10.0, summary["nees_good_mean"]


def test_regression_forward_motion_is_detected_not_silently_wrong(scenarios):
    """The headline behaviour: under forward motion the system must know it is blind."""
    summary = aggregate(run_scenario(scenarios["forward_motion"], seeds=8))
    assert summary["n_degenerate"] > 0
    assert summary["degenerate_recall"] > 0.9, summary["degenerate_recall"]


def test_regression_heading_bias_breaks_calibration(scenarios):
    """A constant yaw bias is unobservable to the filter, and NEES must show it.

    This asserts a *failure*. If it ever starts passing the calibration check,
    either the noise injection stopped working or someone inflated the
    covariance to make the metric look good.
    """
    summary = aggregate(run_scenario(scenarios["heading_bias"], seeds=8))
    assert not summary["nees_calibrated"]
    assert summary["nees_good_mean"] > 50.0, summary["nees_good_mean"]
    assert summary["median_good"] > 1.0


def test_regression_clutter_does_not_destroy_association(scenarios):
    summary = aggregate(run_scenario(scenarios["cluttered"], seeds=8))
    assert summary["track_purity"] > 0.90, summary["track_purity"]
    assert summary["median_good"] < 1.5, summary["median_good"]


def test_degenerate_geometry_is_reported_with_larger_error_than_good(scenarios):
    """The split must be meaningful, not just a label."""
    forward = aggregate(run_scenario(scenarios["forward_motion"], seeds=8))
    clean = aggregate(run_scenario(scenarios["clean_lateral"], seeds=8))
    assert forward["median_degenerate"] > 10 * clean["median_good"]


# --- the one-command deliverable ---------------------------------------------


def test_sweep_runs_and_error_grows_with_bearing_noise(scenarios):
    sweep = Sweep(
        name="t", scenario="clean_lateral", seeds=4,
        axes={"bearing_sigma": [0.0, 0.002, 0.008]},
    )
    result = run_sweep(sweep, scenarios)
    medians = [p.summary["median_good"] for p in result.points]
    assert medians[0] < medians[1] < medians[2], medians


def test_one_command_produces_a_report_with_the_required_sections(tmp_path, scenarios):
    """Phase 3 acceptance."""
    subset = {k: scenarios[k] for k in ("clean_lateral", "forward_motion", "heading_bias")}
    runs = {name: run_scenario(s, seeds=3) for name, s in subset.items()}
    sweep = Sweep(
        name="bearing_noise", scenario="clean_lateral", seeds=3,
        axes={"bearing_sigma": [0.0, 0.003]},
    )
    sweeps = [run_sweep(sweep, scenarios)]

    out = build_report(runs, sweeps, tmp_path / "eval.md", plots_dir=tmp_path / "plots")
    text = out.read_text()
    for section in (
        "# Geolocation evaluation", "## Scenarios", "## Degenerate-geometry detection",
        "## Filter calibration (NEES)", "## Error vs injected noise", "## Limitations",
    ):
        assert section in text, f"missing section: {section}"
    assert "bounding-box centroid" in text.lower()
    assert "wgs84" in text.lower()

    csv_path = write_runs_csv([r for rs in runs.values() for r in rs], tmp_path / "runs.csv")
    assert csv_path.read_text().count("\n") > 3
    assert write_sweep_csv(sweeps, tmp_path / "sweeps.csv").exists()


def test_report_plots_are_written(tmp_path, scenarios):
    pytest.importorskip("matplotlib")
    runs = {"clean_lateral": run_scenario(scenarios["clean_lateral"], seeds=3)}
    build_report(runs, [], tmp_path / "eval.md", plots_dir=tmp_path / "plots")
    assert (tmp_path / "plots" / "geometry_split.png").exists()


def test_metrics_survive_a_run_with_no_tracks():
    """Empty input must produce NaNs, not an exception."""
    session = SyntheticSession(SyntheticScenario(path="lateral", n_frames=40))
    result = run_pipeline(session, noise=NoiseModel(detection_dropout=1.0), seed=0)
    metrics = score_result(result)
    assert metrics.n_tracks == 0
    summary = aggregate([metrics])
    assert np.isnan(summary["rmse_all"])
