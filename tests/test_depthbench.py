"""depthbench harness. Tests the scoring, not the models.

The failure mode a depth benchmark has is not crashing -- it is quietly being
unfair, so that the ranking reflects the harness rather than the models. These
pin the places where that could happen.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from depthbench.metrics import ModelScore, saturation_span, score_run
from depthbench.schema import GtObject, Manifest, Prediction, RefObject, RunResult, Sample

sys.path.insert(0, str(Path("depthbench/runners").resolve()))
import _common  # noqa: E402

K = [[1250.0, 0.0, 800.0], [0.0, 1250.0, 450.0], [0.0, 0.0, 1.0]]


def make_manifest(depths=(5.0, 15.0, 25.0, 35.0, 45.0)):
    objects = [
        GtObject(obj_id=f"o{i}", bbox=[700.0, 400.0, 760.0, 480.0], cls="car",
                 surface_depth_m=d, centroid_depth_m=d + 1.0, height_m=1.55)
        for i, d in enumerate(depths)
    ]
    return Manifest(
        source="test", samples=[Sample(image="x.jpg", width=1600, height=900, K=K,
                                       objects=objects)]
    )


def make_run(pred_fn, model="m"):
    manifest = make_manifest()
    preds = [
        Prediction(obj_id=o.obj_id, image="x.jpg", pred_depth_m=pred_fn(o.surface_depth_m))
        for s in manifest.samples for o in s.objects
    ]
    return RunResult(model=model, predictions=preds, seconds_per_image=1.0, device="cpu")


# --- depth sampling ----------------------------------------------------------


def test_sampling_uses_the_central_region_not_the_whole_box():
    """A box contains background through windows and around outlines.

    Averaging the whole box would charge every model for the scene behind the
    object, which has nothing to do with depth estimation.
    """
    depth = np.full((900, 1600), 100.0)          # distant background
    depth[420:460, 720:740] = 10.0               # the object, central only
    value = _common.sample_depth(depth, [700, 400, 760, 480], (1600, 900))
    assert value == pytest.approx(10.0)


def test_sampling_is_resolution_independent():
    """Models return depth at their own working resolution."""
    full = np.full((900, 1600), 50.0)
    full[420:461, 715:746] = 12.0
    # The object must fill the sampled region at BOTH resolutions, or the median
    # is background and the test measures the fixture rather than the sampler.
    small = np.full((225, 400), 50.0)
    small[105:116, 179:187] = 12.0
    a = _common.sample_depth(full, [700, 400, 760, 480], (1600, 900))
    b = _common.sample_depth(small, [700, 400, 760, 480], (1600, 900))
    assert a == pytest.approx(12.0)
    assert b == pytest.approx(12.0)


def test_sampling_ignores_non_finite_values():
    depth = np.full((900, 1600), np.nan)
    depth[430:450, 725:735] = 7.0
    assert _common.sample_depth(depth, [700, 400, 760, 480], (1600, 900)) == pytest.approx(7.0)


def test_sampling_returns_nan_when_nothing_is_usable():
    assert not np.isfinite(
        _common.sample_depth(np.full((900, 1600), np.nan), [700, 400, 760, 480], (1600, 900))
    )


# --- relative rescaling ------------------------------------------------------


def test_reference_rescale_recovers_metric_depth_from_known_size():
    """Z = fy*H/h on the reference, then the ratio to the model's relative value."""
    height_m, true_depth = 1.55, 20.0
    h_px = 1250.0 * height_m / true_depth
    y1 = 450.0 - h_px / 2
    sample = {
        "K": K, "width": 1600, "height": 900,
        "reference": {"bbox": [760.0, y1, 840.0, y1 + h_px], "height_m": height_m, "cls": "car"},
    }
    # Inverse-depth map: value = 1/metres.
    relative = np.full((900, 1600), 1.0 / 40.0)
    relative[int(y1) : int(y1 + h_px), 760:840] = 1.0 / true_depth

    scale = _common.reference_scale(relative, sample, inverse=True)
    assert np.isfinite(scale)
    # The reference itself must come back at its own depth.
    assert _common.apply_scale(1.0 / true_depth, scale, inverse=True) == pytest.approx(
        true_depth, rel=0.02
    )
    # And another object at 40 m must come back near 40 m.
    assert _common.apply_scale(1.0 / 40.0, scale, inverse=True) == pytest.approx(40.0, rel=0.02)


def test_rescale_declines_without_a_reference():
    sample = {"K": K, "width": 1600, "height": 900, "reference": None}
    assert not np.isfinite(_common.reference_scale(np.ones((900, 1600)), sample, inverse=True))


def test_apply_scale_rejects_impossible_values():
    assert not np.isfinite(_common.apply_scale(0.0, 10.0, inverse=True))
    assert not np.isfinite(_common.apply_scale(-1.0, 10.0, inverse=False))
    assert not np.isfinite(_common.apply_scale(5.0, float("nan"), inverse=False))


# --- scoring -----------------------------------------------------------------


def test_perfect_model_scores_zero_error():
    score = score_run(make_run(lambda d: d), make_manifest())
    assert score.median_abs_err == pytest.approx(0.0)
    assert score.delta1 == pytest.approx(1.0)
    assert score.spearman == pytest.approx(1.0)
    assert score.coverage == pytest.approx(1.0)


def test_declining_hard_objects_shows_up_as_coverage_not_accuracy():
    """A model must not look good by refusing to answer.

    This is the main way a depth benchmark lies: predictions that come back
    non-finite get dropped, the remaining easy objects score well, and the model
    looks better than one that attempted everything.
    """
    score = score_run(
        make_run(lambda d: d if d < 20 else float("nan")), make_manifest()
    )
    assert score.median_abs_err == pytest.approx(0.0)   # perfect on what it answered
    assert score.coverage < 0.5                          # but it answered almost nothing


def test_saturation_is_detected():
    """The characteristic monocular failure: same answer regardless of distance."""
    flat = score_run(make_run(lambda d: 20.0), make_manifest())
    honest = score_run(make_run(lambda d: d * 1.1), make_manifest())
    assert saturation_span(flat) == pytest.approx(0.0, abs=0.05)
    assert saturation_span(honest) == pytest.approx(1.1, rel=0.1)


def test_rank_correlation_catches_an_inverted_depth_map():
    """Reading an inverse-depth map as if it were depth inverts the ordering."""
    inverted = score_run(make_run(lambda d: 50.0 - d), make_manifest())
    assert inverted.spearman < -0.9


def test_scoring_against_centroid_differs_from_surface():
    """Both are reported because the gap is comparable to the errors measured."""
    manifest = make_manifest()
    run = make_run(lambda d: d)  # exactly right on SURFACE depth
    surface = score_run(run, manifest, depth_key="surface_depth_m")
    centroid = score_run(run, manifest, depth_key="centroid_depth_m")
    assert surface.median_abs_err == pytest.approx(0.0)
    assert centroid.median_abs_err == pytest.approx(1.0)  # the fixture's offset


def test_failed_run_is_reported_not_silently_dropped():
    score = score_run(RunResult(model="m", failed=True, error="boom"), make_manifest())
    assert score.failed
    assert "boom" in score.error


def test_run_with_no_usable_predictions_is_a_failure():
    score = score_run(make_run(lambda d: float("nan")), make_manifest())
    assert score.failed


def test_buckets_cover_the_declared_range():
    score = score_run(make_run(lambda d: d + 1.0), make_manifest())
    expected = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    assert [(b.lo, b.hi) for b in score.buckets] == expected
    assert all(b.median_abs_err == pytest.approx(1.0) for b in score.buckets)


# --- manifest ----------------------------------------------------------------


def test_manifest_round_trips(tmp_path):
    manifest = make_manifest()
    manifest.samples[0].reference = RefObject(bbox=[1.0, 2.0, 3.0, 4.0], height_m=1.55, cls="car")
    path = manifest.write(tmp_path / "m.json")
    back = Manifest.read(path)
    assert len(back.samples[0].objects) == len(manifest.samples[0].objects)
    assert back.samples[0].reference.height_m == pytest.approx(1.55)


def test_manifest_rejects_a_version_mismatch(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"version": 99, "samples": []}))
    with pytest.raises(ValueError, match="version"):
        Manifest.read(path)


def test_runner_contract_needs_no_depthbench_import():
    """Runners execute in foreign venvs and must not import this package."""
    for runner in Path("depthbench/runners").glob("*.py"):
        source = runner.read_text()
        assert "from depthbench" not in source, runner
        assert "import depthbench" not in source, runner


def test_every_model_env_points_at_a_real_runner():
    from depthbench.envs import MODELS

    for name, env in MODELS.items():
        assert (Path("depthbench/runners") / env.runner).exists(), name
        assert env.packages, name


def test_report_renders_without_a_model(tmp_path):
    from depthbench.report import build_report

    scores = [
        score_run(make_run(lambda d: d * 1.2, model="good"), make_manifest()),
        ModelScore(model="broken", variant="", n=0, coverage=0.0,
                   median_abs_err=float("nan"), p90_abs_err=float("nan"),
                   median_rel_err=float("nan"), delta1=float("nan"), spearman=float("nan"),
                   seconds_per_image=float("nan"), device="", failed=True, error="did not load"),
    ]
    out = build_report(scores, make_manifest(), tmp_path / "r.md")
    text = out.read_text()
    assert "## Results" in text
    assert "## Saturation" in text
    assert "Did not run" in text and "did not load" in text
