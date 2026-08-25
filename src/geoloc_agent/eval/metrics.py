"""Scoring.

Two choices here matter more than the rest.

**Robust statistics alongside RMSE.** The geolocation error distribution is
heavy-tailed by construction: a short track on a small perpendicular baseline
can land badly wrong while reporting a small covariance, because the covariance
is linearised about the estimate. RMSE over a modest number of runs then measures
how many outliers landed in the sample rather than how accurate the system is.
So median and p90 are reported next to it, and the gross-error rate is its own
number rather than something averaged into silence.

**Geometry classes come from truth, not from the flag being tested.** Splitting
"good" from "degenerate" using the pipeline's own degeneracy flag would make the
split unfalsifiable -- the flag would be grading its own homework. The true
parallax is computed from the true object position and the actual camera centres,
and the flag is then scored against it as a detection problem with a recall and a
false-alarm rate.
"""

from __future__ import annotations

import collections
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats

from geoloc_agent.contracts import TrackStatus
from geoloc_agent.fuse.degenerate import DEFAULT_MAX_RELATIVE_RANGE_SIGMA
from geoloc_agent.geometry import perpendicular_projector
from geoloc_agent.pipeline import PipelineResult

GROSS_ERROR_M = 5.0
CONVERGENCE_SIGMA_M = 2.0


@dataclass
class TrackScore:
    track_id: int
    truth_id: str | None
    n_obs: int
    error_m: float
    nees: float
    sigma_horizontal: float
    cep50: float
    purity: float
    flagged_degenerate: bool
    truly_degenerate: bool
    true_relative_range_sigma: float
    perp_baseline: float
    time_to_converge: float
    class_correct: bool
    class_entropy: float
    confirmed: bool = True


@dataclass
class ScenarioMetrics:
    """Everything one scenario run produces. Serialises straight to a CSV row."""

    scenario: str
    seed: int
    n_tracks: int = 0
    n_truth: int = 0
    n_detections: int = 0
    n_false_positives: int = 0

    rmse_all: float = float("nan")
    median_all: float = float("nan")
    rmse_good: float = float("nan")
    median_good: float = float("nan")
    p90_good: float = float("nan")
    rmse_degenerate: float = float("nan")
    median_degenerate: float = float("nan")

    n_tentative: int = 0
    nees_mean: float = float("nan")
    nees_good_mean: float = float("nan")
    nees_lower: float = float("nan")
    nees_upper: float = float("nan")
    nees_calibrated: bool = False

    track_purity: float = float("nan")
    fragmentation: float = float("nan")
    truth_coverage: float = float("nan")
    false_track_rate: float = float("nan")
    time_to_converge: float = float("nan")

    degenerate_recall: float = float("nan")
    degenerate_precision: float = float("nan")
    gross_error_rate: float = float("nan")
    class_accuracy: float = float("nan")

    noise: dict = field(default_factory=dict)
    tracks: list[TrackScore] = field(default_factory=list)

    def to_row(self) -> dict:
        row = {k: v for k, v in asdict(self).items() if k not in ("tracks", "noise")}
        row.update({f"noise_{k}": v for k, v in self.noise.items()})
        return row


def _true_perpendicular_baseline(origins: list[np.ndarray], truth: np.ndarray) -> float:
    """Perpendicular baseline measured against the *true* object position."""
    if len(origins) < 2:
        return 0.0
    centre = np.mean(origins, axis=0)
    line_of_sight = np.asarray(truth, dtype=float) - centre
    norm = float(np.linalg.norm(line_of_sight))
    if norm < 1e-9:
        return 0.0
    projector = perpendicular_projector(line_of_sight / norm)
    projected = [projector @ o for o in origins]
    best = 0.0
    for i in range(len(projected)):
        for j in range(i + 1, len(projected)):
            best = max(best, float(np.linalg.norm(projected[i] - projected[j])))
    return best


def _true_relative_range_sigma(
    origins: list[np.ndarray], truth: np.ndarray, bearing_sigma: float, gps_sigma: float = 0.0
) -> tuple[float, float]:
    """Fractional range uncertainty the geometry *actually* supports.

    ``sqrt(2) R^2 sigma / B`` divided by ``R`` gives ``sqrt(2) R sigma / B``,
    which is scale-free and is the honest way to ask whether a fix was ever
    possible -- independent of where the estimator happened to put it.

    Camera-position noise has to enter here, not just bearing noise. From a
    single view the two are indistinguishable, and ``sigma_gps / R`` is the
    angular error a position error of that size induces. Leaving it out would
    file tracks as "good geometry" that metre-level GPS noise has already made
    hopeless, and then report their large errors as an accuracy failure rather
    than as the geometry limit it actually is.
    """
    perp = _true_perpendicular_baseline(origins, truth)
    if not origins:
        return 0.0, float("inf")
    range_m = float(np.linalg.norm(np.asarray(truth, float) - origins[-1]))
    if perp <= 1e-9:
        return perp, float("inf")
    effective = float(np.hypot(bearing_sigma, gps_sigma / max(range_m, 1e-6)))
    return perp, float(np.sqrt(2.0) * range_m * effective / perp)


def score_result(
    result: PipelineResult,
    scenario: str = "",
    seed: int = 0,
    truly_degenerate_threshold: float = DEFAULT_MAX_RELATIVE_RANGE_SIGMA,
    convergence_sigma: float = CONVERGENCE_SIGMA_M,
) -> ScenarioMetrics:
    metrics = ScenarioMetrics(
        scenario=scenario or result.session_name,
        seed=seed,
        n_truth=len(result.truth),
        n_detections=result.n_detections,
        n_false_positives=result.n_false_positives,
        noise=result.noise.to_dict(),
    )

    scores: list[TrackScore] = []
    for track in result.final_tracks:
        record = result.track_records.get(track.track_id)
        truth_id = track.truth_id
        truth_position = result.truth.get(truth_id) if truth_id else None

        if record and record.truth_ids:
            counts = collections.Counter(record.truth_ids)
            top_id, top_count = counts.most_common(1)[0]
            purity = top_count / len(record.truth_ids)
        else:
            purity = float("nan")

        if truth_position is None:
            # A track with no truth behind it is a false track, scored as such.
            scores.append(
                TrackScore(
                    track_id=track.track_id, truth_id=None, n_obs=track.n_obs,
                    error_m=float("nan"), nees=float("nan"),
                    sigma_horizontal=track.sigma_horizontal, cep50=track.cep50,
                    purity=purity, flagged_degenerate=track.degenerate,
                    truly_degenerate=False, true_relative_range_sigma=float("nan"),
                    perp_baseline=track.max_perp_baseline, time_to_converge=float("nan"),
                    class_correct=False, class_entropy=track.class_entropy,
                    confirmed=track.status
                    in (TrackStatus.CONFIRMED, TrackStatus.COASTING),
                )
            )
            continue

        delta = track.mean - truth_position
        error = float(np.linalg.norm(delta))
        try:
            nees = float(delta @ np.linalg.solve(track.cov, delta))
        except np.linalg.LinAlgError:
            nees = float("nan")

        origins = record.origins if record else []
        sigma = float(np.mean(record.bearing_sigmas)) if record and record.bearing_sigmas else 0.0
        perp, relative = _true_relative_range_sigma(
            origins, truth_position, sigma, gps_sigma=result.noise.gps_sigma
        )

        converge = float("nan")
        if record:
            for t, sigma_h in record.sigma_history:
                if sigma_h <= convergence_sigma:
                    converge = t - record.first_t
                    break

        scores.append(
            TrackScore(
                track_id=track.track_id, truth_id=truth_id, n_obs=track.n_obs,
                error_m=error, nees=nees, sigma_horizontal=track.sigma_horizontal,
                cep50=track.cep50, purity=purity, flagged_degenerate=track.degenerate,
                truly_degenerate=relative > truly_degenerate_threshold,
                true_relative_range_sigma=relative, perp_baseline=perp,
                time_to_converge=converge,
                class_correct=track.top_class[0] == result.truth_classes.get(truth_id, "?"),
                class_entropy=track.class_entropy,
                confirmed=track.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING),
            )
        )

    metrics.tracks = scores
    metrics.n_tracks = len(scores)
    metrics.n_tentative = sum(1 for s in scores if not s.confirmed)
    _summarise(metrics, result)
    return metrics


def _rms(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")


def _summarise(metrics: ScenarioMetrics, result: PipelineResult) -> None:
    # Only confirmed tracks are scored. A tentative track is internal filter
    # state, not an output: it has two or three observations, has not earned a
    # fix, and would never reach an operator. Averaging it into the accuracy
    # numbers measures the tracker's scratch space rather than its product.
    # Tentative tracks are still counted, under n_tentative.
    real = [s for s in metrics.tracks if s.truth_id is not None and s.confirmed]
    good = [s for s in real if not s.truly_degenerate]
    degenerate = [s for s in real if s.truly_degenerate]

    metrics.rmse_all = _rms([s.error_m for s in real])
    metrics.median_all = float(np.median([s.error_m for s in real])) if real else float("nan")
    metrics.rmse_good = _rms([s.error_m for s in good])
    metrics.rmse_degenerate = _rms([s.error_m for s in degenerate])
    if good:
        metrics.median_good = float(np.median([s.error_m for s in good]))
        metrics.p90_good = float(np.percentile([s.error_m for s in good], 90))
    if degenerate:
        metrics.median_degenerate = float(np.median([s.error_m for s in degenerate]))

    nees_values = [s.nees for s in real if np.isfinite(s.nees)]
    good_nees = [s.nees for s in good if np.isfinite(s.nees)]
    if nees_values:
        metrics.nees_mean = float(np.mean(nees_values))
    if good_nees:
        metrics.nees_good_mean = float(np.mean(good_nees))
        # Two-sided 95% interval for the mean of N chi-square(3) samples.
        lower, upper = stats.chi2.ppf([0.025, 0.975], 3 * len(good_nees))
        metrics.nees_lower = float(lower / len(good_nees))
        metrics.nees_upper = float(upper / len(good_nees))
        metrics.nees_calibrated = bool(
            metrics.nees_lower <= metrics.nees_good_mean <= metrics.nees_upper
        )

    purities = [s.purity for s in real if np.isfinite(s.purity)]
    metrics.track_purity = float(np.mean(purities)) if purities else float("nan")

    # Fragmentation: how many tracks each truth object was split across.
    if result.truth:
        per_truth = collections.Counter(s.truth_id for s in real)
        metrics.fragmentation = float(
            np.mean([per_truth.get(t, 0) for t in result.truth if per_truth.get(t, 0) > 0])
            if any(per_truth.values())
            else 0.0
        )
        metrics.truth_coverage = len(set(s.truth_id for s in real)) / len(result.truth)
    metrics.false_track_rate = (
        (len(metrics.tracks) - len(real)) / len(metrics.tracks) if metrics.tracks else 0.0
    )

    converge = [s.time_to_converge for s in good if np.isfinite(s.time_to_converge)]
    metrics.time_to_converge = float(np.mean(converge)) if converge else float("nan")

    # Degeneracy flag scored as a detector against truth-derived geometry.
    truly = [s for s in real if s.truly_degenerate]
    flagged = [s for s in real if s.flagged_degenerate]
    hits = [s for s in real if s.flagged_degenerate and s.truly_degenerate]
    metrics.degenerate_recall = len(hits) / len(truly) if truly else float("nan")
    metrics.degenerate_precision = len(hits) / len(flagged) if flagged else float("nan")

    if real:
        metrics.gross_error_rate = float(
            np.mean([s.error_m > GROSS_ERROR_M for s in real if np.isfinite(s.error_m)])
        )
        metrics.class_accuracy = float(np.mean([s.class_correct for s in real]))


def aggregate(runs: list[ScenarioMetrics]) -> dict:
    """Pool many seeds into one honest summary.

    Errors are pooled across every track of every seed rather than averaged
    per-seed, so a seed with one track does not carry the same weight as a seed
    with six. The NEES band is recomputed for the pooled sample size, which is
    what makes "calibrated" a statement about the run rather than about luck.
    """
    real = [s for run in runs for s in run.tracks if s.truth_id is not None and s.confirmed]
    good = [s for s in real if not s.truly_degenerate]
    degenerate = [s for s in real if s.truly_degenerate]

    summary: dict = {
        "n_runs": len(runs),
        "n_tracks": len(real),
        "n_tentative": sum(1 for run in runs for s in run.tracks if not s.confirmed),
        "n_good": len(good),
        "n_degenerate": len(degenerate),
        "rmse_all": _rms([s.error_m for s in real]),
        "rmse_good": _rms([s.error_m for s in good]),
        "rmse_degenerate": _rms([s.error_m for s in degenerate]),
        "median_good": float(np.median([s.error_m for s in good])) if good else float("nan"),
        "p90_good": float(np.percentile([s.error_m for s in good], 90)) if good else float("nan"),
        "median_degenerate": (
            float(np.median([s.error_m for s in degenerate])) if degenerate else float("nan")
        ),
        "gross_error_rate": (
            float(np.mean([s.error_m > GROSS_ERROR_M for s in real])) if real else float("nan")
        ),
        "track_purity": (
            float(np.mean([s.purity for s in real if np.isfinite(s.purity)]))
            if real
            else float("nan")
        ),
        "class_accuracy": float(np.mean([s.class_correct for s in real])) if real else float("nan"),
    }

    good_nees = [s.nees for s in good if np.isfinite(s.nees)]
    if good_nees:
        lower, upper = stats.chi2.ppf([0.025, 0.975], 3 * len(good_nees))
        summary["nees_good_mean"] = float(np.mean(good_nees))
        summary["nees_lower"] = float(lower / len(good_nees))
        summary["nees_upper"] = float(upper / len(good_nees))
        summary["nees_calibrated"] = bool(
            summary["nees_lower"] <= summary["nees_good_mean"] <= summary["nees_upper"]
        )
        summary["nees_median"] = float(np.median(good_nees))

    truly = [s for s in real if s.truly_degenerate]
    flagged = [s for s in real if s.flagged_degenerate]
    hits = [s for s in real if s.flagged_degenerate and s.truly_degenerate]
    summary["degenerate_recall"] = len(hits) / len(truly) if truly else float("nan")
    summary["degenerate_precision"] = len(hits) / len(flagged) if flagged else float("nan")

    converge = [s.time_to_converge for s in good if np.isfinite(s.time_to_converge)]
    summary["time_to_converge"] = float(np.mean(converge)) if converge else float("nan")
    fragments = [r.fragmentation for r in runs if np.isfinite(r.fragmentation)]
    summary["fragmentation"] = float(np.mean(fragments)) if fragments else float("nan")
    return summary
