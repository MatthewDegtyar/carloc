"""Scoring.

Absolute error in metres is the headline, because that is the question: how far
off is it. Relative error is reported alongside since a 3 m error means different
things at 8 m and 48 m.

Two diagnostics beyond the error tables, both there to catch failure modes that a
mean error hides:

**Saturation.** A model that compresses everything distant into a narrow band can
still post a respectable average while being unable to tell 20 m from 45 m. The
median prediction per true-depth bucket exposes it immediately, and it is the
characteristic failure of monocular depth.

**Rank correlation.** Whether the model orders objects correctly at all. A near-
zero value means the metric error is meaningless; a negative one means the depth
map was read upside down, which is a real risk with inverse-depth checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BUCKETS = ((0, 10), (10, 20), (20, 30), (30, 40), (40, 50))
DELTA_THRESHOLD = 1.25


def make_buckets(size: float, max_range: float = 50.0) -> tuple[tuple[float, float], ...]:
    """Range bins of a given width.

    Re-binning never requires re-running a model: runners write one prediction per
    object, so any interval can be recovered from the stored results. Narrower bins
    show where a model turns over more precisely, at the cost of fewer objects per
    bin -- which is why every table carries its own n.
    """
    edges = np.arange(0.0, max_range + size, size)
    return tuple((float(a), float(b)) for a, b in zip(edges[:-1], edges[1:], strict=False))


@dataclass
class BucketScore:
    lo: float
    hi: float
    n: int
    median_abs_err: float
    p90_abs_err: float
    median_rel_err: float
    median_pred: float
    median_true: float
    delta1: float


@dataclass
class ModelScore:
    model: str
    variant: str
    n: int
    coverage: float
    median_abs_err: float
    p90_abs_err: float
    median_rel_err: float
    delta1: float
    spearman: float
    seconds_per_image: float
    device: str
    usable_range_m: float = float("nan")
    """Furthest range at which median error stays within tolerance.

    Computed on fine bins independent of whatever binning is displayed, so the
    headline number does not move when someone changes the table's resolution."""

    reverses_at_m: float = float("nan")
    """Range beyond which predictions start DECREASING as true depth increases.

    Worse than saturation. A saturating model at least preserves ordering, so a
    downstream filter can still rank objects; a reversing one reports distant
    objects as nearer than closer ones, and there is nothing left to salvage."""

    notes: str = ""
    buckets: list[BucketScore] = field(default_factory=list)
    failed: bool = False
    error: str = ""

    @property
    def label(self) -> str:
        return f"{self.model} ({self.variant})" if self.variant else self.model


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def score_run(
    run, manifest, depth_key: str = "surface_depth_m", buckets=BUCKETS
) -> ModelScore:
    truth = {
        obj.obj_id: getattr(obj, depth_key)
        for sample in manifest.samples
        for obj in sample.objects
    }
    if run.failed:
        return ModelScore(
            model=run.model, variant=run.variant, n=0, coverage=0.0,
            median_abs_err=float("nan"), p90_abs_err=float("nan"),
            median_rel_err=float("nan"), delta1=float("nan"), spearman=float("nan"),
            seconds_per_image=run.seconds_per_image, device=run.device,
            notes=run.notes, failed=True, error=run.error,
        )

    pred, true = [], []
    attempted = 0
    for p in run.predictions:
        if p.obj_id not in truth:
            continue
        attempted += 1
        if np.isfinite(p.pred_depth_m) and p.pred_depth_m > 0:
            pred.append(p.pred_depth_m)
            true.append(truth[p.obj_id])
    pred, true = np.array(pred), np.array(true)
    # Coverage is tracked so a model that declines the hard objects cannot look
    # more accurate than one that attempts them.
    coverage = len(pred) / attempted if attempted else 0.0

    if len(pred) == 0:
        return ModelScore(
            model=run.model, variant=run.variant, n=0, coverage=0.0,
            median_abs_err=float("nan"), p90_abs_err=float("nan"),
            median_rel_err=float("nan"), delta1=float("nan"), spearman=float("nan"),
            seconds_per_image=run.seconds_per_image, device=run.device,
            notes=run.notes, failed=True, error="no usable predictions",
        )

    abs_err = np.abs(pred - true)
    rel_err = abs_err / true
    ratio = np.maximum(pred / true, true / pred)

    usable, reverses = _usable_and_reversal(pred, true, manifest.max_range_m)

    bucket_scores = []
    for lo, hi in buckets:
        m = (true >= lo) & (true < hi)
        if m.sum() == 0:
            continue
        bucket_scores.append(
            BucketScore(
                lo=lo, hi=hi, n=int(m.sum()),
                median_abs_err=float(np.median(abs_err[m])),
                p90_abs_err=float(np.percentile(abs_err[m], 90)),
                median_rel_err=float(np.median(rel_err[m])),
                median_pred=float(np.median(pred[m])),
                median_true=float(np.median(true[m])),
                delta1=float(np.mean(ratio[m] < DELTA_THRESHOLD)),
            )
        )

    return ModelScore(
        model=run.model, variant=run.variant, n=len(pred), coverage=coverage,
        median_abs_err=float(np.median(abs_err)), p90_abs_err=float(np.percentile(abs_err, 90)),
        median_rel_err=float(np.median(rel_err)), delta1=float(np.mean(ratio < DELTA_THRESHOLD)),
        spearman=_spearman(pred, true), seconds_per_image=run.seconds_per_image,
        device=run.device, notes=run.notes, buckets=bucket_scores,
        usable_range_m=usable, reverses_at_m=reverses,
    )


USABLE_ABS_M = 2.0
USABLE_REL = 0.15
FINE_BIN_M = 5.0


def _usable_and_reversal(pred: np.ndarray, true: np.ndarray, max_range: float):
    """Furthest bin meeting tolerance, and where the response turns over.

    The two scans are deliberately separate. Combining them hides every reversal:
    the usable-range scan stops at the first bin that fails tolerance, and a model
    only reverses well past that point, so a shared loop never looks at the bins
    where it happens.
    """
    edges = np.arange(0.0, max_range + FINE_BIN_M, FINE_BIN_M)
    stats = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (true >= lo) & (true < hi)
        if m.sum() < 3:
            stats.append(None)
            continue
        error = np.abs(pred[m] - true[m])
        stats.append({
            "hi": float(hi),
            "abs": float(np.median(error)),
            "rel": float(np.median(error / true[m])),
            "pred": float(np.median(pred[m])),
        })

    usable = float("nan")
    for entry in stats:
        if entry is None:
            continue
        if entry["abs"] <= USABLE_ABS_M or entry["rel"] <= USABLE_REL:
            usable = entry["hi"]
        else:
            break

    # Scan the WHOLE range for a turnover, not just the usable prefix.
    reverses = float("nan")
    finite = [e for e in stats if e is not None]
    for previous, current in zip(finite[:-1], finite[1:], strict=False):
        if current["pred"] < previous["pred"]:
            reverses = current["hi"]
            break
    return usable, reverses


def saturation_span(score: ModelScore) -> float:
    """Predicted span divided by true span across buckets.

    1.0 means the model tracks range correctly. Near 0 means it returns roughly
    the same answer regardless of distance -- the failure that an average error
    conceals.
    """
    if len(score.buckets) < 2:
        return float("nan")
    pred_span = score.buckets[-1].median_pred - score.buckets[0].median_pred
    true_span = score.buckets[-1].median_true - score.buckets[0].median_true
    return float(pred_span / true_span) if true_span else float("nan")
