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


def score_run(run, manifest, depth_key: str = "surface_depth_m") -> ModelScore:
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

    buckets = []
    for lo, hi in BUCKETS:
        m = (true >= lo) & (true < hi)
        if m.sum() == 0:
            continue
        buckets.append(
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
        device=run.device, notes=run.notes, buckets=buckets,
    )


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
