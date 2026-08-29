"""Confidence intervals on the parked-car count.

Two knobs, one battery. The count depends on how strict you are — the
``min_frames`` frame-confidence threshold (a car must be tracked across at least
that many frames to be counted). And for any fixed strictness, the count has
sampling noise — resample the frames and it wobbles. :func:`sweep` reports both:
the count at each threshold, with a bootstrap CI from frame resampling.

Run detection once (:func:`carloc.video.detect_segment`) and pass the raw
detections in — the sweep only re-runs the cheap tracking step, so a whole battery
costs one model pass.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from carloc.video import track_parked


@dataclass
class ConfidenceRow:
    min_frames: int          # the frame-confidence threshold
    count: int               # count at full sampling
    ci_lo: int               # bootstrap CI, low
    ci_hi: int               # bootstrap CI, high
    boot_median: int


def _count(left, right, lateral_m, min_frames) -> int:
    return (len(track_parked(left, "left", lateral_m, min_frames))
            + len(track_parked(right, "right", lateral_m, min_frames)))


def sweep(left: list[dict], right: list[dict], thresholds=(2, 3, 4, 5, 6),
          lateral_m: float = 7.0, n_boot: int = 60, keep_frac: float = 0.9,
          ci: float = 0.90, seed: int = 0) -> list[ConfidenceRow]:
    """Count vs frame-confidence threshold, each with a frame-resampling CI.

    ``keep_frac`` of frames are kept in each of ``n_boot`` resamples; the spread of
    the resulting counts is the CI. (Fewer frames catch fewer cars, so the band
    leans below the full-sampling point — that asymmetry is honest: it shows how
    much the count depends on how densely you sampled.)
    """
    rng = random.Random(seed)
    frames = sorted({d["frame"] for d in left + right})
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    rows = []
    for mf in thresholds:
        point = _count(left, right, lateral_m, mf)
        boots = []
        for _ in range(n_boot):
            keep = {f for f in frames if rng.random() < keep_frac}
            boots.append(_count([d for d in left if d["frame"] in keep],
                                [d for d in right if d["frame"] in keep], lateral_m, mf))
        boots.sort()
        n = len(boots)
        rows.append(ConfidenceRow(
            min_frames=mf, count=point,
            ci_lo=boots[max(0, int(lo_q * n))],
            ci_hi=boots[min(n - 1, int(hi_q * n))],
            boot_median=boots[n // 2]))
    return rows
