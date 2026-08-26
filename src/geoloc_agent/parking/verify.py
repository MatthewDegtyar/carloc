"""Is the car in this bay the same one that was here last pass?

Each cue contributes a log-likelihood ratio -- how much more probable this
observation is under "same vehicle" than under "different vehicle" -- and they
sum. The result is a posterior odds that can be quoted, thresholded, and
defended, rather than a similarity score that can only be tuned.

That distinction is the whole product. An enforcement decision gets appealed, and
"the algorithm matched them" is not an answer. "Given the observed agreement in
colour, length and bay position, a different vehicle would produce this match
once in N" is.

The `different` distributions are **measured from the local vehicle population**,
not assumed. What counts as a coincidence depends entirely on where you are: in a
street of silver sedans, colour is nearly worthless, and the model should say so
rather than being tuned once and shipped everywhere.

What this cannot do
-------------------
It cannot detect a vehicle that left and returned to the same bay in the same
pose. That is indistinguishable by construction, and if it matters the answer is
a shorter revisit interval, not a cleverer matcher.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geoloc_agent.parking.signature import VehicleSignature

MAX_LOG_LR_PER_CUE = 4.0
"""Ceiling on any single cue's contribution, about 55:1.

No cue here is reliable enough to carry a decision alone, and without a cap a
near-exact agreement in one dimension -- easy to get by chance on a coarse
measurement -- would swamp disagreement everywhere else. Capping makes the fusion
robust to one cue being wrong, at the cost of some power when it is genuinely
decisive."""

CITE_LOG_ODDS = 4.6
"""Threshold to act on a match: about 99:1, or a 1% false-match rate.

Deliberately a stated policy number rather than a tuned one. A city can move it,
and should be told what it costs in both directions."""


@dataclass(frozen=True)
class Population:
    """How much cars in *this* location vary. Fitted, not assumed."""

    length_sd_m: float
    width_sd_m: float
    value_sd: float
    tone_frequencies: dict[str, float]
    class_frequencies: dict[str, float]
    bay_span_m: float
    n: int

    @classmethod
    def fit(cls, signatures: list[VehicleSignature], bay_span_m: float = 5.5) -> Population:
        lengths = np.array([s.length_m for s in signatures if np.isfinite(s.length_m)])
        widths = np.array([s.width_m for s in signatures if np.isfinite(s.width_m)])
        values = np.array([s.value for s in signatures if np.isfinite(s.value)])
        tones = [s.tone for s in signatures]
        classes = [s.cls for s in signatures]

        def frequencies(items: list[str]) -> dict[str, float]:
            total = max(len(items), 1)
            # Floored: a category never seen is rare, not impossible, and a zero
            # would make one disagreement infinitely decisive.
            return {k: max(items.count(k) / total, 1.0 / (total + 1)) for k in set(items)}

        return cls(
            length_sd_m=float(np.std(lengths)) if len(lengths) > 2 else 0.9,
            width_sd_m=float(np.std(widths)) if len(widths) > 2 else 0.35,
            value_sd=float(np.std(values)) if len(values) > 2 else 0.18,
            tone_frequencies=frequencies(tones),
            class_frequencies=frequencies(classes),
            bay_span_m=bay_span_m,
            n=len(signatures),
        )


@dataclass(frozen=True)
class MatchVerdict:
    log_odds: float
    contributions: dict[str, float]
    same: bool
    reason: str

    @property
    def odds(self) -> float:
        return float(np.exp(np.clip(self.log_odds, -50, 50)))

    @property
    def false_match_rate(self) -> float:
        """One in how many, if this were a coincidence."""
        return float(1.0 / (1.0 + np.exp(self.log_odds)))

    def explain(self) -> str:
        parts = ", ".join(f"{k} {v:+.1f}" for k, v in sorted(
            self.contributions.items(), key=lambda kv: -abs(kv[1])))
        return f"log-odds {self.log_odds:+.2f} ({parts})"


def _gaussian_log_lr(delta: float, same_sd: float, different_sd: float) -> float:
    """Log ratio of two zero-mean Gaussians evaluated at the same difference.

    The measurement noise on the cue is ``same_sd``; the spread of the population
    is ``different_sd``. A cue only discriminates to the extent the second
    exceeds the first, which is why a precise measurement of a uniform quantity
    is worth nothing and a coarse measurement of a varied one can be worth a lot.
    """
    if not np.isfinite(delta) or same_sd <= 0 or different_sd <= 0:
        return 0.0
    log_lr = (np.log(different_sd / same_sd)
              - 0.5 * delta**2 * (1.0 / same_sd**2 - 1.0 / different_sd**2))
    return float(np.clip(log_lr, -MAX_LOG_LR_PER_CUE, MAX_LOG_LR_PER_CUE))


def compare(before: VehicleSignature, after: VehicleSignature,
            population: Population, threshold: float = CITE_LOG_ODDS) -> MatchVerdict:
    """Same vehicle, or did the bay turn over?"""
    if not (before.usable and after.usable):
        return MatchVerdict(0.0, {}, False,
                            "one signature is not usable -- too few looks, or a fix "
                            "too loose to place a car inside a bay")

    contributions: dict[str, float] = {}

    # Class. Coarse, and it is the population frequency that sets its worth:
    # agreeing on "car" where everything is a car says almost nothing.
    if before.cls == after.cls:
        frequency = population.class_frequencies.get(before.cls, 0.5)
        contributions["class"] = float(np.clip(-np.log(frequency), 0, MAX_LOG_LR_PER_CUE))
    else:
        contributions["class"] = -MAX_LOG_LR_PER_CUE

    if before.tone == after.tone and before.tone != "unknown":
        frequency = population.tone_frequencies.get(before.tone, 0.3)
        contributions["tone"] = float(np.clip(-np.log(frequency), 0, MAX_LOG_LR_PER_CUE))
    elif before.tone != "unknown" and after.tone != "unknown":
        contributions["tone"] = -2.0

    contributions["value"] = _gaussian_log_lr(
        before.value - after.value, same_sd=0.05, different_sd=population.value_sd)
    contributions["length"] = _gaussian_log_lr(
        before.length_m - after.length_m, same_sd=0.35, different_sd=population.length_sd_m)
    contributions["width"] = _gaussian_log_lr(
        before.width_m - after.width_m, same_sd=0.25, different_sd=population.width_sd_m)

    # The chalk mark. Compared *relative to the bay*, so the errors common to
    # both passes -- pose, GNSS, surface model -- cancel, and what is left is far
    # tighter than either absolute fix.
    same_sd = max(np.hypot(before.position_sigma_m, after.position_sigma_m) * 0.5, 0.10)
    contributions["bay_offset"] = _gaussian_log_lr(
        np.hypot(before.offset_along_m - after.offset_along_m,
                 before.offset_across_m - after.offset_across_m),
        same_sd=same_sd, different_sd=population.bay_span_m / np.sqrt(12.0))

    heading_delta = abs((before.heading_deg - after.heading_deg + 180) % 360 - 180)
    contributions["heading"] = _gaussian_log_lr(heading_delta, same_sd=3.0, different_sd=12.0)

    total = float(sum(contributions.values()))
    same = total >= threshold
    if same:
        reason = f"same vehicle: {total:.1f} log-odds, false match about 1 in {np.exp(total):.0f}"
    else:
        reason = (f"not established: {total:.1f} log-odds, under the {threshold:.1f} bar "
                  f"-- do not cite")
    return MatchVerdict(total, contributions, same, reason)


def collision_rate(signatures: list[VehicleSignature], population: Population,
                   threshold: float = CITE_LOG_ODDS) -> dict:
    """How often would two *different* vehicles be called the same?

    The number that decides whether this can be deployed, and it needs no ground
    truth: every pair drawn from a single pass is by construction a different
    pair, so any match among them is a false one. Bay position is excluded --
    two cars cannot occupy one bay, so including it would flatter the result by
    testing a case that never arises.
    """
    usable = [s for s in signatures if s.usable]
    if len(usable) < 2:
        return {"pairs": 0, "false_matches": 0, "rate": float("nan")}

    false_matches = 0
    pairs = 0
    scores = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            verdict = compare(usable[i], usable[j], population, threshold)
            appearance_only = verdict.log_odds - verdict.contributions.get("bay_offset", 0.0) \
                - verdict.contributions.get("heading", 0.0)
            scores.append(appearance_only)
            pairs += 1
            if appearance_only >= threshold:
                false_matches += 1
    return {
        "pairs": pairs,
        "false_matches": false_matches,
        "rate": false_matches / pairs if pairs else float("nan"),
        "p95_log_odds": float(np.percentile(scores, 95)) if scores else float("nan"),
        "max_log_odds": float(np.max(scores)) if scores else float("nan"),
    }
