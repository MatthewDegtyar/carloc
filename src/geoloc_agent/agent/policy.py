"""The surfacing policy.

This is the reference implementation of the decision the whole pipeline exists
to support: given N tracks with covariances and class posteriors, which ones
reach an operator, in what order, and which ones must not be decided
automatically at all.

The policy is a pure function of a TrackState so that it can serve three roles
at once: the default decision engine when no model is in the loop, the oracle
that the eval scores a model against, and an executable statement of the
intended behaviour. Writing the intent as prose in a prompt and separately as
code in a scorer is how the two quietly drift apart.

The central rule is the interaction between the two kinds of uncertainty:

* Wide covariance alone is not a reason to escalate. It is a reason to ask for
  better geometry if the geometry is the cause, or to suppress if it is not.
* Ambiguous class alone is not a reason to escalate either -- a well-localised
  unknown object is still a useful thing to put on a map.
* **Both together is the escalate case.** An object that might be a person and
  might be a car, whose position is only known to within tens of metres, is
  precisely where an automated guess is worse than admitting ignorance.
"""

from __future__ import annotations

from dataclasses import dataclass

from geoloc_agent.contracts import Action, Decision, TrackState


@dataclass(frozen=True)
class PolicyThresholds:
    surface_sigma_m: float = 5.0
    """Above this horizontal sigma a track is not precise enough to publish."""

    escalate_sigma_m: float = 10.0
    """Wide enough that an operator should look, if the class is also unclear."""

    ambiguous_entropy: float = 0.6
    """Class-posterior entropy in nats above which the class is genuinely unclear."""

    min_confidence: float = 0.5
    min_observations: int = 3
    high_value_classes: tuple[str, ...] = ("pedestrian", "person")


DEFAULT_THRESHOLDS = PolicyThresholds()


def full_view(track: TrackState) -> dict:
    """Everything the policy can use, as a plain dict."""
    cls, confidence = track.top_class
    return {
        "track_id": track.track_id,
        "class": cls,
        "class_confidence": confidence,
        "class_entropy": track.class_entropy,
        "sigma_horizontal_m": track.sigma_horizontal,
        "cep50_m": track.cep50,
        "n_observations": track.n_obs,
        "degenerate_geometry": track.degenerate,
        "degeneracy_reason": track.degeneracy_reason,
    }


def decide_from_view(
    view: dict, thresholds: PolicyThresholds = DEFAULT_THRESHOLDS
) -> Decision:
    """Decide from whatever the agent actually gathered.

    Fields may be absent, and that is the point. An orchestration pattern that
    never calls ``classify`` genuinely does not know a track's class entropy, and
    must fall back to top-class confidence as a weaker proxy for ambiguity. Making
    the policy accept a partial view is what lets the pattern comparison measure
    the cost of *not looking* instead of quietly handing every pattern the same
    omniscient input.
    """
    track_id = int(view["track_id"])
    cls = view.get("class", "unknown")
    confidence = float(view.get("class_confidence", 1.0))
    sigma = float(view.get("sigma_horizontal_m", float("inf")))
    n_obs = int(view.get("n_observations", 0))
    degenerate = bool(view.get("degenerate_geometry", False))

    entropy = view.get("class_entropy")
    if entropy is None:
        # Never called classify. Confidence alone is a blunt instrument: a
        # posterior split 0.45/0.40/0.15 and one split 0.45/0.05/0.05... both
        # show "0.45", but only the first is genuinely ambiguous.
        ambiguous = confidence < thresholds.min_confidence
        entropy_note = "class entropy not retrieved; using top-class confidence only"
    else:
        entropy = float(entropy)
        ambiguous = entropy > thresholds.ambiguous_entropy or confidence < thresholds.min_confidence
        entropy_note = f"entropy {entropy:.2f}"

    if n_obs < thresholds.min_observations:
        return Decision(
            track_id=track_id, action=Action.SUPPRESS,
            rationale=(
                f"only {n_obs} observations; not enough evidence to act on either its "
                f"position or its class"
            ),
            priority=0.0,
        )

    # The adversarial case: unsure what it is AND unsure where it is.
    if ambiguous and sigma > thresholds.escalate_sigma_m:
        return Decision(
            track_id=track_id, action=Action.ESCALATE,
            rationale=(
                f"ambiguous class (top '{cls}' at {confidence:.2f}, {entropy_note}) combined "
                f"with {sigma:.1f} m position uncertainty; guessing here would be worse than "
                f"handing it to an operator"
            ),
            priority=1.0,
        )

    # Wide, but the geometry explains why -- and geometry can be fixed by moving.
    if sigma > thresholds.surface_sigma_m and degenerate:
        reason = view.get("degeneracy_reason") or "insufficient parallax"
        return Decision(
            track_id=track_id, action=Action.ESCALATE,
            rationale=(
                f"{sigma:.1f} m uncertainty caused by degenerate geometry ({reason}); "
                f"requesting a manoeuvre perpendicular to the line of sight rather than "
                f"publishing a position this weak"
            ),
            priority=0.8,
        )

    if sigma > thresholds.surface_sigma_m:
        # Suppressing a wide track without having checked its class entropy is
        # the quiet failure mode: if the class had turned out to be ambiguous
        # this should have escalated instead, and nobody would ever find out.
        # The caveat goes in the rationale so it is visible in the audit trail.
        caveat = "" if entropy is not None else f"; NOTE: {entropy_note}"
        return Decision(
            track_id=track_id, action=Action.SUPPRESS,
            rationale=(
                f"{sigma:.1f} m uncertainty exceeds the {thresholds.surface_sigma_m:.0f} m "
                f"publish threshold and the geometry is adequate, so more manoeuvring will "
                f"not help; measurement noise is the limit{caveat}"
            ),
            priority=0.1,
        )

    if ambiguous:
        return Decision(
            track_id=track_id, action=Action.SURFACE,
            rationale=(
                f"well localised to {sigma:.1f} m but class is unclear ({entropy_note}); "
                f"surfacing as an unclassified object, which is still actionable because "
                f"the position is trustworthy"
            ),
            priority=_priority_from(cls, confidence, sigma, thresholds) * 0.6,
        )

    return Decision(
        track_id=track_id, action=Action.SURFACE,
        rationale=(
            f"{cls} at {confidence:.2f} confidence, localised to {sigma:.1f} m "
            f"(CEP50 {float(view.get('cep50_m', sigma)):.1f} m) over {n_obs} observations"
        ),
        priority=_priority_from(cls, confidence, sigma, thresholds),
    )


def decide(track: TrackState, thresholds: PolicyThresholds = DEFAULT_THRESHOLDS) -> Decision:
    """The policy with full information. This is the oracle the eval scores against."""
    return decide_from_view(full_view(track), thresholds)


def _priority_from(
    cls: str, confidence: float, sigma: float, thresholds: PolicyThresholds
) -> float:
    """Ordering, not filtering.

    Precision and class both matter, and they matter multiplicatively: a
    confident person at 2 m outranks a confident car at 2 m, and both outrank
    anything known only to within 10 m.
    """
    precision = 1.0 / (1.0 + sigma)
    weight = 1.0 if cls in thresholds.high_value_classes else 0.7
    return float(min(1.0, precision * confidence * weight * 2.0))


def decide_all(
    tracks: list[TrackState], thresholds: PolicyThresholds = DEFAULT_THRESHOLDS
) -> list[Decision]:
    """Decide every track and return them in the order an operator should see them."""
    decisions = [decide(t, thresholds) for t in tracks]
    rank = {Action.ESCALATE: 0, Action.SURFACE: 1, Action.SUPPRESS: 2}
    return sorted(decisions, key=lambda d: (rank[d.action], -d.priority))


def expected_tool_for(decision: Decision, track: TrackState) -> str:
    """Which tool a correct agent should have called for this decision.

    Used to score tool-call correctness. An escalation caused by geometry should
    come through ``request_better_geometry`` rather than a bare escalation,
    because the two lead to different operator actions.
    """
    if decision.action is Action.SURFACE:
        return "publish_entity"
    if decision.action is Action.SUPPRESS:
        return "suppress"
    if track.degenerate and "geometry" in decision.rationale:
        return "request_better_geometry"
    return "escalate_to_human"
