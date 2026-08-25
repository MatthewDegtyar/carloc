"""Agent evaluation: task success, tool-call correctness, trajectory quality.

Scenarios are built from TrackStates directly rather than by running the whole
perception pipeline, so that the agent layer is scored on decision quality
rather than on whether that day's tracker happened to produce a good fix. The
adversarial slice in particular needs a track with a *specific* combination of
uncertainties, which is far easier to construct than to provoke.

Three metrics, measuring different failures:

**Task success** -- did the final action match the oracle policy? This is what an
operator experiences.

**Tool-call correctness** -- was the right tool used to express it? Escalating a
degenerate-geometry track through ``escalate_to_human`` rather than
``request_better_geometry`` reaches the same human but loses the actionable part:
one asks a person to look, the other asks the platform to move.

**Trajectory quality** -- did it gather what it needed, without gathering what it
did not? Scored in both directions, because an agent that calls every tool on
every track is not doing well, it is just expensive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geoloc_agent.agent.patterns import PATTERNS, Orchestrator, PatternResult
from geoloc_agent.agent.policy import (
    DEFAULT_THRESHOLDS,
    PolicyThresholds,
    decide,
    expected_tool_for,
)
from geoloc_agent.agent.tools import TrackStore
from geoloc_agent.contracts import Action, TrackState, TrackStatus


def _track(
    track_id: int,
    position: tuple[float, float, float],
    sigma: float,
    posterior: dict[str, float],
    n_obs: int = 20,
    degenerate: bool = False,
    reason: str = "",
    baseline: float = 15.0,
) -> TrackState:
    return TrackState(
        track_id=track_id,
        mean=np.array(position, dtype=float),
        cov=np.diag([sigma**2 / 2, sigma**2 / 2, (sigma / 2) ** 2]),
        class_posterior=dict(posterior),
        n_obs=n_obs,
        age=n_obs * 0.1,
        status=TrackStatus.CONFIRMED,
        degenerate=degenerate,
        degeneracy_reason=reason,
        max_perp_baseline=baseline,
    )


@dataclass
class AgentScenario:
    name: str
    tracks: list[TrackState]
    description: str = ""
    adversarial: bool = False


def build_scenarios() -> list[AgentScenario]:
    """The eval set. Each scenario isolates one decision the policy must get right."""
    return [
        AgentScenario(
            name="clear_picture",
            description="Well-localised, confidently classified tracks. Everything publishes.",
            tracks=[
                _track(1, (10, 30, 0), 1.2, {"car": 0.95, "pedestrian": 0.03, "clutter": 0.02}),
                _track(2, (-8, 25, 0), 0.8, {"pedestrian": 0.92, "car": 0.05, "clutter": 0.03}),
                _track(3, (20, 40, 0), 2.0, {"car": 0.88, "truck": 0.09, "clutter": 0.03}),
            ],
        ),
        AgentScenario(
            name="degenerate_geometry",
            description=(
                "Wide fixes caused by no perpendicular baseline. These must request a "
                "manoeuvre, not a human -- the geometry is fixable by moving."
            ),
            tracks=[
                _track(
                    1, (0, 80, 0), 25.0, {"car": 0.91, "truck": 0.06, "clutter": 0.03},
                    degenerate=True, reason="parallax 0.4 deg below 1.0 deg threshold",
                    baseline=0.3,
                ),
                _track(2, (12, 22, 0), 1.0, {"car": 0.94, "clutter": 0.06}),
            ],
        ),
        AgentScenario(
            name="noisy_but_well_conditioned",
            description=(
                "Wide fixes with adequate geometry. Manoeuvring will not help, so these "
                "suppress rather than requesting geometry."
            ),
            tracks=[
                _track(
                    1, (5, 45, 0), 18.0, {"car": 0.93, "clutter": 0.07},
                    degenerate=False, baseline=22.0,
                ),
                _track(2, (-3, 20, 0), 1.5, {"pedestrian": 0.9, "car": 0.1}),
            ],
        ),
        AgentScenario(
            name="adversarial_ambiguous_and_wide",
            adversarial=True,
            description=(
                "THE case the system must not guess on: class posterior nearly flat AND "
                "position known only to tens of metres. Must escalate to a human. An agent "
                "that publishes its top class here is confidently wrong, which is the worst "
                "possible output."
            ),
            tracks=[
                # The trap: top class 0.55 clears any naive confidence floor, so
                # the listing looks like a confident "car". The posterior is
                # actually near-even between car and pedestrian, and ONLY
                # `classify` reveals that. An agent that trusts the listing sees
                # a confident class over a wide fix and quietly suppresses it --
                # the operator never learns there was something there.
                _track(
                    1, (2, 60, 0), 22.0,
                    {"car": 0.55, "pedestrian": 0.42, "clutter": 0.03},
                    degenerate=False, baseline=18.0,
                ),
                _track(
                    2, (-6, 55, 0), 16.0,
                    {"pedestrian": 0.53, "car": 0.44, "clutter": 0.03},
                    degenerate=False, baseline=20.0,
                ),
                _track(3, (9, 18, 0), 1.1, {"car": 0.96, "clutter": 0.04}),
            ],
        ),
        AgentScenario(
            name="ambiguous_but_precise",
            description=(
                "Flat class posterior but a tight fix. Should still surface -- an "
                "unidentified object at a known location is actionable."
            ),
            tracks=[
                _track(
                    1, (4, 28, 0), 1.4,
                    {"car": 0.37, "pedestrian": 0.35, "clutter": 0.28},
                    baseline=19.0,
                ),
            ],
        ),
        AgentScenario(
            name="thin_evidence",
            description="Too few observations to act on at all.",
            tracks=[
                _track(1, (3, 35, 0), 4.0, {"car": 0.8, "clutter": 0.2}, n_obs=2),
                _track(2, (7, 24, 0), 1.0, {"car": 0.93, "clutter": 0.07}, n_obs=25),
            ],
        ),
        AgentScenario(
            name="mixed_load",
            description="Everything at once, to check ordering under load.",
            tracks=[
                _track(1, (10, 30, 0), 1.0, {"car": 0.95, "clutter": 0.05}),
                _track(2, (-5, 40, 0), 0.7, {"pedestrian": 0.93, "car": 0.07}),
                _track(
                    3, (0, 90, 0), 30.0, {"car": 0.9, "clutter": 0.1},
                    degenerate=True, reason="parallax 0.3 deg", baseline=0.4,
                ),
                _track(
                    4, (6, 65, 0), 20.0,
                    {"car": 0.54, "pedestrian": 0.43, "clutter": 0.03}, baseline=17.0,
                ),
                _track(5, (15, 22, 0), 2.5, {"car": 0.87, "truck": 0.13}),
                _track(6, (2, 12, 0), 3.0, {"pedestrian": 0.6, "car": 0.4}, n_obs=2),
            ],
        ),
    ]


@dataclass
class PatternScore:
    pattern: str
    scenario: str
    n_tracks: int
    task_success: float
    tool_correctness: float
    trajectory_quality: float
    n_tool_calls: int
    adversarial: bool = False
    adversarial_failures: list[str] = field(default_factory=list)
    dangerous_publishes: int = 0
    missed_escalations: int = 0


def _needed_lookups(track: TrackState, thresholds: PolicyThresholds) -> set[str]:
    """Which lookups a competent agent had to make for this track.

    Only counts what the decision genuinely depends on. A confidently classified,
    tightly localised track needs nothing beyond the listing, and an agent that
    inspects it anyway is spending calls for no decision benefit.
    """
    needed: set[str] = set()
    _, confidence = track.top_class
    if confidence < 0.9:
        needed.add("classify")
    if track.sigma_horizontal > thresholds.surface_sigma_m or track.degenerate:
        needed.add("get_track_detail")
    return needed


def score_pattern(
    result: PatternResult,
    scenario: AgentScenario,
    store: TrackStore,
    thresholds: PolicyThresholds = DEFAULT_THRESHOLDS,
) -> PatternScore:
    successes, tool_hits, trajectory = [], [], []
    adversarial_failures: list[str] = []
    dangerous = 0
    missed = 0

    calls_by_track: dict[int, set[str]] = {}
    for call in result.calls:
        track_id = call.arguments.get("track_id")
        if track_id is not None:
            calls_by_track.setdefault(int(track_id), set()).add(call.name)

    for track in scenario.tracks:
        oracle = decide(track, thresholds)
        actual = result.decisions.get(track.track_id)

        matched = actual is not None and actual.action is oracle.action
        successes.append(float(matched))

        expected_tool = expected_tool_for(oracle, track)
        used = actual.tool_calls[0] if actual and actual.tool_calls else None
        tool_hits.append(float(used == expected_tool))

        needed = _needed_lookups(track, thresholds)
        made = calls_by_track.get(track.track_id, set()) & {"classify", "get_track_detail"}
        if not needed and not made:
            trajectory.append(1.0)
        else:
            union = needed | made
            # Jaccard: penalises both missing a needed lookup and making
            # unnecessary ones.
            trajectory.append(len(needed & made) / len(union) if union else 1.0)

        # The failures that matter most, and they are different failures.
        # Publishing something that should have been escalated is a confident
        # answer to a question the system cannot answer. Suppressing it is
        # quieter and arguably worse: nobody ever finds out it was there.
        if oracle.action is Action.ESCALATE and actual and actual.action is not Action.ESCALATE:
            missed += 1
            verb = "published" if actual.action is Action.SURFACE else "silently suppressed"
            if actual.action is Action.SURFACE:
                dangerous += 1
            if scenario.adversarial:
                adversarial_failures.append(
                    f"track {track.track_id}: {verb} '{track.top_class[0]}' "
                    f"({track.top_class[1]:.2f}, entropy {track.class_entropy:.2f}) at "
                    f"{track.sigma_horizontal:.1f} m sigma instead of escalating"
                )

    return PatternScore(
        pattern=result.pattern,
        scenario=scenario.name,
        n_tracks=len(scenario.tracks),
        task_success=float(np.mean(successes)) if successes else float("nan"),
        tool_correctness=float(np.mean(tool_hits)) if tool_hits else float("nan"),
        trajectory_quality=float(np.mean(trajectory)) if trajectory else float("nan"),
        n_tool_calls=result.n_tool_calls,
        adversarial=scenario.adversarial,
        adversarial_failures=adversarial_failures,
        dangerous_publishes=dangerous,
        missed_escalations=missed,
    )


def run_pattern(
    pattern: str | Orchestrator, scenario: AgentScenario, thresholds=DEFAULT_THRESHOLDS
) -> tuple[PatternResult, PatternScore]:
    orchestrator = PATTERNS[pattern](thresholds) if isinstance(pattern, str) else pattern
    store = TrackStore.from_tracks(scenario.tracks)
    result = orchestrator.run(store)
    return result, score_pattern(result, scenario, store, thresholds)


def run_all(
    patterns: list[str] | None = None, scenarios: list[AgentScenario] | None = None
) -> list[PatternScore]:
    patterns = patterns or list(PATTERNS)
    scenarios = scenarios or build_scenarios()
    return [
        run_pattern(pattern, scenario)[1] for scenario in scenarios for pattern in patterns
    ]


def summarise(scores: list[PatternScore]) -> dict[str, dict]:
    """Pool per-pattern. Adversarial results are kept separate, never averaged in."""
    out: dict[str, dict] = {}
    for pattern in sorted({s.pattern for s in scores}):
        rows = [s for s in scores if s.pattern == pattern]
        adversarial = [s for s in rows if s.adversarial]
        out[pattern] = {
            "task_success": float(np.mean([s.task_success for s in rows])),
            "tool_correctness": float(np.mean([s.tool_correctness for s in rows])),
            "trajectory_quality": float(np.mean([s.trajectory_quality for s in rows])),
            "tool_calls_per_track": float(
                np.sum([s.n_tool_calls for s in rows]) / max(np.sum([s.n_tracks for s in rows]), 1)
            ),
            "dangerous_publishes": int(np.sum([s.dangerous_publishes for s in rows])),
            "missed_escalations": int(np.sum([s.missed_escalations for s in rows])),
            "adversarial_success": (
                float(np.mean([s.task_success for s in adversarial]))
                if adversarial
                else float("nan")
            ),
            "adversarial_failures": [f for s in adversarial for f in s.adversarial_failures],
        }
    return out
