"""Orchestration patterns behind one interface.

Three ways to spend tool calls on the same problem:

``linear``            One pass. Read the listing, decide each track from it.
                      Cheapest, and blind to anything the listing omits.
``planner_executor``  A triage step sorts tracks into buckets by what is unclear
                      about them, then an executor gathers only what that bucket
                      needs before deciding.
``graph``             Conditional routing with a re-check: each track is sent
                      down a branch by what is uncertain, and a track that is
                      uncertain in more than one way visits more than one node.

The patterns differ in *what they look at*, not in the policy they apply. That
is deliberate. Holding the decision rule fixed and varying only information
gathering is what isolates the contribution of orchestration; letting each
pattern use a different policy would produce a comparison table that says
nothing about orchestration at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from geoloc_agent.agent.policy import DEFAULT_THRESHOLDS, PolicyThresholds, decide_from_view
from geoloc_agent.agent.tools import ToolCall, TrackStore

CONFIDENCE_LOOK_CLOSER = 0.9


@dataclass
class PatternResult:
    pattern: str
    decisions: dict
    calls: list[ToolCall] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)

    @property
    def n_tool_calls(self) -> int:
        return len(self.calls)


class Orchestrator(ABC):
    name: str = "pattern"

    def __init__(self, thresholds: PolicyThresholds = DEFAULT_THRESHOLDS) -> None:
        self.thresholds = thresholds

    @abstractmethod
    def run(self, store: TrackStore) -> PatternResult:
        """Gather what this pattern gathers, then decide every track."""

    def _commit(self, store: TrackStore, view: dict) -> None:
        decision = decide_from_view(view, self.thresholds)
        track = store.tracks[decision.track_id]
        from geoloc_agent.agent.policy import expected_tool_for

        tool = expected_tool_for(decision, track)
        payload = {"track_id": decision.track_id}
        if tool == "publish_entity":
            payload.update({"rationale": decision.rationale, "priority": decision.priority})
        else:
            payload["reason"] = decision.rationale
        store.call(tool, payload)


class LinearPattern(Orchestrator):
    """One listing, one decision each. The baseline to beat."""

    name = "linear"

    def run(self, store: TrackStore) -> PatternResult:
        listing = store.call("query_tracks", {})
        for entry in listing["tracks"]:
            self._commit(store, entry)
        return PatternResult(self.name, dict(store.decisions), store.calls)


class PlannerExecutorPattern(Orchestrator):
    """Triage first, then gather only what each bucket needs."""

    name = "planner_executor"

    def run(self, store: TrackStore) -> PatternResult:
        listing = store.call("query_tracks", {})
        plan: list[str] = []
        buckets: dict[str, list[dict]] = {"clear": [], "class_unclear": [], "geometry_unclear": []}

        for entry in listing["tracks"]:
            wide = entry["sigma_horizontal_m"] > self.thresholds.surface_sigma_m
            unsure = entry["class_confidence"] < CONFIDENCE_LOOK_CLOSER
            if entry["degenerate_geometry"] or wide:
                buckets["geometry_unclear"].append(entry)
            elif unsure:
                buckets["class_unclear"].append(entry)
            else:
                buckets["clear"].append(entry)

        plan.append(
            f"triage: {len(buckets['clear'])} clear, {len(buckets['class_unclear'])} need "
            f"classification, {len(buckets['geometry_unclear'])} need geometry inspection"
        )

        for entry in buckets["clear"]:
            self._commit(store, entry)

        for entry in buckets["class_unclear"]:
            view = dict(entry)
            view.update(store.call("classify", {"track_id": entry["track_id"]}))
            view["class_entropy"] = view.get("entropy")
            self._commit(store, view)

        # Anything wide or degenerate could ALSO be class-ambiguous, and that
        # combination is the escalate case, so this bucket needs both lookups.
        for entry in buckets["geometry_unclear"]:
            view = dict(entry)
            view.update(store.call("get_track_detail", {"track_id": entry["track_id"]}))
            classification = store.call("classify", {"track_id": entry["track_id"]})
            view["class_entropy"] = classification.get("entropy")
            self._commit(store, view)

        return PatternResult(self.name, dict(store.decisions), store.calls, plan)


class GraphPattern(Orchestrator):
    """Conditional routing: each track visits the nodes its uncertainty demands."""

    name = "graph"

    def run(self, store: TrackStore) -> PatternResult:
        listing = store.call("query_tracks", {})
        plan: list[str] = []

        for entry in listing["tracks"]:
            view = dict(entry)
            route = ["triage"]

            # Node: classify. Entered whenever the top class is not decisive.
            if entry["class_confidence"] < CONFIDENCE_LOOK_CLOSER:
                classification = store.call("classify", {"track_id": entry["track_id"]})
                view["class_entropy"] = classification.get("entropy")
                route.append("classify")

            # Node: geometry. Entered whenever the fix is wide or flagged.
            if (
                entry["sigma_horizontal_m"] > self.thresholds.surface_sigma_m
                or entry["degenerate_geometry"]
            ):
                view.update(store.call("get_track_detail", {"track_id": entry["track_id"]}))
                route.append("geometry")
                # Re-check: a wide track that has not been classified yet could
                # still be the ambiguous-and-wide case, which must escalate. The
                # linear pattern cannot reach this branch at all.
                if "class_entropy" not in view:
                    classification = store.call("classify", {"track_id": entry["track_id"]})
                    view["class_entropy"] = classification.get("entropy")
                    route.append("classify:recheck")

            plan.append(f"track {entry['track_id']}: {' -> '.join(route)}")
            self._commit(store, view)

        return PatternResult(self.name, dict(store.decisions), store.calls, plan)


PATTERNS: dict[str, type[Orchestrator]] = {
    "linear": LinearPattern,
    "planner_executor": PlannerExecutorPattern,
    "graph": GraphPattern,
}
