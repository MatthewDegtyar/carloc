"""Model backends.

Two implementations behind one interface:

``RuleBackend``   The deterministic reference policy, driving the same tools.
                  This is the default, and it is why the eval harness runs in CI
                  with no API key, no network, and no run-to-run variance.
``ClaudeBackend`` Claude via the Anthropic SDK, in a standard tool-use loop.

Keeping both behind one interface is what makes the pattern comparison mean
something: the rule backend is the floor. A pattern that cannot beat a hundred
lines of thresholds does not justify a model call, and without the floor in the
table there is no way to notice that.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from geoloc_agent.agent.policy import DEFAULT_THRESHOLDS, PolicyThresholds, decide
from geoloc_agent.agent.tools import TOOL_SCHEMAS, TrackStore

MODEL_ID = "claude-opus-5"

SYSTEM_PROMPT = """\
You are a track-surfacing agent for a geolocation system. You receive object \
tracks derived from posed video. Every track carries a position AND an \
uncertainty, and the uncertainty is the point: your job is to decide which \
tracks are worth an operator's attention given how well they are actually known.

Rules:
- A track localised to within a few metres and confidently classified should be \
published, ordered by how precise and how important it is.
- A track whose position uncertainty is large because of DEGENERATE GEOMETRY \
(no perpendicular baseline) will not improve by waiting. Request better geometry.
- A track whose position uncertainty is large from measurement noise, with \
adequate geometry, should be suppressed. More manoeuvring will not help.
- A track that is BOTH ambiguously classified AND poorly localised must be \
escalated to a human. Do not guess. Escalation is a correct answer, not a failure.
- Every decision needs a rationale that names the numbers behind it.

Call tools to inspect tracks, then record exactly one decision per track: \
publish_entity, suppress, escalate_to_human, or request_better_geometry.
"""


@dataclass
class BackendResult:
    decisions: dict
    n_model_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    notes: list[str] = field(default_factory=list)


class Backend(ABC):
    name: str = "backend"

    @abstractmethod
    def run(self, store: TrackStore, instruction: str) -> BackendResult:
        """Inspect the store through its tools and record a decision per track."""

    @property
    def available(self) -> bool:
        return True


class RuleBackend(Backend):
    """Deterministic reference policy. No network, no variance, no API key."""

    name = "rules"

    def __init__(self, thresholds: PolicyThresholds = DEFAULT_THRESHOLDS) -> None:
        self.thresholds = thresholds

    def run(self, store: TrackStore, instruction: str = "") -> BackendResult:
        listing = store.call("query_tracks", {})
        for entry in listing["tracks"]:
            track_id = entry["track_id"]
            track = store.tracks[track_id]
            # Look closer before deciding anything that might be ambiguous,
            # exactly as the prompt asks a model to.
            if entry["degenerate_geometry"] or entry["class_confidence"] < 0.9:
                store.call("classify", {"track_id": track_id})
                store.call("get_track_detail", {"track_id": track_id})
            decision = decide(track, self.thresholds)
            _record(store, decision, track)
        return BackendResult(decisions=dict(store.decisions))


class ClaudeBackend(Backend):
    """Claude in a standard tool-use loop.

    Skipped rather than failed when no credentials are present, so the harness
    still produces a complete report offline -- with the model rows marked
    unavailable instead of silently missing.
    """

    name = "claude"

    def __init__(self, model: str = MODEL_ID, max_iterations: int = 12) -> None:
        self.model = model
        self.max_iterations = max_iterations

    @property
    def available(self) -> bool:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def run(self, store: TrackStore, instruction: str = "") -> BackendResult:
        import anthropic

        client = anthropic.Anthropic()
        messages = [{"role": "user", "content": instruction or _default_instruction(store)}]
        calls = tokens_in = tokens_out = 0

        for _ in range(self.max_iterations):
            response = client.messages.create(
                model=self.model,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            calls += 1
            tokens_in += response.usage.input_tokens
            tokens_out += response.usage.output_tokens
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                output = store.call(block.name, block.input)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(output, default=str),
                    }
                )
            # All tool results go back in ONE user message; splitting them
            # teaches the model to stop calling tools in parallel.
            messages.append({"role": "user", "content": results})

        return BackendResult(
            decisions=dict(store.decisions),
            n_model_calls=calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


def _default_instruction(store: TrackStore) -> str:
    return (
        f"There are {len(store.tracks)} active tracks. Review them and record exactly one "
        f"decision for each. Start with query_tracks."
    )


def _record(store: TrackStore, decision, track) -> None:
    """Route a policy decision through the same tools a model would call."""
    from geoloc_agent.agent.policy import expected_tool_for

    tool = expected_tool_for(decision, track)
    if tool == "publish_entity":
        store.call(
            "publish_entity",
            {
                "track_id": decision.track_id,
                "rationale": decision.rationale,
                "priority": decision.priority,
            },
        )
    elif tool == "suppress":
        store.call("suppress", {"track_id": decision.track_id, "reason": decision.rationale})
    elif tool == "request_better_geometry":
        store.call(
            "request_better_geometry",
            {"track_id": decision.track_id, "reason": decision.rationale},
        )
    else:
        store.call(
            "escalate_to_human", {"track_id": decision.track_id, "reason": decision.rationale}
        )
