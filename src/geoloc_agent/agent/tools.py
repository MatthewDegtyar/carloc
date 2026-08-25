"""Agent tools.

Schemas are written in Anthropic tool-use format so the same definitions can be
handed to Claude or driven by the deterministic reference policy. There is one
set of tools, not one per backend -- otherwise the comparison between
orchestration patterns measures differences in tool surface rather than
differences in orchestration.

The important design decision is what ``query_tracks`` returns. It returns
covariance, not a point. Collapsing a track to a coordinate before the agent
sees it would throw away the only information that makes the surfacing decision
non-trivial: a 3 m fix on a possible vehicle and a 40 m fix on a possible vehicle
are completely different situations, and they are indistinguishable once you drop
the uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from geoloc_agent.contracts import Action, Decision, TrackState
from geoloc_agent.geo import GeoOrigin

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "query_tracks",
        "description": (
            "List current tracks with their positions AND uncertainties. Always returns "
            "covariance information (sigma_horizontal_m, cep50_m) alongside position -- a "
            "position without its uncertainty is not actionable. Use this first to see "
            "what is out there."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_confidence": {
                    "type": "number",
                    "description": (
                        "Only return tracks whose top class probability is at least this."
                    ),
                },
                "max_sigma_m": {
                    "type": "number",
                    "description": "Only return tracks at least this well localised, in metres.",
                },
                "cls": {"type": "string", "description": "Filter to a single class."},
                "include_degenerate": {
                    "type": "boolean",
                    "description": "Include tracks flagged as having degenerate geometry.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_track_detail",
        "description": (
            "Full detail for one track: full class posterior, covariance matrix, "
            "observation count, age, geometry diagnosis and the reason for any degeneracy "
            "flag. Use before deciding on a track that looks ambiguous."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"track_id": {"type": "integer"}},
            "required": ["track_id"],
        },
    },
    {
        "name": "classify",
        "description": (
            "Return the class posterior and its entropy for a track. High entropy means "
            "the classification is genuinely ambiguous, not merely low-scoring."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"track_id": {"type": "integer"}},
            "required": ["track_id"],
        },
    },
    {
        "name": "publish_entity",
        "description": (
            "Publish a track to the common operating picture as an entity. Only call this "
            "for tracks that are well enough localised to be actionable. Publishing a "
            "wide-covariance track puts a confident-looking marker on an operator's map "
            "for something whose position is not actually known."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "track_id": {"type": "integer"},
                "rationale": {"type": "string", "description": "Why this one is worth surfacing."},
                "priority": {
                    "type": "number",
                    "description": "0-1. Higher means show it sooner.",
                },
            },
            "required": ["track_id", "rationale"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Hand a track to a human operator instead of deciding. Use when the "
            "classification is ambiguous AND the position uncertainty is large -- that "
            "combination is exactly the case where an automated guess is worse than no "
            "answer. Escalating is a valid, expected outcome, not a failure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "track_id": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["track_id", "reason"],
        },
    },
    {
        "name": "request_better_geometry",
        "description": (
            "Request a manoeuvre that would improve the geometry on a track: the platform "
            "moves perpendicular to the line of sight to build parallax. Use for tracks "
            "whose uncertainty is caused by degenerate geometry rather than by noise, "
            "because those cannot be fixed by waiting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "track_id": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["track_id", "reason"],
        },
    },
    {
        "name": "suppress",
        "description": (
            "Explicitly decide a track is not worth surfacing. Recording this is what "
            "makes the trajectory scoreable -- silence is indistinguishable from an "
            "oversight."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "track_id": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["track_id", "reason"],
        },
    },
]

TOOL_NAMES = tuple(schema["name"] for schema in TOOL_SCHEMAS)


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: Any = None
    error: str | None = None


@dataclass
class TrackStore:
    """The tools' view of the world, plus a recording of what they did."""

    tracks: dict[int, TrackState] = field(default_factory=dict)
    origin: GeoOrigin | None = None
    calls: list[ToolCall] = field(default_factory=list)
    decisions: dict[int, Decision] = field(default_factory=dict)

    @classmethod
    def from_tracks(
        cls, tracks: list[TrackState], origin: GeoOrigin | None = None
    ) -> TrackStore:
        return cls(tracks={t.track_id: t for t in tracks}, origin=origin)

    # -- serialisation ----------------------------------------------------

    def summarise(self, track: TrackState, full: bool = False) -> dict:
        """The listing view deliberately omits class entropy.

        Entropy is what ``classify`` is for. If every listing carried it, an agent
        that never inspects anything would look identical to one that does, and
        the pattern comparison would measure nothing.
        """
        cls_name, confidence = track.top_class
        payload = {
            "track_id": track.track_id,
            "class": cls_name,
            "class_confidence": round(confidence, 3),
            "position_enu_m": [round(float(v), 2) for v in track.mean],
            # Uncertainty travels with the position, always.
            "sigma_horizontal_m": round(track.sigma_horizontal, 2),
            "cep50_m": round(track.cep50, 2),
            "n_observations": track.n_obs,
            "age_s": round(track.age, 2),
            "degenerate_geometry": track.degenerate,
            "status": track.status.value,
        }
        if full:
            payload["class_entropy"] = round(track.class_entropy, 3)
        if self.origin is not None:
            lat, lon, alt = self.origin.enu_to_wgs84(track.mean)
            payload["lat"] = round(lat, 7)
            payload["lon"] = round(lon, 7)
            payload["geo_note"] = f"relative to assumed origin '{self.origin.name}'"
        return payload

    # -- tool implementations ---------------------------------------------

    def call(self, name: str, arguments: dict | None = None) -> Any:
        arguments = dict(arguments or {})
        record = ToolCall(name=name, arguments=arguments)
        self.calls.append(record)
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            record.error = f"unknown tool: {name}"
            return {"error": record.error}
        try:
            record.result = handler(**arguments)
        except TypeError as exc:
            record.error = f"bad arguments for {name}: {exc}"
            record.result = {"error": record.error}
        except KeyError as exc:
            record.error = f"unknown track {exc}"
            record.result = {"error": record.error}
        return record.result

    def _tool_query_tracks(
        self,
        min_confidence: float = 0.0,
        max_sigma_m: float | None = None,
        cls: str | None = None,
        include_degenerate: bool = True,
    ) -> dict:
        out = []
        for track in self.tracks.values():
            name, confidence = track.top_class
            if confidence < min_confidence:
                continue
            if max_sigma_m is not None and track.sigma_horizontal > max_sigma_m:
                continue
            if cls is not None and name != cls:
                continue
            if not include_degenerate and track.degenerate:
                continue
            out.append(self.summarise(track))
        out.sort(key=lambda t: t["sigma_horizontal_m"])
        return {"n_tracks": len(out), "tracks": out}

    def _tool_get_track_detail(self, track_id: int) -> dict:
        track = self.tracks[track_id]
        detail = self.summarise(track, full=True)
        detail.update(
            {
                "class_posterior": {k: round(v, 4) for k, v in track.class_posterior.items()},
                "covariance": [[round(float(v), 4) for v in row] for row in track.cov],
                "sigma_xyz_m": [round(float(v), 3) for v in track.sigma_xyz],
                "max_perpendicular_baseline_m": round(track.max_perp_baseline, 2),
                "degeneracy_reason": track.degeneracy_reason or None,
            }
        )
        return detail

    def _tool_classify(self, track_id: int) -> dict:
        track = self.tracks[track_id]
        name, confidence = track.top_class
        return {
            "track_id": track_id,
            "class": name,
            "confidence": round(confidence, 4),
            "entropy": round(track.class_entropy, 4),
            "posterior": {k: round(v, 4) for k, v in track.class_posterior.items()},
            "ambiguous": track.class_entropy > 0.6,
        }

    def _tool_publish_entity(self, track_id: int, rationale: str, priority: float = 0.5) -> dict:
        track = self.tracks[track_id]
        self.decisions[track_id] = Decision(
            track_id=track_id, action=Action.SURFACE, rationale=rationale,
            priority=float(priority), tool_calls=("publish_entity",),
        )
        return {"published": True, "track_id": track_id, "cep50_m": round(track.cep50, 2)}

    def _tool_escalate_to_human(self, track_id: int, reason: str) -> dict:
        self.tracks[track_id]  # existence check
        self.decisions[track_id] = Decision(
            track_id=track_id, action=Action.ESCALATE, rationale=reason,
            priority=1.0, tool_calls=("escalate_to_human",),
        )
        return {"escalated": True, "track_id": track_id}

    def _tool_request_better_geometry(self, track_id: int, reason: str) -> dict:
        track = self.tracks[track_id]
        self.decisions[track_id] = Decision(
            track_id=track_id, action=Action.ESCALATE, rationale=f"manoeuvre requested: {reason}",
            priority=0.8, tool_calls=("request_better_geometry",),
        )
        return {
            "requested": True,
            "track_id": track_id,
            "current_baseline_m": round(track.max_perp_baseline, 2),
            "suggested": "translate perpendicular to the line of sight to build parallax",
        }

    def _tool_suppress(self, track_id: int, reason: str) -> dict:
        self.tracks[track_id]
        self.decisions[track_id] = Decision(
            track_id=track_id, action=Action.SUPPRESS, rationale=reason,
            priority=0.0, tool_calls=("suppress",),
        )
        return {"suppressed": True, "track_id": track_id}
