"""Phase 6: tools, policy, patterns, and the adversarial slice."""

import numpy as np
import pytest

from geoloc_agent.agent.backend import ClaudeBackend, RuleBackend
from geoloc_agent.agent.eval import build_scenarios, run_all, run_pattern, summarise
from geoloc_agent.agent.patterns import PATTERNS
from geoloc_agent.agent.policy import decide, decide_from_view
from geoloc_agent.agent.report import build_agent_report
from geoloc_agent.agent.tools import TOOL_NAMES, TOOL_SCHEMAS, TrackStore
from geoloc_agent.contracts import Action, TrackState, TrackStatus
from geoloc_agent.geo import GeoOrigin


def track(track_id=1, sigma=1.0, posterior=None, n_obs=20, degenerate=False, reason=""):
    return TrackState(
        track_id=track_id,
        mean=np.array([5.0, 30.0, 0.0]),
        cov=np.diag([sigma**2 / 2, sigma**2 / 2, 0.25]),
        class_posterior=posterior or {"car": 0.95, "clutter": 0.05},
        n_obs=n_obs,
        status=TrackStatus.CONFIRMED,
        degenerate=degenerate,
        degeneracy_reason=reason,
    )


# --- tools -------------------------------------------------------------------


def test_tool_schemas_are_well_formed():
    """They are handed to a model verbatim, so the shape has to be right."""
    for schema in TOOL_SCHEMAS:
        assert {"name", "description", "input_schema"} <= set(schema)
        assert schema["input_schema"]["type"] == "object"
        assert "properties" in schema["input_schema"]
        for required in schema["input_schema"].get("required", []):
            assert required in schema["input_schema"]["properties"]
        assert len(schema["description"]) > 40, f"{schema['name']} needs a real description"


def test_query_tracks_always_returns_uncertainty():
    """Design rule: covariance reaches the agent, never a bare point."""
    store = TrackStore.from_tracks([track()])
    result = store.call("query_tracks", {})
    for entry in result["tracks"]:
        assert "sigma_horizontal_m" in entry
        assert "cep50_m" in entry
        assert entry["sigma_horizontal_m"] > 0


def test_query_tracks_withholds_entropy_so_classify_has_a_job():
    store = TrackStore.from_tracks([track()])
    listing = store.call("query_tracks", {})
    assert "class_entropy" not in listing["tracks"][0]
    assert "entropy" in store.call("classify", {"track_id": 1})
    assert "class_entropy" in store.call("get_track_detail", {"track_id": 1})


def test_tools_emit_latlon_only_when_an_origin_is_supplied():
    without = TrackStore.from_tracks([track()])
    assert "lat" not in without.call("query_tracks", {})["tracks"][0]
    with_origin = TrackStore.from_tracks(
        [track()], origin=GeoOrigin(42.336849, -71.05785, name="boston-seaport")
    )
    entry = with_origin.call("query_tracks", {})["tracks"][0]
    assert "lat" in entry and "lon" in entry
    assert "assumed origin" in entry["geo_note"]


def test_unknown_tool_and_bad_arguments_are_reported_not_raised():
    store = TrackStore.from_tracks([track()])
    assert "error" in store.call("nope", {})
    assert "error" in store.call("get_track_detail", {"wrong": 1})
    assert "error" in store.call("get_track_detail", {"track_id": 999})


def test_every_tool_is_reachable():
    store = TrackStore.from_tracks([track()])
    for name in TOOL_NAMES:
        assert hasattr(store, f"_tool_{name}"), f"{name} has a schema but no implementation"


def test_query_tracks_filters():
    tracks = [
        track(1, sigma=1.0),
        track(2, sigma=30.0, degenerate=True),
        track(3, sigma=2.0, posterior={"pedestrian": 0.9, "car": 0.1}),
    ]
    store = TrackStore.from_tracks(tracks)
    assert store.call("query_tracks", {"max_sigma_m": 5.0})["n_tracks"] == 2
    assert store.call("query_tracks", {"include_degenerate": False})["n_tracks"] == 2
    assert store.call("query_tracks", {"cls": "pedestrian"})["n_tracks"] == 1


# --- policy ------------------------------------------------------------------


def test_precise_and_confident_surfaces():
    assert decide(track(sigma=1.0)).action is Action.SURFACE


def test_ambiguous_and_wide_escalates():
    """The adversarial rule, stated directly."""
    decision = decide(
        track(sigma=25.0, posterior={"car": 0.5, "pedestrian": 0.45, "clutter": 0.05})
    )
    assert decision.action is Action.ESCALATE
    assert "ambiguous" in decision.rationale


def test_wide_but_degenerate_asks_for_geometry_not_a_human():
    decision = decide(track(sigma=25.0, degenerate=True, reason="parallax 0.3 deg"))
    assert decision.action is Action.ESCALATE
    assert "geometry" in decision.rationale
    from geoloc_agent.agent.policy import expected_tool_for

    assert expected_tool_for(decision, track(sigma=25.0, degenerate=True)) == (
        "request_better_geometry"
    )


def test_wide_with_good_geometry_suppresses_because_manoeuvring_will_not_help():
    decision = decide(track(sigma=25.0, degenerate=False))
    assert decision.action is Action.SUPPRESS
    assert "noise" in decision.rationale


def test_ambiguous_but_precise_still_surfaces():
    decision = decide(track(sigma=1.0, posterior={"car": 0.5, "pedestrian": 0.45, "clutter": 0.05}))
    assert decision.action is Action.SURFACE


def test_thin_evidence_suppresses():
    assert decide(track(n_obs=2)).action is Action.SUPPRESS


def test_every_decision_carries_numbers_in_its_rationale():
    for state in (track(sigma=1.0), track(sigma=25.0), track(sigma=25.0, degenerate=True)):
        rationale = decide(state).rationale
        assert any(ch.isdigit() for ch in rationale), rationale


def test_partial_view_falls_back_to_confidence_and_says_so():
    view = {
        "track_id": 1, "class": "car", "class_confidence": 0.55,
        "sigma_horizontal_m": 22.0, "n_observations": 20, "degenerate_geometry": False,
    }
    decision = decide_from_view(view)
    assert "not retrieved" in decision.rationale
    # With entropy it becomes the escalate case; without it, it does not.
    assert decision.action is Action.SUPPRESS
    view["class_entropy"] = 0.8
    assert decide_from_view(view).action is Action.ESCALATE


def test_priority_orders_high_value_and_precise_first():
    from geoloc_agent.agent.policy import decide_all

    tracks = [
        track(1, sigma=4.0, posterior={"car": 0.9, "clutter": 0.1}),
        track(2, sigma=0.5, posterior={"pedestrian": 0.95, "car": 0.05}),
        track(3, sigma=25.0, posterior={"car": 0.5, "pedestrian": 0.45, "clutter": 0.05}),
    ]
    ordered = decide_all(tracks)
    assert ordered[0].action is Action.ESCALATE  # escalations first
    surfaced = [d for d in ordered if d.action is Action.SURFACE]
    assert surfaced[0].track_id == 2  # precise pedestrian outranks vaguer car


# --- patterns ----------------------------------------------------------------


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
def test_every_pattern_decides_every_track(pattern):
    for scenario in build_scenarios():
        result, _ = run_pattern(pattern, scenario)
        assert set(result.decisions) == {t.track_id for t in scenario.tracks}
        assert all(d.rationale.strip() for d in result.decisions.values())


def test_patterns_that_inspect_beat_the_one_that_does_not():
    """The point of the comparison, asserted."""
    summary = summarise(run_all())
    assert summary["graph"]["task_success"] == 1.0
    assert summary["planner_executor"]["task_success"] == 1.0
    assert summary["linear"]["task_success"] < 1.0
    # Linear is genuinely cheaper -- the tradeoff is real, not a strawman.
    assert summary["linear"]["tool_calls_per_track"] < summary["graph"]["tool_calls_per_track"]


def test_linear_misses_the_adversarial_case_and_the_others_do_not():
    summary = summarise(run_all())
    assert summary["linear"]["adversarial_success"] < 1.0
    assert summary["linear"]["missed_escalations"] > 0
    assert summary["linear"]["adversarial_failures"]
    for pattern in ("graph", "planner_executor"):
        assert summary[pattern]["adversarial_success"] == 1.0
        assert summary[pattern]["missed_escalations"] == 0


def test_graph_is_at_least_as_efficient_as_planner_executor():
    summary = summarise(run_all())
    assert (
        summary["graph"]["tool_calls_per_track"]
        <= summary["planner_executor"]["tool_calls_per_track"]
    )
    assert summary["graph"]["trajectory_quality"] >= summary["planner_executor"][
        "trajectory_quality"
    ]


def test_no_pattern_publishes_a_track_that_should_escalate():
    """The dangerous failure. Silence is bad; a confident wrong marker is worse."""
    for score in run_all():
        assert score.dangerous_publishes == 0, f"{score.pattern}/{score.scenario}"


# --- backends ----------------------------------------------------------------


def test_rule_backend_decides_everything_offline():
    scenario = build_scenarios()[-1]
    store = TrackStore.from_tracks(scenario.tracks)
    result = RuleBackend().run(store, "")
    assert set(result.decisions) == {t.track_id for t in scenario.tracks}
    assert result.n_model_calls == 0


def test_claude_backend_reports_unavailable_rather_than_exploding(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert ClaudeBackend().available is False


def test_claude_backend_targets_the_current_model():
    from geoloc_agent.agent.backend import MODEL_ID

    assert MODEL_ID == "claude-opus-5"


# --- report ------------------------------------------------------------------


def test_agent_report_contains_the_comparison_table(tmp_path):
    """Phase 6 acceptance."""
    out = build_agent_report(run_all(), tmp_path / "agent_eval.md")
    text = out.read_text()
    assert "# Agent layer evaluation" in text
    assert "## Head-to-head" in text
    for pattern in PATTERNS:
        assert f"`{pattern}`" in text
    assert "task success" in text
    assert "adversarial" in text.lower()
    assert "silently suppressed" in text
