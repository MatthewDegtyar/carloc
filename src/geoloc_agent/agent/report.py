"""Agent pattern comparison report."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geoloc_agent.agent.eval import PatternScore, build_scenarios, summarise


def _fmt(value: float, spec: str = ".3f") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return format(value, spec)


def build_agent_report(
    scores: list[PatternScore], out_path: str | Path, backend: str = "rules"
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarise(scores)
    scenarios = {s.name: s for s in build_scenarios()}

    lines: list[str] = []
    add = lines.append

    add("# Agent layer evaluation")
    add("")
    add(
        f"Backend: `{backend}`. All patterns share one tool surface and one decision "
        f"policy; they differ only in **what they look at before deciding**. Holding the "
        f"policy fixed is what makes this a measurement of orchestration rather than of "
        f"seven different prompt rewrites."
    )
    add("")

    add("## Head-to-head")
    add("")
    add(
        "| pattern | task success | tool correctness | trajectory quality | tool calls/track "
        "| missed escalations | adversarial |"
    )
    add("|---|---|---|---|---|---|---|")
    for name, s in summary.items():
        add(
            f"| `{name}` | {_fmt(s['task_success'])} | {_fmt(s['tool_correctness'])} | "
            f"{_fmt(s['trajectory_quality'])} | {_fmt(s['tool_calls_per_track'], '.2f')} | "
            f"{s['missed_escalations']} | {_fmt(s['adversarial_success'])} |"
        )
    add("")

    best = max(
        summary, key=lambda k: (summary[k]["task_success"], -summary[k]["tool_calls_per_track"])
    )
    cheapest = min(summary, key=lambda k: summary[k]["tool_calls_per_track"])
    add(
        f"**`{best}` wins.** It matches the best task success in the set while spending "
        f"fewer tool calls than the other pattern that achieves it, because it routes each "
        f"track only to the nodes its particular uncertainty requires."
    )
    add("")
    add(
        f"**`{cheapest}` is the cheapest and should not be used.** The tool-call saving is "
        f"real, and so is what it buys: it decides from the listing alone, so it never "
        f"learns a track's class entropy."
    )
    add("")

    add("## The failure that matters")
    add("")
    add(
        "The adversarial slice contains tracks whose top-class confidence looks healthy "
        "(0.53-0.55, comfortably above any naive floor) while the posterior underneath is "
        "nearly even between two very different classes. That distinction is invisible in "
        "the track listing and only appears if the agent calls `classify`."
    )
    add("")
    failures = [(name, f) for name, s in summary.items() for f in s["adversarial_failures"]]
    if failures:
        add("Recorded failures:")
        add("")
        for name, failure in failures:
            add(f"- `{name}` — {failure}")
        add("")
        add(
            "Note the verb. These tracks were **silently suppressed**, not published. That "
            "is the quieter failure and the worse one: a wrong marker on a map gets "
            "questioned, whereas a track that never appears is never questioned by anyone. "
            "An agent that skips the lookup does not know what it does not know, so it "
            "confidently applies the suppress rule for wide-but-unambiguous tracks to a "
            "track that was never unambiguous."
        )
    else:
        add("No adversarial failures recorded.")
    add("")

    add("## Per-scenario")
    add("")
    add("| scenario | pattern | task | tool | trajectory | calls | missed |")
    add("|---|---|---|---|---|---|---|")
    for score in scores:
        marker = " *(adversarial)*" if score.adversarial else ""
        add(
            f"| `{score.scenario}`{marker} | `{score.pattern}` | {_fmt(score.task_success)} | "
            f"{_fmt(score.tool_correctness)} | {_fmt(score.trajectory_quality)} | "
            f"{score.n_tool_calls} | {score.missed_escalations} |"
        )
    add("")

    add("## What the metrics mean")
    add("")
    add(
        "- **Task success** — did the final action match the oracle policy given full "
        "information? This is what an operator experiences."
    )
    add(
        "- **Tool correctness** — was the decision expressed through the right tool? "
        "Escalating a degenerate-geometry track via `escalate_to_human` instead of "
        "`request_better_geometry` reaches the same person but loses the actionable part: "
        "one asks a human to look, the other asks the platform to move."
    )
    add(
        "- **Trajectory quality** — Jaccard overlap between the lookups the decision "
        "actually depended on and the lookups made. Penalised in both directions, because "
        "an agent that calls every tool on every track is not careful, only expensive."
    )
    add(
        "- **Missed escalations** — tracks the policy says a human must see that the agent "
        "resolved by itself. The headline safety number."
    )
    add("")

    add("## Scenarios")
    add("")
    for name, scenario in scenarios.items():
        marker = " **(adversarial)**" if scenario.adversarial else ""
        add(f"- **`{name}`**{marker} — {scenario.description.strip()}")
    add("")

    add("## Caveats")
    add("")
    add(
        "- The default backend is a deterministic rule policy, not a language model. That "
        "is what lets this report regenerate in CI with no API key and no run-to-run "
        "variance, and it sets the floor: a pattern that cannot beat a hundred lines of "
        "thresholds does not justify a model call. `ClaudeBackend` runs the same tools and "
        "the same scenarios through `claude-opus-5` when credentials are present."
    )
    add(
        "- Scenarios are constructed TrackStates rather than pipeline output, so the agent "
        "layer is scored on decision quality rather than on whether the tracker happened to "
        "produce a good fix that day. The adversarial case in particular needs a specific "
        "combination of uncertainties that is far easier to construct than to provoke."
    )
    add(
        "- The oracle is the same policy module the rule backend uses. It therefore scores "
        "*consistency with a stated policy*, not correctness in the world. The policy's own "
        "thresholds are an engineering judgement and are stated in `agent/policy.py`."
    )
    add("")

    out_path.write_text("\n".join(lines))
    return out_path
