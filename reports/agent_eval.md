# Agent layer evaluation

Backend: `rules`. All patterns share one tool surface and one decision policy; they differ only in **what they look at before deciding**. Holding the policy fixed is what makes this a measurement of orchestration rather than of seven different prompt rewrites.

## Head-to-head

| pattern | task success | tool correctness | trajectory quality | tool calls/track | missed escalations | adversarial |
|---|---|---|---|---|---|---|
| `graph` | 1.000 | 1.000 | 1.000 | 2.11 | 0 | 1.000 |
| `linear` | 0.881 | 0.881 | 0.405 | 1.37 | 3 | 0.333 |
| `planner_executor` | 1.000 | 1.000 | 0.917 | 2.26 | 0 | 1.000 |

**`graph` wins.** It matches the best task success in the set while spending fewer tool calls than the other pattern that achieves it, because it routes each track only to the nodes its particular uncertainty requires.

**`linear` is the cheapest and should not be used.** The tool-call saving is real, and so is what it buys: it decides from the listing alone, so it never learns a track's class entropy.

## The failure that matters

The adversarial slice contains tracks whose top-class confidence looks healthy (0.53-0.55, comfortably above any naive floor) while the posterior underneath is nearly even between two very different classes. That distinction is invisible in the track listing and only appears if the agent calls `classify`.

Recorded failures:

- `linear` — track 1: silently suppressed 'car' (0.55, entropy 0.80) at 22.0 m sigma instead of escalating
- `linear` — track 2: silently suppressed 'pedestrian' (0.53, entropy 0.80) at 16.0 m sigma instead of escalating

Note the verb. These tracks were **silently suppressed**, not published. That is the quieter failure and the worse one: a wrong marker on a map gets questioned, whereas a track that never appears is never questioned by anyone. An agent that skips the lookup does not know what it does not know, so it confidently applies the suppress rule for wide-but-unambiguous tracks to a track that was never unambiguous.

## Per-scenario

| scenario | pattern | task | tool | trajectory | calls | missed |
|---|---|---|---|---|---|---|
| `clear_picture` | `linear` | 1.000 | 1.000 | 0.667 | 4 | 0 |
| `clear_picture` | `planner_executor` | 1.000 | 1.000 | 1.000 | 5 | 0 |
| `clear_picture` | `graph` | 1.000 | 1.000 | 1.000 | 5 | 0 |
| `degenerate_geometry` | `linear` | 1.000 | 1.000 | 0.500 | 3 | 0 |
| `degenerate_geometry` | `planner_executor` | 1.000 | 1.000 | 0.750 | 5 | 0 |
| `degenerate_geometry` | `graph` | 1.000 | 1.000 | 1.000 | 4 | 0 |
| `noisy_but_well_conditioned` | `linear` | 1.000 | 1.000 | 0.500 | 3 | 0 |
| `noisy_but_well_conditioned` | `planner_executor` | 1.000 | 1.000 | 0.750 | 5 | 0 |
| `noisy_but_well_conditioned` | `graph` | 1.000 | 1.000 | 1.000 | 4 | 0 |
| `adversarial_ambiguous_and_wide` *(adversarial)* | `linear` | 0.333 | 0.333 | 0.333 | 4 | 2 |
| `adversarial_ambiguous_and_wide` *(adversarial)* | `planner_executor` | 1.000 | 1.000 | 1.000 | 8 | 0 |
| `adversarial_ambiguous_and_wide` *(adversarial)* | `graph` | 1.000 | 1.000 | 1.000 | 8 | 0 |
| `ambiguous_but_precise` | `linear` | 1.000 | 1.000 | 0.000 | 2 | 0 |
| `ambiguous_but_precise` | `planner_executor` | 1.000 | 1.000 | 1.000 | 3 | 0 |
| `ambiguous_but_precise` | `graph` | 1.000 | 1.000 | 1.000 | 3 | 0 |
| `thin_evidence` | `linear` | 1.000 | 1.000 | 0.500 | 3 | 0 |
| `thin_evidence` | `planner_executor` | 1.000 | 1.000 | 1.000 | 4 | 0 |
| `thin_evidence` | `graph` | 1.000 | 1.000 | 1.000 | 4 | 0 |
| `mixed_load` | `linear` | 0.833 | 0.833 | 0.333 | 7 | 1 |
| `mixed_load` | `planner_executor` | 1.000 | 1.000 | 0.917 | 13 | 0 |
| `mixed_load` | `graph` | 1.000 | 1.000 | 1.000 | 12 | 0 |

## What the metrics mean

- **Task success** — did the final action match the oracle policy given full information? This is what an operator experiences.
- **Tool correctness** — was the decision expressed through the right tool? Escalating a degenerate-geometry track via `escalate_to_human` instead of `request_better_geometry` reaches the same person but loses the actionable part: one asks a human to look, the other asks the platform to move.
- **Trajectory quality** — Jaccard overlap between the lookups the decision actually depended on and the lookups made. Penalised in both directions, because an agent that calls every tool on every track is not careful, only expensive.
- **Missed escalations** — tracks the policy says a human must see that the agent resolved by itself. The headline safety number.

## Scenarios

- **`clear_picture`** — Well-localised, confidently classified tracks. Everything publishes.
- **`degenerate_geometry`** — Wide fixes caused by no perpendicular baseline. These must request a manoeuvre, not a human -- the geometry is fixable by moving.
- **`noisy_but_well_conditioned`** — Wide fixes with adequate geometry. Manoeuvring will not help, so these suppress rather than requesting geometry.
- **`adversarial_ambiguous_and_wide`** **(adversarial)** — THE case the system must not guess on: class posterior nearly flat AND position known only to tens of metres. Must escalate to a human. An agent that publishes its top class here is confidently wrong, which is the worst possible output.
- **`ambiguous_but_precise`** — Flat class posterior but a tight fix. Should still surface -- an unidentified object at a known location is actionable.
- **`thin_evidence`** — Too few observations to act on at all.
- **`mixed_load`** — Everything at once, to check ordering under load.

## Caveats

- The default backend is a deterministic rule policy, not a language model. That is what lets this report regenerate in CI with no API key and no run-to-run variance, and it sets the floor: a pattern that cannot beat a hundred lines of thresholds does not justify a model call. `ClaudeBackend` runs the same tools and the same scenarios through `claude-opus-5` when credentials are present.
- Scenarios are constructed TrackStates rather than pipeline output, so the agent layer is scored on decision quality rather than on whether the tracker happened to produce a good fix that day. The adversarial case in particular needs a specific combination of uncertainties that is far easier to construct than to provoke.
- The oracle is the same policy module the rule backend uses. It therefore scores *consistency with a stated policy*, not correctness in the world. The policy's own thresholds are an engineering judgement and are stated in `agent/policy.py`.
