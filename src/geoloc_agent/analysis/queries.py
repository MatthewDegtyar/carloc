"""Operator queries over tracks.

Each query is a predicate plus a rationale. The rationale matters as much as the
match: an operator shown a highlighted box needs to know *why* it is highlighted,
and "the system decided" is not an answer. It is the same discipline the surfacing
policy uses -- every decision carries the numbers behind it.

Queries read only what the pipeline produces: class posterior, position,
covariance, sampled appearance, and physical extent derived from box and range.
Nothing here reads dataset annotations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from geoloc_agent.analysis.appearance import Appearance
from geoloc_agent.analysis.concealment import Concealment
from geoloc_agent.contracts import TrackState


@dataclass
class Candidate:
    track: TrackState
    appearance: Appearance | None
    concealment: Concealment | None
    range_m: float


@dataclass
class Query:
    name: str
    caption: str
    predicate: Callable[[Candidate], bool]
    rationale: Callable[[Candidate], str]
    colour: str = "#4ec9b0"

    colour_of: Callable[[Candidate], str] | None = None
    """Per-track colour, where the query is about a property that varies across
    matches. Only "everything" uses it, to encode uncertainty -- with one flat
    colour that section shows what is present but not which fixes are worth
    acting on, which is the distinction this whole system exists to draw."""

    def match(self, c: Candidate) -> tuple[bool, str]:
        try:
            hit = self.predicate(c)
        except Exception:  # noqa: BLE001 - a query must never break the render
            return False, ""
        return hit, (self.rationale(c) if hit else "")


def _is(cls: str, c: Candidate, min_p: float = 0.4) -> bool:
    name, p = c.track.top_class
    return name == cls and p >= min_p


def _tone(c: Candidate) -> str:
    return c.appearance.tone if c.appearance else "unknown"


GOOD, WARN, BAD = "#4ec9b0", "#e2b93d", "#e05561"
CONFIDENT_SIGMA_M = 2.0
"""Below this horizontal 1-sigma a fix is worth acting on directly. It is the
same bar the surfacing policy uses, so the picture and the policy agree."""


def _uncertainty_colour(c: Candidate) -> str:
    if c.track.degenerate:
        return BAD
    return GOOD if c.track.sigma_horizontal < CONFIDENT_SIGMA_M else WARN


QUERIES: list[Query] = [
    Query(
        name="everything",
        caption="Every confirmed track -- the whole picture the other six queries "
                "select from. Colour encodes position uncertainty: green under 2 m "
                "horizontal 1-sigma, amber beyond it, red where the geometry cannot "
                "support a fix at all.",
        predicate=lambda c: True,
        rationale=lambda c: (f"{c.track.top_class[0]} at {c.range_m:.0f} m "
                             f"+/- {c.track.sigma_horizontal:.1f} m"),
        colour=GOOD,
        colour_of=_uncertainty_colour,
    ),
    Query(
        name="light vehicles",
        caption="Vehicles whose bodywork samples light -- white, silver, pale grey. "
                "Colour is the median over the track's whole life, not one frame, "
                "because a single look catches sun and shadow.",
        predicate=lambda c: _is("car", c) and _tone(c) == "light",
        rationale=lambda c: (f"light bodywork (value {c.appearance.value:.2f}), "
                             f"{c.range_m:.0f} m"),
        colour="#e8e8e8",
    ),
    Query(
        name="dark vehicles",
        caption="The same scene, same tracks, inverted selection. Nothing was "
                "re-detected -- only the query changed.",
        predicate=lambda c: _is("car", c) and _tone(c) == "dark",
        rationale=lambda c: (f"dark bodywork (value {c.appearance.value:.2f}), "
                             f"{c.range_m:.0f} m"),
        colour="#7aa2f7",
    ),
    Query(
        name="concealment",
        caption="Objects large enough to conceal a STANDING person, plus those cut "
                "off by the frame edge whose height cannot be measured -- unknown "
                "size is not the same as harmless. The wedge is ground the camera "
                "cannot see behind them: where someone COULD be, not that anyone is.",
        predicate=lambda c: bool(c.concealment and (c.concealment.hides_standing
                                                    or c.concealment.indeterminate)),
        rationale=lambda c: c.concealment.reason,
        colour="#e0af68",
    ),
    Query(
        name="partial cover",
        caption="Lower bar: objects that would conceal a CROUCHING person. Most "
                "cars qualify; pedestrians do not, since you cannot meaningfully "
                "hide behind someone your own size.",
        predicate=lambda c: bool(c.concealment and c.concealment.hides_crouching
                                 and not c.concealment.hides_standing
                                 and not c.concealment.indeterminate),
        rationale=lambda c: c.concealment.reason,
        colour="#d19a66",
    ),
    Query(
        name="people",
        caption="Pedestrians, with the uncertainty on each fix. A person at 30 m "
                "with a 3 m error ellipse is a different report from one at 8 m "
                "with 30 cm.",
        predicate=lambda c: _is("pedestrian", c),
        rationale=lambda c: (f"pedestrian, {c.range_m:.0f} m "
                             f"+/- {c.track.sigma_horizontal:.1f} m"),
        colour="#c678dd",
    ),
    Query(
        name="unreliable",
        caption="Tracks the system does NOT trust: degenerate geometry, or a fix "
                "whose uncertainty is the size of the estimate. Surfacing these as "
                "confident positions is the failure this project exists to avoid.",
        predicate=lambda c: c.track.degenerate,
        rationale=lambda c: c.track.degeneracy_reason.split(";")[0][:70],
        colour="#e05561",
    ),
]

BY_NAME = {q.name: q for q in QUERIES}
