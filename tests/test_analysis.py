"""Appearance, concealment and the operator queries built on them.

The queries make claims an operator would act on -- "a person could be hidden
behind this" -- so the tests here are mostly about the claim NOT being made when
the evidence does not support it.
"""

import numpy as np
import pytest

from geoloc_agent.analysis.appearance import AppearanceMemory, sample_appearance
from geoloc_agent.analysis.concealment import (
    CLASS_CONSISTENCY,
    HULL_MARGIN,
    MIN_WIDTH_M,
    PERSON_STAND_M,
    assess,
    physical_extent,
    shadow_polygon,
)

FX = FY = 1250.0


def box_for(width_m: float, height_m: float, range_m: float, cx: float = 800.0,
            cy: float = 450.0) -> np.ndarray:
    """The inverse of `physical_extent`: a box that derives to this size."""
    w_px = width_m * FX / range_m
    h_px = height_m * FY / range_m
    return np.array([cx - w_px / 2, cy - h_px / 2, cx + w_px / 2, cy + h_px / 2])


# -- extent -----------------------------------------------------------------

def test_physical_extent_inverts_the_pinhole_relation():
    bbox = box_for(1.8, 1.5, 20.0)
    w, h = physical_extent(bbox, 20.0, FX, FY)
    assert w == pytest.approx(1.8, rel=1e-6)
    assert h == pytest.approx(1.5, rel=1e-6)


def test_same_object_further_away_derives_the_same_size():
    """Range and apparent size must cancel, or every query becomes range-dependent."""
    near = physical_extent(box_for(2.0, 1.8, 10.0), 10.0, FX, FY)
    far = physical_extent(box_for(2.0, 1.8, 40.0), 40.0, FX, FY)
    assert near == pytest.approx(far, rel=1e-6)


# -- concealment ------------------------------------------------------------

def test_a_van_conceals_a_standing_person():
    c = assess(box_for(2.1, 2.2, 25.0), 25.0, FX, FY, cls="car")
    assert c.hides_standing and c.hides_crouching
    assert c.level == "full"
    assert c.shadow_area_m2 > 0


def test_a_sedan_conceals_only_a_crouching_person():
    c = assess(box_for(4.5, 1.5, 25.0), 25.0, FX, FY, cls="car")
    assert c.hides_crouching
    assert not c.hides_standing
    assert c.level == "partial"


def test_a_person_is_not_concealment_for_another_person():
    """The failure this width gate exists for: a 1.0 m derived width is a person."""
    c = assess(box_for(1.0, 1.9, 25.0), 25.0, FX, FY, cls="pedestrian")
    assert not c.hides_crouching and not c.hides_standing
    assert c.level == "none"
    assert c.shadow_area_m2 == 0.0


def test_the_width_gate_sits_between_the_two_populations():
    """Measured derived widths: pedestrian p90 1.22 m, car p10 3.29 m."""
    assert 1.22 < MIN_WIDTH_M < 3.29


def test_hull_margin_keeps_an_ordinary_car_roof_below_the_standing_bar():
    """A 1.5 m sedan derives ~1.15x taller; without the margin that clears 1.75 m."""
    derived = 1.5 * 1.15
    assert derived > PERSON_STAND_M * 0.9      # the bias is real and large
    assert derived < PERSON_STAND_M * HULL_MARGIN   # and the margin absorbs it


def test_size_that_contradicts_the_class_makes_no_claim():
    """A pedestrian track deriving 3.4 m tall is a broken pairing, not a tall person."""
    c = assess(box_for(2.0, 3.4, 25.0), 25.0, FX, FY, cls="pedestrian")
    assert not c.hides_standing and not c.hides_crouching
    assert "disagree" in c.reason
    assert c.shadow_area_m2 == 0.0


def test_the_consistency_check_never_grants_concealment():
    """It can only withhold. A wide, tall car still qualifies on geometry alone."""
    bbox = box_for(2.2, 2.3, 25.0)
    with_class = assess(bbox, 25.0, FX, FY, cls="car")
    without = assess(bbox, 25.0, FX, FY)
    assert with_class.hides_standing == without.hides_standing is True


def test_an_unknown_class_is_judged_on_geometry_alone():
    c = assess(box_for(2.5, 2.4, 30.0), 30.0, FX, FY, cls="shipping-container")
    assert c.hides_standing


def test_consistency_bound_is_loose_enough_for_a_tall_real_car():
    """A 2.0 m van is a real car height and must survive the check."""
    c = assess(box_for(2.2, 2.0 * 1.15, 25.0), 25.0, FX, FY, cls="car")
    assert "disagree" not in c.reason
    assert CLASS_CONSISTENCY > 1.0


# -- shadow geometry --------------------------------------------------------

def test_shadow_wedge_lies_behind_the_object_and_widens():
    origin = np.zeros(3)
    centre = np.array([0.0, 20.0, 0.0])
    poly = shadow_polygon(origin, centre, width_m=2.0, depth_m=2.0)
    assert poly.shape == (4, 2)
    # Every vertex is at or beyond the object, none between it and the camera.
    assert np.all(poly[:, 1] >= 20.0 - 1e-9)
    near_width = np.linalg.norm(poly[0] - poly[1])
    far_width = np.linalg.norm(poly[2] - poly[3])
    assert far_width > near_width


def test_shadow_wedge_follows_the_bearing_not_the_axes():
    poly = shadow_polygon(np.zeros(3), np.array([20.0, 0.0, 0.0]), width_m=2.0)
    assert np.all(poly[:, 0] >= 20.0 - 1e-9)


def test_degenerate_shadow_inputs_return_empty_rather_than_raising():
    assert len(shadow_polygon(np.zeros(3), np.zeros(3), 2.0)) == 0
    assert len(shadow_polygon(np.zeros(3), np.array([0.0, 10.0, 0.0]), 0.0)) == 0


# -- appearance -------------------------------------------------------------

def _flat_image(value: int, shape=(900, 1600, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_a_white_panel_samples_light_and_a_black_one_dark():
    bbox = np.array([600.0, 300.0, 900.0, 500.0])
    assert sample_appearance(_flat_image(235), bbox).tone == "light"
    assert sample_appearance(_flat_image(20), bbox).tone == "dark"


def test_appearance_memory_survives_a_single_bad_look():
    """One frame of glare must not move a track between buckets."""
    memory = AppearanceMemory()
    bbox = np.array([600.0, 300.0, 900.0, 500.0])
    for _ in range(9):
        memory.observe(1, sample_appearance(_flat_image(20), bbox))
    memory.observe(1, sample_appearance(_flat_image(250), bbox))
    assert memory.get(1).tone == "dark"


def test_appearance_of_an_offscreen_box_is_none_not_a_guess():
    bbox = np.array([2000.0, 1200.0, 2100.0, 1300.0])
    assert sample_appearance(_flat_image(128), bbox) is None


# -- truncation -------------------------------------------------------------

IMAGE = (1600.0, 900.0)


def test_a_box_cut_off_by_the_frame_edge_is_not_called_too_small():
    """The near, large vehicle whose box runs off the bottom of the image.

    Its derived height is a lower bound, so failing the standing-person bar says
    nothing. Reporting that as "too small to conceal a person" is a false
    negative on the one query where that is the dangerous way to be wrong.
    """
    bbox = np.array([200.0, 700.0, 900.0, 899.5])   # runs into the bottom edge
    c = assess(bbox, 10.0, FX, FY, cls="car", image_size=IMAGE)
    assert c.truncated
    assert c.indeterminate
    assert c.level == "unknown"
    assert "not measurable" in c.reason
    assert "too small" not in c.reason


def test_a_truncated_object_still_shadows_ground():
    bbox = np.array([200.0, 700.0, 900.0, 899.5])
    c = assess(bbox, 10.0, FX, FY, cls="car", image_size=IMAGE)
    assert c.shadow_area_m2 > 0


def test_truncation_does_not_rescue_something_narrower_than_a_person():
    """Width survives a vertical clip, so the width gate still has to be met."""
    bbox = np.array([700.0, 700.0, 760.0, 899.5])
    c = assess(bbox, 25.0, FX, FY, cls="pedestrian", image_size=IMAGE)
    assert c.truncated
    assert not c.indeterminate
    assert c.level == "none"


def test_an_interior_box_is_never_truncated():
    c = assess(box_for(4.5, 1.5, 25.0), 25.0, FX, FY, cls="car", image_size=IMAGE)
    assert not c.truncated and not c.indeterminate
    assert c.level == "partial"


def test_a_truncated_box_that_already_clears_the_bar_is_a_confident_match():
    """A lower bound above the threshold is still above the threshold."""
    bbox = np.array([300.0, 0.5, 450.0, 175.0])   # clipped at the top
    c = assess(bbox, 25.0, FX, FY, cls="truck", image_size=IMAGE)
    assert c.truncated
    assert c.hides_standing
    assert not c.indeterminate      # measured, not merely unknown
    assert c.level == "full"


def test_without_image_size_nothing_is_treated_as_truncated():
    """Callers that cannot say where the frame edge is get the old behaviour."""
    bbox = np.array([200.0, 700.0, 900.0, 899.5])
    c = assess(bbox, 10.0, FX, FY, cls="car")
    assert not c.truncated and not c.indeterminate


# -- queries ----------------------------------------------------------------

from geoloc_agent.analysis.queries import BY_NAME, QUERIES, Candidate  # noqa: E402


class _Track:
    """Minimal stand-in: the queries touch only these fields."""

    def __init__(self, cls="car", p=0.9, sigma=0.5, degenerate=False, track_id=1):
        self.track_id = track_id
        self.top_class = (cls, p)
        self.sigma_horizontal = sigma
        self.degenerate = degenerate
        self.degeneracy_reason = "parallax below floor" if degenerate else ""


def candidate(cls="car", tone_value=0.2, sigma=0.5, degenerate=False,
              conceal=None, range_m=20.0) -> Candidate:
    from geoloc_agent.analysis.appearance import Appearance

    return Candidate(
        track=_Track(cls=cls, sigma=sigma, degenerate=degenerate),
        appearance=Appearance(value=tone_value, saturation=0.02, hue_deg=0.0,
                              rgb=(60.0, 60.0, 60.0), n_pixels=900),
        concealment=conceal,
        range_m=range_m,
    )


def test_every_query_has_a_distinct_name_and_colour():
    assert len({q.name for q in QUERIES}) == len(QUERIES)
    assert len({q.colour for q in QUERIES}) == len(QUERIES)


def test_light_and_dark_are_mutually_exclusive():
    light = candidate(tone_value=0.75)
    dark = candidate(tone_value=0.10)
    assert BY_NAME["light vehicles"].match(light)[0]
    assert not BY_NAME["dark vehicles"].match(light)[0]
    assert BY_NAME["dark vehicles"].match(dark)[0]
    assert not BY_NAME["light vehicles"].match(dark)[0]


def test_a_track_with_no_appearance_matches_no_colour_query():
    """Unknown must not fall into a bucket by default."""
    c = candidate()
    c.appearance = None
    assert not BY_NAME["light vehicles"].match(c)[0]
    assert not BY_NAME["dark vehicles"].match(c)[0]


def test_concealment_and_partial_cover_never_both_match():
    """Including the truncated case, which was in both before it was excluded."""
    for bbox, rng, size in (
        (box_for(2.2, 2.3, 25.0), 25.0, None),                 # full
        (box_for(4.5, 1.5, 25.0), 25.0, None),                 # partial
        (np.array([200.0, 700.0, 900.0, 899.5]), 10.0, IMAGE),  # truncated
    ):
        cz = assess(bbox, rng, FX, FY, cls="car", image_size=size)
        c = candidate(conceal=cz)
        both = (BY_NAME["concealment"].match(c)[0]
                and BY_NAME["partial cover"].match(c)[0])
        assert not both, cz.reason


def test_a_matched_query_always_states_why():
    cz = assess(box_for(2.2, 2.3, 25.0), 25.0, FX, FY, cls="car")
    for query in QUERIES:
        hit, why = query.match(candidate(conceal=cz))
        if hit:
            assert why.strip(), f"{query.name} matched without a rationale"


def test_a_broken_predicate_does_not_break_the_render():
    """A query raising must drop out, not take the whole frame with it."""
    from geoloc_agent.analysis.queries import Query

    q = Query(name="boom", caption="", predicate=lambda c: 1 / 0,
              rationale=lambda c: "never")
    assert q.match(candidate()) == (False, "")


def test_everything_colours_by_uncertainty_not_one_flat_colour():
    from geoloc_agent.analysis.queries import BAD, GOOD, WARN

    q = BY_NAME["everything"]
    assert q.colour_of is not None
    assert q.colour_of(candidate(sigma=0.5)) == GOOD
    assert q.colour_of(candidate(sigma=9.0)) == WARN
    assert q.colour_of(candidate(sigma=0.5, degenerate=True)) == BAD


def test_a_degenerate_track_reads_red_however_tight_its_covariance():
    """The failure mode this project exists to avoid: small and wrong together."""
    from geoloc_agent.analysis.queries import BAD

    c = candidate(sigma=0.05, degenerate=True)
    assert BY_NAME["everything"].colour_of(c) == BAD
    assert BY_NAME["unreliable"].match(c)[0]
