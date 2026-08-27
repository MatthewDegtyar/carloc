# Street View sign survey — feasibility test

Tested the "traverse Street View, find the plaques, mark them down" plan on
downtown Miami. **It half works, and the half that fails is the half that matters.**

## What worked

Landed on NE 2nd St at 25.77616, -80.19083 (Aug 2024 imagery) and could see
clearly:

- **kerbside parallel parking along the south side**, cars present, unambiguous
- **an MPA plaque** mid-block on the north side — same blue/white shape as the
  40703 sign, spotted at roughly 60-80 m

So Street View does answer *"does this block face have parking"* and *"is there a
plaque"*. Those are real, and that is one block face verified.

## What did not work

**Reading the zone number.** At 60-80 m the plaque is about 12 px of sign. Zoom
just enlarges blur. You have to move to it.

**Moving to it.** Street View advances by clicking the road ahead or the arrows.
Single-click, double-click and hand-built heading URLs all failed to change
panorama in this automation context — the view stayed put every time. Only the
initial navigate-to-a-pano-ID worked, and pano IDs are not predictable, so I
cannot jump to "the pano nearest that sign" without already knowing its ID.

## The cost, measured not guessed

**~15 tool calls for one location, ending without a zone number.** Extrapolating
to the 5-location test is 75+ calls for maybe 2-3 readable plaques, and to
downtown-wide is thousands.

There is also a terms problem at scale. Looking at a handful of places is using
Maps as intended. Systematically traversing Street View to extract a dataset of
sign locations is the thing Google's terms prohibit, and this would be building
exactly that.

## What this changes

Nothing about the goal, one thing about the route.

You want to know **whether a car is physically present at a location at a given
time**. That needs three things, and the zone number is the least of them:

1. **which block faces have parking** — Street View *can* answer this, and so can
   aerial imagery, faster. This is the useful survivor from the test.
2. **the zone number on each** — Street View is the wrong tool. The records
   request is the right one.
3. **cars detected and located** — the pipeline, blocked on pose for the Miami
   video.

Item 3 is the product. Items 1 and 2 are labels you hang on the answer, and
neither needs to be complete for a demo: a working occupancy readout on ten
verified block faces is more convincing than a complete zone map with no cars in
it.

## Verified so far

| block face | parking | plaque seen | zone |
|---|---|---|---|
| NE 2nd St, south side, ~189 NE 2nd St | yes, parallel | yes, north side mid-block | unread |
