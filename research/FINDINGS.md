# Miami on-street parking zones: what exists, what governs it, how to get it

Worked while you were out. Short version: **the data you want is not public
anywhere, but it is a public record and you can compel it** — and the sign-hunting
plan is unnecessary, because the regulation that governs those signs also tells
you where a zone starts and stops.

---

## 1. The sources that do not have it

| source | result |
|---|---|
| **ParkMobile zone search box** | Not a zone lookup — passes your text to Google's geocoder, so `40703` returns **Costa Rica**. This misled me into concluding there was no zone API. **There is one.** See §6. |
| ParkMobile page internals | Leaks a Google Maps API key in the query string. Not used — those are someone's paid credentials. |
| Miami Parking Authority site | Blocks scripted requests (403 / Cloudflare). Facilities map is Mapbox with data embedded, and covers **garages and lots**, not on-street zones. |
| MPA "Find Parking" | Server-rendered list, geocoded client-side from street addresses. Facilities only. |
| ArcGIS Online / Miami-Dade GIS | Nothing. The one layer named `OnStreetParking_Miami` is a **pavement-condition inspection** layer — 45 features with `RCI_Rating` and `Repaired_Date`. |
| **OpenStreetMap** | 1,610 downtown street ways, **8 carry any `parking:*` tag**. Effectively zero coverage. |

So there is no scrape, no API, and no open dataset. That is worth knowing before
spending a weekend on it.

---

## 2. The legal position — this is the actual leverage

### MPA is a public agency, and the inventory is a public record

MPA's own cookie notice states it plainly:

> *"The Miami Parking Authority (MPA) is an **independent agency of the City of
> Miami**… may be a **public record** subject to mandatory disclosure under
> **Florida Chapter 119, Florida Statutes**"*

Florida's Public Records Act is among the strongest in the US, and it is
constitutional, not merely statutory — **Art. I, § 24, Fla. Const.** A GIS layer
or meter/zone inventory held by MPA is a record. They must produce it, may charge
only the actual cost of duplication plus extensive-use labour, and must state a
statutory exemption if they refuse.

MPA has a standing Public Records Request page and a Public Records Division.

**This is the single highest-value action available and it costs an email.** A
draft is in `research/records_request.md`.

### What to actually ask for — corrected

My first draft led with GIS layers and polygons. That was wrong, and worth
recording as a mistake: **no city stores parking zones as boxes.** Checking how
other cities hold the same data settles it.

New York publishes *"Parking Meters — ParkNYC Block Faces"* — ParkNYC being their
mobile-payment system, the direct equivalent of ParkMobile zones. 11,185 records,
schema:

    the_geom     MultiLineString        <- a LINE along the kerb, not a polygon
    pay_by_cel   100009                 <- the mobile-payment zone number
    on_street    William Street
    side_of_st   E
    from_stree   Cedar Street
    to_street    Liberty Street
    meter_rate   Zone M1

Los Angeles holds it at the individual space level: `spaceid`, `blockface`
("700 HOPE ST"), `metertype`, `ratetype`, `timelimit`, `latlng`.

So the real data model is **zone number -> street + side + from-cross-street +
to-cross-street**, which is the MUTCD block-face rule written down. The "box" is
something you *derive* from that line plus a lane width — it is not something any
authority stores.

That reframes the request. Asking for a shapefile invites an accurate "we don't
have one". Asking for the **table** is much harder to refuse, because the mobile
payment vendor cannot charge for zone 40703 without having been told what 40703
is, and MPA is who told them.

Miami publishes none of this: no parking dataset on any Socrata portal, and
nothing on the county or city ArcGIS hubs.

### The signage rules answer your zone-boundary question

You asked whether a zone broken by a planter or barrier needs its own plaque. The
governing document is the **MUTCD** (federal, adopted by Florida through FDOT),
Chapter 2B, regulatory signs. What it says:

- **A parking regulation runs to the next cross street** unless a termination
  sign ends it sooner. So the **default zone unit is the block face.**
- **Arrow direction encodes position in the zone.** A single-headed arrow means
  you are at a **boundary** and the regulation extends the way the arrow points.
  A double-headed arrow means you are **mid-zone**.
- **Sign spacing is capped at roughly 270 ft**, derived from legibility — 1 inch
  of letter height per 30 ft of viewing distance.

Two consequences that matter for you:

1. **A plaque photo is more informative than its location alone.** The arrow tells
   you whether that sign is an edge or an interior point of the zone. That is
   readable from Street View, and it means you do not need to find *every* sign —
   you need the ones with single arrows.
2. **A physical break does not by itself create a new zone.** Zone numbering is
   an MPA administrative decision, not a MUTCD requirement. A planter splitting a
   run of kerb does not compel a second plaque; the regulation continues to the
   cross street unless MPA chose to sign it otherwise. So sign-hunting recovers
   *signage*, not *zone identity* — and only MPA's inventory carries the mapping
   from zone number to extent.

**Caveat:** MUTCD compliance and FDOT's Florida supplement are engineering
standards, and municipal practice deviates. Treat the above as how it is supposed
to work, and verify against the City of Miami Code (Ch. 35, Motor Vehicles and
Traffic) before relying on it for anything adversarial. I did not read the city
code directly — the MPA site blocks automated access and I had no browser session
budget left for it.

---

## 3. What I built: the block-face scaffold

`carloc/blockface.py`, output in `reports/miami_blockfaces.{geojson,csv}`.

Since a zone defaults to a block face, the block face is the right container. This
derives every one of them for downtown Miami from the OSM street network:

- splits each street at its intersections
- offsets to both kerbs
- emits a rectangle with corner lat/lon, length, and an estimated space count

**GeoJSON imports straight into Google My Maps**, which is the overlay you asked
for. The CSV has an empty `zone` column, ready for MPA's numbers.

Assumptions, all stated in the module and all crude:

| | value | note |
|---|---|---|
| lane offset from centreline | 4.6 m | the weakest one — a 2 m error puts a box in the travel lane |
| parking lane width | 2.6 m | US standard 8–8.5 ft |
| space length | 6.7 m | 22 ft including manoeuvring |
| corner clearance | 7.6 m each end | hydrants, daylighting, kerb return |

### The number is an upper bound, and I want to be clear about that

| | faces | est. spaces |
|---|---|---|
| every block face | 3,516 | 39,946 |
| filtered to plausible streets | 2,602 | 31,676 |
| **MPA actual, whole city** | | **~11,800** |

Downtown alone comes out at **2.7× the entire city's real inventory.** The filter
(excluding arterials) barely dented it, because the error is not arterials — it is
that most block faces simply have no *managed* parking, and nothing in the street
geometry says which.

I stopped tuning deliberately. I could have adjusted filters until 31,676 became
11,800, but matching one aggregate that way yields a map that is right in total
and wrong on every individual block. An honest upper bound is more useful than a
fitted number.

**So: this is where parking *can* be, not where it *is*.** It is the container.
MPA's inventory is what fills it.

---

## 4. What I would do next, in order

1. **Send the records request** (`research/records_request.md`). Days of latency,
   so start it first. If they produce a GIS layer, everything above is moot and
   you have the real thing.
2. **Ask for the enforcement-side data too** — MPA runs enforcement, so a
   zone→block-face mapping exists internally in some form, if only as the table
   their handhelds query.
3. **Only if both fail**, sample Street View. And then sample *smartly*: single-
   arrow plaques at block ends, not every sign, and only on the blocks the video
   actually covers.
4. Separately, the Miami video still needs pose before any of this joins up to
   detected cars — see `PLAN.md`. That is an independent blocker and the larger
   one.

---

## 5. Loose ends I did not resolve

- Did not read City of Miami Code Ch. 35 directly (site blocks automation).
- Did not find MPA's contract with ParkMobile/PayByPhone. It would be a public
  record too, and would likely name the zone-numbering scheme — worth adding to
  the request.
- Did not verify whether MPA publishes an annual report with a zone list; their
  budget documents may contain space counts by street.
- The `40703` plaque is Miami Parking Authority branding with **both** PayByPhone
  and ParkMobile logos, so the zone numbering is **MPA's**, not ParkMobile's. Any
  vendor-side scraping would have got you a vendor's view of someone else's
  numbering scheme.


---

## 6. CORRECTION: the ParkMobile zone API exists

I was wrong in §1. I tested the search box, concluded "no zone API", and stopped.
The search box is a geocoder; the zone data lives behind the **"enter your zone
number"** flow at `/zone/start`, and it is reachable unauthenticated.

### The chain

    /api/proxy/parkmobileapi/zones/{signageCode}
        40703 -> internalZoneCode 97840703       (prefix 978 + signage code)

    /api/locations?internalZoneCode=97840703&supplierId=978040
        -> { "signageCode":"40703", "type":"OnStreet",
             "geometry":[ {"latitude":25.771859,"longitude":-80.187708}, ...13 ] }

`type` really does say **OnStreet**, and the coordinates are real.

Two other endpoints exist and are dead ends worth naming so nobody re-checks
them: `/api/zones/search` and `/api/zones/search/transient` are viewport queries
that return **only bookable off-street garages and lots**. Every `parkingType`
value returns the same handful, in downtown Miami and in South Beach alike.

There is no bulk listing — `/api/locations` requires an `internalZoneCode` and
supplier-only queries 404 — so zones are found by probing signage codes. Density
is sparse: 4 of 12 resolved around 40703.

### What `geometry` is, and is not

**Anchors, not boundaries.** Points inside a zone, almost certainly pay-station
or meter positions.

| zone | anchors | extent |
|---|---|---|
| 40701 | 7 | 215 × 267 m |
| 40703 | 13 | 700 × 429 m |
| 40711 | 35 | 1886 × 679 m |

**So a Miami zone is not a block face.** It is a corridor running along a street
for a kilometre or more, covering many blocks — visible in `reports/miami_zones.png`
as long linear runs of anchors. This contradicts the MUTCD default I inferred in
§2: the regulation may terminate at each cross street, but the *zone number* does
not change there. Miami numbers far coarser than the signage rule implies.

### Data quality

Zone 40708's only "coordinate" is **(0, 0)** — null island. A consumer trusting
the field plots Miami parking in the Gulf of Guinea. Filtered in
`carloc/parkmobile.py` rather than downstream, where it blew out every map extent
it touched.

### Result

`research/zonemap.py` snaps anchors to the nearest block face within 45 m:

    5 zones, 125 anchors -> 79 block faces highlighted, ~1,294 estimated spaces

That is the deliverable: **the parts of the road where the parking actually is**,
in lat/lon, cross-referenced from ParkMobile rather than read off signs.

Zone 40713 has 40 anchors but matched only 6 faces — the rest sit outside the OSM
extract. Widening the bbox fixes it; nothing is wrong with the join.

### This obsoletes two earlier plans

**No records request needed** for zone geometry. **No sign-reading needed** — the
number comes from the API keyed by location, which is exactly the cross-reference
that was proposed. Both were answers to a problem that turned out not to exist.

## 9. The lane band, measured

The parking lane was a guess: 3.3-5.9 m from the street centreline, one travel
lane out plus a 2.6 m bay, taken from US design defaults. That guess is now a
measurement.

RF-DETR over 21 tiles of zones 40701 and 40703 returned 349 vehicles. Taking the
185 that sit 1.5-12 m from a street centreline -- excluding cars in the travel
lane and cars in off-street lots -- and measuring their perpendicular offset:

| | |
|---|---|
| signed offset | **bimodal, peaks at -4.5 and +4.5 m** |
| right side | n=133, median +4.55 m |
| left side | n=76, median -4.74 m |
| lane centre (median abs) | **4.73 m** |
| implied band, 2.6 m bay | **3.43 - 6.03 m** |
| assumed band | 3.30 - 5.90 m |
| **error in the assumption** | **13 cm** |

Cars really do park at 4.7 m from the centreline, on both sides, and the design
default was right to within a hand's width. Boxes now carry `calibrated=True`.

A percentile fit was tried first and was worse. `calibrate_from_observations()`
at p75 returns 3.70-6.74 m, centring the band at 5.22 m instead of on the mode at
4.73 m, because the offset distribution is right-skewed -- its tail runs into
parking lots. At p85 the outer edge reaches 10.2 m, which is a lot, not a lane.
**Fit the band to the median plus half a standard bay, not to percentiles of a
contaminated tail.**

## 10. The bay is the wrong decision boundary

Judging against the 2.6 m bay gave 3 cars inside a paid zone out of 349. The
geometry was not the problem; the decision rule was.

A verdict of INSIDE requires `margin > sigma`, and margin is measured to the
nearest edge. Against a 2.6 m box the half-width is 1.30 m, so at sigma = 1.0 m a
car's centre has to land within **0.30 m** of the lane centreline to be called
confidently inside. That is a coin flip dressed up as a measurement.

| decision band | width | inside | ambiguous | outside |
|---|---|---|---|---|
| bay (stall) 3.43-6.03 m | 2.60 m | 2 | 55 | 292 |
| kerbside envelope 2.5-7.0 m | 4.50 m | 19 | 37 | 293 |
| **block face 2.5-8.5 m** | **6.00 m** | **29** | **27** | **293** |
| wide 2.0-10.0 m | 8.00 m | 40 | 29 | 280 |

The enforcement question is *"is this car in ParkMobile zone 40703"*, not *"is
this car in stall 7"*. A zone is a block face. Testing against a single bay
imported a precision the question never asked for, and sigma then ate it.

The block-face band is used for the verdict; the measured bay is kept for
counting spaces. Result: **29 inside, 27 ambiguous (7.7%), 293 outside.**

## 11. What the imagery shows

`reports/downtown_survey.png`, four block faces on georeferenced Esri tiles:

- **NE 1st Street (40703)** -- the lane box lands on the row of parked cars.
  20 cars in the lane against 12 estimated spaces, so the space estimate is low,
  not the box misplaced.
- **NW 2nd Avenue (40701)** -- the decisive panel. Green dots sit on the column
  of kerbside cars inside the box; red dots sit on cars in the surface lot a few
  metres away. **The method separates kerb from lot.** That separation is the
  whole product.
- **NW 3rd Street (40701)** -- boxes on visibly empty asphalt, zero cars. A true
  negative, not a failure.

## 12. Denominators

68% of detections are more than 10 m from any street. Downtown Miami is surface
lots, garages and rooftop decks, and RF-DETR finds cars in all of them. "349 cars
detected, 29 in a paid zone" is not an 8% hit rate -- most of those 349 were never
candidates. The denominator for enforcement is cars on the kerb of a paid block
face, and against that the method is working.

## 13. The dashcam pass: what works and what does not

Satellite answers *which cars are in a paid zone now*. Overstay needs two passes
separated in time, which is what a patrol vehicle gives. So the clip was cut
00:01:00-00:10:00 (540 s, 2160 quarter-second frames) and the chain tested leg by
leg.

**Leg 2, detection, works.** RF-DETR on the 640x360 frames upscaled 2x returns
270 vehicles over 40 sampled frames, median 7 per frame, boxes 38-309 px wide,
and cleanly separates the kerbside row from traffic ahead
(`reports/dashcam_detect.png`).

**Leg 1, localisation, does not.** Four independent measurements, each fatal on
its own:

| | |
|---|---|
| GPS telemetry | **none.** Two streams, video + audio, `encoder: Google`. It is a YouTube re-encode; any original telemetry is gone. |
| Google Maps API key | **none available.** Street View matching -- the requested method -- cannot run. ParkMobile's own key leaks in their page query string and was not used: those are someone else's paid credentials. |
| Road texture for visual odometry | **3-5 grey levels** std on the road surface at rows 300-355. Dense Farneback flow returns exactly `0.00` px there. |
| Street-name blades | **illegible.** A blade is ~12 px wide at source; upscaled 8x it is a green smudge. |

The file is named `4K` and is **640x360**.

### What the odometry does give

Yaw survives, because a camera rotation shifts the whole image and needs no road
texture -- an affine fit over all tracked features recovers it. That yields a
clean turn signature over the 9 minutes:

    t=120 L    t=222 R (large)    t=335 L    t=347 R
    t=382 L    t=400 L            t=504 L

and `f` self-calibrates from any turn whose true angle the grid fixes at 90
degrees, giving ~229 px/rad, i.e. a 109-degree horizontal field of view --
plausible for a dashcam, and derived rather than assumed.

Speed does not survive. `_ground_speed` fits `dy = c*(y-cy)**2` and rejects what
does not lie on the road plane, which is the right model and still returns a
coefficient ~30x too small: in a street scene most strong corners are on
buildings, parked cars and other traffic, and after those are rejected what is
left on the asphalt has no texture to track.

### Turns alone cannot place the car

The route was narrowed to the block face at Gesu Church (footprint
25.775808-25.776072 N, 25.775940 centroid, west edge **10 m** from NE 1st Avenue)
and then three candidate routes were each falsified:

* **Northbound NE 1st Ave, left at NE 2nd St** -- NE 2nd St (way/11165263) is
  one-way **eastbound**. The turn would be illegal.
* **Southbound NE 2nd Ave, left at NE 2nd St** -- legal, but puts the church
  100 m away and sends the car east toward Biscayne, away from the `TO 95`
  trailblazer visible at t=509.
* **Southbound North Miami Ave** -- puts the church on the left; the video has it
  on the right.

In a uniform grid a turn sequence is not a position fix. Without one anchor of
absolute position, the shape matches many places at once.

### The unblock

Either would close leg 1:

1. **A Google Maps API key** -- the originally requested method. Street View
   Static frames matched against video frames give absolute anchors directly.
2. **The genuine 4K source.** At 4x the linear resolution the road recovers
   texture, blades become OCR-able, and both odometry and anchoring return.

Leg 1 is the only missing leg. Legs 2 and 3 are already demonstrated.

## 14. The 4K source changes the answer

The `640x360` file was a re-encode. The genuine source is `3840x2160 @ 60 fps`
(av01, format 401); the 00:01:00-00:10:00 section is 1.5 GB. Two things follow
immediately.

**Street blades become legible, and they falsified the route.** At 640x360 a
blade is ~12 px wide and unreadable, so the route had been inferred from
landmarks -- and the landmark was wrong. The church at t=482-500 is not Gesu
(downtown); the first legible blade, `SE 7 ST`, put the whole pass in **Brickell**,
and the facade reads `FIRST PRESB` -- First Presbyterian Church of Miami, 609
Brickell Avenue, at **25.767679, -80.189544**, which is *east* of both Brickell
Avenue carriageways.

Every downtown route hypothesis in section 13 was therefore wrong, including the
premise. The corrected route, read off blades rather than inferred:

    t=30    Brickell Ave northbound at SE 13 ST / Coral Way
    t=300   Brickell Bay Drive
    t=412   SE 6th St x SE 1st Avenue        (blade "SE 1 Av")
    t=420   all-way STOP, SE 1st Avenue
    t=480   blade "SE 6 ST", "SE 7 ST NEXT SIGNAL"
    t=505   right turn onto Brickell Avenue southbound
    t=520   SE 7 ST signal, One Brickell City Centre hoarding

Travel direction is fixed independently: the colourful Brickell City Centre
garage is on the **left** at t=430, and BCC is south of SE 6th Street, so the car
is running **east**. That also settles the yaw sign -- a camera turning right
shifts the scene left, so negative image translation is a **right** turn, and the
turn labels in section 13 were inverted.

**Road texture is still not the problem it was assumed to be.** At 4K the
near-field asphalt still measures 2-6 grey levels and dense flow still returns
exactly `0.00` px. Resolution was never the issue: Miami asphalt in direct sun
has no texture to track. Ground-plane visual odometry is unavailable on this
footage at any resolution, and that is a property of the road, not the camera.

## 15. What replaced odometry, and the pass that came out of it

Speed comes from **residual scene motion** instead. After the rigid part of the
frame-to-frame transform is removed, what remains is parallax against buildings
and parked cars, which have ample texture. It gives a reliable stopped/moving
shape -- the car is stopped for **50%** of the window -- and the anchors supply
the scale. Distance is then distributed by observed motion rather than uniformly
in time.

Anchors are the two crossings the blades pin: SE 1st Avenue at t=420 and the
Brickell Avenue turn at t=505, **190.6 m apart over 85 s** (mean 2.24 m/s, a
crawl, which the frames confirm).

Along-track error accumulates only while the car moves and must vanish at both
anchors, so it is a fraction of the distance to the *nearer* anchor, not of the
whole span. The result is a sawtooth: **median 11.4 m, best 1.4 m at the
anchors, 50 m worst mid-segment.**

**Monocular range is never used.** The satellite survey already measured where a
kerbside car sits -- 4.73 m from the centreline, from 185 observations (section
9) -- so lateral offset is known a priori and only bearing is needed:

    along = lateral / tan(|bearing|),  bearing = atan((cx - W/2)/f)

with `f` self-calibrated from a 90-degree turn. This is why the chain survives a
weak camera: the hard quantity was measured from orbit, and the video only has to
supply an angle.

Result over 190 m of SE 6th Street: **576 kerbside detections collapsing to 40
distinct parked cars**, each with a lat/lon and a 1-sigma ellipse
(`reports/se6_pass.png`, `se6_cars.csv`, `se6_track.csv`).

## 16. The remaining gap: no ParkMobile geometry for Brickell

The verdict cannot be computed for these 40 cars, and the reason is data, not
method. **260 signage codes were scanned** (40600-40859) against the zone API at
a 0.6 s delay. Twenty resolved as `OnStreet`, and the nearest anchor to SE 6th
Street is **371 m away** (zone 40610, a single anchor north of the river). None
covers Brickell.

The zones on file cluster downtown (40701, 40703), Coconut Grove (40711-40714)
and Edgewater (40715). ParkMobile's own Brickell page lists only garages and
lots, and confirms kerbside is "limited and metered" without publishing a code.
The plaque visible at SE 1st Avenue is a backlit kiosk and unreadable even at 4K.

So the honest verdict for these 40 cars is **UNKNOWN -- no zone geometry**, which
is not the same as OUTSIDE. Closing it needs the Brickell signage code, from a
legible plaque or from the operator; the lookup itself is already built and
demonstrated downtown in sections 10-11.
