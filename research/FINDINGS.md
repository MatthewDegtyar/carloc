# Miami on-street parking zones: what exists, what governs it, how to get it

Worked while you were out. Short version: **the data you want is not public
anywhere, but it is a public record and you can compel it** — and the sign-hunting
plan is unnecessary, because the regulation that governs those signs also tells
you where a zone starts and stops.

---

## 1. The sources that do not have it

| source | result |
|---|---|
| **ParkMobile zone search** | **No zone API.** The box sends your text to Google's geocoder. `40703` returned **Costa Rica** (9.98, −84.17). Zone→location is not exposed. |
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
draft is in `research/records_request.md` — written narrowly on purpose, because
MPA's own guidance warns that broad requests return thousands of documents.

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
