# Draft public records request — MPA on-street parking zone inventory

Send to MPA's Public Records Division (address on
`miamiparking.com/public-records-request/`).

**Why it is written this way.** An earlier draft led with "GIS layer, shapefile,
geodatabase" — which was a mistake. A parking authority is not a GIS shop, and
leading with a format they probably do not hold invites an accurate "we don't have
that" that closes the conversation.

What they must hold is the **table**, because the mobile-payment vendor cannot
charge for zone 40703 without being told what 40703 is. That mapping was created
by MPA and handed over. So this asks for the table first, names the exact fields
by reference to a city that publishes the same thing, and explicitly says a
spreadsheet is fine and that they need not create anything.

It also cites the constitutional provision alongside the statute, and asks for any
exemption to be stated in writing, which converts a vague refusal into a specific
one you can evaluate.

Two things worth doing before you send: replace the bracketed fields, and decide
whether to include item 5 — the vendor contracts are the most likely thing to
draw a delay or a trade-secret objection, and you may prefer to get the geometry
first and ask for contracts separately.

---

**Subject:** Public records request — on-street parking zone inventory (Ch. 119, Fla. Stat.)

To the Public Records Custodian, Miami Parking Authority:

Under Chapter 119, Florida Statutes and Article I, Section 24 of the Florida
Constitution, I request copies of the following records. I am requesting records,
not asking questions, and I do not seek any personal information about any
individual.

1. **The on-street parking zone inventory**, in whatever form it is maintained.
   Specifically, the record that maps each mobile-payment zone number (for
   example zone 40703) to its physical location. A tabular export is entirely
   sufficient. Based on how other cities maintain the same information, the
   fields I am seeking are:

   - the zone / pay-by-phone number
   - the street the zone is on
   - which side of the street
   - the cross street the zone begins at
   - the cross street the zone ends at
   - the number of parking spaces
   - the rate or rate class, if held in the same record

   For reference, this is the same information the City of New York publishes as
   *"Parking Meters - ParkNYC Block Faces"* (NYC Open Data, 11,185 records), which
   carries the fields `pay_by_cel`, `on_street`, `side_of_st`, `from_stree`,
   `to_street` and a centreline geometry. **I am not asking you to create a
   dataset in that format** — I am asking for whatever equivalent record you
   already hold, in its existing format, including a spreadsheet or database
   export.

2. **Parking meter and pay station asset locations.** Any list or export of meter
   or multi-space pay station assets with asset identifier, street address or
   coordinates, and associated zone number.

3. **Any geospatial version of the above**, if one exists — shapefile,
   geodatabase, KML, or GeoJSON of on-street parking spaces, zones, or meter
   locations. If no GIS version exists, items 1 and 2 are what I am after and I
   do not need you to produce one.

4. **The sign inventory,** if maintained: parking regulation sign or zone plaque
   locations with the zone number displayed.

5. **The current agreements** between the Miami Parking Authority and ParkMobile
   LLC and/or PayByPhone Technologies for mobile parking payment services,
   including any exhibit, appendix, schedule, or data specification describing the
   parking zone numbering scheme or how zone identifiers are communicated to the
   vendor.

I request these in electronic format, delivered by email or download link, in the
format in which they are ordinarily maintained.

If any portion is exempt or confidential, please produce the remainder and state
in writing the specific statutory exemption relied on for each withheld portion,
as required by § 119.07(1)(e)–(f), Fla. Stat.

If fulfilling this request will require extensive use of information technology
resources or clerical assistance such that a special service charge applies under
§ 119.07(4)(d), please provide a written estimate before incurring the cost, and I
will confirm or narrow the request. I am equally happy to narrow the geographic
scope — for example to the downtown core only — if that materially reduces the
burden.

Please acknowledge receipt and provide an estimated response date.

Thank you,

[NAME]
[EMAIL]
[PHONE]

---

## If they push back

- **"We don't have it in that format."** Ch. 119 requires production of existing
  records, not creation of new ones — but an export from a system they already
  run is generally production, not creation. Ask what format it *is* held in and
  request that.
- **"It's a trade secret."** Plausible only for item 5, and even then usually only
  for pricing terms. Ask them to redact the exempt portions and produce the rest.
- **Delay.** Florida requires a response in a reasonable time, with no fixed
  deadline, but unjustified delay is itself a violation. A polite follow-up citing
  the acknowledgement request usually moves it.
- **A fee estimate that looks high.** Ask for the breakdown; the statute allows
  only actual duplication cost plus, for extensive requests, the labour rate of
  the lowest-paid capable employee.

**None of this is legal advice** — it is how the statute reads and how these
requests usually go. If it becomes adversarial, a Florida public-records attorney
is cheap relative to the time already spent.
