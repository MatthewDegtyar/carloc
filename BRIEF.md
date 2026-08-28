# Parking enforcement from a dashcam — a plain-language brief

*For Al Lehman. What this is, whether it works, and what it's worth.*

## The idea in one line

A car with a camera drives down a street; software finds every parked car, drops
each one on a map, and checks it against ParkMobile's paid-zone database — so you
can tell paying customers from evaders **without ever reading a licence plate**.

## Why "without the plate" matters

The instinct is to read plates. You don't need to, and not needing to is the whole
advantage. ParkMobile already knows which *zones* are paid and, through its own
app's back-end, where those zones are. So the only thing missing is **where each
parked car physically is**. Get that accurately and you cross-reference location
against ParkMobile's records — no plate, far less privacy and legal baggage, and a
system a patrol vehicle can run on a normal drive-by.

## Does it work? Yes — with one honest caveat

We built and tested the full chain on real Miami dashcam footage. Three things are
solid:

1. **Finding the cars.** The software reliably detects parked cars from a moving
   camera — both sides of the street, correctly counted (no double-counting the
   same car as it drives past), even labelling colour and type so a car can be
   recognised again on a later pass. On a clean block it counted 20 parked cars
   exactly, atomically.

2. **The paid-zone check.** We reverse-engineered ParkMobile's zone geometry and
   built the "is this car in a paid zone?" verdict, with a measured margin of
   error, and it correctly flags residential streets where the approach *can't*
   work (you can't tell a resident from a violator there).

3. **Overstay.** Driving the same block twice, 30 minutes apart, the system
   matches the cars present both times — i.e. it catches the ones that overstayed.
   We demonstrated this on two genuine passes of the same street.

**The one caveat is *precise location*.** Putting a car on the map to the exact
metre needs a reference the street can offer. Where there are readable street-name
signs, we anchor to them and hit **metre accuracy** — cars land on individual
parking spots. Where there aren't (small neighbourhood streets), we get the count
and the block right but the exact position can drift by up to a block.

**The fix for that is trivial and boring: a GPS feed on the vehicle.** Every
enforcement vehicle already has one. With GPS, the "up to a block" caveat
disappears everywhere and the whole thing is metre-accurate by default. Everything
we built is what you can squeeze out of *just the video* — GPS makes it easy.

## What we deliberately proved *doesn't* work

So you know the limits are real and tested, not glossed over:

- **Drone/aerial-only** imagery can't resolve individual cars well enough.
- **Existing public traffic cameras** don't overlap street parking — of 383 city
  cameras, exactly one sits near a paid kerb, and it watches a highway. You can't
  piggyback on infrastructure; you need the drive-by.
- **Matching video to public street-imagery** (Google/Mapillary) to get location
  without GPS is unreliable in fast-changing neighbourhoods. GPS is the answer.

## The bottom line

The hard, interesting part — turning ordinary dashcam video into an accurate,
plate-free, mapped inventory of who's parked where and who's overstaying — works.
It's demonstrated end to end on real footage. The last mile (exact position on
featureless streets) is solved the moment the capture vehicle shares its GPS,
which every candidate vehicle already has.

It's a licensable capability for ParkMobile or any operator whose paid parking is
the default: give an existing enforcement drive-by the ability to see, count, and
time-check kerbside parking against the paid-zone database, hands-free and
plate-free.

---

*Technical detail: [`README.md`](README.md). Full engineering decision log,
including every dead end: [`research/FINDINGS.md`](research/FINDINGS.md).*
