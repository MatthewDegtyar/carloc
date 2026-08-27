"""Build the sightings log from the SE 6th Street pass and exercise its query."""
import json
import math
from datetime import datetime, timedelta

from carloc.sightings import Sighting, SightingLog, synthetic_ts, zone_for

# Fabricated clock. The video has no chronology, so this epoch is invented; the
# source's own creation time is used as the base so the invention is at least
# anchored to something real. Every record is stamped synthetic=True.
EPOCH = datetime(2025, 6, 2, 9, 0, 0)          # 09:00:00, synthetic
SOURCE = "Magic City Driving Tour 4K @ SE 6th St eastbound"

with open("reports/se6_cars.json") as fh:
    cars = json.load(fh)

# ParkMobile anchors we actually hold geometry for (downtown + grove); none are
# in Brickell, so zone_for will correctly return None here.
with open("reports/miami_zones.geojson") as fh:
    Z = json.load(fh)
anchors = []
for f in Z["features"]:
    if f["geometry"]["type"] != "Point":
        continue
    lo, la = f["geometry"]["coordinates"]
    anchors.append((la, lo, str(f["properties"].get("zone")
                                or f["properties"].get("signage_code"))))

log = SightingLog()
for i, c in enumerate(sorted(cars, key=lambda c: c["video_t"])):
    log.add(Sighting(
        sighting_id=f"SE6-{i:03d}",
        ts=synthetic_ts(EPOCH, c["video_t"]),
        video_t=c["video_t"], lat=c["lat"], lon=c["lon"],
        heading_deg=c["heading_deg"],
        sigma_along_m=c["sigma_along_m"], sigma_cross_m=c["sigma_cross_m"],
        vehicle_class=c["vehicle_class"], color=c["color"],
        size_px=c["size_px"], color_rgb=tuple(c["color_rgb"]),
        zone=zone_for(c["lat"], c["lon"], anchors), source=SOURCE))

log.to_csv("reports/sightings.csv")
log.to_json("reports/sightings.json")
print(f"logged {len(log.sightings)} sightings")
span = (log.sightings[-1].ts - log.sightings[0].ts).total_seconds()
print(f"synthetic time span: {log.sightings[0].ts:%H:%M:%S} .. {log.sightings[-1].ts:%H:%M:%S} "
      f"({span:.0f} s)")
print(f"zones matched: {sum(1 for s in log.sightings if s.zone)} "
      f"(expected 0 -- no Brickell geometry on file)")

print("\n--- QUERY 1: was a car at a known parked spot around 09:07:30?")
hit_car = sorted(cars, key=lambda c: c["sigma_along_m"])[0]      # best-localised car
when = EPOCH + timedelta(seconds=hit_car["video_t"])
res = log.near(hit_car["lat"], hit_car["lon"], when=when, window_s=90, gate=2.0)
print(f"    query point {hit_car['lat']:.6f}, {hit_car['lon']:.6f} at {when:%H:%M:%S}")
for s, d in res[:4]:
    print(f"    HIT  {s.sighting_id}  {s.color} {s.vehicle_class}  seen {s.ts:%H:%M:%S}  "
          f"maha={d:.2f}sigma  (along±{s.sigma_along_m:.0f}m, cross±{s.sigma_cross_m:.0f}m)")
if not res:
    print("    (no car here)")

print("\n--- QUERY 2: was a car out in the traffic lane? (cross-track, should miss)")
mid = log.sightings[len(log.sightings) // 2]
th = math.radians(mid.heading_deg)
# 6 m to the RIGHT of travel = out into the carriageway, away from the north kerb
LANE_M = 6.0
q_lat = mid.lat + (-LANE_M * math.sin(th)) / 110540.0
q_lon = mid.lon + (LANE_M * math.cos(th)) / (111320.0 * math.cos(math.radians(mid.lat)))
res2 = log.near(q_lat, q_lon, gate=2.0)
d_self = mid.mahalanobis(q_lat, q_lon)
print(f"    a point {LANE_M:.0f} m cross-track from {mid.sighting_id} sits at "
      f"{d_self:.1f} sigma -> HITs: {len(res2)}")

print("\n--- OVERSTAY check on this single pass (should be 0 candidates)")
cand = log.overstay(min_gap_s=1200, gate=2.0)
print(f"    candidates: {len(cand)}  -- correct: every car seen exactly once in one pass")
