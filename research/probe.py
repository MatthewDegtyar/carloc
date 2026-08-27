"""Probe ParkMobile signage codes to learn Miami's numbering density."""
import json, time, urllib.request, sys

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

def get(url, timeout=20):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": "application/json",
        "Referer": "https://app.parkmobile.io/search"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def resolve(sign):
    """signage code -> (internalZoneCode, locationName) or None"""
    try:
        d = get(f"https://app.parkmobile.io/api/proxy/parkmobileapi/zones/{sign}")
    except Exception:
        return None
    z = (d or {}).get("zones") or []
    if not z:
        return None
    return z[0].get("internalZoneCode"), z[0].get("locationName")

codes = [int(c) for c in sys.argv[1:]] or list(range(40700, 40712))
hits = 0
for c in codes:
    r = resolve(c)
    if r:
        hits += 1
        print(f"  {c}  -> {r[0]:>12}  {r[1]}")
    else:
        print(f"  {c}  -> -")
    time.sleep(0.6)
print(f"\n{hits}/{len(codes)} resolved")
