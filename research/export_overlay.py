"""Export the SE 6th Street pass as KML for Google My Maps / Google Earth.

The point is verification on imagery the viewer already trusts: drop this into
mymaps.google.com (or Google Earth) and every parked car lands on Google's own
satellite, so "is that a real parking spot" is answered by eye against a source
that is not ours. Three folders -- the cars (coloured by what colour they were
detected as), the 4 Hz camera track, and the two anchors that pin it.
"""
from __future__ import annotations

import json
from xml.sax.saxutils import escape

SWATCH = {"black": "1a1a1c", "white": "e8e8e8", "silver": "aaacaf",
          "grey": "6e7073", "red": "c0342c", "blue": "3550a0",
          "green": "3aa35a", "tan": "b6a06e"}


def kml_colour(hex_rgb: str) -> str:
    """#RRGGBB -> KML aabbggrr (opaque)."""
    r, g, b = hex_rgb[0:2], hex_rgb[2:4], hex_rgb[4:6]
    return f"ff{b}{g}{r}".lower()


def main():
    with open("reports/se6_cars.json") as fh:
        cars = json.load(fh)
    with open("reports/se6_track.json") as fh:
        track = json.load(fh)
    with open("reports/sightings.json") as fh:
        sights = {s["video_t"]: s for s in json.load(fh)}

    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           '<name>SE 6th Street pass — parked cars</name>']

    # one icon style per detected colour
    for name, hexc in SWATCH.items():
        out.append(
            f'<Style id="c_{name}"><IconStyle><color>{kml_colour(hexc)}</color>'
            '<scale>1.1</scale><Icon><href>http://maps.google.com/mapfiles/kml/'
            'shapes/placemark_circle.png</href></Icon></IconStyle>'
            '<LabelStyle><scale>0.7</scale></LabelStyle></Style>')
    out.append('<Style id="anchor"><IconStyle><color>ff66d1ff</color><scale>1.4</scale>'
               '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/star.png</href>'
               '</Icon></IconStyle></Style>')
    out.append('<Style id="trk"><LineStyle><color>ffffd166</color>'
               '<width>3</width></LineStyle></Style>')

    # cars
    out.append(f'<Folder><name>Parked cars ({len(cars)})</name>')
    for i, c in enumerate(sorted(cars, key=lambda c: c["video_t"])):
        s = sights.get(c["video_t"], {})
        ts = s.get("ts", "")
        desc = (f"id: {c.get('id') or f'SE6-{i:03d}'}<br/>"
                f"synthetic time: {escape(str(ts))}<br/>"
                f"class: {c['vehicle_class']}<br/>colour: {c['color']}<br/>"
                f"tracked across: {c.get('ndet', 0)} frames<br/>"
                f"tracklets: {c.get('n_tracklets', 1)}"
                f"{' (occlusion-split)' if c.get('n_tracklets', 1) > 1 else ''}<br/>"
                f"along-track sigma: {c['sigma_along_m']} m<br/>"
                f"cross-track sigma: {c['sigma_cross_m']} m")
        out.append(
            f'<Placemark><name>{c.get("id") or f"SE6-{i:03d}"} '
            f'· {c["color"]} {c["vehicle_class"]}</name>'
            f'<description><![CDATA[{desc}]]></description>'
            f'<styleUrl>#c_{c["color"]}</styleUrl>'
            f'<Point><coordinates>{c["lon"]:.7f},{c["lat"]:.7f},0</coordinates></Point>'
            '</Placemark>')
    out.append('</Folder>')

    # camera track
    line = " ".join(f'{p["lon"]:.7f},{p["lat"]:.7f},0' for p in track)
    out.append('<Folder><name>Camera track (4 Hz)</name>'
               f'<Placemark><name>SE 6th St pass</name><styleUrl>#trk</styleUrl>'
               f'<LineString><tessellate>1</tessellate><coordinates>{line}'
               '</coordinates></LineString></Placemark></Folder>')

    # anchors
    out.append('<Folder><name>Anchors</name>')
    for lon, lat, lab in [(-80.192180, 25.768365, "SE 1st Ave (t=420)"),
                          (-80.190327, 25.767980, "Brickell Ave turn (t=505)")]:
        out.append(f'<Placemark><name>{lab}</name><styleUrl>#anchor</styleUrl>'
                   f'<Point><coordinates>{lon:.7f},{lat:.7f},0</coordinates></Point></Placemark>')
    out.append('</Folder>')

    out.append('</Document></kml>')
    with open("reports/se6_overlay.kml", "w") as fh:
        fh.write("\n".join(out))
    print(f"wrote reports/se6_overlay.kml ({len(cars)} cars, {len(track)} track pts)")


if __name__ == "__main__":
    main()
