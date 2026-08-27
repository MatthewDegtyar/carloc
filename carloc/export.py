"""Export in formats a mapping tool cannot misread.

GeoJSON mandates `[longitude, latitude]`. Half the tools that consume it assume
`[latitude, longitude]`, and because Miami's longitude is -80.19 and its latitude
is 25.77, a reader with the axes swapped does not produce a visibly silly answer
-- it produces **Antarctica**, silently, at a plausible-looking coordinate.

That is worth defending against rather than documenting, because the failure has
no symptom until someone looks at the map.

So this writes:

* **KML**, which Google My Maps and Earth parse natively and unambiguously
* **CSV with named `latitude` / `longitude` columns**, which cannot be transposed
  because the columns say which is which
* **WKT** in the CSV for the polygon, for tools that want the full shape

Between them there is no configuration to get wrong.
"""

from __future__ import annotations

import csv
from pathlib import Path
from xml.sax.saxutils import escape

KML_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>{name}</name>
"""

KML_STYLE = """<Style id="{sid}">
  <LineStyle><color>{line}</color><width>2</width></LineStyle>
  <PolyStyle><color>{fill}</color></PolyStyle>
</Style>
"""

KML_FOOTER = "</Document>\n</kml>\n"


def _abgr(hex_rgb: str, alpha: str = "80") -> str:
    """#RRGGBB -> KML's aabbggrr, which is byte-reversed from the usual order."""
    h = hex_rgb.lstrip("#")
    return f"{alpha}{h[4:6]}{h[2:4]}{h[0:2]}".lower()


def polygons_to_kml(features: list[dict], path: str | Path, name: str = "carloc",
                    colour: str = "#4ec9b0") -> Path:
    """`features` = [{"name":…, "polygon":[(lon,lat)…], "props":{…}}, …]"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [KML_HEADER.format(name=escape(name)),
           KML_STYLE.format(sid="box", line=_abgr(colour, "ff"), fill=_abgr(colour, "66"))]

    for feature in features:
        props = feature.get("props") or {}
        description = "<![CDATA[" + "<br/>".join(
            f"<b>{escape(str(k))}</b>: {escape(str(v))}" for k, v in props.items()) + "]]>"
        ring = " ".join(f"{lon:.7f},{lat:.7f},0" for lon, lat in feature["polygon"])
        out.append(
            "<Placemark>\n"
            f"  <name>{escape(str(feature.get('name', '')))}</name>\n"
            f"  <description>{description}</description>\n"
            "  <styleUrl>#box</styleUrl>\n"
            "  <Polygon><outerBoundaryIs><LinearRing>\n"
            f"    <coordinates>{ring}</coordinates>\n"
            "  </LinearRing></outerBoundaryIs></Polygon>\n"
            "</Placemark>\n"
        )
    out.append(KML_FOOTER)
    path.write_text("".join(out))
    return path


def points_to_kml(points: list[dict], path: str | Path, name: str = "carloc") -> Path:
    """`points` = [{"name":…, "lon":…, "lat":…, "props":{…}}, …]"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [KML_HEADER.format(name=escape(name))]
    for point in points:
        props = point.get("props") or {}
        description = "<![CDATA[" + "<br/>".join(
            f"<b>{escape(str(k))}</b>: {escape(str(v))}" for k, v in props.items()) + "]]>"
        out.append(
            "<Placemark>\n"
            f"  <name>{escape(str(point.get('name', '')))}</name>\n"
            f"  <description>{description}</description>\n"
            f"  <Point><coordinates>{point['lon']:.7f},{point['lat']:.7f},0</coordinates></Point>\n"
            "</Placemark>\n"
        )
    out.append(KML_FOOTER)
    path.write_text("".join(out))
    return path


def polygons_to_csv(features: list[dict], path: str | Path) -> Path:
    """CSV with named lat/lon columns plus WKT.

    The column names are the defence: a tool that asks which column is latitude
    cannot be told the wrong one by accident.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for feature in features:
        for k in (feature.get("props") or {}):
            if k not in keys:
                keys.append(k)

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "latitude", "longitude", "wkt", *keys])
        for feature in features:
            ring = feature["polygon"]
            lat = sum(p[1] for p in ring[:4]) / 4
            lon = sum(p[0] for p in ring[:4]) / 4
            wkt = "POLYGON((" + ", ".join(f"{x:.7f} {y:.7f}" for x, y in ring) + "))"
            props = feature.get("props") or {}
            writer.writerow([feature.get("name", ""), f"{lat:.7f}", f"{lon:.7f}", wkt,
                             *[props.get(k, "") for k in keys]])
    return path


def boxes_to_all(boxes, stem: str | Path, name: str = "carloc parking lanes") -> dict:
    """Write KML, CSV and GeoJSON for a list of ZoneBox, and say what each is for."""
    import json

    from carloc.zonebox import to_geojson

    stem = Path(stem)
    features = [{
        "name": f"zone {b.zone} · {b.street} ({b.side})",
        "polygon": b.polygon,
        "props": {"zone": b.zone, "street": b.street, "side": b.side,
                  "length_m": b.length_m, "est_spaces": b.spaces,
                  "ambiguous_band_m": round(b.ambiguous_band_m, 2),
                  "lane_offset": "ASSUMED 3.3-5.9 m from centreline"},
    } for b in boxes]

    written = {
        "kml": polygons_to_kml(features, stem.with_suffix(".kml"), name=name),
        "csv": polygons_to_csv(features, stem.with_suffix(".csv")),
    }
    geojson_path = stem.with_suffix(".geojson")
    geojson_path.write_text(json.dumps(to_geojson(boxes)))
    written["geojson"] = geojson_path
    return written
