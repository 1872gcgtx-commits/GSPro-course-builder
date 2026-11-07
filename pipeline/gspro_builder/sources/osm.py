"""Course layout from OpenStreetMap (golf=* tagging) via the Overpass API.

OSM golf tagging (https://wiki.openstreetmap.org/wiki/Key:golf):
  golf=hole (line, tee -> green, tags ref=<hole no>, par=, handicap=)
  golf=tee | fairway | green | bunker | rough | water_hazard |
       lateral_water_hazard | cartpath | driving_range | pin (node)
  natural=water, natural=wood / landuse=forest are also captured.

Data is (c) OpenStreetMap contributors, ODbL. Many local courses are partially
mapped; anything missing you trace yourself in QGIS / geojson.io and save as
courses/<slug>/features.geojson with the same properties.
"""
from __future__ import annotations

import requests

from ..geo import CourseFrame
from .arcgis import USER_AGENT

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def bbox_lonlat(frame: CourseFrame) -> tuple[float, float, float, float]:
    h = frame.size_m / 2
    corners = [frame.to_lonlat(x, z) for x in (-h, h) for z in (-h, h)]
    lons, lats = zip(*corners)
    return min(lats), min(lons), max(lats), max(lons)   # Overpass order: s,w,n,e


def fetch_osm(frame: CourseFrame) -> dict:
    s, w, n, e = bbox_lonlat(frame)
    q = f"""[out:json][timeout:120];
(
  nwr["golf"]({s},{w},{n},{e});
  nwr["natural"="water"]({s},{w},{n},{e});
  nwr["natural"="wood"]({s},{w},{n},{e});
  nwr["landuse"="forest"]({s},{w},{n},{e});
);
out geom;"""
    r = requests.post(OVERPASS_URL, data={"data": q}, timeout=180, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return overpass_to_geojson(r.json())


def _tags_to_props(tags: dict) -> dict:
    props = dict(tags)
    golf = tags.get("golf")
    if golf is None:
        if tags.get("natural") == "water":
            golf = "water_hazard"
        elif tags.get("natural") == "wood" or tags.get("landuse") == "forest":
            golf = "trees"
    props["golf"] = golf
    return props


def overpass_to_geojson(data: dict) -> dict:
    feats = []
    for el in data.get("elements", []):
        props = _tags_to_props(el.get("tags", {}))
        props["osm_id"] = f"{el['type']}/{el['id']}"
        if el["type"] == "node":
            geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif el["type"] == "way" and "geometry" in el:
            coords = [[p["lon"], p["lat"]] for p in el["geometry"]]
            closed = len(coords) > 3 and coords[0] == coords[-1]
            if closed and props["golf"] not in ("hole", "cartpath", "path"):
                geom = {"type": "Polygon", "coordinates": [coords]}
            else:
                geom = {"type": "LineString", "coordinates": coords}
        elif el["type"] == "relation":
            outers = [[[p["lon"], p["lat"]] for p in m["geometry"]]
                      for m in el.get("members", []) if m.get("role") == "outer" and "geometry" in m]
            inners = [[[p["lon"], p["lat"]] for p in m["geometry"]]
                      for m in el.get("members", []) if m.get("role") == "inner" and "geometry" in m]
            if not outers:
                continue
            geom = {"type": "MultiPolygon", "coordinates": [[o] + inners for o in outers]}
        else:
            continue
        feats.append({"type": "Feature", "properties": props, "geometry": geom})
    return {"type": "FeatureCollection", "features": feats}
