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
OSM_API_URL = "https://api.openstreetmap.org/api/0.6/map.json"


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
    try:
        r = requests.post(OVERPASS_URL, data={"data": q}, timeout=180, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        return overpass_to_geojson(r.json())
    except requests.RequestException as exc:
        print(f"[gspro-builder]   Overpass unavailable ({type(exc).__name__}); using the OSM API instead", flush=True)
        return fetch_osm_api((s, w, n, e))


def _wanted(tags: dict) -> bool:
    return ("golf" in tags or tags.get("natural") in ("water", "wood") or tags.get("landuse") == "forest")


def fetch_osm_api(bbox_snwe, depth: int = 0) -> dict:
    """Fallback: the main OSM API `map` call (bbox <= 0.25 deg2, <= 50k nodes; split if
    refused), filtered to golf/water/wood features and converted to Overpass-style geometry."""
    s, w, n, e = bbox_snwe
    r = requests.get(OSM_API_URL, params={"bbox": f"{w},{s},{e},{n}"}, timeout=180,
                     headers={"User-Agent": USER_AGENT})
    if r.status_code == 400 and depth < 3:          # too many nodes: quarter the box
        mlat, mlon = (s + n) / 2, (w + e) / 2
        feats, seen = [], set()
        for q in ((s, w, mlat, mlon), (s, mlon, mlat, e), (mlat, w, n, mlon), (mlat, mlon, n, e)):
            for f in fetch_osm_api(q, depth + 1)["features"]:
                if f["properties"]["osm_id"] not in seen:
                    seen.add(f["properties"]["osm_id"])
                    feats.append(f)
        return {"type": "FeatureCollection", "features": feats}
    r.raise_for_status()
    els = r.json()["elements"]
    nodes = {el["id"]: el for el in els if el["type"] == "node"}
    ways = {el["id"]: el for el in els if el["type"] == "way"}

    def geom(way):
        return [{"lon": nodes[i]["lon"], "lat": nodes[i]["lat"]} for i in way["nodes"] if i in nodes]

    out = []
    for el in els:
        tags = el.get("tags", {})
        if not _wanted(tags):
            continue
        if el["type"] == "node":
            out.append(el)
        elif el["type"] == "way":
            out.append({**el, "geometry": geom(el)})
        elif el["type"] == "relation":
            members = [{"type": m["type"], "role": m.get("role", ""), "geometry": geom(ways[m["ref"]])}
                       for m in el.get("members", []) if m["type"] == "way" and m["ref"] in ways]
            out.append({**el, "members": members})
    return overpass_to_geojson({"elements": out})


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
