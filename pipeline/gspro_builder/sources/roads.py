"""Streets from OpenStreetMap (OSM API `map` calls, tiled), used to orient homes:
front door and garage face the nearest street, driveways run out to it."""
from __future__ import annotations

import requests

from ..geo import CourseFrame

API = "https://api.openstreetmap.org/api/0.6/map.json"
ROAD_KINDS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
              "living_street", "service", "road"}


def fetch_roads(frame: CourseFrame, tiles: int = 4, log=print) -> dict:
    h = frame.size_m / 2
    lon0, lat0 = frame.to_lonlat(-h, -h)
    lon1, lat1 = frame.to_lonlat(h, h)
    nodes, ways = {}, {}
    for i in range(tiles):
        for j in range(tiles):
            w = lon0 + (lon1 - lon0) * i / tiles; e = lon0 + (lon1 - lon0) * (i + 1) / tiles
            s = lat0 + (lat1 - lat0) * j / tiles; n = lat0 + (lat1 - lat0) * (j + 1) / tiles
            r = requests.get(API, params={"bbox": f"{w},{s},{e},{n}"}, timeout=180,
                             headers={"User-Agent": "gspro-course-builder"})
            r.raise_for_status()
            for el in r.json().get("elements", []):
                if el["type"] == "node":
                    nodes[el["id"]] = (el["lon"], el["lat"])
                elif el["type"] == "way" and (el.get("tags") or {}).get("highway") in ROAD_KINDS:
                    ways[el["id"]] = el
    feats = []
    for wid, el in ways.items():
        pts = [nodes[n] for n in el["nodes"] if n in nodes]
        if len(pts) >= 2:
            feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": pts},
                          "properties": {"highway": el["tags"]["highway"], "name": el["tags"].get("name"), "osm_id": wid}})
    log(f"  {len(feats)} streets")
    return {"type": "FeatureCollection", "features": feats}
