"""Building footprints: Microsoft Global ML Building Footprints (open data, ODbL).

The dataset is split into zoom-9 quadkey tiles listed in dataset-links.csv; each
tile is a gzipped GeoJSON-lines file. Each footprint carries an estimated height
(metres, -1 when unknown). We download the tiles covering the course frame and
keep the footprints whose centre falls inside it.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math

import requests

from ..geo import CourseFrame

LINKS = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"


def quadkey(lat: float, lon: float, zoom: int = 9) -> str:
    x = (lon + 180) / 360
    s = math.sin(math.radians(lat))
    y = 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)
    n = 2 ** zoom
    tx, ty = int(x * n), int(y * n)
    key = ""
    for i in range(zoom, 0, -1):
        m = 1 << (i - 1)
        key += str((1 if tx & m else 0) + (2 if ty & m else 0))
    return key


def fetch_ms_buildings(frame: CourseFrame, region: str = "UnitedStates", log=print) -> dict:
    h = frame.size_m / 2
    corners = [frame.to_lonlat(x, z) for x in (-h, 0, h) for z in (-h, 0, h)]
    keys = {quadkey(lat, lon) for lon, lat in corners}
    rows = list(csv.DictReader(io.StringIO(requests.get(LINKS, timeout=120).text)))
    urls = [r["Url"] for r in rows if r["QuadKey"] in keys and r["Location"] == region]
    if not urls:
        raise RuntimeError(f"no {region} footprint tiles for quadkeys {sorted(keys)}")
    lons = [c[0] for c in corners]; lats = [c[1] for c in corners]
    feats = []
    for u in urls:
        log(f"  footprint tile {u.split('quadkey=')[1].split('/')[0]}")
        raw = gzip.decompress(requests.get(u, timeout=600).content).decode()
        for line in raw.splitlines():
            try:
                f = json.loads(line)
            except ValueError:
                continue
            ring = f["geometry"]["coordinates"][0]
            lo = sum(p[0] for p in ring) / len(ring); la = sum(p[1] for p in ring) / len(ring)
            if min(lons) <= lo <= max(lons) and min(lats) <= la <= max(lats):
                feats.append(f)
    return {"type": "FeatureCollection", "features": feats,
            "source": "Microsoft Global ML Building Footprints (ODbL)"}
