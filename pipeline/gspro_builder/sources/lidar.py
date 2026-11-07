"""LiDAR point clouds -> DTM (bare earth), DSM, canopy height, class masks.

Two ways in:
  lidar_ept   : streams the USGS 3DEP point cloud from the public Entwine (EPT)
                bucket on AWS (usgs-lidar-public, EPSG:3857). No key; only the
                octree nodes covering the course are downloaded.
  lidar_files : .las/.laz tiles you downloaded yourself (apps.nationalmap.gov
                downloader, state portals). Any CRS pyproj understands, incl.
                State Plane in US survey feet.

Why the point cloud rather than the 1 m DEM: we grid ground returns at 0.5 m
(greens, bunkers and tee pads need it), and the first returns give the canopy
height model used to place real trees.

ASPRS classes used: 2 ground, 6 building, 7/18 noise, 9 water.
"""
from __future__ import annotations

import glob
import io
import json
import re
from pathlib import Path

import numpy as np
import requests

from ..geo import CourseFrame
from .arcgis import USER_AGENT

EPT_INDEX = "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson"


class LidarGrid:
    """Streaming accumulator: feed points in UTM metres, get rasters out."""

    def __init__(self, frame: CourseFrame, res_m: float):
        self.frame, self.res = frame, res_m
        self.n = int(round(frame.size_m / res_m))
        self.xmin, _, _, self.ymax = frame.utm_bbox
        shp = (self.n, self.n)
        self.g_sum = np.zeros(shp, np.float64)
        self.g_cnt = np.zeros(shp, np.int32)
        self.dsm = np.full(shp, -np.inf, np.float32)
        self.water = np.zeros(shp, np.int32)
        self.building = np.zeros(shp, np.int32)
        self.points = 0

    def add(self, x, y, z, cls) -> None:
        col = np.floor((x - self.xmin) / self.res).astype(np.int64)
        row = np.floor((self.ymax - y) / self.res).astype(np.int64)
        ok = (col >= 0) & (col < self.n) & (row >= 0) & (row < self.n) & (cls != 7) & (cls != 18)
        col, row, z, cls = col[ok], row[ok], z[ok].astype(np.float64), cls[ok]
        self.points += int(ok.sum())
        idx = row * self.n + col
        g = (cls == 2) | (cls == 8)
        np.add.at(self.g_sum.ravel(), idx[g], z[g])
        np.add.at(self.g_cnt.ravel(), idx[g], 1)
        np.maximum.at(self.dsm.ravel(), idx, z.astype(np.float32))
        np.add.at(self.water.ravel(), idx[cls == 9], 1)
        np.add.at(self.building.ravel(), idx[cls == 6], 1)

    def finalize(self) -> dict[str, np.ndarray]:
        from scipy import ndimage
        if self.g_cnt.sum() == 0:
            raise ValueError("No ground-classified (class 2) LiDAR points inside the course tile")
        dtm = np.where(self.g_cnt > 0, self.g_sum / np.maximum(self.g_cnt, 1), np.nan)
        dtm = fill_gaps(dtm)
        dsm = np.where(np.isfinite(self.dsm), self.dsm, dtm)
        chm = np.clip(dsm - dtm, 0, 80)
        coverage = float((self.g_cnt > 0).mean())
        water = ndimage.binary_closing(self.water > 0, iterations=2)
        building = ndimage.binary_closing(self.building > 0, iterations=1)
        return {"dtm": dtm.astype(np.float32), "dsm": dsm.astype(np.float32), "chm": chm.astype(np.float32),
                "water": water, "building": building, "ground_coverage": np.float32(coverage)}


def fill_gaps(a: np.ndarray, smooth_sigma: float = 2.0) -> np.ndarray:
    """Fill NaN cells (under buildings, water, dense canopy) from the nearest
    valid ground, then smooth only the filled cells so the patches aren't blocky."""
    from scipy import ndimage
    mask = ~np.isfinite(a)
    if not mask.any():
        return a
    if mask.all():
        raise ValueError("raster is entirely empty")
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled = a[tuple(idx)]
    sm = ndimage.gaussian_filter(filled, smooth_sigma)
    return np.where(mask, sm, a)


# ---------------------------------------------------------------- local files
def _z_factor(crs, override: str | None) -> float:
    units = {"m": 1.0, "ft": 0.3048, "usft": 1200 / 3937}
    if override and override != "auto":
        return units[override]
    try:
        vert = [c for c in getattr(crs, "sub_crs_list", []) if c.is_vertical]
        axis = (vert[0] if vert else crs).axis_info[-1 if vert else 0]
        return float(axis.unit_conversion_factor or 1.0)
    except Exception:
        return 1.0


def read_las_files(patterns: list[str], root: Path, frame: CourseFrame, res_m: float,
                   crs_override: str | None = None, z_units: str | None = "auto") -> dict:
    import laspy
    from pyproj import CRS, Transformer
    files = sorted({f for p in patterns for f in glob.glob(str(root / p))})
    if not files:
        raise FileNotFoundError(f"No LiDAR files match {patterns} under {root}")
    grid = LidarGrid(frame, res_m)
    for f in files:
        with laspy.open(f) as reader:
            crs = CRS.from_user_input(crs_override) if crs_override else reader.header.parse_crs()
            if crs is None:
                raise ValueError(f"{f} has no CRS; set [terrain] lidar_crs = \"EPSG:xxxx\"")
            horiz = next((c for c in getattr(crs, "sub_crs_list", []) if not c.is_vertical), crs)
            tr = Transformer.from_crs(horiz, frame.epsg, always_xy=True)
            zf = _z_factor(crs, z_units)
            for pts in reader.chunk_iterator(2_000_000):
                x, y = tr.transform(np.asarray(pts.x), np.asarray(pts.y))
                grid.add(np.asarray(x), np.asarray(y), np.asarray(pts.z) * zf, np.asarray(pts.classification))
        print(f"[gspro-builder]   {Path(f).name}: {grid.points:,} points in tile so far", flush=True)
    return grid.finalize()


# ---------------------------------------------------------------- USGS EPT
def _years(name: str) -> int:
    """Acquisition year from a USGS project name: 'TX_Bell_2011', 'TX_CentralEast_1_A23' (=2023)."""
    ys = [int(y) for y in re.findall(r"(?:19|20)\d\d", name)]
    ys += [2000 + int(y) for y in re.findall(r"_[A-Z](\d\d)$", name)]
    return max(ys) if ys else 0


def _polygons(geom: dict):
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return geom["coordinates"]
    return []


def find_ept_datasets(frame: CourseFrame) -> list[dict]:
    """USGS EPT projects whose real coverage polygon (not just its bounding box)
    contains the whole course tile, newest first."""
    from ..course_meshes import points_in_ring
    h = frame.size_m / 2
    probe = np.array([frame.to_lonlat(x, z) for x in (-h, 0, h) for z in (-h, 0, h)])
    r = requests.get(EPT_INDEX, timeout=120, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    hits = []
    for f in r.json()["features"]:
        inside = np.zeros(len(probe), bool)
        for poly in _polygons(f["geometry"]):
            ring = np.asarray(poly[0])
            if ring[:, 0].min() > probe[:, 0].max() or ring[:, 0].max() < probe[:, 0].min() \
                    or ring[:, 1].min() > probe[:, 1].max() or ring[:, 1].max() < probe[:, 1].min():
                continue
            m = points_in_ring(probe, ring)
            for hole in poly[1:]:
                m &= ~points_in_ring(probe, np.asarray(hole))
            inside |= m
        if inside.all():
            hits.append(f["properties"])
    return sorted(hits, key=lambda p: _years(p["name"]), reverse=True)


def read_ept(frame: CourseFrame, res_m: float, ept_url: str | None = None, log=print) -> dict:
    import laspy
    from pyproj import Transformer
    if not ept_url:
        ds = find_ept_datasets(frame)
        if not ds:
            raise ValueError("No USGS EPT LiDAR dataset fully covers this tile; download .laz "
                             "tiles and use [terrain] source = \"lidar_files\"")
        log(f"  LiDAR dataset: {ds[0]['name']} (candidates: {', '.join(d['name'] for d in ds[:4])})")
        ept_url = ds[0]["url"]
    base = ept_url.rsplit("/", 1)[0]
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    meta = sess.get(ept_url, timeout=60).json()
    b = meta["bounds"]
    to_ept = Transformer.from_crs(frame.epsg, 3857, always_xy=True)
    to_utm = Transformer.from_crs(3857, frame.epsg, always_xy=True)
    xmin, ymin, xmax, ymax = frame.utm_bbox
    cx, cy = to_ept.transform([xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
    qb = (min(cx) - 5, min(cy) - 5, max(cx) + 5, max(cy) + 5)

    def node_bounds(key):
        d, x, y, _ = map(int, key.split("-"))
        span = (b[3] - b[0]) / 2 ** d
        return b[0] + x * span, b[1] + y * span, b[0] + (x + 1) * span, b[1] + (y + 1) * span

    def hit(key):
        nb = node_bounds(key)
        return nb[0] <= qb[2] and nb[2] >= qb[0] and nb[1] <= qb[3] and nb[3] >= qb[1]

    keys, pending = [], ["0-0-0-0"]
    while pending:
        hk = pending.pop()
        h = sess.get(f"{base}/ept-hierarchy/{hk}.json", timeout=60).json()
        for k, cnt in h.items():
            if not hit(k):
                continue
            if cnt == -1:
                pending.append(k)
            elif cnt > 0:
                keys.append(k)
    keys = sorted(set(keys))
    log(f"  downloading {len(keys)} EPT nodes")
    grid = LidarGrid(frame, res_m)
    for i, k in enumerate(keys):
        r = sess.get(f"{base}/ept-data/{k}.laz", timeout=180)
        r.raise_for_status()
        las = laspy.read(io.BytesIO(r.content))
        x, y = to_utm.transform(np.asarray(las.x), np.asarray(las.y))
        grid.add(np.asarray(x), np.asarray(y), np.asarray(las.z), np.asarray(las.classification))
        if (i + 1) % 25 == 0:
            log(f"    {i + 1}/{len(keys)} nodes, {grid.points:,} points")
    return grid.finalize()


def save_lidar(products: dict, data_dir: Path) -> None:
    np.save(data_dir / "dem.npy", products["dtm"])
    np.save(data_dir / "chm.npy", products["chm"])
    np.savez_compressed(data_dir / "lidar_masks.npz", water=products["water"], building=products["building"])
    (data_dir / "lidar.json").write_text(json.dumps({"ground_coverage": float(products["ground_coverage"])}))
