"""Conformed course-surface meshes (greens, tees, fairways, bunkers, water planes).

Each layout polygon becomes a triangle mesh draped on the *graded* terrain:
boundary densified + interior grid points -> Delaunay -> keep triangles inside
the polygon (and outside its holes). Grid spacing is finer where it matters
(greens 0.5 m, fairways 1 m). Water bodies become flat planes at the graded
water level.

Object names follow the OPCD material-token convention, Hole##_<Surface>
(e.g. Hole07_Green, Hole07_Bunker_2, Hole99_WaterPlane_1), so OPCD's
"Set Mesh Materials" step can recognise them.

Coordinates: internally x east, y up, z north (the Unity frame). The OBJ file is
written right-handed (x east, y up, z SOUTH, CCW faces) so that Blender's default
OBJ import (forward -Z, up Y) puts north on Blender +Y. Unity's model importer
mirrors X when it reads OBJ/FBX, so the Unity importer places the model rotated
180 deg about Y, which restores east/north exactly. water.json is written in
Unity's own frame with Unity (clockwise-from-above) winding.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .layout import Layout, Surface
from .terrain import DEM

SPACING_M = {"green": 0.5, "tee": 0.5, "bunker": 0.5, "fairway": 1.0, "water": 2.0}
TOKEN = {"green": "Green", "tee": "Tee", "bunker": "Bunker", "fairway": "Fairway", "water": "WaterPlane"}


def _densify(ring, step):
    pts = []
    ring = list(ring)
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    for i in range(len(ring)):
        a, b = np.array(ring[i]), np.array(ring[(i + 1) % len(ring)])
        k = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        for t in range(k):
            pts.append(a + (b - a) * t / k)
    return np.array(pts)


def points_in_ring(pts: np.ndarray, ring) -> np.ndarray:
    r = np.asarray(ring, dtype=np.float64)
    x, z = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    j = len(r) - 1
    for i in range(len(r)):
        xi, zi = r[i]
        xj, zj = r[j]
        cross = ((zi > z) != (zj > z)) & (x < (xj - xi) * (z - zi) / ((zj - zi) + 1e-12) + xi)
        inside ^= cross
        j = i
    return inside


def triangulate(surface: Surface, spacing: float):
    """2-D triangulation of a polygon with holes. Returns (pts[N,2], tris[M,3])."""
    from scipy.spatial import Delaunay
    step = spacing * 0.5
    boundary = np.vstack([_densify(r, step) for r in surface.rings])
    outer = np.asarray(surface.rings[0])
    xs = np.arange(outer[:, 0].min(), outer[:, 0].max() + spacing, spacing)
    zs = np.arange(outer[:, 1].min(), outer[:, 1].max() + spacing, spacing)
    gx, gz = np.meshgrid(xs, zs)
    grid = np.stack([gx.ravel(), gz.ravel()], 1)
    keep = points_in_ring(grid, surface.rings[0])
    for hole in surface.rings[1:]:
        keep &= ~points_in_ring(grid, hole)
    grid = grid[keep]
    # drop interior points too close to the boundary (sliver triangles)
    if len(grid):
        from scipy.spatial import cKDTree
        d, _ = cKDTree(boundary).query(grid)
        grid = grid[d > spacing * 0.35]
    pts = np.vstack([boundary, grid])
    pts = np.unique(np.round(pts, 4), axis=0)
    if len(pts) < 3:
        return pts, np.zeros((0, 3), int)
    tri = Delaunay(pts)
    c = pts[tri.simplices].mean(1)
    ok = points_in_ring(c, surface.rings[0])
    for hole in surface.rings[1:]:
        ok &= ~points_in_ring(c, hole)
    # drop slivers (collinear boundary points) - they flip under rounding and shade badly
    t = tri.simplices
    a, b, cc = pts[t[:, 0]], pts[t[:, 1]], pts[t[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cc[:, 0] - a[:, 0]))
    ok &= area > 0.02 * spacing * spacing
    return pts, t[ok]


def _orient_ccw(pts, tris):
    """Make triangles counter-clockwise in plan (x east, z north)."""
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    tris = tris.copy()
    tris[cross < 0] = tris[cross < 0][:, ::-1]
    return tris


def _sample_level(water_level: np.ndarray, frame, x, z) -> np.ndarray:
    """Water surface at points (nearest wet cell, so edge vertices don't fall off)."""
    from scipy import ndimage
    n = water_level.shape[0]
    res = frame.size_m / n
    h = frame.size_m / 2
    col = np.clip(((x + h) / res).astype(int), 0, n - 1)
    row = np.clip(((h - z) / res).astype(int), 0, n - 1)
    r0, r1, c0, c1 = max(row.min() - 5, 0), min(row.max() + 6, n), max(col.min() - 5, 0), min(col.max() + 6, n)
    win = water_level[r0:r1, c0:c1]
    wet = np.isfinite(win)
    if not wet.any():
        return np.full(len(x), np.nan)
    idx = ndimage.distance_transform_edt(~wet, return_distances=False, return_indices=True)
    filled = win[tuple(idx)]
    return filled[row - r0, col - c0]


def build_meshes(layout: Layout, dem: DEM, base_elev: float, water_bodies: list[dict],
                 offset_m: float = 0.0, water_level: np.ndarray | None = None) -> list[dict]:
    objs = []
    counters: dict[str, int] = {}
    water_ids = {id(w["surface"]): w["level_m"] for w in water_bodies}
    for s in layout.surfaces:
        if s.kind not in TOKEN:
            continue
        pts, tris = triangulate(s, SPACING_M[s.kind])
        if len(tris) == 0:
            continue
        if s.kind == "water":
            level = water_ids.get(id(s))
            if level is None:
                continue
            if water_level is not None:
                y = _sample_level(water_level, dem.frame, pts[:, 0], pts[:, 1]) - base_elev
                y = np.where(np.isfinite(y), y, level - base_elev)
            else:
                y = np.full(len(pts), level - base_elev)
        else:
            y = dem.sample_smooth(pts[:, 0], pts[:, 1]) - base_elev + offset_m
        hole = s.hole if s.hole is not None else 99
        key = f"Hole{hole:02d}_{TOKEN[s.kind]}"
        counters[key] = counters.get(key, 0) + 1
        name = key if counters[key] == 1 else f"{key}_{counters[key]}"
        objs.append({"name": name, "kind": s.kind, "hole": hole,
                     "vertices": np.column_stack([pts[:, 0], y, pts[:, 1]]),
                     "triangles": _orient_ccw(pts, tris)})
    return objs


def write_obj(objs: list[dict], path: Path, size_m: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = size_m / 2
    with open(path, "w") as f:
        f.write("# GSPro course surface meshes: metres, right-handed Y-up, x east, z SOUTH\n")
        base = 1
        for o in objs:
            v = o["vertices"] * np.array([1.0, 1.0, -1.0])   # z north -> z south
            f.write(f"o {o['name']}\ng {o['name']}\nusemtl {o['kind']}\n")
            np.savetxt(f, v, fmt="v %.3f %.3f %.3f")
            np.savetxt(f, np.column_stack([(v[:, 0] + h) / size_m, (h - v[:, 2]) / size_m]), fmt="vt %.6f %.6f")
            t = o["triangles"] + base
            np.savetxt(f, np.column_stack([t[:, 0], t[:, 0], t[:, 1], t[:, 1], t[:, 2], t[:, 2]]),
                       fmt="f %d/%d %d/%d %d/%d")
            base += len(v)
    return path


def write_water_json(objs: list[dict], water_bodies: list[dict], base_elev: float, path: Path) -> None:
    bodies = []
    for o in (o for o in objs if o["kind"] == "water"):
        v = o["vertices"]
        bodies.append({"name": o["name"], "hole": o["hole"], "level_y": round(float(v[0, 1]), 3),
                       "vertices": [{"x": round(float(a), 3), "y": round(float(b), 3), "z": round(float(c), 3)}
                                    for a, b, c in v],
                       "triangles": o["triangles"][:, ::-1].ravel().tolist()})
    path.write_text(json.dumps({"base_elev_m": base_elev, "bodies": bodies}))
