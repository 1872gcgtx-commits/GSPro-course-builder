"""Terrain grid mesh export (Wavefront OBJ) for Blender / OPCD conforming.

Right-handed Y-up metres: x east, y above terrain base, z SOUTH (see
course_meshes.py for why). Blender's default OBJ import puts north on +Y.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .terrain import DEM


def write_obj(dem: DEM, path: Path, base_elev: float, cell_m: float = 2.0) -> Path:
    size = dem.frame.size_m
    n = int(round(size / cell_m)) + 1
    h = size / 2
    coords = -h + np.arange(n) * (size / (n - 1))
    xx, zz = np.meshgrid(coords, coords)
    yy = dem.sample_smooth(xx, zz) - base_elev
    path.parent.mkdir(parents=True, exist_ok=True)
    uv = (coords + h) / size
    with open(path, "w") as f:
        f.write(f"# terrain grid {n}x{n}, {cell_m} m cells, right-handed Y-up, x east, z south\no terrain\n")
        verts = np.stack([xx.ravel(), yy.ravel(), -zz.ravel()], 1)
        np.savetxt(f, verts, fmt="v %.3f %.3f %.3f")
        uu, vv = np.meshgrid(uv, uv)
        np.savetxt(f, np.stack([uu.ravel(), vv.ravel()], 1), fmt="vt %.6f %.6f")
        idx = np.arange(n * n).reshape(n, n) + 1
        a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
        c, d = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
        # counter-clockwise in plan (rows run south->north, cols west->east) = +Y normals
        faces = np.stack([a, a, b, b, c, c, a, a, c, c, d, d], 1).reshape(-1, 6)
        np.savetxt(f, faces, fmt="f %d/%d %d/%d %d/%d")
    return path
