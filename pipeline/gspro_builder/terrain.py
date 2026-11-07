"""DEM -> Unity terrain heightmap (16-bit RAW) + metadata.

Conventions (documented in terrain.json so every consumer agrees):
  * RAW is little-endian uint16 ("Windows" byte order in Unity's import dialog).
  * N x N samples, N in {513, 1025, 2049, 4097}; samples span the tile edge to edge.
  * Row 0 is the SOUTH edge (z = -size/2), matching TerrainData.SetHeights([z, x]).
    In Unity's "Import Raw" dialog that means *Flip Vertically = off*.
  * height01 = (elev - min_elev) / height_range; Unity terrain size.y = height_range.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .geo import CourseFrame


@dataclass
class TerrainInfo:
    resolution: int
    size_m: float
    min_elev_m: float
    max_elev_m: float
    height_range_m: float
    center_lon: float
    center_lat: float
    utm_epsg: int
    raw_file: str
    byte_order: str = "little"
    row_order: str = "south_to_north"
    unity_flip_vertically: bool = False


def bilinear(dem: np.ndarray, col: np.ndarray, row: np.ndarray) -> np.ndarray:
    h, w = dem.shape
    col = np.clip(col, 0, w - 1)
    row = np.clip(row, 0, h - 1)
    c0 = np.floor(col).astype(int)
    r0 = np.floor(row).astype(int)
    c1, r1 = np.minimum(c0 + 1, w - 1), np.minimum(r0 + 1, h - 1)
    fc, fr = col - c0, row - r0
    top = dem[r0, c0] * (1 - fc) + dem[r0, c1] * fc
    bot = dem[r1, c0] * (1 - fc) + dem[r1, c1] * fc
    return top * (1 - fr) + bot * fr


class DEM:
    """North-up DEM covering exactly the course frame, at res_m per pixel."""

    def __init__(self, data: np.ndarray, frame: CourseFrame):
        if data.shape[0] != data.shape[1]:
            raise ValueError("DEM must be square (covering the course frame)")
        self.data = data.astype(np.float32)
        self.frame = frame
        self.res_m = frame.size_m / data.shape[0]

    def sample(self, x, z):
        """Elevation at local coordinates (scalars or arrays), bilinear."""
        h = self.frame.size_m / 2
        x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
        col = (x + h) / self.res_m - 0.5
        row = (h - z) / self.res_m - 0.5
        return bilinear(self.data, col, row)

    def sample_smooth(self, x, z):
        """Cubic-spline elevation (no facets between DEM cells) for heightmaps and meshes."""
        from scipy import ndimage
        if getattr(self, "_coef", None) is None:
            self._coef = ndimage.spline_filter(self.data.astype(np.float64), order=3, mode="nearest")
        h = self.frame.size_m / 2
        x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
        col = (x + h) / self.res_m - 0.5
        row = (h - z) / self.res_m - 0.5
        out = ndimage.map_coordinates(self._coef, [row.ravel(), col.ravel()], order=3,
                                      mode="nearest", prefilter=False)
        return out.reshape(np.shape(x))

    def heightmap(self, n: int) -> np.ndarray:
        """N x N float elevations, row 0 = south."""
        h = self.frame.size_m / 2
        coords = -h + np.arange(n) * (self.frame.size_m / (n - 1))
        xx, zz = np.meshgrid(coords, coords)       # zz[i, :] = south..north
        return self.sample_smooth(xx, zz)


def write_heightmap(dem: DEM, n: int, out_dir: Path, name: str = "heightmap") -> TerrainInfo:
    out_dir.mkdir(parents=True, exist_ok=True)
    hm = dem.heightmap(n)
    lo, hi = float(hm.min()), float(hm.max())
    rng = max(hi - lo, 0.01)
    q = np.round((hm - lo) / rng * 65535.0).clip(0, 65535).astype("<u2")
    raw = out_dir / f"{name}_{n}.raw"
    q.tofile(raw)
    info = TerrainInfo(n, dem.frame.size_m, lo, hi, rng, dem.frame.center_lon,
                       dem.frame.center_lat, dem.frame.epsg, raw.name)
    (out_dir / "terrain.json").write_text(json.dumps(asdict(info), indent=2))
    # 8-bit preview with simple hillshade for a quick visual check
    _hillshade_png(hm[::-1], dem.frame.size_m / (n - 1), out_dir / "heightmap_preview.png")
    return info


def _hillshade_png(hm_north_up: np.ndarray, cell: float, path: Path) -> None:
    from PIL import Image
    gz, gx = np.gradient(hm_north_up, cell)
    slope = np.arctan(np.hypot(gx, gz))
    aspect = np.arctan2(-gx, gz)
    az, alt = np.radians(315), np.radians(45)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    img = (np.clip(shade, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(img)
    if im.width > 2048:
        im = im.resize((2048, 2048))
    im.save(path)


def read_raw(path: Path, n: int) -> np.ndarray:
    return np.fromfile(path, dtype="<u2").reshape(n, n)
