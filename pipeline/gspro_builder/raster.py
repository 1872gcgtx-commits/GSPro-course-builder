"""Raster helpers shared by grading, surfaces, foliage and holes."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .geo import CourseFrame


def rasterize_instances(surfaces, frame: CourseFrame, n: int) -> np.ndarray:
    """int32 raster (north-up, n x n): 0 = none, i+1 = surfaces[i]. Later surfaces win."""
    img = Image.new("I", (n, n), 0)
    draw = ImageDraw.Draw(img)
    for i, s in enumerate(surfaces):
        px = [frame.local_to_pixel(x, z, n) for x, z in s.rings[0]]
        draw.polygon(px, fill=i + 1)
        for inner in s.rings[1:]:
            draw.polygon([frame.local_to_pixel(x, z, n) for x, z in inner], fill=0)
    return np.asarray(img, dtype=np.int32)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def feather_weight(mask: np.ndarray, feather_px: float) -> np.ndarray:
    """1 inside mask, smooth fall-off to 0 at feather_px outside."""
    from scipy import ndimage
    if feather_px <= 0:
        return mask.astype(np.float32)
    d = ndimage.distance_transform_edt(~mask)
    return (1 - smoothstep(d / feather_px)).astype(np.float32)


def cell_centers(frame: CourseFrame, n: int, rows: slice, cols: slice):
    """Local (x, z) metre coordinates of cell centres for a window of an n x n north-up raster."""
    res = frame.size_m / n
    h = frame.size_m / 2
    r = np.arange(rows.start, rows.stop)
    c = np.arange(cols.start, cols.stop)
    x = -h + (c + 0.5) * res
    z = h - (r + 0.5) * res
    return np.meshgrid(x, z)


def grow_window(sl: tuple[slice, slice], pad: int, n: int) -> tuple[slice, slice]:
    return tuple(slice(max(0, s.start - pad), min(n, s.stop + pad)) for s in sl)


def fit_plane(x, z, y):
    """Least-squares plane y = a + b*x + c*z. Returns (a, b, c)."""
    A = np.stack([np.ones_like(x), x, z], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def slope_grid(dem: np.ndarray, res: float) -> np.ndarray:
    """Slope magnitude (rise/run) per cell."""
    gz, gx = np.gradient(dem, res)
    return np.hypot(gx, gz)
