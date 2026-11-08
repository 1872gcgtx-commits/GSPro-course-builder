"""Where water collects and flows, from the LiDAR terrain (suggestions for review).

  drainage   D8 flow directions on the depression-filled terrain (2 m grid), flow
             accumulation -> upstream area per cell. Channels draining >= 40 ha become
             creek suggestions (3 / 4.5 / 6 m wide by catchment); smaller swales are reported.
  ponds      LiDAR records a pond's surface as flat ground, so ponds come from the aerial:
             dark, low near-infrared, nothing standing (no LiDAR height), not a roof.
Everything is a suggestion: it goes to the review page as dashed water for the owner
to keep or delete.
"""
from __future__ import annotations

import numpy as np


def drainage(dem_data, frame, cell_m=2.0):
    """(upstream area in ha, depression depth in m, grid res) on a cell_m grid."""
    import pyproj
    from affine import Affine
    from pysheds.grid import Grid
    from pysheds.sview import Raster, ViewFinder
    if not hasattr(np, "in1d"):
        np.in1d = np.isin                     # pysheds 0.5 on numpy 2
    k = max(1, int(round(cell_m / (frame.size_m / dem_data.shape[0]))))
    n = dem_data.shape[0] // k
    D = dem_data[:n * k, :n * k].astype(np.float64).reshape(n, k, n, k).mean((1, 3))
    res = frame.size_m / n; h = frame.size_m / 2
    vf = ViewFinder(affine=Affine(res, 0, -h, 0, -res, h), shape=D.shape, nodata=-9999.0,
                    crs=pyproj.Proj("EPSG:32614"))
    g = Grid(viewfinder=vf)
    pit = g.fill_pits(Raster(D, viewfinder=vf))
    flooded = g.fill_depressions(pit)
    acc = g.accumulation(g.flowdir(g.resolve_flats(flooded)))
    return np.asarray(acc, np.float32) * res * res / 1e4, (np.asarray(flooded) - np.asarray(pit)).astype(np.float32), res


def creek_suggestions(area_ha, res, frame, zone=None, existing_water=None, min_ha=40.0, shift=(0.0, 0.0)):
    """Channel polygons (page/photo frame) for drainage >= min_ha, not already water."""
    import shapely
    from scipy import ndimage
    from shapely.geometry import LineString
    from .paths import _skeleton_lines
    n = area_ha.shape[0]; h = frame.size_m / 2
    m = area_ha >= min_ha
    if zone is not None:
        yy, xx = np.mgrid[0:n, 0:n]
        m &= shapely.contains_xy(zone, -h + (xx + 0.5) * res, h - (yy + 0.5) * res)
    if existing_water is not None:
        m &= ~existing_water
    out = []
    amax = ndimage.maximum_filter(area_ha, 5)
    for ln in _skeleton_lines(ndimage.binary_dilation(m, iterations=1), res, h, min_px=8):
        ln = np.asarray(ln)
        r = np.clip(((h - ln[:, 1]) / res).astype(int), 0, n - 1); c = np.clip(((ln[:, 0] + h) / res).astype(int), 0, n - 1)
        a = float(amax[r, c].max())
        w = 3.0 if a < 100 else 4.5 if a < 300 else 6.0
        L = LineString(ln - np.asarray(shift))
        if L.length >= 25:
            out.append({"polygon": L.buffer(w / 2).simplify(0.5), "catchment_ha": round(a), "width_m": w,
                        "length_m": round(L.length)})
    return out
