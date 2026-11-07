"""Bare-earth elevation sources.

usgs_3dep : USGS 3D Elevation Program dynamic DEM service (public domain, CONUS/
            AK/HI/territories). Serves the best available 3DEP DEM, which is
            1 m LiDAR-derived bare earth for most of the US.
file      : a local GeoTIFF you produced yourself (e.g. from .laz tiles in QGIS/
            PDAL/LAStools). Must be in the tile's UTM zone, metres. Needs rasterio.
"""
from __future__ import annotations

import numpy as np

from ..geo import CourseFrame
from .arcgis import export_image

USGS_3DEP_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
NODATA_THRESHOLD = -1000.0


def fetch_usgs_3dep(frame: CourseFrame, res_m: float) -> np.ndarray:
    n = int(round(frame.size_m / res_m))
    dem = export_image(USGS_3DEP_URL, frame.utm_bbox, frame.epsg, n, n, pixel_type="F32")
    return dem.astype(np.float32)


def read_dem_file(path: str, frame: CourseFrame, res_m: float) -> np.ndarray:
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise RuntimeError("terrain.source = 'file' needs rasterio: pip install rasterio") from exc
    n = int(round(frame.size_m / res_m))
    with rasterio.open(path) as ds:
        if ds.crs and ds.crs.to_epsg() != frame.epsg:
            raise ValueError(f"{path} is EPSG:{ds.crs.to_epsg()}, expected EPSG:{frame.epsg} "
                             f"(reproject with: gdalwarp -t_srs EPSG:{frame.epsg} in.tif out.tif)")
        win = from_bounds(*frame.utm_bbox, transform=ds.transform)
        a = ds.read(1, window=win, out_shape=(n, n), resampling=Resampling.bilinear,
                    boundless=True, fill_value=ds.nodata if ds.nodata is not None else -9999)
        if ds.nodata is not None:
            a = np.where(a == ds.nodata, np.nan, a)
    return a.astype(np.float32)


def fill_nodata(dem: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """Replace NaN / sentinel cells by iterative neighbour averaging."""
    dem = dem.astype(np.float32).copy()
    dem[dem < NODATA_THRESHOLD] = np.nan
    if np.all(np.isnan(dem)):
        raise ValueError("DEM is entirely nodata - is the course inside the data coverage?")
    for _ in range(max_iter):
        mask = np.isnan(dem)
        if not mask.any():
            break
        p = np.pad(dem, 1, constant_values=np.nan)
        nb = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        cnt = (~np.isnan(nb)).sum(0)
        s = np.nansum(nb, 0)
        fill = mask & (cnt > 0)
        dem[fill] = s[fill] / cnt[fill]
    return np.nan_to_num(dem, nan=float(np.nanmean(dem)))
