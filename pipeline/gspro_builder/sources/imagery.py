"""Aerial imagery. Default: USDA NAIP (public domain, 0.6 m-1 m, US only).

Imagery is used as the trace reference / overlay (the OPCD SVG image layer and
the Unity preview). Avoid commercial basemaps (Google/Bing/Esri World Imagery):
their terms don't allow use as course textures.
"""
from __future__ import annotations

import numpy as np

from ..geo import CourseFrame
from .arcgis import export_image

NAIP_URLS = [
    "https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer",
    "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer",  # USGS mirror
]


def fetch_naip(frame: CourseFrame, res_m: float, max_px: int = 8192) -> np.ndarray:
    n = min(max_px, int(round(frame.size_m / res_m)))
    last = None
    for url in NAIP_URLS:
        try:
            return export_image(url, frame.utm_bbox, frame.epsg, n, n, fmt="jpg",
                                extra={"bandIds": "0,1,2"})
        except Exception as exc:     # try the next mirror
            print(f"[gspro-builder]   {url.split('/')[2]} failed ({type(exc).__name__}), trying next source", flush=True)
            last = exc
    raise RuntimeError("all NAIP imagery sources failed") from last
