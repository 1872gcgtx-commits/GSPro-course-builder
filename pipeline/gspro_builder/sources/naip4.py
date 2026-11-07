"""4-band NAIP (red, green, blue, near-infrared), uncompressed, at the service's
finest resolution (0.3-0.6 m), for a window of the course frame.

The colour-only JPEG used for the map overlay is too blocky to classify turf;
near-infrared separates irrigated fairway/green turf from dry rough, and
NDVI < 0 picks out open water.
"""
from __future__ import annotations

import io
import time

import numpy as np
import requests

from ..geo import CourseFrame
from .arcgis import USER_AGENT

NAIP4_URL = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"


def fetch_naip4(frame: CourseFrame, x0: float, x1: float, z0: float, z1: float, res: float = 0.3,
                chunk: int = 2000, log=print) -> np.ndarray:
    """(H, W, 4) uint8, north-up, covering local x0..x1, z0..z1 (metres)."""
    import tifffile
    cx, cy = frame.center_utm
    W, H = int(round((x1 - x0) / res)), int(round((z1 - z0) / res))
    out = np.zeros((H, W, 4), np.uint8)
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    n = 0
    for r0 in range(0, H, chunk):
        for c0 in range(0, W, chunk):
            r1, c1 = min(H, r0 + chunk), min(W, c0 + chunk)
            bb = (cx + x0 + c0 * res, cy + z1 - r1 * res, cx + x0 + c1 * res, cy + z1 - r0 * res)
            p = {"bbox": ",".join(f"{v:.3f}" for v in bb), "bboxSR": frame.epsg, "imageSR": frame.epsg,
                 "size": f"{c1 - c0},{r1 - r0}", "format": "tiff", "pixelType": "U8", "bandIds": "0,1,2,3",
                 "compression": "None", "interpolation": "RSP_BilinearInterpolation", "f": "image"}
            for attempt in range(5):
                try:
                    r = s.get(NAIP4_URL, params=p, timeout=300)
                    r.raise_for_status()
                    a = tifffile.imread(io.BytesIO(r.content))
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            out[r0:r1, c0:c1] = a[: r1 - r0, : c1 - c0, :4]
            n += 1
    log(f"  NAIP 4-band: {W}x{H} px at {res} m in {n} requests")
    return out
