"""Generic ArcGIS ImageServer `exportImage` client.

Used for both USGS 3DEP elevation (float32 DEM) and USDA NAIP imagery (RGB).
Large requests are split into tiles (ImageServers cap the output size, commonly
4000-8000 px) and stitched back together.
"""
from __future__ import annotations

import io
import time

import numpy as np
import requests
from PIL import Image

USER_AGENT = "gspro-course-builder/0.1 (local golf course pipeline)"
MAX_TILE_PX = 4000


def _get(url: str, params: dict, retries: int = 4) -> bytes:
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=180, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            if "json" in ctype or "html" in ctype:
                raise RuntimeError(f"ImageServer returned an error: {r.text[:500]}")
            return r.content
        except (requests.RequestException, RuntimeError):
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def export_image(service_url: str, bbox: tuple[float, float, float, float], epsg: int,
                 width: int, height: int, *, pixel_type: str | None = None,
                 fmt: str = "tiff", interpolation: str = "RSP_BilinearInterpolation",
                 extra: dict | None = None) -> np.ndarray:
    """Fetch bbox (xmin, ymin, xmax, ymax in `epsg`) as an array, row 0 = north.

    Float rasters come back as (H, W) float32; imagery as (H, W, bands) uint8.
    """
    xmin, ymin, xmax, ymax = bbox
    out = None
    nx = -(-width // MAX_TILE_PX)
    ny = -(-height // MAX_TILE_PX)
    tw, th = -(-width // nx), -(-height // ny)
    px_w, px_h = (xmax - xmin) / width, (ymax - ymin) / height
    for j in range(ny):
        for i in range(nx):
            c0, r0 = i * tw, j * th
            c1, r1 = min(width, c0 + tw), min(height, r0 + th)
            tb = (xmin + c0 * px_w, ymax - r1 * px_h, xmin + c1 * px_w, ymax - r0 * px_h)
            params = {
                "bbox": ",".join(f"{v:.3f}" for v in tb),
                "bboxSR": epsg, "imageSR": epsg,
                "size": f"{c1 - c0},{r1 - r0}",
                "format": fmt, "f": "image",
                "interpolation": interpolation,
            }
            if pixel_type:
                params["pixelType"] = pixel_type
            if fmt == "tiff":
                params["compression"] = "None"
            params.update(extra or {})
            arr = decode_image(_get(service_url.rstrip("/") + "/exportImage", params))
            if out is None:
                shape = (height, width) + arr.shape[2:]
                out = np.zeros(shape, dtype=arr.dtype)
            out[r0:r1, c0:c1] = arr[: r1 - r0, : c1 - c0]
    assert out is not None
    return out


def decode_image(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        if img.mode in ("F", "I;16", "I"):
            return np.asarray(img, dtype=np.float32)
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception:
        try:  # multi-band / compressed GeoTIFFs Pillow can't read
            import rasterio
            from rasterio.io import MemoryFile
        except ImportError as exc:
            raise RuntimeError("Could not decode image; `pip install rasterio` for GeoTIFF support") from exc
        with MemoryFile(data) as mf, mf.open() as ds:
            a = ds.read()
            return a[0].astype(np.float32) if a.shape[0] == 1 else np.moveaxis(a[:3], 0, -1)
