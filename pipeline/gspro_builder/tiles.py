"""Map tile pyramid for the routing tracer page.

Downloads NAIP imagery (0.5 m) and the USGS 3DEP bare-earth DEM (1 m) for a
large square around the course centre (default 6 km) and cuts both into a
1024 px JPEG tile pyramid the tracer page loads on demand as you pan/zoom:

  tiles/aerial/<level>/<col>_<row>.jpg
  tiles/relief/<level>/<col>_<row>.jpg
  tiles/tiles.json   extent (course-frame metres), levels, metres-per-pixel

Level L has metres/pixel = finest * 2**(max_level - L); level 0 is one tile.
Coordinates stay in the course frame (x east, z north from course.toml's
centre), so points traced on the page drop straight into the pipeline.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .geo import CourseFrame

TILE = 1024


def _pyramid(img: Image.Image, finest_m: float, out: Path, quality: int) -> int:
    levels = max(0, math.ceil(math.log2(max(img.width, img.height) / TILE)))
    for lv in range(levels, -1, -1):
        scale = 2 ** (levels - lv)
        im = img if scale == 1 else img.resize((math.ceil(img.width / scale), math.ceil(img.height / scale)),
                                               Image.LANCZOS)
        d = out / str(lv)
        d.mkdir(parents=True, exist_ok=True)
        for r in range(math.ceil(im.height / TILE)):
            for c in range(math.ceil(im.width / TILE)):
                t = im.crop((c * TILE, r * TILE, min(im.width, (c + 1) * TILE), min(im.height, (r + 1) * TILE)))
                t.convert("RGB").save(d / f"{c}_{r}.jpg", quality=quality)
    return levels


def relief_image(dem: np.ndarray, res: float) -> Image.Image:
    gz, gx = np.gradient(dem.astype(np.float64), res)
    s = np.arctan(np.hypot(gx, gz) * 2.5)
    asp = np.arctan2(-gx, gz)
    sh = np.clip(np.sin(np.radians(40)) * np.cos(s) + np.cos(np.radians(40)) * np.sin(s) * np.cos(np.radians(315) - asp), 0, 1)
    lo, hi = np.percentile(dem, [1, 99])
    e = np.clip((dem - lo) / max(hi - lo, 1e-3), 0, 1)
    rgb = np.stack([sh * (0.75 + 0.25 * e), sh * (0.8 + 0.15 * e), sh * (0.85 - 0.1 * e)], -1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def _existing_levels(d: Path) -> int | None:
    lv = [int(p.name) for p in d.glob("*") if p.name.isdigit()] if d.exists() else []
    return max(lv) if lv and (d / "0" / "0_0.jpg").exists() else None


def _aerial(big, n, res, out_dir, log, export_image, urls) -> int:
    log(f"aerial: {big.size_m:g} m square at {res} m ({n} px)")
    for url in urls:
        try:
            aerial = export_image(url, big.utm_bbox, big.epsg, n, n, fmt="jpg", extra={"bandIds": "0,1,2"})
            return _pyramid(Image.fromarray(aerial), res, out_dir / "aerial", 80)
        except Exception as exc:  # next mirror
            log(f"  {url.split('/')[2]} failed ({type(exc).__name__})")
    raise RuntimeError("no NAIP source answered")


def build_tiles(center: CourseFrame, area_m: float, out_dir: Path, log=print,
                aerial_res: float = 0.5, relief_res: float = 1.0) -> dict:
    from .sources.arcgis import export_image
    from .sources.elevation import USGS_3DEP_URL, fill_nodata
    from .sources.imagery import NAIP_URLS
    big = CourseFrame(center.center_lon, center.center_lat, area_m)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(round(area_m / aerial_res))
    la = _existing_levels(out_dir / "aerial")
    if la is not None:
        log("aerial: tiles already built, skipping")
    else:
        la = _aerial(big, n, aerial_res, out_dir, log, export_image, NAIP_URLS)

    m = int(round(area_m / relief_res))
    log(f"relief: 3DEP DEM at {relief_res} m ({m} px)")
    # the DEM service errors on large requests: ask for it in 1000 px pieces
    dem = fill_nodata(export_image(USGS_3DEP_URL, big.utm_bbox, big.epsg, m, m, pixel_type="F32", max_tile=1000))
    lr = _pyramid(relief_image(dem, relief_res), relief_res, out_dir / "relief", 78)


    meta = {"tile": TILE, "x0": -area_m / 2, "z1": area_m / 2, "size_m": area_m,
            "center_lat": center.center_lat, "center_lon": center.center_lon,
            "layers": {"aerial": {"levels": la, "finest_m": aerial_res},
                       "relief": {"levels": lr, "finest_m": relief_res}}}
    (out_dir / "tiles.json").write_text(json.dumps(meta, indent=2))
    count = sum(1 for _ in out_dir.rglob("*.jpg"))
    log(f"tiles: {count} files -> {out_dir}")
    return meta
