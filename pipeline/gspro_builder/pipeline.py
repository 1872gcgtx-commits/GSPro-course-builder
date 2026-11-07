"""Pipeline orchestration: fetch -> build.

fetch : downloads DEM, imagery and OSM layout into courses/<slug>/data/
build : turns data/ into courses/<slug>/output/ (all offline, re-runnable)

output/
  terrain/heightmap_<N>.raw, terrain.json, heightmap_preview.png, terrain.obj
  surfaces/splat_control0.png, splat_control1.png, surface_labels.png, surfaces.json, surface_preview.png
  overlay.jpg
  <slug>_opcd.svg
  holes.json, greenkeeper.csv
  build_report.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .config import CourseConfig
from .holes import build_holes, write_holes
from .layout import load_layout
from .mesh import write_obj
from .surfaces import write_surfaces
from .svg_opcd import write_svg
from .terrain import DEM, write_heightmap


def log(msg: str) -> None:
    print(f"[gspro-builder] {msg}", flush=True)


def fetch(cfg: CourseConfig, what: tuple[str, ...] = ("dem", "imagery", "layout")) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    if "dem" in what:
        from .sources.elevation import fetch_usgs_3dep, fill_nodata, read_dem_file
        if cfg.dem_source == "file":
            log(f"reading DEM {cfg.dem_file}")
            dem = read_dem_file(str(cfg.root / cfg.dem_file), cfg.frame, cfg.dem_res_m)
        else:
            log(f"downloading USGS 3DEP DEM ({cfg.frame.size_m:g} m @ {cfg.dem_res_m:g} m)")
            dem = fetch_usgs_3dep(cfg.frame, cfg.dem_res_m)
        np.save(cfg.data_dir / "dem.npy", fill_nodata(dem))
    if "imagery" in what and cfg.imagery_source == "naip":
        from PIL import Image
        from .sources.imagery import fetch_naip
        log("downloading USDA NAIP imagery")
        Image.fromarray(fetch_naip(cfg.frame, cfg.imagery_res_m)).save(cfg.data_dir / "overlay.jpg", quality=92)
    if "layout" in what and cfg.layout_source == "osm":
        from .sources.osm import fetch_osm
        log("downloading OpenStreetMap golf features")
        gj = fetch_osm(cfg.frame)
        (cfg.data_dir / "osm.geojson").write_text(json.dumps(gj))
        log(f"  {len(gj['features'])} features")


def _layout_geojson(cfg: CourseConfig) -> dict:
    feats = []
    osm = cfg.data_dir / "osm.geojson"
    if cfg.layout_source == "osm" and osm.exists():
        feats += json.loads(osm.read_text())["features"]
    local = cfg.root / cfg.layout_file
    if local.exists():   # hand-traced features add to / override OSM
        extra = json.loads(local.read_text())["features"]
        drop = {f["properties"].get("replaces") for f in extra if f.get("properties")} - {None}
        feats = [f for f in feats if f["properties"].get("osm_id") not in drop] + extra
    if not feats:
        raise FileNotFoundError(f"No layout: run `fetch` (OSM) or create {local}")
    return {"type": "FeatureCollection", "features": feats}


def build(cfg: CourseConfig) -> dict:
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    dem_path = cfg.data_dir / "dem.npy"
    if not dem_path.exists():
        raise FileNotFoundError(f"{dem_path} missing - run `fetch` first")
    dem = DEM(np.load(dem_path), cfg.frame)

    log(f"terrain -> {cfg.heightmap_res}x{cfg.heightmap_res} RAW")
    tinfo = write_heightmap(dem, cfg.heightmap_res, out / "terrain")
    write_obj(dem, out / "terrain" / "terrain.obj", tinfo.min_elev_m)

    layout = load_layout(_layout_geojson(cfg), cfg.frame)
    log(f"layout: {len(layout.holes)} holes, {len(layout.surfaces)} surfaces")
    smeta = write_surfaces(layout, cfg.frame, cfg.splat_res, out / "surfaces")

    overlay = None
    if (cfg.data_dir / "overlay.jpg").exists():
        shutil.copy(cfg.data_dir / "overlay.jpg", out / "overlay.jpg")
        overlay = "overlay.jpg"
    write_svg(layout, cfg.frame, out / f"{cfg.slug}_opcd.svg", overlay, cfg.svg_colors)

    holes = build_holes(layout, dem, tinfo.min_elev_m, cfg.tee_sets)
    write_holes(holes, cfg.name, out)

    warnings = []
    numbers = [h["number"] for h in holes]
    if len(numbers) not in (9, 18):
        warnings.append(f"{len(numbers)} holes found (expected 9 or 18)")
    for n in range(1, max(numbers, default=0) + 1):
        if n not in numbers:
            warnings.append(f"hole {n} missing a golf=hole line")
    for h in holes:
        if not any(s.kind == "green" and s.hole == h["number"] for s in layout.surfaces):
            warnings.append(f"hole {h['number']}: no green polygon")
        if h["par_estimated"]:
            warnings.append(f"hole {h['number']}: par estimated from yardage ({h['par']})")
    report = {"course": cfg.name, "terrain": tinfo.__dict__, "surfaces": smeta["layers"],
              "holes": len(holes), "par": sum(h["par"] for h in holes), "warnings": warnings}
    (out / "build_report.json").write_text(json.dumps(report, indent=2))
    for w in warnings:
        log(f"WARNING: {w}")
    log(f"done -> {out}")
    return report
