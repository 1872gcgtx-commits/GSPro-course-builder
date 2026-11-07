"""Command line: python -m gspro_builder <command> <course dir>

  new   <dir> --lat --lon [--size 2000] [--name]   scaffold courses/<slug>/course.toml
  fetch <dir> [--only dem,imagery,layout]          download public data
  build <dir>                                      produce all outputs
  all   <dir>                                      fetch + build
  demo  <dir>                                      synthetic offline course, then build
  import-routing <dir> <json file or folder>       hole lines from the routing tracer page
  import-surfaces <dir> <draft.json> [<edits folder>]  reviewed tees/greens/fairways/bunkers
  tiles <dir> [--area 6000]                        map tiles (aerial + LiDAR relief) for the tracer page
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .config import load_config
from .pipeline import build, fetch

TEMPLATE = """[course]
name = "{name}"
slug = "{slug}"
tee_sets = ["Black", "Blue", "White", "Red"]

[terrain]
# Square terrain tile centred here, axis-aligned to the local UTM grid.
center_lat = {lat}
center_lon = {lon}
size_m = {size}            # cover the whole property plus ~100 m
heightmap_res = 4097       # 513 | 1025 | 2049 | 4097  (4097 over 2000 m = 0.49 m spacing)
dem_res_m = 0.5            # LiDAR ground grid; greens/bunkers need <= 0.5 m
splat_res = 4096
source = "lidar_ept"       # lidar_ept   : USGS 3DEP point cloud streamed from AWS (best)
                           # lidar_files : your own .laz/.las tiles (lidar_files = ["data/laz/*.laz"])
                           # usgs_3dep   : 1 m bare-earth DEM only (no trees/water from LiDAR)
                           # file        : your own DEM GeoTIFF (dem_file = "...")
# lidar_crs = "EPSG:3735"  # only if your .laz has no CRS
# lidar_z_units = "auto"   # auto | m | ft | usft

[imagery]
source = "naip"            # USDA NAIP, public domain; "none" to skip
resolution_m = 0.5

[layout]
source = "osm"             # OpenStreetMap golf=* features; "geojson" to use only your file
file = "features.geojson"  # your hand-traced additions/overrides (QGIS or geojson.io)

[grading]                  # defaults shown; delete lines you don't change
tee_max_slope = 0.015      # tee pads levelled to <= 1.5 %
tee_raise_m = 0.0
green_smooth_m = 0.6       # removes LiDAR noise, keeps contours
green_slope_warn = 0.03
fairway_smooth_m = 0.5
bunker_mode = "auto"       # auto (keep LiDAR bowl if present) | carve | lidar
bunker_depth_m = 0.6
bunker_lip_m = 0.25        # lip on the green-facing side
water_freeboard_m = 0.3    # water surface below the lowest bank
water_depth_m = 1.8

[surfaces]
fringe_width_m = 1.0       # collar around greens
first_cut_width_m = 2.0    # intermediate rough around fairways

[foliage]
source = "auto"            # auto (LiDAR canopy, else scatter in `trees` areas) | lidar | scatter | none
min_height_m = 3.0
keep_clear_m = 1.0

[meshes]
offset_m = 0.0             # lift course-surface meshes above terrain (e.g. 0.02) if they z-fight
"""


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="gspro_builder", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("dir")
    n.add_argument("--lat", type=float, required=True)
    n.add_argument("--lon", type=float, required=True)
    n.add_argument("--size", type=float, default=2000)
    n.add_argument("--name", default=None)
    f = sub.add_parser("fetch")
    f.add_argument("dir")
    f.add_argument("--only", default="dem,imagery,layout")
    for c in ("build", "all", "demo"):
        sub.add_parser(c).add_argument("dir")
    tl = sub.add_parser("tiles")
    tl.add_argument("dir")
    tl.add_argument("--area", type=float, default=6000)
    isf = sub.add_parser("import-surfaces")
    isf.add_argument("dir")
    isf.add_argument("draft")
    isf.add_argument("edits", nargs="?")
    ir = sub.add_parser("import-routing")
    ir.add_argument("dir")
    ir.add_argument("source")
    a = p.parse_args(argv)
    d = Path(a.dir)

    if a.cmd == "new":
        d.mkdir(parents=True, exist_ok=True)
        name = a.name or d.name.replace("-", " ").title()
        slug = re.sub(r"[^a-z0-9]+", "-", d.name.lower()).strip("-")
        (d / "course.toml").write_text(TEMPLATE.format(name=name, slug=slug, lat=a.lat, lon=a.lon, size=a.size))
        print(f"created {d / 'course.toml'}")
    elif a.cmd == "fetch":
        fetch(load_config(d), tuple(a.only.split(",")))
    elif a.cmd == "build":
        build(load_config(d))
    elif a.cmd == "all":
        cfg = load_config(d)
        fetch(cfg)
        build(cfg)
    elif a.cmd == "tiles":
        from .tiles import build_tiles
        cfg = load_config(d)
        build_tiles(cfg.frame, a.area, cfg.data_dir / "tiles")
    elif a.cmd == "import-surfaces":
        import json
        from .routing import merge_surfaces, reviewed_shapes, surfaces_to_geojson
        cfg = load_config(d)
        draft = json.loads(Path(a.draft).read_text())
        edits = []
        if a.edits:
            for f in sorted(Path(a.edits).rglob("*.json")):
                v = json.loads(f.read_text())
                v = v.get("data", v)
                v.setdefault("id", f.stem)
                edits.append(v)
        shapes = reviewed_shapes(draft, edits)
        n = merge_surfaces(cfg.root / cfg.layout_file, surfaces_to_geojson(shapes, cfg.frame))
        print(f"imported {n} surfaces ({len(edits)} edits) into {cfg.root / cfg.layout_file}")
    elif a.cmd == "import-routing":
        import json
        from .routing import merge_hole_lines, tracer_to_geojson
        cfg = load_config(d)
        src = Path(a.source)
        files = sorted(src.rglob("*.json")) if src.is_dir() else [src]
        docs = []
        for f in files:
            v = json.loads(f.read_text())
            docs += v if isinstance(v, list) else [v.get("data", v)]
        docs = [x for x in docs if isinstance(x, dict) and "nine" in x and "points" in x]
        n = merge_hole_lines(cfg.root / cfg.layout_file, tracer_to_geojson(docs, cfg.frame))
        print(f"imported {n} hole lines into {cfg.root / cfg.layout_file}")
    elif a.cmd == "demo":
        from .demo import make_demo
        make_demo(d)
        build(load_config(d))


if __name__ == "__main__":
    main()
