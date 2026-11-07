"""Command line: python -m gspro_builder <command> <course dir>

  new   <dir> --lat --lon [--size 2000] [--name]   scaffold courses/<slug>/course.toml
  fetch <dir> [--only dem,imagery,layout]          download public data
  build <dir>                                      produce all outputs
  all   <dir>                                      fetch + build
  demo  <dir>                                      synthetic offline course, then build
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
size_m = {size}            # cover the whole property plus a margin
heightmap_res = 2049       # 513 | 1025 | 2049 | 4097
dem_res_m = 1.0            # USGS 3DEP is 1 m LiDAR bare-earth in most of the US
splat_res = 2048
source = "usgs_3dep"       # or "file" with dem_file = "data/my_lidar_dem.tif" (UTM, metres)

[imagery]
source = "naip"            # USDA NAIP, public domain; "none" to skip
resolution_m = 0.5

[layout]
source = "osm"             # OpenStreetMap golf=* features; "geojson" to use only your file
file = "features.geojson"  # your hand-traced additions/overrides (QGIS or geojson.io)
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
    elif a.cmd == "demo":
        from .demo import make_demo
        make_demo(d)
        build(load_config(d))


if __name__ == "__main__":
    main()
