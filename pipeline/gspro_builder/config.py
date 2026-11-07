"""course.toml loading and validation."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .geo import CourseFrame

UNITY_HEIGHTMAP_SIZES = (513, 1025, 2049, 4097)


@dataclass
class CourseConfig:
    root: Path                       # courses/<slug>/
    name: str
    slug: str
    frame: CourseFrame
    heightmap_res: int = 2049
    dem_res_m: float = 1.0
    splat_res: int = 2048
    imagery_res_m: float = 0.5
    dem_source: str = "usgs_3dep"     # usgs_3dep | file
    dem_file: str | None = None       # GeoTIFF in the tile's UTM zone (needs rasterio)
    imagery_source: str = "naip"      # naip | none
    layout_source: str = "osm"        # osm | geojson
    layout_file: str = "features.geojson"
    tee_sets: list[str] = field(default_factory=lambda: ["Black", "Blue", "White", "Red"])
    svg_colors: dict = field(default_factory=dict)   # override OPCD fill colours

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def out_dir(self) -> Path:
        return self.root / "output"


def load_config(path: str | Path) -> CourseConfig:
    path = Path(path)
    if path.is_dir():
        path = path / "course.toml"
    raw = tomllib.loads(path.read_text())
    c, t = raw.get("course", {}), raw.get("terrain", {})
    img, lay = raw.get("imagery", {}), raw.get("layout", {})
    for key in ("center_lat", "center_lon", "size_m"):
        if key not in t:
            raise ValueError(f"{path}: [terrain] is missing '{key}'")
    cfg = CourseConfig(
        root=path.parent,
        name=c.get("name", path.parent.name),
        slug=c.get("slug", path.parent.name),
        frame=CourseFrame(float(t["center_lon"]), float(t["center_lat"]), float(t["size_m"])),
        heightmap_res=int(t.get("heightmap_res", 2049)),
        dem_res_m=float(t.get("dem_res_m", 1.0)),
        splat_res=int(t.get("splat_res", 2048)),
        dem_source=t.get("source", "usgs_3dep"),
        dem_file=t.get("dem_file"),
        imagery_source=img.get("source", "naip"),
        imagery_res_m=float(img.get("resolution_m", 0.5)),
        layout_source=lay.get("source", "osm"),
        layout_file=lay.get("file", "features.geojson"),
        tee_sets=list(c.get("tee_sets", ["Black", "Blue", "White", "Red"])),
        svg_colors=dict(raw.get("svg", {}).get("colors", {})),
    )
    if cfg.heightmap_res not in UNITY_HEIGHTMAP_SIZES:
        raise ValueError(f"heightmap_res must be one of {UNITY_HEIGHTMAP_SIZES}")
    if cfg.frame.size_m <= 0 or cfg.frame.size_m > 8000:
        raise ValueError("size_m must be in (0, 8000]")
    return cfg
