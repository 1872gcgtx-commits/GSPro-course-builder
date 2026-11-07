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
    heightmap_res: int = 4097
    dem_res_m: float = 0.5
    splat_res: int = 4096
    imagery_res_m: float = 0.5
    dem_source: str = "lidar_ept"     # lidar_ept | lidar_files | usgs_3dep | file
    dem_file: str | None = None       # GeoTIFF in the tile's UTM zone (needs rasterio)
    lidar_files: list[str] = field(default_factory=lambda: ["data/laz/*.laz", "data/laz/*.las"])
    lidar_crs: str | None = None      # override when .laz has no CRS, e.g. "EPSG:6565"
    lidar_z_units: str = "auto"       # auto | m | ft | usft
    lidar_ept_url: str | None = None  # pin a specific USGS EPT dataset (ept.json URL)
    imagery_source: str = "naip"      # naip | none
    layout_source: str = "osm"        # osm | geojson
    layout_file: str = "features.geojson"
    tee_sets: list[str] = field(default_factory=lambda: ["Black", "Blue", "White", "Red"])
    svg_colors: dict = field(default_factory=dict)   # override OPCD fill colours
    grading: dict = field(default_factory=dict)      # -> GradingConfig
    surfaces: dict = field(default_factory=dict)     # -> SurfaceConfig
    foliage: dict = field(default_factory=dict)      # -> FoliageConfig (+ "source")
    mesh_offset_m: float = 0.0
    nine_names: dict = field(default_factory=dict)   # {1: "Creek 1", ...}
    boundary: list = field(default_factory=list)     # property outline [[lon, lat], ...]
    routings: list = field(default_factory=list)     # [{name, front, back}]; empty = every pair

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
        heightmap_res=int(t.get("heightmap_res", 4097)),
        dem_res_m=float(t.get("dem_res_m", 0.5)),
        splat_res=int(t.get("splat_res", 4096)),
        dem_source=t.get("source", "lidar_ept"),
        dem_file=t.get("dem_file"),
        lidar_files=list(t.get("lidar_files", ["data/laz/*.laz", "data/laz/*.las"])),
        lidar_crs=t.get("lidar_crs"),
        lidar_z_units=t.get("lidar_z_units", "auto"),
        lidar_ept_url=t.get("lidar_ept_url"),
        imagery_source=img.get("source", "naip"),
        imagery_res_m=float(img.get("resolution_m", 0.5)),
        layout_source=lay.get("source", "osm"),
        layout_file=lay.get("file", "features.geojson"),
        tee_sets=list(c.get("tee_sets", ["Black", "Blue", "White", "Red"])),
        svg_colors=dict(raw.get("svg", {}).get("colors", {})),
        grading=dict(raw.get("grading", {})),
        surfaces=dict(raw.get("surfaces", {})),
        foliage=dict(raw.get("foliage", {})),
        mesh_offset_m=float(raw.get("meshes", {}).get("offset_m", 0.0)),
        nine_names={int(k): str(v) for k, v in raw.get("nines", {}).items()},
        boundary=[[float(a), float(b)] for a, b in c.get("boundary", [])],
        routings=[{"name": r["name"], "front": int(r["front"]), "back": int(r["back"])}
                  for r in raw.get("routing", [])],
    )
    if cfg.dem_source not in ("lidar_ept", "lidar_files", "usgs_3dep", "file"):
        raise ValueError(f"unknown [terrain] source '{cfg.dem_source}'")
    if cfg.heightmap_res not in UNITY_HEIGHTMAP_SIZES:
        raise ValueError(f"heightmap_res must be one of {UNITY_HEIGHTMAP_SIZES}")
    if cfg.frame.size_m <= 0 or cfg.frame.size_m > 8000:
        raise ValueError("size_m must be in (0, 8000]")
    return cfg
