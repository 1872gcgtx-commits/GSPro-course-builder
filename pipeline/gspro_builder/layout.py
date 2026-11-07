"""Course layout model: GeoJSON features -> local-frame surfaces and holes."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .geo import CourseFrame

# Normalised surface kinds, in paint order (later paints over earlier).
SURFACE_ORDER = ["rough", "trees", "fairway", "tee", "green", "bunker", "water", "cartpath"]

KIND_ALIASES = {
    "rough": "rough", "semi_rough": "rough", "driving_range": "rough",
    "trees": "trees", "native": "trees", "deep_rough": "trees",
    "fairway": "fairway", "tee": "tee", "green": "green", "bunker": "bunker",
    "water_hazard": "water", "lateral_water_hazard": "water", "water": "water",
    "cartpath": "cartpath", "path": "cartpath",
    "hole": "hole", "pin": "pin",
}

Pt = tuple[float, float]


@dataclass
class Surface:
    kind: str
    rings: list[list[Pt]]            # rings[0] outer, rest holes (local metres)
    hole: int | None = None
    props: dict = field(default_factory=dict)

    @property
    def centroid(self) -> Pt:
        ring = self.rings[0]
        return sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)


@dataclass
class HoleLine:
    number: int
    path: list[Pt]                    # tee -> (dog-leg aim points) -> green
    par: int | None = None
    handicap: int | None = None
    name: str | None = None
    props: dict = field(default_factory=dict)


@dataclass
class Layout:
    surfaces: list[Surface]
    holes: list[HoleLine]
    pins: list[tuple[Pt, int | None]]
    paths: list[tuple[str, list[Pt]]]   # line features rendered with width (cart paths)


def _int(v) -> int | None:
    try:
        return int(str(v).strip().split(";")[0])
    except (TypeError, ValueError):
        return None


def _polys(geom: dict) -> list[list[list[Pt]]]:
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return geom["coordinates"]
    return []


def point_segment_dist(p: Pt, a: Pt, b: Pt) -> float:
    ax, az = a
    dx, dz = b[0] - ax, b[1] - az
    L2 = dx * dx + dz * dz
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - az) * dz) / L2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (az + t * dz))


def point_polyline_dist(p: Pt, line: list[Pt]) -> float:
    if len(line) == 1:
        return math.dist(p, line[0])
    return min(point_segment_dist(p, line[i], line[i + 1]) for i in range(len(line) - 1))


def polyline_length(line: list[Pt]) -> float:
    return sum(math.dist(line[i], line[i + 1]) for i in range(len(line) - 1))


def load_layout(geojson: dict | str | Path, frame: CourseFrame) -> Layout:
    if not isinstance(geojson, dict):
        geojson = json.loads(Path(geojson).read_text())
    loc = lambda c: frame.to_local(c[0], c[1])  # noqa: E731
    surfaces, holes, pins, paths = [], [], [], []
    for f in geojson.get("features", []):
        props, geom = f.get("properties") or {}, f.get("geometry")
        if not geom:
            continue
        kind = KIND_ALIASES.get(str(props.get("golf") or props.get("surface") or "").lower())
        if kind is None:
            continue
        hole_no = _int(props.get("hole", props.get("ref")))
        if kind == "hole" and geom["type"] == "LineString":
            if hole_no is None:
                continue
            holes.append(HoleLine(hole_no, [loc(c) for c in geom["coordinates"]],
                                  _int(props.get("par")), _int(props.get("handicap")),
                                  props.get("name"), props))
        elif kind == "pin" and geom["type"] == "Point":
            pins.append((loc(geom["coordinates"]), hole_no))
        elif geom["type"] == "LineString":
            paths.append((kind, [loc(c) for c in geom["coordinates"]]))
        else:
            for poly in _polys(geom):
                rings = [[loc(c) for c in ring] for ring in poly]
                if len(rings[0]) >= 3:
                    surfaces.append(Surface(kind, rings, hole_no, props))
    holes.sort(key=lambda h: h.number)
    _assign_holes(surfaces, holes)
    return Layout(surfaces, holes, pins, paths)


def _assign_holes(surfaces: list[Surface], holes: list[HoleLine], max_dist: float = 80.0) -> None:
    """Give un-numbered tees/fairways/greens/bunkers the nearest hole line.
    Rough, trees, water and paths further than max_dist stay course-wide (None)."""
    if not holes:
        return
    for s in surfaces:
        if s.hole is not None:
            continue
        c = s.centroid
        best = min(holes, key=lambda h: point_polyline_dist(c, h.path))
        d = point_polyline_dist(c, best.path)
        if s.kind in ("tee", "fairway", "green", "bunker") or d <= max_dist / 2:
            if d <= max_dist or s.kind in ("tee", "green"):
                s.hole = best.number
