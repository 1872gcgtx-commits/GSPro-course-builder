"""Hole data: tees, pins, aim points, yardages - in the Unity course frame.

Unity frame: x = metres east, z = metres north of the terrain centre,
y = metres above the terrain's lowest point (terrain placed at y = 0).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .layout import HoleLine, Layout, Surface, polyline_length
from .terrain import DEM

M_TO_YD = 1.0936133


def point_in_ring(p, ring) -> bool:
    x, z = p
    inside = False
    for i in range(len(ring)):
        x1, z1 = ring[i]
        x2, z2 = ring[(i + 1) % len(ring)]
        if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1) + x1:
            inside = not inside
    return inside


def estimate_par(yards: float) -> int:
    return 3 if yards <= 250 else 4 if yards <= 470 else 5


def _surfaces(layout: Layout, kind: str, hole: int) -> list[Surface]:
    return [s for s in layout.surfaces if s.kind == kind and s.hole == hole]


def build_holes(layout: Layout, dem: DEM, base_elev: float, tee_sets: list[str]) -> list[dict]:
    y = lambda p: round(float(dem.sample(p[0], p[1])) - base_elev, 3)  # noqa: E731
    v3 = lambda p: {"x": round(p[0], 3), "y": y(p), "z": round(p[1], 3)}  # noqa: E731
    out = []
    for h in layout.holes:
        greens = _surfaces(layout, "green", h.number)
        green = min(greens, key=lambda g: math.dist(g.centroid, h.path[-1])) if greens else None
        pin = next((p for p, n in layout.pins if n == h.number), None)
        if pin is None and green is not None:
            pin = next((p for p, n in layout.pins if n is None and point_in_ring(p, green.rings[0])), None)
        if pin is None:
            pin = green.centroid if green and point_in_ring(green.centroid, green.rings[0]) else h.path[-1]
        aims = h.path[1:-1]

        tee_boxes = _surfaces(layout, "tee", h.number)
        tee_pts = sorted((t.centroid for t in tee_boxes),
                         key=lambda c: -polyline_length([c, *aims, pin])) or [h.path[0]]
        tees = []
        for i, name in enumerate(tee_sets):
            tp = tee_pts[min(i, len(tee_pts) - 1)]
            yards = polyline_length([tp, *aims, pin]) * M_TO_YD
            tees.append({"name": name, "position": v3(tp), "yards": round(yards)})
        back = tees[0]["yards"]
        out.append({
            "number": h.number,
            "name": h.name or f"Hole {h.number}",
            "par": h.par or estimate_par(back),
            "par_estimated": h.par is None,
            "handicap": h.handicap or 0,
            "tees": tees,
            "pin": v3(pin),
            "aim_points": [v3(a) for a in aims],
            "green_center": v3(green.centroid) if green else v3(pin),
            "elevation_change_m": round(y(pin) - tees[0]["position"]["y"], 2),
        })
    return out


def write_holes(holes: list[dict], course_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "course": course_name,
        "frame": "Unity metres: x east, y up from terrain base, z north; origin = terrain centre",
        "par_total": sum(h["par"] for h in holes),
        "holes": holes,
    }
    (out_dir / "holes.json").write_text(json.dumps(doc, indent=2))
    with open(out_dir / "greenkeeper.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hole", "par", "handicap", "tee", "yards", "tee_x", "tee_y", "tee_z",
                    "pin_x", "pin_y", "pin_z", "aim_points"])
        for h in holes:
            for t in h["tees"]:
                xyz = lambda v: (v["x"], v["y"], v["z"])  # noqa: E731
                aims = " | ".join("{},{},{}".format(*xyz(a)) for a in h["aim_points"])
                w.writerow([h["number"], h["par"], h["handicap"] or "", t["name"], t["yards"],
                            *xyz(t["position"]), *xyz(h["pin"]), aims])
