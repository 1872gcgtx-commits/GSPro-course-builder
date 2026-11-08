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


def pin_positions(green: Surface, approach_from, dem: DEM, max_slope: float = 0.025,
                  min_edge_m: float = 3.5, step: float = 0.5) -> dict:
    """Front / middle / back hole locations on the flattest parts of the green.

    Candidates sit at least `min_edge_m` inside the edge, on slope <= max_slope
    (relaxed if the green has no such area). Front/middle/back are the 20th/50th/
    80th percentile along the line of play; among nearby candidates the flattest wins.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from .course_meshes import _densify, points_in_ring
    ring = np.asarray(green.rings[0])
    xs = np.arange(ring[:, 0].min(), ring[:, 0].max(), step)
    zs = np.arange(ring[:, 1].min(), ring[:, 1].max(), step)
    gx, gz = np.meshgrid(xs, zs)
    pts = np.column_stack([gx.ravel(), gz.ravel()])
    pts = pts[points_in_ring(pts, green.rings[0])]
    if len(pts) == 0:
        return {}
    edge = cKDTree(_densify(green.rings[0], 0.25)).query(pts)[0]
    e = 0.5
    sx = (dem.sample_smooth(pts[:, 0] + e, pts[:, 1]) - dem.sample_smooth(pts[:, 0] - e, pts[:, 1])) / (2 * e)
    sz = (dem.sample_smooth(pts[:, 0], pts[:, 1] + e) - dem.sample_smooth(pts[:, 0], pts[:, 1] - e)) / (2 * e)
    slope = np.hypot(sx, sz)
    for edge_req, slope_req in ((min_edge_m, max_slope), (min_edge_m * 0.6, max_slope * 1.5), (0, 1e9)):
        ok = (edge >= edge_req) & (slope <= slope_req)
        if ok.sum() >= 3:
            break
    cand, cslope = pts[ok], slope[ok]
    c = np.asarray(green.centroid)
    d = c - np.asarray(approach_from)
    d = d / max(np.linalg.norm(d), 1e-6)
    along = (cand - c) @ d
    out = {}
    for name, q in (("front", 20), ("middle", 50), ("back", 80)):
        target = np.percentile(along, q)
        near = np.abs(along - target) <= max(1.5, 0.1 * (along.max() - along.min()))
        idx = np.flatnonzero(near)
        best = idx[np.argmin(cslope[idx] + 0.002 * np.abs(along[idx] - target))]
        out[name] = (tuple(cand[best]), float(cslope[best]))
    return out


def _surfaces(layout: Layout, kind: str, hole: int) -> list[Surface]:
    return [s for s in layout.surfaces if s.kind == kind and s.hole == hole]


def build_holes(layout: Layout, dem: DEM, base_elev: float, tee_sets: list[str]) -> list[dict]:
    y = lambda p: round(float(dem.sample(p[0], p[1])) - base_elev, 3)  # noqa: E731
    v3 = lambda p: {"x": round(p[0], 3), "y": y(p), "z": round(p[1], 3)}  # noqa: E731
    out = []
    for h in layout.holes:
        greens = _surfaces(layout, "green", h.number)
        green = min(greens, key=lambda g: math.dist(g.centroid, h.path[-1])) if greens else None
        aims = h.path[1:-1]
        approach = (aims or h.path[:1])[-1]
        pins = pin_positions(green, approach, dem) if green is not None else {}
        pin = next((p for p, n in layout.pins if n == h.number), None)
        if pin is None and green is not None:
            pin = next((p for p, n in layout.pins if n is None and point_in_ring(p, green.rings[0])), None)
        if pin is None and pins:
            pin = pins["middle"][0]
        if pin is None:
            pin = green.centroid if green and point_in_ring(green.centroid, green.rings[0]) else h.path[-1]

        tee_boxes = _surfaces(layout, "tee", h.number)
        named = {t.props["tee_set"]: t.centroid for t in tee_boxes if t.props.get("tee_set")}
        tees = []
        if named:
            # tee boxes labelled in the review page: one tee per labelled set, back to front
            order = ["Black", "Blue", "White", "Gold", "Red", "Green"]
            for name in sorted(named, key=lambda n: order.index(n) if n in order else 99):
                tp = named[name]
                yards = polyline_length([tp, *aims, pin]) * M_TO_YD
                tees.append({"name": name, "position": v3(tp), "yards": round(yards)})
            tees.sort(key=lambda t: -t["yards"])
        else:
            tee_pts = sorted((t.centroid for t in tee_boxes),
                             key=lambda c: -polyline_length([c, *aims, pin])) or [h.path[0]]
            for i, name in enumerate(tee_sets):
                tp = tee_pts[min(i, len(tee_pts) - 1)]
                yards = polyline_length([tp, *aims, pin]) * M_TO_YD
                tees.append({"name": name, "position": v3(tp), "yards": round(yards)})
        back = tees[0]["yards"]
        out.append({
            "number": h.number,
            "nine": int(h.props["nine"]) if h.props.get("nine") else None,
            "nine_hole": int(h.props["nine_hole"]) if h.props.get("nine_hole") else None,
            "name": h.name or f"Hole {h.number}",
            "par": h.par or estimate_par(back),
            "par_estimated": h.par is None,
            "handicap": h.handicap or 0,
            "tees": tees,
            "pin": v3(pin),
            "pin_positions": [{"name": k, "position": v3(p), "slope_pct": round(sl * 100, 2)}
                              for k, (p, sl) in pins.items()],
            "aim_points": [v3(a) for a in aims],
            "green_center": v3(green.centroid) if green else v3(pin),
            "elevation_change_m": round(y(pin) - tees[0]["position"]["y"], 2),
        })
    return out


def write_holes(holes: list[dict], course_name: str, out_dir: Path, tee_sets: list[dict] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "course": course_name,
        "tee_sets": tee_sets or [],                    # [{name, color}] in card order, from the review page
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
