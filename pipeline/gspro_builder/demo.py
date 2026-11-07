"""Synthetic 3-hole demo course so the whole chain (including the Unity 6 importer)
can be exercised offline without downloading any data."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .geo import CourseFrame

DEMO_TOML = """[course]
name = "Demo Links (synthetic)"
slug = "demo"

[terrain]
center_lat = 40.0
center_lon = -83.0
size_m = 1200
heightmap_res = 2049
dem_res_m = 0.5
splat_res = 2048

[imagery]
source = "none"

[layout]
source = "geojson"
file = "features.geojson"
"""


def _circle(c, r, n=32):
    return [(c[0] + r * math.cos(2 * math.pi * i / n), c[1] + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def _ellipse(c, rx, rz, ang, n=40):
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x, z = rx * math.cos(t), rz * math.sin(t)
        pts.append((c[0] + x * ca - z * sa, c[1] + x * sa + z * ca))
    return pts


def _strip(a, b, w0, w1):
    dx, dz = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dz)
    nx, nz = -dz / L, dx / L
    return [(a[0] + nx * w0 / 2, a[1] + nz * w0 / 2), (b[0] + nx * w1 / 2, b[1] + nz * w1 / 2),
            (b[0] - nx * w1 / 2, b[1] - nz * w1 / 2), (a[0] - nx * w0 / 2, a[1] - nz * w0 / 2)]


def _along(a, b, d):
    L = math.dist(a, b)
    return (a[0] + (b[0] - a[0]) * d / L, a[1] + (b[1] - a[1]) * d / L)


HOLES = [  # number, par, handicap, path (local metres)
    (1, 4, 1, [(-420, -420), (-400, -90), (-330, -20)]),
    (2, 3, 3, [(-250, 120), (-110, 210)]),
    (3, 5, 2, [(20, 260), (210, 110), (320, -170)]),
]


def make_demo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "course.toml").write_text(DEMO_TOML)
    frame = CourseFrame(-83.0, 40.0, 1200)
    feats = []

    def add(golf, ring=None, line=None, point=None, **props):
        if ring is not None:
            ll = [list(frame.to_lonlat(*p)) for p in ring]
            geom = {"type": "Polygon", "coordinates": [ll + [ll[0]]]}
        elif line is not None:
            geom = {"type": "LineString", "coordinates": [list(frame.to_lonlat(*p)) for p in line]}
        else:
            geom = {"type": "Point", "coordinates": list(frame.to_lonlat(*point))}
        feats.append({"type": "Feature", "properties": {"golf": golf, **props}, "geometry": geom})

    for num, par, hcp, path in HOLES:
        add("hole", line=path, ref=str(num), par=str(par), handicap=str(hcp), name=f"Demo {num}")
        tee, green = path[0], path[-1]
        nxt = path[1]
        add("tee", ring=_strip(_along(tee, nxt, -12), _along(tee, nxt, 12), 14, 14))
        add("tee", ring=_strip(_along(tee, nxt, 25), _along(tee, nxt, 40), 12, 12))
        if par > 3:
            for a, b in zip(path[:-1], path[1:]):
                start = _along(a, b, 180) if a == tee else a
                add("fairway", ring=_strip(start, _along(a, b, math.dist(a, b) - 25), 38, 34))
        prev = path[-2]
        ang = math.atan2(green[1] - prev[1], green[0] - prev[0])
        add("green", ring=_ellipse(green, 18, 13, ang))
        add("pin", point=_along(prev, green, math.dist(prev, green) + 4), ref=str(num))
        side = (-math.sin(ang), math.cos(ang))
        add("bunker", ring=_ellipse((green[0] + side[0] * 24, green[1] + side[1] * 24), 9, 5, ang))
        add("bunker", ring=_ellipse((green[0] - side[0] * 22 - 10 * math.cos(ang),
                                     green[1] - side[1] * 22 - 10 * math.sin(ang)), 7, 4, ang + 0.6))
    add("water_hazard", ring=_ellipse((-180, 160), 45, 25, 0.5))
    add("cartpath", line=[(-430, -440), (-440, -100), (-300, 60), (-260, 140), (-230, 215), (-40, 250), (230, 140), (350, -150)])
    add("trees", ring=_ellipse((-120, -250), 140, 90, 0.3))
    (root / "features.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))

    # Rolling synthetic terrain at 0.5 m with LiDAR-like noise (+-4 cm) and no
    # bunkers/pond carved, so the grading step has real work to do.
    n = 2400
    c = np.linspace(-600 + 0.25, 600 - 0.25, n)
    xx, zz = np.meshgrid(c, c[::-1])   # north-up
    rng = np.random.default_rng(1)
    dem = 250 + 6 * np.sin(xx / 140) * np.cos(zz / 170) + 0.012 * zz + 3 * np.sin((xx + zz) / 60)
    # Built-up greens: real greens are shaped to ~1-2.5 % slopes, so blend a gently
    # tilted pad (draining toward the front) into the rolling ground at each green.
    for _, _, _, path in HOLES:
        gx, gz = path[-1]
        px, pz = path[-2]
        L = math.dist((gx, gz), (px, pz))
        ux, uz = (gx - px) / L, (gz - pz) / L
        d = np.hypot(xx - gx, zz - gz)
        w = np.clip(1 - (d - 22) / 15, 0, 1)
        w = w * w * (3 - 2 * w)
        r, cc = int((600 - gz) / 0.5), int((gx + 600) / 0.5)
        pad = dem[r, cc] + 0.3 + 0.015 * ((xx - gx) * ux + (zz - gz) * uz)
        dem = dem * (1 - w) + pad * w
    dem = dem + rng.normal(0, 0.04, xx.shape)     # LiDAR vertical noise
    (root / "data").mkdir(exist_ok=True)
    np.save(root / "data" / "dem.npy", dem.astype(np.float32))

    # Synthetic canopy height model: a wood plus scattered specimen trees.
    chm = np.zeros_like(dem, dtype=np.float32)
    tree_xy = [(x, z) for x in range(-240, 0, 14) for z in range(-330, -170, 14)
               if ((x + 120) / 140) ** 2 + ((z + 250) / 90) ** 2 < 1]
    tree_xy += [(-470, -200), (-360, -300), (150, 300), (380, 50), (100, -350)]
    for (tx, tz) in tree_xy:
        hgt, rad = 15 + 6 * rng.random(), 3.5 + 1.5 * rng.random()
        r0, c0 = int((600 - tz) / 0.5), int((tx + 600) / 0.5)
        k = int(rad * 2 / 0.5)
        sl = (slice(max(0, r0 - k), r0 + k), slice(max(0, c0 - k), c0 + k))
        d2 = (xx[sl] - tx) ** 2 + (zz[sl] - tz) ** 2
        chm[sl] = np.maximum(chm[sl], hgt * np.exp(-d2 / (2 * (rad / 1.2) ** 2)))
    np.save(root / "data" / "chm.npy", chm)
    (root / "data" / "demo_trees.json").write_text(json.dumps({"planted": len(tree_xy)}))

    # Fake "aerial" overlay = coloured surfaces + noise, so the image layer is exercised.
    from .layout import load_layout
    from .surfaces import PREVIEW_COLORS, TERRAIN_LAYERS, rasterize_labels
    labels = rasterize_labels(load_layout(root / "features.geojson", frame), frame, 1200)
    pal = np.array([PREVIEW_COLORS[k] for k in TERRAIN_LAYERS], dtype=np.int16)
    rgb = pal[labels] + rng.integers(-12, 12, (1200, 1200, 1))
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(root / "data" / "overlay.jpg", quality=90)
    return root
