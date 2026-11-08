"""Trees: individual trees from the LiDAR canopy height model (CHM).

1. CHM (first-return surface minus ground) is lightly smoothed.
2. Tree tops = local maxima, with a search radius that grows with tree height
   (small trees close together, big trees spread out).
3. Crown radius measured along 8 directions to where the canopy drops below
   half the tree height.
4. Type: narrow crowns (radius/height < 0.2) -> conifer, else deciduous.
5. Trees inside playing surfaces, water, cart paths or buildings are rejected
   (LiDAR sees flags, ball washers, carts and roof edges as "trees").

If there's no LiDAR canopy (DEM-only source), trees are scattered at natural
spacing inside the layout's `trees` polygons instead.

Output trees.csv (Unity frame): x, y, z, height_m, crown_radius_m, type, source.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .layout import Layout
from .raster import rasterize_instances
from .terrain import DEM

NO_TREE_KINDS = ("green", "tee", "fairway", "bunker", "water")


@dataclass
class FoliageConfig:
    enabled: bool = True
    min_height_m: float = 3.0
    max_height_m: float = 60.0
    conifer_ratio: float = 0.2
    keep_clear_m: float = 1.0           # around playing surfaces
    scatter_spacing_m: float = 7.0      # fallback without LiDAR
    boundary_margin_m: float = 80.0     # keep trees this far outside the property outline
    seed: int = 7

    @classmethod
    def from_dict(cls, d: dict) -> "FoliageConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _exclusion(layout: Layout, frame, n: int, clear_m: float, building=None):
    from scipy import ndimage
    surf = [s for s in layout.surfaces if s.kind in NO_TREE_KINDS]
    m = rasterize_instances(surf, frame, n) > 0
    if clear_m > 0:
        m = ndimage.binary_dilation(m, iterations=max(1, int(round(clear_m / (frame.size_m / n)))))
    if building is not None and building.shape == m.shape:
        m |= ndimage.binary_dilation(building, iterations=2)
    return m


def detect_trees(chm: np.ndarray, layout: Layout, dem: DEM, base_elev: float,
                 cfg: FoliageConfig, building: np.ndarray | None = None) -> list[dict]:
    from scipy import ndimage
    frame = dem.frame
    n = chm.shape[0]
    res = frame.size_m / n
    sm = ndimage.gaussian_filter(chm.astype(np.float32), 0.6 / res)
    peaks = np.zeros_like(sm, bool)
    for hmin, hmax, r_m in ((cfg.min_height_m, 10, 2.0), (10, 20, 3.0), (20, cfg.max_height_m, 4.5)):
        size = 2 * max(1, int(round(r_m / res))) + 1
        mf = ndimage.maximum_filter(sm, size=size)
        peaks |= (sm == mf) & (sm >= hmin) & (sm < hmax)
    peaks &= ~_exclusion(layout, frame, n, cfg.keep_clear_m, building)
    # flat-topped crowns give several equal neighbouring maxima: one peak per blob
    lab, nlab = ndimage.label(peaks, structure=np.ones((3, 3)))
    pos = ndimage.maximum_position(sm, lab, range(1, nlab + 1)) if nlab else []
    # non-maximum suppression: a tree top can't sit inside a taller neighbour's core
    order = sorted(pos, key=lambda rc: -sm[rc])
    kept: list[tuple[int, int]] = []
    if order:
        from scipy.spatial import cKDTree
        pts = np.array(order, dtype=float) * res
        tree = cKDTree(pts)
        alive = np.ones(len(order), bool)
        for i, rc in enumerate(order):
            if not alive[i]:
                continue
            kept.append(rc)
            for j in tree.query_ball_point(pts[i], max(2.0, 0.2 * float(sm[rc]))):
                if j > i:
                    alive[j] = False
    rows = np.array([r for r, _ in kept], dtype=int)
    cols = np.array([c for _, c in kept], dtype=int)
    h = frame.size_m / 2
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    max_steps = int(12 / res)
    trees = []
    for r, c in zip(rows, cols):
        ht = float(sm[r, c])
        radii = []
        for dr, dc in dirs:
            step_len = res * (1.4142 if dr and dc else 1.0)
            k = 1
            while k < max_steps:
                rr, cc = r + dr * k, c + dc * k
                if not (0 <= rr < n and 0 <= cc < n) or sm[rr, cc] < 0.5 * ht:
                    break
                k += 1
            radii.append(k * step_len)
        crown = float(np.median(radii))
        x, z = -h + (c + 0.5) * res, h - (r + 0.5) * res
        trees.append({"x": x, "z": z, "height_m": round(ht, 2), "crown_radius_m": round(crown, 2),
                      "type": "conifer" if crown / max(ht, 0.1) < cfg.conifer_ratio else "deciduous",
                      "source": "lidar"})
    _set_y(trees, dem, base_elev)
    return trees


def scatter_trees(layout: Layout, dem: DEM, base_elev: float, cfg: FoliageConfig) -> list[dict]:
    """Fallback: natural-looking (Poisson-disk) planting inside `trees` polygons."""
    from .course_meshes import points_in_ring
    rng = np.random.default_rng(cfg.seed)
    trees = []
    sp = cfg.scatter_spacing_m
    cs = sp / 1.4143                      # at most one tree per grid cell
    grid: dict[tuple[int, int], tuple[float, float]] = {}

    def free(p) -> bool:
        gi, gj = int(p[0] // cs), int(p[1] // cs)
        for di in (-2, -1, 0, 1, 2):
            for dj in (-2, -1, 0, 1, 2):
                q = grid.get((gi + di, gj + dj))
                if q is not None and (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 < sp * sp:
                    return False
        return True

    for s in (s for s in layout.surfaces if s.kind == "trees"):
        ring = np.asarray(s.rings[0])
        area = 0.5 * abs(np.dot(ring[:, 0], np.roll(ring[:, 1], 1)) - np.dot(ring[:, 1], np.roll(ring[:, 0], 1)))
        target = int(area / (cfg.scatter_spacing_m ** 2 * 1.2))
        cand = np.column_stack([rng.uniform(ring[:, 0].min(), ring[:, 0].max(), target * 6),
                                rng.uniform(ring[:, 1].min(), ring[:, 1].max(), target * 6)])
        cand = cand[points_in_ring(cand, s.rings[0])]
        for p in cand:
            if not free(p):
                continue
            grid[(int(p[0] // cs), int(p[1] // cs))] = (float(p[0]), float(p[1]))
            ht = float(rng.uniform(10, 24))
            conifer = rng.random() < 0.3
            trees.append({"x": float(p[0]), "z": float(p[1]), "height_m": round(ht, 2),
                          "crown_radius_m": round(ht * (0.15 if conifer else 0.3), 2),
                          "type": "conifer" if conifer else "deciduous", "source": "scatter"})
            if len(trees) >= 20000:
                break
    _set_y(trees, dem, base_elev)
    return trees


def _set_y(trees, dem, base_elev):
    if trees:
        xs = np.array([t["x"] for t in trees])
        zs = np.array([t["z"] for t in trees])
        ys = dem.sample(xs, zs) - base_elev
        for t, y in zip(trees, ys):
            t["x"], t["z"], t["y"] = round(t["x"], 2), round(t["z"], 2), round(float(y), 3)


def clip_to_boundary(trees: list[dict], boundary_local: list, margin_m: float) -> list[dict]:
    """Keep trees inside the property outline or within margin_m of it."""
    if not trees or len(boundary_local) < 3:
        return trees
    from scipy.spatial import cKDTree
    from .course_meshes import _densify, points_in_ring
    pts = np.array([(t["x"], t["z"]) for t in trees])
    inside = points_in_ring(pts, boundary_local)
    near = cKDTree(_densify(boundary_local, 2.0)).query(pts)[0] <= margin_m
    return [t for t, keep in zip(trees, inside | near) if keep]


TREE_COLUMNS = ["x", "y", "z", "height_m", "crown_radius_m", "type", "source", "species", "species_conf",
                "crown_base_m", "size_class", "tint_r", "tint_g", "tint_b", "rotation_deg"]


def write_trees_csv(trees: list[dict], path: Path) -> None:
    """Unity frame. The first 7 columns are the original format; species columns follow."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TREE_COLUMNS)
        for t in trees:
            tint = t.get("tint", [1.0, 1.0, 1.0])
            w.writerow([t["x"], t["y"], t["z"], t["height_m"], t["crown_radius_m"], t["type"], t["source"],
                        t.get("species", t["type"]), t.get("species_conf", ""), t.get("crown_base_m", ""),
                        t.get("size_class", ""), tint[0], tint[1], tint[2], t.get("rotation_deg", 0)])


def write_shrubs_csv(shrubs: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "height_m", "kind", "rotation_deg"])
        for s in shrubs:
            w.writerow([s["x"], s["y"], s["z"], s["height_m"], s["kind"], s["rotation_deg"]])
