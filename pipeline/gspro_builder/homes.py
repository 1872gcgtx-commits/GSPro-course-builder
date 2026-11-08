"""Homes around the course: footprints -> low-poly houses with hipped roofs.

Footprints come from Microsoft's open building footprints (they include houses
built after the LiDAR was flown). Each house gets:

  height   measured roof top from the LiDAR (canopy-height grid inside the footprint)
           where the LiDAR saw a building there; otherwise the footprint's own height
           estimate; otherwise a single-storey default
  roof     a true hipped roof: every point rises at the roof pitch with its distance
           from the eave line (the straight-skeleton roof of any footprint shape),
           with an eave overhang; big flat-topped buildings get a capped hip
  walls    from the lowest ground under the footprint up to the eave
  colour   roof colour sampled from the aerial photo; walls drawn from the Salado
           palette (cream limestone, rustic stone, tan/red brick, white stucco)

Footprints are registered onto the LiDAR first (best shift that puts them on the
LiDAR's building returns). Footprints on course surfaces (greens, fairways, tees,
bunkers, water) are dropped. Output: homes.obj/.mtl (one object per 250 m tile,
one material per wall/roof finish; same axis convention as course_surfaces.obj)
and homes.json (one record per house).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WALLS = [("Home_Wall_Limestone", (0.86, 0.81, 0.70), 0.38), ("Home_Wall_Stone", (0.74, 0.66, 0.55), 0.18),
         ("Home_Wall_BrickTan", (0.72, 0.58, 0.45), 0.20), ("Home_Wall_BrickRed", (0.58, 0.33, 0.26), 0.12),
         ("Home_Wall_Stucco", (0.90, 0.88, 0.83), 0.12)]
ROOFS = {"Home_Roof_Charcoal": (0.24, 0.24, 0.25), "Home_Roof_Grey": (0.45, 0.45, 0.44),
         "Home_Roof_Brown": (0.36, 0.29, 0.23), "Home_Roof_Tan": (0.58, 0.52, 0.42),
         "Home_Roof_Terracotta": (0.55, 0.32, 0.24), "Home_Roof_MetalLight": (0.78, 0.78, 0.76),
         "Home_Roof_MetalGreen": (0.27, 0.37, 0.31)}


@dataclass
class HomesConfig:
    enabled: bool = True
    margin_m: float = 400.0          # keep homes within this distance of the property outline
    overhang_m: float = 0.5
    pitch: float = 0.42              # roof rise per metre (5:12)
    max_rise_m: float = 3.5
    min_eave_m: float = 2.6
    default_height_m: float = 6.0
    min_area_m2: float = 12.0
    roof_grid_m: float = 1.0          # spacing of the roof's inset rings
    tile_m: float = 250.0
    seed: int = 7

    @classmethod
    def from_dict(cls, d: dict) -> "HomesConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _raster(polys, n, frame, shift=(0.0, 0.0)):
    from PIL import Image, ImageDraw
    res, h = frame.size_m / n, frame.size_m / 2
    im = Image.new("L", (n, n), 0); dr = ImageDraw.Draw(im)
    for p in polys:
        dr.polygon([((x + shift[0] + h) / res, (h - z - shift[1]) / res) for x, z in p.exterior.coords], fill=1)
    return np.asarray(im, bool)


def register(polys, building_mask, frame, max_m=4.0, step_m=0.5):
    """Shift (dx, dz) that best puts footprints on the LiDAR building returns."""
    if building_mask is None or not building_mask.any():
        return (0.0, 0.0), None
    n = building_mask.shape[0]; res = frame.size_m / n
    fp = _raster(polys, n, frame)
    best, score0 = (0.0, 0.0, -1.0), None
    k = int(round(max_m / res)); s = max(1, int(round(step_m / res)))
    rows, cols = np.nonzero(fp)
    for dr in range(-k, k + 1, s):
        for dc in range(-k, k + 1, s):
            r2, c2 = rows + dr, cols + dc
            ok = (r2 >= 0) & (r2 < n) & (c2 >= 0) & (c2 < n)
            sc = building_mask[r2[ok], c2[ok]].mean()
            if dr == 0 and dc == 0:
                score0 = sc
            if sc > best[2]:
                best = (dc * res, -dr * res, sc)
    return (best[0], best[1]), {"overlap_before": round(float(score0), 3), "overlap_after": round(float(best[2]), 3)}


def roof_probability(naip4, meta, footprints, step=3, seed=0):
    """Per-pixel roof probability from the 4-band aerial (about 0.9 m cells).

    A random forest learns roofs from the footprint dataset itself: pixels inside known
    footprints are roofs, pixels well away from any footprint are not. It then finds
    roofs the dataset misses (new houses, houses it skipped) and rejects things that
    only look like roofs to simple rules (shadowed tree crowns, pools, lawn)."""
    from PIL import Image, ImageDraw
    from scipy import ndimage
    from sklearn.ensemble import RandomForestClassifier
    a = np.asarray(naip4[::step, ::step]).astype(np.float32) / 255
    res = meta["res"] * step
    H, W = a.shape[:2]
    R, G, B, N = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    ndvi = (N - R) / np.maximum(N + R, 1e-3); lum = (R + G + B) / 3
    lm = lambda x, s: ndimage.uniform_filter(x, s)                       # noqa: E731
    ls = lambda x, s: np.sqrt(np.maximum(lm(x * x, s) - lm(x, s) ** 2, 0))  # noqa: E731
    F = np.stack([R, G, B, N, ndvi, lum, G - R, B - R, lm(lum, 5), ls(lum, 5), lm(lum, 15), ls(lum, 15),
                  lm(ndvi, 15), lm(ndvi, 31), ls(ndvi, 9), ndimage.sobel(lum, 0) ** 2 + ndimage.sobel(lum, 1) ** 2],
                 -1).reshape(H * W, -1)
    im = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(im)
    for p in footprints:
        dr.polygon([((x - meta["x0"]) / res, (meta["z1"] - z) / res) for x, z in p.exterior.coords], fill=1)
    fp = np.asarray(im, bool)
    pos = np.flatnonzero(ndimage.binary_erosion(fp, iterations=1))
    far = ~ndimage.binary_dilation(fp, iterations=int(5 / res))
    hard = np.flatnonzero((far & (ndimage.uniform_filter(ndvi, 3) < 0.25)).ravel())
    anyneg = np.flatnonzero(far.ravel())
    rng = np.random.default_rng(seed)
    pick = lambda ix, k: rng.choice(ix, min(len(ix), k), replace=False)  # noqa: E731
    pos, hard, anyneg = pick(pos, 60000), pick(hard, 60000), pick(anyneg, 40000)
    rf = RandomForestClassifier(60, max_depth=16, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    rf.fit(F[np.r_[pos, hard, anyneg]], np.r_[np.ones(len(pos)), np.zeros(len(hard) + len(anyneg))])
    P = np.empty(H * W, np.float32)
    for s0 in range(0, H * W, 2_000_000):
        P[s0:s0 + 2_000_000] = rf.predict_proba(F[s0:s0 + 2_000_000])[:, 1]
    return P.reshape(H, W), {"res": res, "x0": meta["x0"], "z1": meta["z1"]}


def mask_footprints(mask, res, x0, z1, min_area_m2=60.0, max_area_m2=2500.0):
    """House-shaped outlines from a mask on a north-up grid: boxy blobs become their
    minimum rotated rectangle, others a simplified outline; roads, walls, bluffs and
    other long or ragged shapes are skipped."""
    import shapely
    from shapely.geometry import Polygon
    from scipy import ndimage
    from skimage.measure import find_contours
    lab, k = ndimage.label(mask)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        blob = lab[sl] == i
        if not (min_area_m2 <= blob.sum() * res * res <= max_area_m2):
            continue
        c = max(find_contours(np.pad(blob.astype(float), 1), 0.5), key=len)
        pts = [(x0 + (sl[1].start + cc - 1 + 0.5) * res, z1 - (sl[0].start + rr - 1 + 0.5) * res) for rr, cc in c]
        p = Polygon(pts).buffer(0)
        if p.is_empty or p.geom_type != "Polygon":
            continue
        rect = p.minimum_rotated_rectangle
        rx, rz = np.array(rect.exterior.coords).T
        sides = sorted(np.hypot(np.diff(rx), np.diff(rz))[:2])
        fill = p.area / max(rect.area, 1e-6)
        aspect = sides[1] / max(sides[0], 1e-6)
        if fill < 0.6 or sides[0] < 5.0 or aspect > 7.0 or (aspect > 4.0 and sides[0] < 8.0):
            continue                       # roads, drives, walls: long and narrow (townhouse rows are wide)
        p = rect if fill > 0.8 else p.simplify(0.8, preserve_topology=True)
        out.append(shapely.geometry.polygon.orient(p, 1.0))
    return out


def _roof_class(rgb):
    """Aerial roof colour -> finish. The photo has a green/yellow cast, so roofs with
    little colour are greys by brightness; only clearly coloured roofs get a hue class."""
    r, g, b = (float(v) for v in rgb)
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    sat = max(r, g, b) - min(r, g, b)
    if sat < 0.09:
        return "Home_Roof_Charcoal" if lum < 0.33 else "Home_Roof_Grey" if lum < 0.58 else "Home_Roof_MetalLight"
    if g - max(r, b) > 0.05:
        return "Home_Roof_MetalGreen"
    if r - g > 0.12 and r - b > 0.15:
        return "Home_Roof_Terracotta"
    if lum < 0.40:
        return "Home_Roof_Brown"
    return "Home_Roof_Tan" if lum < 0.65 else "Home_Roof_MetalLight"


def build_homes(footprints: dict, frame, dem, base_elev, layout, cfg: HomesConfig,
                chm=None, building_mask=None, naip=None, naip_shift=(0.0, 0.0), boundary=None, log=print,
                naip4=None, naip4_meta=None, roads=None):
    import shapely
    from shapely.geometry import Polygon
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(cfg.seed)
    polys, src_h = [], []
    for f in footprints["features"]:
        ring = [frame.to_local(lo, la) for lo, la in f["geometry"]["coordinates"][0]]
        p = Polygon(ring)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.geom_type != "Polygon" or p.area < cfg.min_area_m2:
            continue
        polys.append(shapely.geometry.polygon.orient(p, 1.0)); src_h.append(float((f.get("properties") or {}).get("height", -1) or -1))
    h = frame.size_m / 2 - 2
    keep = [i for i, p in enumerate(polys) if max(abs(c) for xy in p.exterior.coords for c in xy) < h]
    if boundary:
        zone = Polygon(boundary).buffer(cfg.margin_m)
        keep = [i for i in keep if zone.contains(polys[i].centroid)]
    surf = [Polygon(s.rings[0]) for s in layout.surfaces if s.kind in ("green", "fairway", "tee", "bunker", "water")
            and len(s.rings[0]) >= 3]
    surf = shapely.union_all([s.buffer(0) for s in surf]) if surf else None
    dropped_course = 0
    if surf is not None:
        k2 = []
        for i in keep:
            if polys[i].intersection(surf).area > 0.1 * polys[i].area:
                dropped_course += 1
            else:
                k2.append(i)
        keep = k2
    polys = [polys[i] for i in keep]; src_h = [src_h[i] for i in keep]

    # roofs in the aerial that the footprint dataset misses (needs the 4-band aerial)
    found = []
    if naip4 is not None:
        P_roof, g = roof_probability(naip4, naip4_meta, polys)
        from scipy import ndimage
        cand = ndimage.binary_opening(ndimage.gaussian_filter(P_roof, 1) > 0.6, iterations=2)
        found = [shapely.affinity.translate(p, naip_shift[0], naip_shift[1])   # photo -> LiDAR frame
                 for p in mask_footprints(cand, g["res"], g["x0"], g["z1"])]
    # remove spikes and slivers (thin wall-like tails on some footprints)
    def _clean(p):
        q = p.buffer(-1.5, join_style="mitre").buffer(1.5, join_style="mitre")   # parts < 3 m wide
        if q.is_empty:
            return None
        if q.geom_type != "Polygon":
            q = max(q.geoms, key=lambda g_: g_.area)
        return shapely.geometry.polygon.orient(q.simplify(0.2), 1.0) if q.area >= cfg.min_area_m2 else None
    cleaned = [(_clean(p), hgt) for p, hgt in zip(polys, src_h)]
    polys = [p for p, _ in cleaned if p is not None]; src_h = [hgt for p, hgt in cleaned if p is not None]
    found = [q for q in (_clean(p) for p in found) if q is not None]
    shift, reg = register(polys, building_mask, frame)
    if any(shift):
        polys = [shapely.affinity.translate(p, shift[0], shift[1]) for p in polys]
    n_ms = len(polys)
    if found:
        have = shapely.union_all([p.buffer(2.0) for p in polys]) if polys else None
        zone = Polygon(boundary).buffer(cfg.margin_m) if boundary else None
        for p in found:
            if have is not None and p.intersects(have):
                continue
            if zone is not None and not zone.contains(p.centroid):
                continue
            if surf is not None and p.intersection(surf).area > 0.1 * p.area:
                continue
            polys.append(p); src_h.append(-1.0)
    log(f"homes: {n_ms} footprints + {len(polys) - n_ms} found in the aerial (dropped {dropped_course} on course surfaces), "
        f"registered {shift[0]:+.1f} m E {shift[1]:+.1f} m N onto the LiDAR")

    n = chm.shape[0] if chm is not None else None
    res = frame.size_m / n if n else None
    if chm is not None:
        from scipy import ndimage
        raised = ndimage.maximum_filter(chm, size=3) >= 2.5            # fills empty cells on roofs
        building_mask = raised if building_mask is None else (raised | building_mask)
    homes = []
    for idx, (P, mh) in enumerate(zip(polys, src_h)):
        # ---- height
        lidar_cov, top = 0.0, None
        if chm is not None:
            inner = P.buffer(-0.5)
            inner = inner if inner.geom_type == "Polygon" and inner.area > 4 else P
            m = _raster([inner], n, frame)
            if m.any():
                bm = building_mask[m] if building_mask is not None else np.zeros(m.sum(), bool)
                lidar_cov = float(bm.mean())
                if lidar_cov >= 0.5:                       # the house stood when the LiDAR was flown
                    vals = chm[m][bm]
                    top = float(np.percentile(vals[vals > 0], 90)) if (vals > 0).any() else None
        src = "lidar" if top is not None else ("footprint" if mh > 2.5 else "default")
        if top is None:
            top = mh if mh > 2.5 else cfg.default_height_m
        top = float(np.clip(top, 3.2, 10.0 if P.area < 900 else 16.0))   # tree crowns over a roof can read high
        # ---- ground
        xs, zs = np.array(P.exterior.coords).T
        g = dem.sample(np.append(xs, P.centroid.x), np.append(zs, P.centroid.y)) - base_elev
        ground = float(np.min(g)) - 0.25
        # ---- roof geometry
        Po = P.buffer(cfg.overhang_m, join_style="mitre", mitre_limit=3.0)
        Po = shapely.geometry.polygon.orient(Po if Po.geom_type == "Polygon" else max(Po.geoms, key=lambda q: q.area), 1.0)
        # roof points: the eave outline and its inward offsets (mitre insets). Inset corners
        # lie exactly on the hip/valley lines, so the triangles between rings are flat planes.
        rings, k = [], 0.0
        while True:
            q = Po if k == 0 else Po.buffer(-k, join_style="mitre", mitre_limit=10.0)
            if q.is_empty:
                break
            for g_ in (q.geoms if q.geom_type == "MultiPolygon" else [q]):
                rings.append(np.array(g_.exterior.coords)[:-1])
            k += cfg.roof_grid_m
        # the ridge: the inset just before it collapses (binary search), i.e. the ridge line
        lo, hi = max(k - cfg.roof_grid_m, 0.0), k
        for _ in range(12):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if not Po.buffer(-mid, join_style="mitre", mitre_limit=10.0).is_empty else (lo, mid)
        q = Po.buffer(-lo, join_style="mitre", mitre_limit=10.0)
        for g_ in (q.geoms if q.geom_type == "MultiPolygon" else [q]):
            if not g_.is_empty:
                rings.append(np.array(g_.exterior.coords)[:-1])
        pts = np.unique(np.round(np.vstack(rings), 3), axis=0)
        dist = shapely.distance(Po.exterior, shapely.points(pts))
        dmax = float(dist.max()) if len(dist) else 1.0
        span = top                                          # LiDAR/footprint height is above ground
        rise = min(cfg.pitch * dmax, cfg.max_rise_m, max(span - cfg.min_eave_m - 0.0, 0.6))
        eave_o = ground + span - rise                       # roof height at the overhang edge
        cap = rise / cfg.pitch
        zr = eave_o + cfg.pitch * np.minimum(dist, cap)
        tri = Delaunay(pts).simplices
        cen = pts[tri].mean(1)
        ok = shapely.contains_xy(Po, cen[:, 0], cen[:, 1])
        for a_, b_ in ((0, 1), (1, 2), (2, 0)):        # every edge must stay inside the outline
            mid = (pts[tri[:, a_]] + pts[tri[:, b_]]) / 2
            ok &= shapely.contains_xy(Po.buffer(0.05), mid[:, 0], mid[:, 1])
        tri = tri[ok]
        wall_top = eave_o + cfg.pitch * min(cfg.overhang_m, cap)
        # ---- colour
        roof_rgb = (0.4, 0.4, 0.4)
        if naip is not None:
            nn = naip.shape[0]; nr = frame.size_m / nn; hh = frame.size_m / 2
            c = P.centroid; w = max(2.0, min(6.0, math.sqrt(P.area) / 3))
            r0 = int((hh - (c.y - naip_shift[1]) - w) / nr); c0 = int((c.x - naip_shift[0] + hh - w) / nr)
            patch = naip[max(r0, 0):r0 + int(2 * w / nr), max(c0, 0):c0 + int(2 * w / nr), :3]
            px = patch.reshape(-1, 3).astype(float)
            if len(px):
                veg = (px[:, 1] > px[:, 0] * 1.04) & (px[:, 1] > px[:, 2] * 1.04)   # overhanging tree crowns
                if (~veg).sum() > 0.3 * len(px):
                    roof_rgb = tuple((np.median(px[~veg], axis=0) / 255).tolist())
        wall = WALLS[int(rng.choice(len(WALLS), p=[w_[2] for w_ in WALLS]))][0]
        homes.append({"id": idx, "footprint": [[round(x, 2), round(z, 2)] for x, z in P.exterior.coords[:-1]],
                      "area_m2": round(P.area, 1), "ground_y": round(ground, 2), "eave_y": round(wall_top, 2),
                      "ridge_y": round(float(zr.max()), 2), "height_m": round(top, 2),
                      "height_source": src, "lidar_coverage": round(lidar_cov, 2),
                      "wall": wall, "roof": _roof_class(roof_rgb), "roof_rgb": [round(v, 3) for v in roof_rgb],
                      "_roof": (np.c_[pts[:, 0], zr, pts[:, 1]], tri), "_wall_ring": np.array(P.exterior.coords)[:-1],
                      "_ground": ground, "_wall_top": wall_top, "_eave_o": eave_o, "_cap": cap, "_Po": Po, "_P": P})
    _dress(homes, roads, dem, base_elev, cfg, rng)
    return homes, {"count": len(homes), "registration_shift_m": list(shift), "registration": reg,
                   "dropped_on_course": dropped_course, "from_footprints": n_ms, "from_aerial": len(polys) - n_ms,
                   "height_sources": {k: sum(h_["height_source"] == k for h_ in homes) for k in ("lidar", "footprint", "default")}}


def _dress(homes, roads, dem, base_elev, cfg, rng):
    """Street side, front door, garage + driveway, chimney for each house."""
    import shapely
    from shapely.geometry import LineString, Point
    from shapely.ops import nearest_points
    net = shapely.union_all([LineString(r) for r in roads if len(r) >= 2]) if roads else None
    for hm in homes:
        P, ring = hm["_P"], hm["_wall_ring"]
        c = np.array([P.centroid.x, P.centroid.y])
        n_e = len(ring)
        edges = [(ring[i], ring[(i + 1) % n_e]) for i in range(n_e)]
        lens = np.array([np.hypot(*(b - a)) for a, b in edges])
        normals = np.array([np.array([(b - a)[1], -(b - a)[0]]) / max(np.hypot(*(b - a)), 1e-6) for a, b in edges])
        road_pt = None
        if net is not None and not net.is_empty:
            q = nearest_points(Point(*c), net)[1]
            if Point(*c).distance(q) < 90:
                road_pt = np.array([q.x, q.y])
        if road_pt is not None:
            d = (road_pt - c) / max(np.linalg.norm(road_pt - c), 1e-6)
            score = normals @ d + 0.02 * np.minimum(lens, 15)
        else:
            score = lens.copy()                                  # no street known: longest wall
        score[lens < 3.5] = -9
        fe = int(np.argmax(score))
        a, b = edges[fe]; L = lens[fe]; t = (b - a) / max(L, 1e-6); nrm = normals[fe]
        hm["front_edge"] = fe
        garage = hm["area_m2"] >= 140 and L >= 9.5 and road_pt is not None and rng.random() < 0.85
        g_at = lambda p: float(dem.sample(np.array([p[0]]), np.array([p[1]]))[0] - base_elev)  # noqa: E731
        parts = {}
        if garage:
            left = rng.random() < 0.5
            s0 = 0.8 if left else L - 0.8 - 4.9
            gc = a + t * (s0 + 2.45)
            parts["garage"] = (a + t * s0, a + t * (s0 + 4.9), g_at(gc + nrm * 1.0))
            door_s = (s0 + 4.9 + 1.6) if left else (s0 - 2.6)
            door_s = float(np.clip(door_s, 0.6, L - 1.6))
            # driveway: from the garage straight out to the street
            q = nearest_points(Point(*(gc + nrm * 0.5)), net)[1]
            end = np.array([q.x, q.y])
            dl = np.linalg.norm(end - gc)
            if 3 < dl < 70:
                k = max(2, int(dl / 2.0))
                cl = [gc + nrm * 0.1 + (end - gc - nrm * 0.1) * i / k for i in range(k + 1)]
                hs = dem.sample(np.array([p[0] for p in cl]), np.array([p[1] for p in cl])) - base_elev
                parts["driveway"] = (np.array(cl), hs + 0.06)
        else:
            door_s = L / 2 - 0.5
        dp = a + t * door_s
        parts["door"] = (dp, dp + t * 1.0, g_at(dp + t * 0.5 + nrm * 1.0))
        # chimney on a third of the larger homes, near a side or back wall
        if hm["area_m2"] >= 180 and rng.random() < 0.35:
            others = [i for i in range(n_e) if i != fe and lens[i] >= 4]
            if others:
                i = others[int(rng.integers(len(others)))]
                ea, eb = edges[i]; et = (eb - ea) / lens[i]
                cpos = ea + et * lens[i] * rng.uniform(0.3, 0.7) - normals[i] * 0.9
                if P.contains(Point(*cpos)):
                    parts["chimney"] = (cpos, et, normals[i])
        hm["_parts"] = parts
        hm["garage"] = "garage" in parts
        hm["chimney"] = "chimney" in parts


class _Mesh:
    """Triangle soup with UVs, a material and a facing hint per triangle (internal frame)."""
    def __init__(self):
        self.V, self.UV, self.T, self.M, self.N = [], [], [], [], []

    def quad(self, p, uv, mat, n):
        b = len(self.V); self.V += list(p); self.UV += list(uv)
        self.T += [(b, b + 1, b + 2), (b, b + 2, b + 3)]; self.M += [mat, mat]; self.N += [n, n]

    def tri(self, p, uv, mat, n):
        b = len(self.V); self.V += list(p); self.UV += list(uv)
        self.T.append((b, b + 1, b + 2)); self.M.append(mat); self.N.append(n)


def _v3(xz, y):
    return np.array([xz[0], y, xz[1]], float)


def _mesh(home, cfg=None):
    """Roof (shingle UVs along each face), soffit, fascia, walls, foundation band,
    front door, garage door, chimney and driveway. Internal frame: x east, y up, z north."""
    from scipy.spatial import Delaunay
    import shapely
    mk = _Mesh()
    up = np.array([0.0, 1.0, 0.0])
    # --- roof: per-triangle UVs (u along the eave, v up the slope) so shingle rows follow each face
    rv, rt = home["_roof"]
    for t_ in rt:
        P3 = rv[t_]
        nrm = np.cross(P3[1] - P3[0], P3[2] - P3[0]); nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
        if nrm[1] < 0:
            nrm = -nrm
        h = np.array([nrm[0], nrm[2]]); hl = np.linalg.norm(h)
        if hl < 1e-3:
            uv = [(p[0] / 4, p[2] / 4) for p in P3]
        else:
            down = h / hl; along = np.array([-down[1], down[0]])
            slope = math.sqrt(1 + (hl / max(nrm[1], 1e-3)) ** 2)
            uv = [(float(np.dot([p[0], p[2]], along)) / 4, -float(np.dot([p[0], p[2]], down)) * slope / 4) for p in P3]
        mk.tri(P3, uv, home["roof"], up)
    ring = home["_wall_ring"]; g, top = home["_ground"], home["_wall_top"]
    eo = home["_eave_o"]; Po, P = home["_Po"], home["_P"]
    # --- soffit (underside of the overhang) and fascia board around the eaves
    sof = Po.difference(P)
    if not sof.is_empty:
        polys = sof.geoms if sof.geom_type == "MultiPolygon" else [sof]
        for pg in polys:
            pts = np.vstack([np.array(r.coords)[:-1] for r in [pg.exterior, *pg.interiors]])
            if len(pts) < 3:
                continue
            tri = Delaunay(pts).simplices
            cen = pts[tri].mean(1)
            tri = tri[shapely.contains_xy(pg, cen[:, 0], cen[:, 1])]
            for t_ in tri:
                mk.tri([_v3(pts[i], eo - 0.02) for i in t_], [(pts[i][0] / 4, pts[i][1] / 4) for i in t_], "Home_Trim", -up)
    pr = np.array(Po.exterior.coords)[:-1]
    for i in range(len(pr)):
        a, b = pr[i], pr[(i + 1) % len(pr)]
        n2 = np.array([(b - a)[1], -(b - a)[0]]); n2 /= max(np.linalg.norm(n2), 1e-6)
        mk.quad([_v3(a, eo - 0.22), _v3(b, eo - 0.22), _v3(b, eo + 0.01), _v3(a, eo + 0.01)],
                [(0, 0), (1, 0), (1, 1), (0, 1)], "Home_Trim", np.array([n2[0], 0, n2[1]]))
    # --- walls (3 m window bays, one storey per 3 m) and a stone foundation band
    along = 0.0
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        L = float(np.hypot(*(b - a)))
        n2 = np.array([(b - a)[1], -(b - a)[0]]) / max(L, 1e-6); n3 = np.array([n2[0], 0, n2[1]])
        u0, u1 = along / 3, (along + L) / 3
        mk.quad([_v3(a, g), _v3(b, g), _v3(b, top), _v3(a, top)],
                [(u0, 0), (u1, 0), (u1, (top - g) / 3), (u0, (top - g) / 3)], home["wall"], n3)
        fa, fb = a + n2 * 0.03, b + n2 * 0.03
        mk.quad([_v3(fa, g), _v3(fb, g), _v3(fb, g + 0.7), _v3(fa, g + 0.7)],
                [(u0, 0), (u1, 0), (u1, 0.93), (u0, 0.93)], "Home_Foundation", n3)
        along += L
    parts = home.get("_parts", {})
    fe = home.get("front_edge", 0)
    a, b = ring[fe], ring[(fe + 1) % len(ring)]
    L = float(np.hypot(*(b - a))); n2 = np.array([(b - a)[1], -(b - a)[0]]) / max(L, 1e-6); n3 = np.array([n2[0], 0, n2[1]])
    if "garage" in parts:
        p0, p1, gy = parts["garage"]; p0, p1 = p0 + n2 * 0.06, p1 + n2 * 0.06
        y0 = max(gy, g + 0.25)
        mk.quad([_v3(p0, y0), _v3(p1, y0), _v3(p1, y0 + 2.15), _v3(p0, y0 + 2.15)],
                [(0, 0), (1, 0), (1, 1), (0, 1)], "Home_GarageDoor", n3)
    if "door" in parts:
        p0, p1, gy = parts["door"]; p0, p1 = p0 + n2 * 0.06, p1 + n2 * 0.06
        y0 = max(gy + 0.15, g + 0.4)
        mk.quad([_v3(p0, y0), _v3(p1, y0), _v3(p1, y0 + 2.3), _v3(p0, y0 + 2.3)],
                [(0, 0), (1, 0), (1, 1), (0, 1)], "Home_FrontDoor", n3)
        # small porch step in concrete
        s0, s1 = p0 - (p1 - p0) * 0.4 + n2 * 0.0, p1 + (p1 - p0) * 0.4
        mk.quad([_v3(s0, y0), _v3(s1, y0), _v3(s1 + n2 * 1.2, y0), _v3(s0 + n2 * 1.2, y0)],
                [(0, 0), (0.5, 0), (0.5, 0.3), (0, 0.3)], "Home_Concrete", up)
    if "chimney" in parts:
        c, et, en = parts["chimney"]
        ridge = home["ridge_y"]
        y0, y1 = home["_wall_top"] - 0.3, ridge + 0.9
        cs = [c - et * 0.9 - en * 0.5, c + et * 0.9 - en * 0.5, c + et * 0.9 + en * 0.5, c - et * 0.9 + en * 0.5]
        for i in range(4):
            q0, q1 = cs[i], cs[(i + 1) % 4]
            sd = (q0 + q1) / 2 - c; sd = sd / max(np.linalg.norm(sd), 1e-6)
            w_ = float(np.linalg.norm(q1 - q0)) / 3
            mk.quad([_v3(q0, y0), _v3(q1, y0), _v3(q1, y1), _v3(q0, y1)],
                    [(0, 0), (w_, 0), (w_, (y1 - y0) / 3), (0, (y1 - y0) / 3)], "Home_Wall_Stone", np.array([sd[0], 0, sd[1]]))
        mk.quad([_v3(q, y1) for q in cs], [(0, 0), (0.3, 0), (0.3, 0.3), (0, 0.3)], "Home_Trim", up)
    if "driveway" in parts:
        cl, hs = parts["driveway"]
        d = cl[-1] - cl[0]; d = d / max(np.linalg.norm(d), 1e-6); side = np.array([-d[1], d[0]]) * 2.75
        for i in range(len(cl) - 1):
            q = [cl[i] - side, cl[i] + side, cl[i + 1] + side, cl[i + 1] - side]
            y = [hs[i], hs[i], hs[i + 1], hs[i + 1]]
            v0, v1 = i * 2.0 / 4, (i + 1) * 2.0 / 4
            mk.quad([_v3(q[k], y[k]) for k in range(4)], [(0, v0), (1.4, v0), (1.4, v1), (0, v1)], "Home_Concrete", up)
    return (np.array(mk.V, float), np.array(mk.UV, float), np.array(mk.T, int), mk.M, np.array(mk.N, float))


def _face(v_obj, tri, hint_obj):
    """Flip triangles so right-handed normals agree with the facing hint."""
    a, b, c = v_obj[tri[:, 0]], v_obj[tri[:, 1]], v_obj[tri[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), hint_obj) < 0
    tri = tri.copy(); tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri


def write_homes(homes: list[dict], out_dir: Path, cfg: HomesConfig) -> dict:
    from .home_textures import write_textures
    out_dir.mkdir(parents=True, exist_ok=True)
    tex = write_textures(out_dir, [w[0] for w in WALLS], list(ROOFS))
    with open(out_dir / "homes.mtl", "w") as f:
        for name, fn in tex.items():
            ks = "0.05 0.05 0.05" if "Roof" in name else "0 0 0"
            f.write(f"newmtl {name}\nKd 1 1 1\nKs {ks}\nmap_Kd {fn}\n\n")
    tiles: dict = {}
    for hm in homes:
        c = np.mean(hm["_wall_ring"], axis=0)
        tiles.setdefault((int(math.floor(c[0] / cfg.tile_m)), int(math.floor(c[1] / cfg.tile_m))), []).append(hm)
    nv = nt = 0
    flipz = np.array([1.0, 1.0, -1.0])
    with open(out_dir / "homes.obj", "w") as f:
        f.write("# Homes around the course: metres, right-handed Y-up, x east, z SOUTH (as course_surfaces.obj)\n")
        f.write("mtllib homes.mtl\n")
        base = 1
        for (tx, tz), hs in sorted(tiles.items()):
            groups: dict = {}
            verts, uvs = [], []
            for hm in hs:
                v, uv, t, mats, hint = _mesh(hm, cfg)
                if not len(t):
                    continue
                vo = v * flipz
                t = _face(vo, t, hint * flipz)
                o = sum(len(x) for x in verts)
                verts.append(vo); uvs.append(uv)
                for tri, m in zip(t + o, mats):
                    groups.setdefault(m, []).append(tri)
            if not verts:
                continue
            V = np.vstack(verts); UV = np.vstack(uvs)
            name = f"Homes_{tx + 50:02d}_{tz + 50:02d}"
            f.write(f"o {name}\ng {name}\n")
            np.savetxt(f, V, fmt="v %.3f %.3f %.3f")
            np.savetxt(f, UV, fmt="vt %.4f %.4f")
            for m, tris in groups.items():
                f.write(f"usemtl {m}\n")
                tt = np.array(tris) + base
                np.savetxt(f, np.column_stack([tt[:, 0], tt[:, 0], tt[:, 1], tt[:, 1], tt[:, 2], tt[:, 2]]), fmt="f %d/%d %d/%d %d/%d")
                nt += len(tt)
            base += len(V); nv += len(V)
    pub = [{k: v for k, v in hm.items() if not k.startswith("_")} for hm in homes]
    (out_dir / "homes.json").write_text(json.dumps({"homes": pub}, indent=0))
    return {"tiles": len(tiles), "vertices": nv, "triangles": nt,
            "garages": sum(h.get("garage", False) for h in homes), "chimneys": sum(h.get("chimney", False) for h in homes)}


def homes_mask_filter(trees: list[dict], homes: list[dict], clear_m: float = 1.0) -> list[dict]:
    """Drop trees standing inside a house footprint (e.g. lots built after the LiDAR)."""
    import shapely
    from shapely.geometry import Polygon
    if not homes or not trees:
        return trees
    zone = shapely.union_all([Polygon(h["_wall_ring"]).buffer(clear_m) for h in homes])
    xs = np.array([t["x"] for t in trees]); zs = np.array([t["z"] for t in trees])
    inside = shapely.contains_xy(zone, xs, zs)
    return [t for t, i in zip(trees, inside) if not i]
