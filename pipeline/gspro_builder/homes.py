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
                naip4=None, naip4_meta=None):
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
        q = p.buffer(-0.6, join_style="mitre").buffer(0.6, join_style="mitre")
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
                      "_ground": ground, "_wall_top": wall_top})
    return homes, {"count": len(homes), "registration_shift_m": list(shift), "registration": reg,
                   "dropped_on_course": dropped_course, "from_footprints": n_ms, "from_aerial": len(polys) - n_ms,
                   "height_sources": {k: sum(h_["height_source"] == k for h_ in homes) for k in ("lidar", "footprint", "default")}}


def _mesh(home):
    """Triangles in the internal frame (x east, y up, z north): roof + walls, with UVs."""
    V, T, UV, M = [], [], [], []
    rv, rt = home["_roof"]
    V.append(rv); UV.append(np.c_[rv[:, 0] / 4, rv[:, 2] / 4]); T.append(rt); M += [home["roof"]] * len(rt)
    ring = home["_wall_ring"]; g, t = home["_ground"], home["_wall_top"]
    off = len(rv); along = 0.0
    wv, wt, wuv = [], [], []
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        L = float(np.hypot(*(b - a)))
        k = len(wv)
        wv += [(a[0], g, a[1]), (b[0], g, b[1]), (b[0], t, b[1]), (a[0], t, a[1])]
        wuv += [(along / 3, 0), ((along + L) / 3, 0), ((along + L) / 3, (t - g) / 3), (along / 3, (t - g) / 3)]
        wt += [(off + k, off + k + 1, off + k + 2), (off + k, off + k + 2, off + k + 3)]
        along += L
    V.append(np.array(wv)); UV.append(np.array(wuv)); T.append(np.array(wt)); M += [home["wall"]] * len(wt)
    return np.vstack(V), np.vstack(UV), np.vstack(T), M


def _outward(v_obj, tri, center_obj, roof_mask):
    """Flip triangles so right-handed normals face out (walls) or up (roofs)."""
    a, b, c = v_obj[tri[:, 0]], v_obj[tri[:, 1]], v_obj[tri[:, 2]]
    nrm = np.cross(b - a, c - a)
    out = np.where(roof_mask[:, None], np.array([[0, 1, 0]]), (a + b + c) / 3 - center_obj)
    flip = np.einsum("ij,ij->i", nrm, out) < 0
    tri = tri.copy(); tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri


def write_homes(homes: list[dict], out_dir: Path, cfg: HomesConfig) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "homes.mtl", "w") as f:
        for name, rgb, _ in WALLS:
            f.write(f"newmtl {name}\nKd {rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f}\nKs 0 0 0\n\n")
        for name, rgb in ROOFS.items():
            f.write(f"newmtl {name}\nKd {rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f}\nKs 0.05 0.05 0.05\n\n")
    tiles: dict = {}
    for hm in homes:
        c = np.mean(hm["_wall_ring"], axis=0)
        tiles.setdefault((int(math.floor(c[0] / cfg.tile_m)), int(math.floor(c[1] / cfg.tile_m))), []).append(hm)
    nv = nt = 0
    with open(out_dir / "homes.obj", "w") as f:
        f.write("# Homes around the course: metres, right-handed Y-up, x east, z SOUTH (as course_surfaces.obj)\n")
        f.write("mtllib homes.mtl\n")
        base = 1
        for (tx, tz), hs in sorted(tiles.items()):
            groups: dict = {}
            verts, uvs = [], []
            for hm in hs:
                v, uv, t, mats = _mesh(hm)
                vo = v * np.array([1.0, 1.0, -1.0])
                cen = np.array([np.mean(hm["_wall_ring"][:, 0]), 0.0, -np.mean(hm["_wall_ring"][:, 1])])
                roof_mask = np.array([m.startswith("Home_Roof") for m in mats])
                cen_rows = np.repeat(cen[None], len(t), 0); cen_rows[:, 1] = vo[t].mean(1)[:, 1]
                t = _outward(vo, t, cen_rows, roof_mask)
                o = sum(len(x) for x in verts)
                verts.append(vo); uvs.append(uv)
                for tri, m in zip(t + o, mats):
                    groups.setdefault(m, []).append(tri)
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
    return {"tiles": len(tiles), "vertices": nv, "triangles": nt}


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
