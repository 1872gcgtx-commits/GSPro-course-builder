"""Tree species and understory for Central Texas courses (Salado / Edwards Plateau edge).

Each LiDAR tree gets evidence from three sources:

  crown shape (LiDAR)   height, crown width / height, how peaked the top is,
                        how solid the crown is (share of LiDAR returns inside it)
  colour (4-band NAIP)  brightness, greenness, blue tint, NDVI and texture inside the crown
  setting               distance to water and height above it (creek bottom vs upland)

and is scored against the local species, each described the way an arborist would:

  live_oak       broad, flat-topped dome, wider than tall; dark glossy green, dense
  ashe_juniper   ("cedar") narrow or peaked; dark blue-green; uplands, often in groves
  cedar_elm      upright oval; mid green; common everywhere, more in the bottoms
  pecan          tall (15 m+), big open crown; light green; creek bottoms
  bald_cypress   tall, narrow, peaked; right on the creek bank
  mesquite       small and airy; light grey-green, see-through crown; open ground
  hackberry      mid-size, irregular, mid green; fence lines and woods edges

There are no labels to learn from, so the scores are rules built on those
descriptions; the winning score is kept as a confidence. Each tree also gets a
crown base height (where the branches start), a colour tint relative to its
species (from the photo) and a size class so Unity can pick young / mature / old
models. Understory (brush and young cedar 0.6-3 m tall) is sampled separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPECIES = {
    #               crown base (share of height), Unity prefab hint
    "live_oak":     (0.22, "Live oak (Quercus virginiana / fusiformis)"),
    "ashe_juniper": (0.05, "Ashe juniper / mountain cedar (Juniperus ashei)"),
    "cedar_elm":    (0.35, "Cedar elm (Ulmus crassifolia)"),
    "pecan":        (0.40, "Pecan (Carya illinoinensis)"),
    "bald_cypress": (0.20, "Bald cypress (Taxodium distichum)"),
    "mesquite":     (0.18, "Honey mesquite (Prosopis glandulosa)"),
    "hackberry":    (0.30, "Sugar hackberry (Celtis laevigata)"),
}


@dataclass
class SpeciesConfig:
    enabled: bool = True
    shrubs: bool = True
    shrub_spacing_m: float = 3.0
    seed: int = 11

    @classmethod
    def from_dict(cls, d: dict) -> "SpeciesConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _naip_sampler(naip4, meta, shift):
    """(x, z) LiDAR-frame metres -> RGBN pixel values (photo is offset by `shift`)."""
    H, W = naip4.shape[:2]

    def at(x, z):
        c = np.clip(((np.asarray(x) - shift[0] - meta["x0"]) / meta["res"]).astype(int), 0, W - 1)
        r = np.clip(((meta["z1"] - (np.asarray(z) - shift[1])) / meta["res"]).astype(int), 0, H - 1)
        return naip4[r, c].astype(np.float32) / 255
    return at


def tree_features(trees, chm, frame, naip4=None, meta=None, shift=(0.0, 0.0), water=None, dem=None):
    """Per-tree feature table (dict of arrays)."""
    from scipy import ndimage
    n = chm.shape[0]; res = frame.size_m / n; h = frame.size_m / 2
    x = np.array([t["x"] for t in trees]); z = np.array([t["z"] for t in trees])
    H = np.array([t["height_m"] for t in trees]); R = np.array([t["crown_radius_m"] for t in trees])
    rows = np.clip(((h - z) / res).astype(int), 0, n - 1); cols = np.clip(((x + h) / res).astype(int), 0, n - 1)
    sm = ndimage.gaussian_filter(chm.astype(np.float32), 0.6 / res)
    # crown shape: mean height on a ring at half the crown radius (peaked crowns drop fast)
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    def ring(frac):
        rr = rows[:, None] - (frac * R[:, None] * np.sin(ang)[None] / res)
        cc = cols[:, None] + (frac * R[:, None] * np.cos(ang)[None] / res)
        return sm[np.clip(rr.astype(int), 0, n - 1), np.clip(cc.astype(int), 0, n - 1)]
    half = ring(0.5).mean(1); edge = ring(0.85)
    f = {"height": H, "radius": R, "spread": R / np.maximum(H, 0.5),
         "peak": (H - half) / np.maximum(H, 0.5),                      # 0 flat top .. 0.5 spire
         "solid": (edge > 0.3 * H[:, None]).mean(1)}                   # crown filled to its edge
    # raw (unsmoothed) returns inside the crown: see-through crowns have holes
    holes = np.zeros(len(trees))
    for k, (r0, c0, rad) in enumerate(zip(rows, cols, R)):
        w = max(1, int(0.6 * rad / res))
        patch = chm[max(r0 - w, 0):r0 + w + 1, max(c0 - w, 0):c0 + w + 1]
        holes[k] = float((patch < 0.3 * H[k]).mean()) if patch.size else 0.0
    f["holes"] = holes
    # surrounding canopy cover (share of ground under >3 m canopy within 15 m): woods vs open
    cover = ndimage.uniform_filter((chm > 3).astype(np.float32), int(30 / res) | 1)
    f["cover"] = cover[rows, cols]
    # trees found only in the aerial have no LiDAR crown: hold their shape scores neutral
    aerial = np.array([t.get("source") == "aerial" for t in trees])
    if aerial.any() and (~aerial).any():
        for k in ("peak", "solid", "holes"):
            f[k][aerial] = np.median(f[k][~aerial])
        f["cover"][aerial] = np.maximum(f["cover"][aerial], np.median(f["cover"][~aerial]) * 0.5)
    if naip4 is not None:
        at = _naip_sampler(naip4, meta, shift)
        off = np.array([(0, 0), (0.35, 0), (-0.35, 0), (0, 0.35), (0, -0.35)])
        px = np.stack([at(x + dx * R, z + dz * R) for dx, dz in off], 1)   # (n, 5, 4)
        rgbn = px.mean(1)
        r_, g_, b_, nir = rgbn.T
        f.update({"lum": (r_ + g_ + b_) / 3, "green": g_ - (r_ + b_) / 2, "blue": b_ - r_,
                  "ndvi": (nir - r_) / np.maximum(nir + r_, 1e-3), "texture": px[..., 3].std(1),
                  "rgb": rgbn[:, :3]})
    if water is not None and water.any():
        d, (wr, wc) = ndimage.distance_transform_edt(~water, return_indices=True)
        f["water_dist"] = d[rows, cols] * res
        if dem is not None:
            g = dem.data
            f["above_water"] = g[rows, cols] - g[wr[rows, cols], wc[rows, cols]]
    return f


def _z(v, lo, hi):
    """0 below lo, 1 above hi, linear between."""
    return np.clip((v - lo) / (hi - lo), 0, 1)


def classify(f) -> tuple[np.ndarray, np.ndarray, dict]:
    """Species per tree from the feature table. Returns (names, confidence, scores).

    Setting decides first (creek bank, creek bottom, open ground, everywhere else),
    then crown shape and photo colour choose among the species that grow there. Shape
    and colour are compared with this course's own trees (robust z-scores): LiDAR
    density, photo season and haze vary between courses, the relative order doesn't."""
    n = len(f["height"])
    def z(v):
        v = np.asarray(v, float)
        return (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))) + 1e-6)
    H = np.asarray(f["height"], float)
    spz, pkz, hol = z(f["spread"]), z(f["peak"]), z(f["holes"])
    lz = z(f.get("lum", np.zeros(n))); gz = z(f.get("green", np.zeros(n))); bz = z(f.get("blue", np.zeros(n)))
    nz = z(f.get("ndvi", np.zeros(n)))
    cov = np.asarray(f.get("cover", np.full(n, 0.5)))
    wd = np.asarray(f.get("water_dist", np.full(n, 500.0))); aw = np.asarray(f.get("above_water", np.full(n, 20.0)))
    bank = (wd < 20) & (aw < 2.5)
    bottom = (wd < 150) & (aw < 5.0)
    open_ = cov < 0.15
    # crown shape + colour scores for the general mix
    s = {
        "live_oak": _z(spz, -0.2, 1.0) + _z(-pkz, -0.3, 0.8) + _z(-lz, -0.3, 0.9) + 0.6 * _z(-hol, -0.3, 0.8),
        "ashe_juniper": _z(-spz, -0.2, 0.9) + _z(pkz, -0.2, 0.9) + _z(-lz, -0.1, 1.0) + _z(bz, -0.3, 0.8)
                        + 0.4 * _z(-cov, -0.6, -0.2) * 0 + 0.3 * _z(cov, 0.2, 0.5),
        "cedar_elm": _z(lz, -0.4, 0.6) + _z(gz, -0.3, 0.7) + 0.6 * _z(H, 6, 10) + 0.4 * (1 - _z(abs(spz), 0.8, 1.6)),
        "hackberry": 0.8 * _z(hol, 0.2, 1.2) + 0.5 * _z(gz, 0.0, 1.0) + 0.5 * _z(lz, 0.0, 1.0) + 0.2,
    }
    s["pecan"] = np.where(bottom & (H >= 12), 4.0 + _z(H, 12, 18), -9.0)
    s["bald_cypress"] = np.where(bank & (H >= 9), 3.5 + _z(pkz, -0.5, 1.0) + _z(-spz, -0.5, 1.0), -9.0)
    s["mesquite"] = np.where(open_ & (H < 7.5) & ~bottom,
                             1.2 + _z(lz, 0.0, 1.0) + _z(-nz, 0.0, 1.0) + 0.6 * _z(hol, 0.0, 1.0), -9.0)
    # the creek bottom has no junipers or mesquite; its general mix is elm and hackberry
    s["ashe_juniper"] = np.where(bottom, -9.0, s["ashe_juniper"])
    s["cedar_elm"] = np.where(bottom, s["cedar_elm"] + 0.8, s["cedar_elm"])
    names = list(s)
    S = np.stack([s[k] for k in names], 1)
    best = S.argmax(1)
    srt = np.sort(S, 1)
    conf = np.clip((srt[:, -1] - srt[:, -2]) / 1.5 + 0.4, 0, 1)
    return np.array(names)[best], conf, s


def assign_species(trees, chm, frame, naip4=None, meta=None, shift=(0.0, 0.0), water=None, dem=None, seed=11):
    """Add species, confidence, crown base, colour tint, size class and rotation to each tree."""
    if not trees:
        return trees, {}
    f = tree_features(trees, chm, frame, naip4, meta, shift, water, dem)
    sp, conf, _ = classify(f)
    rng = np.random.default_rng(seed)
    out_counts: dict = {}
    for k, t in enumerate(trees):
        name = str(sp[k])
        t["species"] = name
        t["type"] = "conifer" if name in ("ashe_juniper", "bald_cypress") else "deciduous"
        t["species_conf"] = round(float(conf[k]), 2)
        t["crown_base_m"] = round(SPECIES[name][0] * t["height_m"], 2)
        t["rotation_deg"] = int(rng.integers(0, 360))
        out_counts[name] = out_counts.get(name, 0) + 1
    # size class within species (young / mature / old) and photo tint relative to the species
    for name in out_counts:
        idx = [k for k, t in enumerate(trees) if t["species"] == name]
        hs = np.array([trees[k]["height_m"] for k in idx])
        q1, q2 = np.percentile(hs, [33, 80])
        rgb = f["rgb"][idx] if "rgb" in f else None
        mean = rgb.mean(0) if rgb is not None else None
        for j, k in enumerate(idx):
            trees[k]["size_class"] = "young" if hs[j] < q1 else ("old" if hs[j] > q2 else "mature")
            if rgb is not None:
                tint = np.clip(rgb[j] / np.maximum(mean, 1e-3), 0.75, 1.25)
                trees[k]["tint"] = [round(float(v), 3) for v in tint]
    return trees, out_counts


def detect_shrubs(chm, frame, naip4, meta, shift, exclude, trees, dem, base_elev, cfg: SpeciesConfig):
    """Understory: 0.6-3 m vegetation (brush, young cedar, yaupon) as instance points."""
    from scipy import ndimage
    n = chm.shape[0]; res = frame.size_m / n; h = frame.size_m / 2
    low = (chm > 0.6) & (chm < 3.0) & ~exclude
    step = max(1, int(round(cfg.shrub_spacing_m / res)))
    rng = np.random.default_rng(cfg.seed)
    rr, cc = np.nonzero(low[::step, ::step])
    rr = rr * step + rng.integers(0, step, len(rr)); cc = cc * step + rng.integers(0, step, len(cc))
    rr = np.clip(rr, 0, n - 1); cc = np.clip(cc, 0, n - 1)
    keep = low[rr, cc]
    rr, cc = rr[keep], cc[keep]
    x = -h + (cc + 0.5) * res; z = h - (rr + 0.5) * res
    if trees:                                     # not inside a tree trunk zone
        from scipy.spatial import cKDTree
        tx = np.array([[t["x"], t["z"]] for t in trees])
        d, _ = cKDTree(tx).query(np.c_[x, z])
        ok = d > 1.5
        x, z, rr, cc = x[ok], z[ok], rr[ok], cc[ok]
    kind = np.full(len(x), "brush", dtype=object)
    if naip4 is not None:
        px = _naip_sampler(naip4, meta, shift)(x, z)
        ndvi = (px[:, 3] - px[:, 0]) / np.maximum(px[:, 3] + px[:, 0], 1e-3)
        keep = ndvi > 0.15                           # living green, not fences or junk
        lum = px[:, :3].mean(1)
        dark = lum < np.percentile(lum[keep], 35) if keep.any() else np.zeros(len(x), bool)
        kind = np.where(dark, "juniper_shrub", "brush")
        x, z, rr, cc, kind = x[keep], z[keep], rr[keep], cc[keep], kind[keep]
    y = dem.sample(x, z) - base_elev
    hgt = chm[rr, cc]
    return [{"x": round(float(a), 2), "y": round(float(b), 3), "z": round(float(c), 2), "height_m": round(float(d), 2),
             "kind": str(k), "rotation_deg": int(rng.integers(0, 360))} for a, b, c, d, k in zip(x, y, z, hgt, kind)]


def drop_ghosts(trees, P, g, min_prob=0.35):
    """Remove trees the LiDAR saw that are gone in the (newer) aerial: no tree crown there
    now (canopy probability from canopy_probability, which tells crowns from green grass)."""
    if not trees or P is None:
        return trees, 0
    Hh, W = P.shape
    x = np.array([t["x"] for t in trees]); z = np.array([t["z"] for t in trees]); R = np.array([t["crown_radius_m"] for t in trees])
    best = np.zeros(len(trees))
    for dx, dz in ((0, 0), (0.4, 0), (-0.4, 0), (0, 0.4), (0, -0.4)):
        c = np.clip(((x + dx * R - g["x0"]) / g["res"]).astype(int), 0, W - 1)
        r = np.clip(((g["z1"] - (z + dz * R)) / g["res"]).astype(int), 0, Hh - 1)
        best = np.maximum(best, P[r, c])
    keep = best >= min_prob
    return [t for t, k in zip(trees, keep) if k], int((~keep).sum())


def canopy_probability(naip4, meta, chm, frame, shift, step=3, seed=0):
    """Tree-canopy probability from the 4-band aerial (~0.9 m), trained on the LiDAR:
    canopy = LiDAR >3 m and green; not canopy = LiDAR <0.5 m (turf, fields, roads, roofs)."""
    from scipy import ndimage
    from sklearn.ensemble import RandomForestClassifier
    a = np.asarray(naip4[::step, ::step]).astype(np.float32) / 255
    res = meta["res"] * step
    Hh, W = a.shape[:2]
    R, G, B, N = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    ndvi = (N - R) / np.maximum(N + R, 1e-3); lum = (R + G + B) / 3
    lm = lambda x, s: ndimage.uniform_filter(x, s)                       # noqa: E731
    ls = lambda x, s: np.sqrt(np.maximum(lm(x * x, s) - lm(x, s) ** 2, 0))  # noqa: E731
    F = np.stack([R, G, B, N, ndvi, lum, G - R, B - R, lm(lum, 5), ls(lum, 5), ls(N, 5), lm(ndvi, 7),
                  ls(ndvi, 7), lm(lum, 15), ls(lum, 15), ndimage.minimum_filter(lum, 5)], -1).reshape(Hh * W, -1)
    # LiDAR canopy height at each aerial cell (photo + shift = LiDAR frame)
    n = chm.shape[0]; cres = frame.size_m / n; h = frame.size_m / 2
    xs = meta["x0"] + (np.arange(W) + 0.5) * res + shift[0]
    zs = meta["z1"] - (np.arange(Hh) + 0.5) * res + shift[1]
    ci = np.clip(((xs + h) / cres).astype(int), 0, n - 1); ri = np.clip(((h - zs) / cres).astype(int), 0, n - 1)
    hmax = ndimage.maximum_filter(chm, 3)[np.ix_(ri, ci)]
    pos = np.flatnonzero(((hmax > 3) & (ndvi > 0.3)).ravel())
    neg = np.flatnonzero((hmax < 0.5).ravel())
    rng = np.random.default_rng(seed)
    pos = rng.choice(pos, min(len(pos), 80000), replace=False); neg = rng.choice(neg, min(len(neg), 80000), replace=False)
    rf = RandomForestClassifier(60, max_depth=16, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    rf.fit(F[np.r_[pos, neg]], np.r_[np.ones(len(pos)), np.zeros(len(neg))])
    P = np.empty(Hh * W, np.float32)
    for s0 in range(0, Hh * W, 2_000_000):
        P[s0:s0 + 2_000_000] = rf.predict_proba(F[s0:s0 + 2_000_000])[:, 1]
    return P.reshape(Hh, W), hmax, {"res": res, "x0": meta["x0"] + shift[0], "z1": meta["z1"] + shift[1]}


def aerial_trees(P, hmax, g, exclude_fn, min_radius_m=1.5, seed=0):
    """Trees grown or planted since the LiDAR flight: canopy in the aerial where the LiDAR
    saw nothing taller than 1.5 m. Crowns are split at local maxima of the distance to the
    canopy edge; height comes from crown size (Central Texas allometry, ~1.5 x radius + 2 m)."""
    from scipy import ndimage
    res = g["res"]
    new = ndimage.binary_opening(ndimage.gaussian_filter(P, 0.8) > 0.6, iterations=1) & (hmax < 1.5)
    dist = ndimage.distance_transform_edt(new) * res
    peaks = (dist == ndimage.maximum_filter(dist, size=int(2 * 2.5 / res) | 1)) & (dist >= min_radius_m)
    lab, k = ndimage.label(peaks)
    pts = ndimage.center_of_mass(peaks, lab, range(1, k + 1)) if k else []
    rng = np.random.default_rng(seed)
    out = []
    for r, c in pts:
        rad = float(dist[int(r), int(c)])
        x = g["x0"] + (c + 0.5) * res; z = g["z1"] - (r + 0.5) * res
        if exclude_fn(x, z):
            continue
        hgt = float(np.clip(1.5 * rad + 2.0 + rng.normal(0, 0.6), 3.0, 14.0))
        out.append({"x": round(x, 2), "z": round(z, 2), "height_m": round(hgt, 2),
                    "crown_radius_m": round(min(rad * 1.15, 9.0), 2), "type": "deciduous", "source": "aerial"})
    return out


def refine_trees(trees, chm, frame, layout, dem, base_elev, naip4, meta, shift, boundary_local, margin_m,
                 exclusion, log=print):
    """Bring the LiDAR tree list up to the date of the aerial: drop trees that are gone,
    add trees grown or planted since the LiDAR flight. `exclusion` is a boolean grid on
    the CHM raster (course surfaces, buildings, water)."""
    if naip4 is None or not trees:
        return trees, {}
    P, hmax, g = canopy_probability(naip4, meta, chm, frame, shift)
    trees, gone = drop_ghosts(trees, P, g)
    n = chm.shape[0]; res = frame.size_m / n; h = frame.size_m / 2
    zone = None
    if boundary_local:
        import shapely
        from shapely.geometry import Polygon
        zone = Polygon(boundary_local).buffer(margin_m); shapely.prepare(zone)

    def excl(x, z):
        r, c = int((h - z) / res), int((x + h) / res)
        if not (0 <= r < n and 0 <= c < n) or exclusion[r, c]:
            return True
        if zone is not None:
            import shapely
            return not shapely.contains_xy(zone, x, z)
        return False
    new = aerial_trees(P, hmax, g, excl)
    if new:
        ys = dem.sample(np.array([t["x"] for t in new]), np.array([t["z"] for t in new])) - base_elev
        for t, y in zip(new, ys):
            t["y"] = round(float(y), 3)
    log(f"trees: {gone} gone since the LiDAR flight (no crown in the aerial), {len(new)} grown or planted since")
    return trees + new, {"removed_gone": gone, "added_from_aerial": len(new)}
