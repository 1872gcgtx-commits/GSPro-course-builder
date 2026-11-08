"""Course land-cover classification + surface drafting (v2).

1. Features on the course's 0.5 m grid from 4-band NAIP (R, G, B, NIR at 0.3 m,
   uncompressed) + LiDAR (canopy height, micro-relief, slope).
2. Training labels come from the course itself, no hand labelling needed:
     mown   : around the owner's green and back-tee clicks, and along the middle
              of each par-4/5 line (only where LiDAR shows no canopy)
     rough  : open ground far (>45 m) from every hole line
     trees  : LiDAR canopy > 3 m
     sand   : very bright, unvegetated, treeless
     water  : LiDAR water returns / negative NDVI
     other  : buildings, pavement
   A random forest learns this course's own look (season, grass type, light).
3. Shapes per hole from the classified map:
     fairway : mown pixels in the hole's own corridor (each pixel goes to its
               nearest hole line, so parallel holes don't merge), connected to
               the line, minus green/tee/bunkers, smoothed
     green   : region grown from the green click on mown turf (0.3 m colour +
               NIR), checked for size, compactness and LiDAR pad lift
     tee     : level LiDAR pad on mown turf at the back-tee click
     bunker  : sand components near the hole
     water   : water components (ponds, creek), attached to the nearest hole
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from .autodraft import _ellipse, _polygon, _rect, _solidity, _strip, _along

CLASSES = ["mown", "rough", "trees", "sand", "water", "other"]


class Grid:
    """A window of the course's north-up raster grid (DEM resolution)."""

    def __init__(self, frame, n, res, r0, r1, c0, c1):
        self.frame, self.n, self.res = frame, n, res
        self.r0, self.r1, self.c0, self.c1 = r0, r1, c0, c1
        self.shape = (r1 - r0, c1 - c0)

    def rc(self, x, z):
        h = self.frame.size_m / 2
        return int((h - z) / self.res) - self.r0, int((x + h) / self.res) - self.c0

    def xz(self, r, c):
        h = self.frame.size_m / 2
        return -h + (c + self.c0 + 0.5) * self.res, h - (r + self.r0 + 0.5) * self.res


def _resample(a: np.ndarray, shape) -> np.ndarray:
    zy, zx = shape[0] / a.shape[0], shape[1] / a.shape[1]
    out = ndimage.zoom(a.astype(np.float32), (zy, zx), order=1)
    o = np.zeros(shape, np.float32)
    h, w = min(shape[0], out.shape[0]), min(shape[1], out.shape[1])
    o[:h, :w] = out[:h, :w]
    return o


def build_features(cfg, log=print):
    d = cfg.data_dir
    ext = json.loads((d / "naip4.json").read_text())
    naip = np.load(d / "naip4.npy", mmap_mode="r")
    dem_full = np.load(d / "dem.npy", mmap_mode="r")
    n = dem_full.shape[0]
    res = cfg.frame.size_m / n
    h = cfg.frame.size_m / 2
    r0, r1 = int((h - ext["z1"]) / res), int((h - ext["z0"]) / res)
    c0, c1 = int((ext["x0"] + h) / res), int((ext["x1"] + h) / res)
    G = Grid(cfg.frame, n, res, r0, r1, c0, c1)
    log(f"  feature grid {G.shape[1]}x{G.shape[0]} at {res} m")
    dem = np.asarray(dem_full[r0:r1, c0:c1], np.float32)
    chm = np.asarray(np.load(d / "chm.npy", mmap_mode="r")[r0:r1, c0:c1], np.float32)
    masks = np.load(d / "lidar_masks.npz")
    lw = masks["water"][r0:r1, c0:c1]
    lb = masks["building"][r0:r1, c0:c1]

    # 0.3 m texture first, then everything to the 0.5 m grid
    nir_hi = naip[..., 3].astype(np.float32)
    bri_hi = naip[..., :3].astype(np.float32).mean(-1)
    def lstd(a, k):
        m = ndimage.uniform_filter(a, k)
        return np.sqrt(np.maximum(ndimage.uniform_filter(a * a, k) - m * m, 0))
    tex_n = _resample(lstd(nir_hi, 5), G.shape)
    tex_b = _resample(lstd(bri_hi, 5), G.shape)
    del nir_hi, bri_hi
    R, Gc, B, N = (_resample(naip[..., i], G.shape) for i in range(4))
    ndvi = (N - R) / (N + R + 1e-3)
    bri = (R + Gc + B) / 3
    exg = 2 * Gc - R - B
    k = lambda m: max(1, int(m / res))  # noqa: E731
    gz, gx = np.gradient(ndimage.gaussian_filter(dem, 1.2), res)
    slope = np.hypot(gx, gz)
    detr = dem - ndimage.gaussian_filter(dem, 8 / res)
    feats = np.stack([R, Gc, B, N, ndvi, bri, exg,
                      ndimage.uniform_filter(ndvi, k(3)), ndimage.uniform_filter(ndvi, k(8)),
                      lstd(ndvi, k(2)), tex_n, tex_b, ndimage.uniform_filter(tex_b, k(4)),
                      chm, detr, slope], -1).astype(np.float32)
    aux = dict(R=R, G=Gc, B=B, N=N, ndvi=ndvi, bri=bri, exg=exg, chm=chm, detr=detr, slope=slope,
               lw=lw, lb=lb, tex_b=tex_b, dem=dem)
    return G, feats, aux


def _line_mask(G: Grid, pts, radius_m) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.new("L", (G.shape[1], G.shape[0]), 0)
    dr = ImageDraw.Draw(img)
    px = [(G.rc(x, z)[1], G.rc(x, z)[0]) for x, z in pts]
    w = max(1, int(2 * radius_m / G.res))
    if len(px) > 1:
        dr.line(px, fill=1, width=w, joint="curve")
    for c, r in px:
        rr = radius_m / G.res
        dr.ellipse([c - rr, r - rr, c + rr, r + rr], fill=1)
    return np.asarray(img, bool)


def _sub_path(path, f0, f1, step=3.0):
    L = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
    return [_along(path, f) for f in np.linspace(f0, f1, max(2, int((f1 - f0) * L / step)))]


def train_and_classify(G, feats, aux, holes, boundary_local=None, log=print, seed=3):
    from sklearn.ensemble import RandomForestClassifier
    chm, ndvi, bri, exg = aux["chm"], aux["ndvi"], aux["bri"], aux["exg"]
    lw, lb = aux["lw"], aux["lb"]
    H, W = G.shape
    # distance to the nearest hole line (all holes)
    allline = np.zeros(G.shape, bool)
    for h in holes:
        allline |= _line_mask(G, _sub_path(h.path, 0, 1, 2.0), 1.0)
    dline = ndimage.distance_transform_edt(~allline) * G.res

    labels = np.full(G.shape, -1, np.int8)
    mown = np.zeros(G.shape, bool)
    for h in holes:
        mown |= _line_mask(G, [h.path[-1]], 4.0)            # green centre
        mown |= _line_mask(G, [h.path[0]], 2.0)             # back tee
        if (h.par or 4) >= 4:
            mown |= _line_mask(G, _sub_path(h.path, 0.55, 0.85), 3.0)
    mown &= (chm < 0.5) & (ndvi > 0.1)
    rough = (dline > 45) & (chm < 0.8) & ~lb & (ndvi > 0.05) & (ndvi < 0.6)
    trees = chm > 3
    sand = (bri > np.percentile(bri, 97)) & (ndvi < 0.12) & (exg < 25) & (chm < 0.5) & ~lb
    sand = ndimage.binary_opening(sand)
    water = (lw & (ndvi < 0.15)) | ((ndvi < -0.05) & (bri < np.percentile(bri, 30)) & (chm < 0.5))
    water = ndimage.binary_opening(water, iterations=2)
    other = lb | ((ndvi < 0.08) & (chm < 0.5) & ~sand & ~water & (bri > np.percentile(bri, 40)))
    for i, m in enumerate([rough, trees, other, water, sand, mown]):   # later wins
        labels[m] = CLASSES.index(["rough", "trees", "other", "water", "sand", "mown"][i])

    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci in range(len(CLASSES)):
        idx = np.flatnonzero(labels.ravel() == ci)
        if len(idx) == 0:
            continue
        take = rng.choice(idx, min(len(idx), 25000), replace=False)
        X.append(feats.reshape(-1, feats.shape[-1])[take]); y.append(np.full(len(take), ci))
        log(f"  training {CLASSES[ci]:6s}: {len(idx):9,d} labelled px, using {len(take):,d}")
    X, y = np.concatenate(X), np.concatenate(y)
    rf = RandomForestClassifier(n_estimators=60, max_depth=18, min_samples_leaf=3, n_jobs=-1,
                                class_weight="balanced", random_state=seed)
    rf.fit(X, y)

    # classify only where it matters: near holes, plus water candidates
    todo = (dline < 110) | water | lw
    flat = feats.reshape(-1, feats.shape[-1])
    idx = np.flatnonzero(todo.ravel())
    prob = np.zeros((H * W, len(CLASSES)), np.float32)
    prob[:, CLASSES.index("other")] = 1.0
    for s in range(0, len(idx), 2_000_000):
        part = idx[s:s + 2_000_000]
        pp = rf.predict_proba(flat[part])
        prob[part] = 0
        for j, cls in enumerate(rf.classes_):
            prob[part, cls] = pp[:, j]
    prob = prob.reshape(H, W, len(CLASSES))
    for j in range(len(CLASSES)):
        prob[..., j] = ndimage.gaussian_filter(prob[..., j], 0.8)
    cls = np.argmax(prob, -1).astype(np.int8)
    log("  class areas (ha): " + ", ".join(f"{c} {float((cls == i).sum()) * G.res ** 2 / 1e4:.1f}"
                                           for i, c in enumerate(CLASSES)))
    return cls, prob, dline


def _rectangularity(m) -> float:
    """Area / minimum-area bounding rectangle (roofs ~1, bunkers lower)."""
    from skimage.measure import regionprops
    props = regionprops(m.astype(np.uint8))
    if not props:
        return 0.0
    p = props[0]
    ma, mi = p.axis_major_length, p.axis_minor_length
    if ma <= 0 or mi <= 0:
        return 0.0
    # for a rectangle, area = (ma*mi) * (pi/ (4*sqrt(3))) ... use the orientation-aligned box instead
    import math as _m
    rr, cc = np.nonzero(m)
    th = p.orientation
    u = cc * _m.cos(th) - rr * _m.sin(th)
    v = cc * _m.sin(th) + rr * _m.cos(th)
    box = (u.max() - u.min() + 1) * (v.max() - v.min() + 1)
    return float(len(rr) / max(box, 1))


def _hole_partition(G, holes, max_m=80.0):
    """Index of the nearest hole line per pixel (-1 beyond max_m)."""
    from scipy.spatial import cKDTree
    from .course_meshes import _densify
    pts, lab = [], []
    for i, h in enumerate(holes):
        dense = _densify(list(h.path) + [h.path[-1]], 1.5)[:-1] if len(h.path) > 1 else np.array(h.path)
        pts.append(dense); lab.append(np.full(len(dense), i))
    tree = cKDTree(np.concatenate(pts))
    lab = np.concatenate(lab)
    rr, cc = np.mgrid[0:G.shape[0], 0:G.shape[1]]
    x = -G.frame.size_m / 2 + (cc + G.c0 + 0.5) * G.res
    z = G.frame.size_m / 2 - (rr + G.r0 + 0.5) * G.res
    d, j = tree.query(np.column_stack([x.ravel(), z.ravel()]), workers=-1)
    part = lab[j].reshape(G.shape)
    part[d.reshape(G.shape) > max_m] = -1
    return part


def _smooth_mask(m, sigma_px=2.0):
    return ndimage.gaussian_filter(m.astype(np.float32), sigma_px) > 0.5


def refine_paths(G, prob, aux, holes, log=print):
    """Line of play per hole: least-cost route tee -> green through the classified
    land cover (mown turf cheap, centre of the fairway cheapest; rough dearer; trees
    much dearer; water/buildings ~blocked), simplified to <= 2 dog-leg aim points.
    Returns {hole.number: [tee, aim..., green]} in local metres."""
    from skimage.graph import route_through_array
    from skimage.measure import approximate_polygon
    I = {c: i for i, c in enumerate(CLASSES)}
    pm, pt = prob[..., I["mown"]], prob[..., I["trees"]]
    blocked = (prob[..., I["water"]] > 0.5) | aux["lb"]
    mown = pm > 0.5
    inside = ndimage.distance_transform_edt(mown) * G.res            # metres from the fairway edge
    cost = 1.0 + 4.0 * (1 - pm) + 30.0 * pt + 400.0 * blocked
    cost = cost + 2.5 / (1.0 + inside / 6.0) * mown                  # prefer the middle of the turf
    out = {}
    for h in holes:
        tee, green = h.path[0], h.path[-1]
        L = math.dist(tee, green)
        if (h.par or 4) < 4 or L < 180:
            out[h.number] = [tee, green]
            continue
        pad = max(70.0, 0.4 * L)
        (tr, tc), (gr, gc) = G.rc(*tee), G.rc(*green)
        r0 = max(0, min(tr, gr) - int(pad / G.res)); r1 = min(G.shape[0], max(tr, gr) + int(pad / G.res))
        c0 = max(0, min(tc, gc) - int(pad / G.res)); c1 = min(G.shape[1], max(tc, gc) + int(pad / G.res))
        sub = cost[r0:r1, c0:c1]
        step = 2                                                        # route on a 1 m grid for speed
        sub2 = sub[::step, ::step]
        a = ((tr - r0) // step, (tc - c0) // step)
        b = ((gr - r0) // step, (gc - c0) // step)
        try:
            idx, _ = route_through_array(sub2, a, b, fully_connected=True, geometric=True)
        except Exception:
            out[h.number] = [tee, green]
            continue
        pts = np.array([G.xz(r0 + r * step, c0 + c * step) for r, c in idx])
        simp = approximate_polygon(pts, tolerance=7.0)
        # keep the interior vertices that really bend the hole (>12 m off the straight line)
        from .layout import point_segment_dist
        inner = [tuple(v) for v in simp[1:-1] if point_segment_dist(tuple(v), tee, green) > 12.0]
        inner.sort(key=lambda v: -point_segment_dist(v, tee, green))
        inner = inner[:2]
        inner.sort(key=lambda v: math.dist(tee, v))
        out[h.number] = [tee] + [(round(x, 1), round(z, 1)) for x, z in inner] + [green]
    bent = sum(1 for v in out.values() if len(v) > 2)
    log(f"  line of play: {bent} of {len(holes)} holes bend (dog-legs)")
    return out


def _fairway_continuous(G, path, ok, gmask, step=4.0, hazard=None, min_start=40.0, min_run=4):
    """One smooth fairway per hole along its line of play.

    ok    : boolean grid of fairway-like turf available to this hole
    gmask : the hole's green (the fairway runs up to its front edge)
    Returns a boolean grid mask (single component, no holes) or None.
    """
    L = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
    if L < 60:
        return None
    # smooth the centreline so offsets don't loop at sharp dog-leg corners
    st = np.arange(0, L, step)
    cen = np.array([_along(path, d / L) for d in st])
    cen = np.column_stack([ndimage.gaussian_filter1d(cen[:, 0], 3, mode="nearest"),
                           ndimage.gaussian_filter1d(cen[:, 1], 3, mode="nearest")])
    tan = np.gradient(cen, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-6)
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    offs = np.arange(-40, 40.01, 0.5)
    mid = len(offs) // 2
    lo, hi = np.full(len(st), np.nan), np.full(len(st), np.nan)
    in_green = np.zeros(len(st), bool)
    for i, (c, n) in enumerate(zip(cen, nrm)):
        r, cc = G.rc(c[0], c[1])
        if 0 <= r < G.shape[0] and 0 <= cc < G.shape[1] and gmask[r, cc]:
            in_green[i] = True
        xs, zs = c[0] + n[0] * offs, c[1] + n[1] * offs
        rr = np.array([G.rc(x, z)[0] for x, z in zip(xs, zs)])
        cc2 = np.array([G.rc(x, z)[1] for x, z in zip(xs, zs)])
        inb = (rr >= 0) & (rr < G.shape[0]) & (cc2 >= 0) & (cc2 < G.shape[1])
        v = np.zeros(len(offs), bool)
        v[inb] = ok[rr[inb], cc2[inb]]
        v = ndimage.binary_closing(v, iterations=6)            # bridge paths/sprinkler gaps
        idx = np.flatnonzero(v)
        if len(idx) == 0 or np.min(np.abs(idx - mid)) * 0.5 > 8:
            continue
        k = idx[np.argmin(np.abs(idx - mid))]
        a = b = k
        while a > 0 and v[a - 1]:
            a -= 1
        while b < len(v) - 1 and v[b + 1]:
            b += 1
        lo[i], hi[i] = offs[a], offs[b]
    width = hi - lo
    good = np.isfinite(width) & (width >= 14) & ~in_green
    wet = np.zeros(len(st), bool)
    if hazard is not None:
        for i, c in enumerate(cen):
            r, cc = G.rc(c[0], c[1])
            if 0 <= r < G.shape[0] and 0 <= cc < G.shape[1] and hazard[r, cc]:
                wet[i] = True
    good &= ~wet
    # start: first run of 3 good stations at least 40 m past the tee; end: last good before the green
    first_green = np.flatnonzero(in_green)
    end_lim = first_green[0] if len(first_green) else len(st)
    g_idx = np.flatnonzero(good & (st > min_start) & (np.arange(len(st)) < end_lim))
    if len(g_idx) < min_run:
        return None
    # split into runs: never bridge >25 m of non-fairway or any water/creek crossing
    runs, cur = [], [g_idx[0]]
    for i in g_idx[1:]:
        gap = slice(cur[-1] + 1, i)
        if (i - cur[-1]) * step > 25 or wet[gap].any():
            runs.append(cur); cur = [i]
        else:
            cur.append(i)
    runs.append(cur)
    best_len = max(len(r) for r in runs)
    cands = [r for r in runs if len(r) >= 0.8 * best_len]
    run = max(cands, key=lambda r: r[-1])                     # tie -> the run nearer the green
    if len(run) < min_run:
        return None
    start, end = run[0], run[-1]
    if len(first_green) and end_lim - end <= 6 and not wet[end:end_lim].any():
        end = end_lim - 1                                     # run right up to the green's front
    sl = slice(start, end + 1)
    x = np.arange(len(st))[sl]
    def fill(a):
        a = a[sl].copy()
        okk = np.isfinite(a)
        a = np.interp(np.arange(len(a)), np.flatnonzero(okk), a[okk])
        return ndimage.gaussian_filter1d(a, 3.0, mode="nearest")
    lo_s, hi_s = fill(lo), fill(hi)
    centre_shift = (lo_s + hi_s) / 2
    half = np.clip((hi_s - lo_s) / 2, 9, 32)
    half = ndimage.gaussian_filter1d(half, 2.0, mode="nearest")
    c = cen[sl] + nrm[sl] * centre_shift[:, None]
    n = nrm[sl]
    left = c + n * half[:, None]
    right = c - n * half[:, None]
    # rounded ends
    def cap(p, t, r, sign):
        ang0 = math.atan2(t[1], t[0])
        return [(p[0] + r * math.cos(ang0 + sign * a), p[1] + r * math.sin(ang0 + sign * a))
                for a in np.linspace(math.pi / 2, -math.pi / 2, 9)]
    t_end, t_start = tan[sl][-1], -tan[sl][0]
    ring = list(map(tuple, left)) + cap(c[-1], t_end, half[-1], 1)[1:-1] + \
        list(map(tuple, right[::-1])) + cap(c[0], t_start, half[0], 1)[1:-1]
    # rasterise -> one filled, smooth component
    from PIL import Image, ImageDraw
    img = Image.new("L", (G.shape[1], G.shape[0]), 0)
    ImageDraw.Draw(img).polygon([(G.rc(px, pz)[1], G.rc(px, pz)[0]) for px, pz in ring], fill=1)
    m = np.asarray(img, bool)
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=4))
    m = _smooth_mask(m, 3.0)
    m &= ~ndimage.binary_dilation(gmask, iterations=2)
    lab, nlab = ndimage.label(m)
    if nlab == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, nlab + 1))
    m = ndimage.binary_fill_holes(lab == 1 + int(np.argmax(sizes)))
    return m


class _Win:
    """Adapter so autodraft._polygon can turn grid masks into local-metre rings."""

    def __init__(self, G):
        self.G = G

    def xz(self, r, c):
        return self.G.xz(r, c)


def draft_v2(cfg, holes, log=print) -> list[dict]:
    G, feats, aux = build_features(cfg, log)
    cls, prob, dline = train_and_classify(G, feats, aux, holes, log=log)
    # follow dog-legs: re-route each hole along its fairway, then partition on those lines
    import copy
    paths = refine_paths(G, prob, aux, holes, log=log)
    orig = {h.number: list(h.path) for h in holes}
    holes = [copy.copy(h) for h in holes]
    for h in holes:
        if len(orig[h.number]) == 2:          # keep any aim points the owner already placed
            h.path = paths[h.number]
    allline = np.zeros(G.shape, bool)
    for h in holes:
        allline |= _line_mask(G, _sub_path(h.path, 0, 1, 2.0), 1.0)
    dline = ndimage.distance_transform_edt(~allline) * G.res
    part = _hole_partition(G, holes)
    P = _Win(G)
    full = (slice(0, G.shape[0]), slice(0, G.shape[1]))
    I = {c: i for i, c in enumerate(CLASSES)}
    mown = prob[..., I["mown"]] > 0.5
    sand = cls == I["sand"]
    water = (cls == I["water"]) | ndimage.binary_opening(aux["lw"], iterations=2)
    water = ndimage.binary_closing(water, iterations=3)
    out = []

    def ring_of(mask, tol):
        r = _polygon(P, mask, full, tol)
        return [(round(x, 1), round(z, 1)) for x, z in r] if r else None

    def bbox_mask(mask, pad=2):
        rr, cc = np.nonzero(mask)
        r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad + 1, G.shape[0])
        c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad + 1, G.shape[1])
        return (slice(r0, r1), slice(c0, c1))

    def poly(mask, tol):
        if not mask.any():
            return None
        win = bbox_mask(mask)
        r = _polygon(P, mask[win], win, tol)
        return [(round(x, 1), round(z, 1)) for x, z in r] if r else None

    rgbN = np.stack([aux["R"], aux["G"], aux["B"], aux["N"]], -1)
    # creek bed / ponds: water plus the bare gravel channel next to it
    hazard = ndimage.binary_dilation(water, iterations=6) | \
        (ndimage.binary_dilation(water, iterations=40) & (aux["ndvi"] < 0.12) & (prob[..., I["mown"]] < 0.3))
    for hi, h in enumerate(holes):
        nine = int(h.props.get("nine", 0) or 0)
        nh = int(h.props.get("nine_hole", 0) or 0)
        tee, green = h.path[0], h.path[-1]
        prev = h.path[-2]
        ang_in = math.atan2(green[1] - prev[1], green[0] - prev[0])
        ang_out = math.atan2(h.path[1][1] - tee[1], h.path[1][0] - tee[0])
        def add(kind, ring, method, conf):
            out.append({"kind": kind, "ring": ring, "method": method, "confidence": round(conf, 2),
                        "hole": h.number, "nine": nine, "nine_hole": nh})

        # ---- green: colour+NIR region grown on mown turf from the click
        gr, gc = G.rc(*green)
        gmask = None
        if 0 <= gr < G.shape[0] and 0 <= gc < G.shape[1]:
            k = int(40 / G.res)
            ws = (slice(max(gr - k, 0), gr + k), slice(max(gc - k, 0), gc + k))
            lr, lc = gr - ws[0].start, gc - ws[1].start
            v = rgbN[ws]
            seed = v[max(lr - 6, 0):lr + 7, max(lc - 6, 0):lc + 7].reshape(-1, 4)
            mu, sd = np.median(seed, 0), seed.std(0) + 2
            dist = np.sqrt((((v - mu) / sd) ** 2).sum(-1))
            allowed = (prob[ws][..., I["mown"]] > 0.3) & (aux["slope"][ws] < 0.15) & ~sand[ws]
            prev_area, best = None, None
            for t in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
                m = ndimage.binary_closing((dist < t) & allowed, iterations=2)
                lab, _ = ndimage.label(m)
                sl = lab[lr, lc] if lab[lr, lc] else 0
                if not sl:
                    dd, ix = ndimage.distance_transform_edt(lab == 0, return_indices=True)
                    if dd[lr, lc] * G.res <= 4:
                        sl = lab[ix[0][lr, lc], ix[1][lr, lc]]
                if not sl:
                    continue
                reg = ndimage.binary_fill_holes(lab == sl)
                area = reg.sum() * G.res ** 2
                if area > 1300 or (prev_area and area > prev_area * 1.7 and prev_area > 250):
                    break
                if area >= 200 and _solidity(reg) < 0.75:
                    break
                if area >= 200:
                    best = reg
                prev_area = area
            if best is not None:
                best = _smooth_mask(ndimage.binary_opening(best, iterations=2), 1.5)
                full_m = np.zeros(G.shape, bool)
                full_m[ws] = best
                ring = poly(full_m, 0.8)
                if ring:
                    lift = float(np.median(aux["detr"][full_m]))
                    add("green", ring, "detected", 0.8 if lift > 0.03 else 0.6)
                    gmask = full_m
        if gmask is None and 0 <= gr < G.shape[0] and 0 <= gc < G.shape[1]:
            # greens are built-up pads: raised micro-relief, gentle slope, turf
            k = int(35 / G.res)
            ws = (slice(max(gr - k, 0), gr + k), slice(max(gc - k, 0), gc + k))
            lr, lc = gr - ws[0].start, gc - ws[1].start
            pad = (aux["detr"][ws] > 0.02) & (aux["slope"][ws] < 0.10) & (aux["chm"][ws] < 1) & (aux["ndvi"][ws] > 0.1) & ~sand[ws]
            pad = ndimage.binary_opening(pad, iterations=2)
            lab, _ = ndimage.label(pad)
            sl = lab[lr, lc]
            if not sl:
                dd, ix = ndimage.distance_transform_edt(lab == 0, return_indices=True)
                if dd[lr, lc] * G.res <= 5:
                    sl = lab[ix[0][lr, lc], ix[1][lr, lc]]
            if sl:
                m = ndimage.binary_fill_holes(lab == sl)
                area = m.sum() * G.res ** 2
                if 200 <= area <= 1300 and _solidity(m) > 0.7:
                    full_m = np.zeros(G.shape, bool)
                    full_m[ws] = _smooth_mask(m, 1.5)
                    ring = poly(full_m, 0.8)
                    if ring:
                        add("green", ring, "detected", 0.55)
                        gmask = full_m
        if gmask is None:
            add("green", [(round(x, 1), round(z, 1)) for x, z in _ellipse(*green, 15, 11, ang_in)], "fallback", 0.2)
            gmask = np.zeros(G.shape, bool)
            gr_, gc_ = G.rc(*green)
            gmask[max(gr_ - 26, 0):gr_ + 26, max(gc_ - 26, 0):gc_ + 26] = True

        # ---- back tee: level pad on turf at the click
        tr, tc = G.rc(*tee)
        tmask = None
        if 0 <= tr < G.shape[0] and 0 <= tc < G.shape[1]:
            k = int(25 / G.res)
            ws = (slice(max(tr - k, 0), tr + k), slice(max(tc - k, 0), tc + k))
            lr, lc = tr - ws[0].start, tc - ws[1].start
            z = aux["dem"][ws].astype(float)
            yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
            nb = (abs(yy - lr) <= 6) & (abs(xx - lc) <= 6)
            A = np.column_stack([np.ones(nb.sum()), xx[nb], yy[nb]])
            coef, *_ = np.linalg.lstsq(A, z[nb], rcond=None)
            resid = np.abs(z - (coef[0] + coef[1] * xx + coef[2] * yy))
            lvl = (resid < 0.12) & (aux["slope"][ws] < 0.045) & (prob[ws][..., I["mown"]] > 0.3) & (aux["chm"][ws] < 1)
            lvl = ndimage.binary_opening(lvl, iterations=2)
            lab, _ = ndimage.label(lvl)
            sl = lab[lr, lc]
            if not sl:
                dd, ix = ndimage.distance_transform_edt(lab == 0, return_indices=True)
                if dd[lr, lc] * G.res <= 3:
                    sl = lab[ix[0][lr, lc], ix[1][lr, lc]]
            if sl:
                m = ndimage.binary_fill_holes(lab == sl)
                area = m.sum() * G.res ** 2
                if 70 <= area <= 1000 and _solidity(m) > 0.7:
                    full_m = np.zeros(G.shape, bool)
                    full_m[ws] = _smooth_mask(m, 1.0)
                    ring = poly(full_m, 0.8)
                    if ring:
                        add("tee", ring, "detected", 0.7)
                        tmask = full_m
        if tmask is None:
            add("tee", [(round(x, 1), round(z, 1)) for x, z in _rect(*tee, 22, 11, ang_out)], "fallback", 0.2)
            tmask = np.zeros(G.shape, bool)
        out[-1]["tee_set"] = "Blue"

        mine = part == hi
        # ---- fairway: ONE smooth continuous shape along the line of play
        fwmask = np.zeros(G.shape, bool)
        if (h.par or 4) >= 4:
            ok = (prob[..., I["mown"]] > 0.4) & mine & (prob[..., I["trees"]] < 0.5) & ~sand & \
                ~ndimage.binary_dilation(tmask, iterations=2)
            m = _fairway_continuous(G, h.path, ok, gmask, hazard=hazard)
            if m is not None and m.sum() * G.res ** 2 >= 800:
                ring = poly(m, 1.6)
                if ring:
                    add("fairway", ring, "detected", 0.75)
                    fwmask = m
            if not fwmask.any():
                a = _along(h.path, 0.42)
                b = _along(h.path, 0.92)
                add("fairway", [(round(x, 1), round(z, 1)) for x, z in _strip(a, b, 32)], "fallback", 0.2)
                fwmask = _line_mask(G, _sub_path(h.path, 0.42, 0.92), 16.0)

        # ---- bunkers: sand in this hole's playing area (near its fairway or green), not roofs
        near_play = (ndimage.distance_transform_edt(~fwmask) * G.res < 40) | \
            (ndimage.distance_transform_edt(~gmask) * G.res < 35)
        bmask = sand & mine & near_play & ~ndimage.binary_dilation(aux["lb"], iterations=4)
        bmask = ndimage.binary_fill_holes(ndimage.binary_closing(bmask, iterations=2))
        lab, _ = ndimage.label(bmask)
        for j, sl in enumerate(ndimage.find_objects(lab)):
            if sl is None:
                continue
            m = lab[sl] == j + 1
            area = m.sum() * G.res ** 2
            if not 8 <= area <= 900 or _solidity(m) < 0.55 or _rectangularity(m) > 0.88:
                continue
            fm = np.zeros(G.shape, bool)
            fm[sl] = m
            # a bunker is surrounded by grass, not pavement/roof/dirt
            ringpx = ndimage.binary_dilation(fm, iterations=14) & ~ndimage.binary_dilation(fm, iterations=6)
            grassy = np.isin(cls[ringpx], [I["mown"], I["rough"]]).mean()
            if grassy < 0.45:
                continue
            ring = poly(_smooth_mask(fm, 0.8) | fm, 0.8)
            if ring:
                bowl = float(np.median(aux["detr"][fm]))
                add("bunker", ring, "detected", 0.8 if bowl < -0.08 else 0.55)

    # ---- water bodies (shared): ponds and the creek
    lab, nlab = ndimage.label(water)
    from .layout import point_polyline_dist
    wi = 0
    for j, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        m = lab[sl] == j + 1
        area = m.sum() * G.res ** 2
        if area < 200:
            continue
        fm = np.zeros(G.shape, bool)
        fm[sl] = m
        lidar_wet = aux["lw"][fm].mean() > 0.2
        dark = np.median(aux["bri"][fm]) < 75
        neg = np.median(aux["ndvi"][fm]) < -0.02
        if not (lidar_wet or (dark and neg)):
            continue                                    # bare dirt, roofs, shadows
        ring = poly(_smooth_mask(fm, 1.5) | fm, 1.5)
        if not ring:
            continue
        cx = sum(p[0] for p in ring) / len(ring)
        cz = sum(p[1] for p in ring) / len(ring)
        best = min(holes, key=lambda hh: point_polyline_dist((cx, cz), hh.path))
        out.append({"kind": "water", "ring": ring, "method": "detected", "confidence": 0.8,
                    "hole": best.number, "nine": int(best.props.get("nine", 0) or 0),
                    "nine_hole": int(best.props.get("nine_hole", 0) or 0)})
        wi += 1
    log(f"  drafted {len(out)} shapes ({wi} water bodies)")
    draft_v2.paths = {h.number: [list(p) for p in h.path] for h in holes}
    (cfg.data_dir / "line_of_play.json").write_text(json.dumps(
        {f"n{int(h.props.get('nine', 0) or 0)}h{int(h.props.get('nine_hole', 0) or 0)}":
         [[round(x, 1), round(z, 1)] for x, z in h.path] for h in holes}))
    np.save(cfg.data_dir / "landcover.npy", cls)
    (cfg.data_dir / "landcover.json").write_text(json.dumps({"classes": CLASSES, "r0": G.r0, "c0": G.c0,
                                                              "res": G.res}))
    return out
