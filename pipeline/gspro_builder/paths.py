"""Cart paths and streets: traced, graded into the terrain, and built as meshes.

Cart paths
  1. A random forest learns concrete path from the 4-band aerial, using the mapped
     (OpenStreetMap) cart paths that really show as concrete as examples, plus a
     "bright narrow line" filter.
  2. Paths are routed the way a course is laid out: previous green -> next tee, and tee
     -> green alongside each hole, as least-cost routes. Cheap: visible concrete and
     mapped paths. Costly: woods, steep ground. Avoided: fairways, greens, tees.
     Blocked: bunkers, houses; water only where it must (bridges).
  3. Overlapping routes are merged into one network (raster union -> skeleton ->
     polylines) and smoothed.
Streets
  OpenStreetMap streets, width by class; concrete or asphalt from the aerial's
  brightness along the street; curbs on residential streets, centre lines on bigger roads.
Both
  Graded into the terrain: each cross-section levelled, the long profile smoothed, the
  shoulders blended. Draped ribbon meshes on the graded ground; bridges where a cart
  path crosses water (deck + rails).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ROAD_WIDTH = {"motorway": 14.0, "trunk": 12.0, "primary": 11.0, "secondary": 10.0, "tertiary": 9.0,
              "unclassified": 7.0, "residential": 8.0, "living_street": 6.0, "service": 5.0, "road": 6.0}
CART_WIDTH = 2.5


# ------------------------------------------------------------------ cart path detection
def cartpath_probability(naip4, meta, osm_lines_photo, step=2, seed=0):
    from PIL import Image, ImageDraw
    from scipy import ndimage
    from sklearn.ensemble import RandomForestClassifier
    a = np.asarray(naip4[::step, ::step]).astype(np.float32) / 255
    res = meta["res"] * step
    H, W = a.shape[:2]
    R, G, B, N = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    ndvi = (N - R) / np.maximum(N + R, 1e-3); lum = (R + G + B) / 3
    lm = lambda x, s: ndimage.uniform_filter(x, s)  # noqa: E731
    ridge = ndimage.gaussian_filter(lum, 1.2) - ndimage.gaussian_filter(lum, 6)
    ridge_d = ndimage.gaussian_filter(-ndvi, 1.2) - ndimage.gaussian_filter(-ndvi, 6)
    F = np.stack([R, G, B, N, ndvi, lum, ridge, ridge_d, lm(lum, 5), lm(ndvi, 5), lm(ndvi, 21),
                  np.sqrt(np.maximum(lm(lum * lum, 5) - lm(lum, 5) ** 2, 0))], -1).reshape(H * W, -1)
    im = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(im)
    for ln in osm_lines_photo:
        dr.line([((x - meta["x0"]) / res, (meta["z1"] - z) / res) for x, z in ln], fill=1, width=max(1, int(1.6 / res)))
    osm = np.asarray(im, bool)
    pos = np.flatnonzero((osm & (ndvi < 0.15)).ravel())
    if len(pos) < 500:
        return None, None
    neg = ~ndimage.binary_dilation(osm, iterations=int(6 / res))
    rng = np.random.default_rng(seed)
    pick = lambda ix, k: rng.choice(ix, min(len(ix), k), replace=False)  # noqa: E731
    hard = np.flatnonzero((neg & (ndvi < 0.2)).ravel()); anyn = np.flatnonzero(neg.ravel())
    pos, hard, anyn = pick(pos, 40000), pick(hard, 60000), pick(anyn, 40000)
    rf = RandomForestClassifier(60, max_depth=16, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    rf.fit(F[np.r_[pos, hard, anyn]], np.r_[np.ones(len(pos)), np.zeros(len(hard) + len(anyn))])
    P = np.empty(H * W, np.float32)
    for s0 in range(0, H * W, 2_000_000):
        P[s0:s0 + 2_000_000] = rf.predict_proba(F[s0:s0 + 2_000_000])[:, 1]
    return P.reshape(H, W), {"res": res, "x0": meta["x0"], "z1": meta["z1"]}


def _chaikin(P, k=2):
    P = np.asarray(P, float)
    for _ in range(k):
        Q = np.empty((2 * len(P) - 2, 2)); Q[0::2] = 0.75 * P[:-1] + 0.25 * P[1:]; Q[1::2] = 0.25 * P[:-1] + 0.75 * P[1:]
        P = np.vstack([P[:1], Q, P[-1:]])
    return P


def _skeleton_lines(mask, R, h, min_px=10):
    """Skeleton of a mask -> smoothed polylines (local metres)."""
    from scipy import ndimage
    from skimage.morphology import skeletonize
    sk = skeletonize(mask)
    n = sk.shape[0]
    nb = ndimage.convolve(sk.astype(int), np.ones((3, 3)), mode="constant") - 1
    nodes = sk & ((nb == 1) | (nb >= 3))
    off = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    seen_px = np.zeros_like(sk)
    lines = []

    def walk(r, c, pr, pc):
        path = [(r, c)]
        while not nodes[r, c] and len(path) < 50000:
            nxt = [(r + dr, c + dc) for dr, dc in off if 0 <= r + dr < n and 0 <= c + dc < n and sk[r + dr, c + dc]
                   and (r + dr, c + dc) != (pr, pc) and (r + dr, c + dc) not in path[-3:]]
            if not nxt:
                break
            pr, pc = r, c; r, c = nxt[0]; path.append((r, c))
        return path
    for r, c in zip(*np.nonzero(nodes)):
        for dr, dc in off:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and sk[rr, cc] and not seen_px[rr, cc]:
                p = [(r, c)] + walk(rr, cc, r, c)
                for q in p[1:-1]:
                    seen_px[q] = True
                if len(p) >= min_px:
                    lines.append(p)
    out = []
    for p in lines:
        xy = np.array([((c + 0.5) * R - h, h - (r + 0.5) * R) for r, c in p])
        if len(xy) > 7:
            k = 5; sm = xy.copy()
            for i in range(len(xy)):
                sm[i] = xy[max(0, i - k):i + k + 1].mean(0)
            sm[0], sm[-1] = xy[0], xy[-1]; xy = sm
        xy = xy[::3] if len(xy) > 6 else xy
        if len(xy) >= 2:
            out.append(_chaikin(xy, 2))
    return out


def route_cartpaths(layout, frame, chm, dem_data, building, water_mask, P=None, g=None, shift=(0.0, 0.0), R=1.0):
    """Cart path network (local metres, LiDAR frame) for every hole of every nine."""
    from PIL import Image, ImageDraw
    from scipy import ndimage
    from skimage.graph import MCP_Geometric
    from .raster import rasterize_instances
    S = frame.size_m; n = int(S / R); h = S / 2
    xs = -h + (np.arange(n) + 0.5) * R; zs = h - (np.arange(n) + 0.5) * R

    def down(a, how):
        k = a.shape[0] // n
        b = a[:n * k, :n * k].reshape(n, k, n, k)
        return b.max((1, 3)) if how == "max" else b.mean((1, 3))
    Pm = np.zeros((n, n), np.float32)
    if P is not None:
        ci = np.clip(((xs - shift[0] - g["x0"]) / g["res"]).astype(int), 0, P.shape[1] - 1)
        ri = np.clip(((g["z1"] - (zs - shift[1])) / g["res"]).astype(int), 0, P.shape[0] - 1)
        inside = ((xs - shift[0] > g["x0"])[None, :] & ((zs - shift[1]) < g["z1"])[:, None])
        Pm = ndimage.gaussian_filter(ndimage.maximum_filter(P, 3)[np.ix_(ri, ci)] * inside, 0.8)
    chm1 = down(chm, "max"); dem1 = down(dem_data, "mean")
    gy, gx = np.gradient(dem1, R); slope = np.hypot(gx, gy)

    def ras(kinds):
        return rasterize_instances([s for s in layout.surfaces if s.kind in kinds], frame, n) > 0
    play, hazard = ras(("fairway", "green", "tee")), ras(("bunker",))
    water = ras(("water",)) | (down(water_mask.astype(np.uint8), "max") > 0 if water_mask is not None else False)
    house = ndimage.binary_dilation(down(building.astype(np.uint8), "max") > 0, iterations=2) if building is not None \
        else np.zeros((n, n), bool)
    im = Image.new("L", (n, n), 0); dr = ImageDraw.Draw(im)
    for kind, ln in layout.paths:
        if kind == "cartpath" and len(ln) > 1:
            dr.line([((x + h) / R, (h - z) / R) for x, z in ln], fill=1, width=2)
    osm = np.asarray(im, bool)
    cost = 0.5 + 9.0 * (1 - np.clip(Pm * 1.4, 0, 1)) ** 2
    cost += 300 * np.clip(slope - 0.08, 0, None)
    cost[chm1 > 3] += 25
    cost = np.where(osm, 0.3, cost)
    cost[play] = 120; cost[hazard] = 3000; cost[house] = 3000; cost[water] = 250

    def ring(pt, r0=6, r1=22):
        c = int((pt[0] + h) / R); r = int((h - pt[1]) / R)
        yy, xx = np.mgrid[r - r1:r + r1 + 1, c - r1:c + r1 + 1]; dd = np.hypot(yy - r, xx - c)
        ok = (dd >= r0) & (dd <= r1) & (yy >= 0) & (yy < n) & (xx >= 0) & (xx < n)
        yy, xx = yy[ok], xx[ok]
        good = ~play[yy, xx] & ~hazard[yy, xx] & ~house[yy, xx] & ~water[yy, xx]
        return list(zip(yy[good], xx[good]))
    holes = {hl.number: hl for hl in layout.holes}
    net = osm.copy()
    for nn in sorted({(k - 1) // 9 for k in holes}):
        hs = [holes[i] for i in range(nn * 9 + 1, nn * 9 + 10) if i in holes]
        for j, hl in enumerate(hs):
            segs = [(hl.path[0], hl.path[-1])]
            if j > 0:
                segs.append((hs[j - 1].path[-1], hl.path[0]))
            for a, b in segs:
                st, en = ring(a), ring(b)
                if not st or not en:
                    continue
                allp = st + en
                r0 = max(min(p[0] for p in allp) - 60, 0); r1 = min(max(p[0] for p in allp) + 60, n - 1)
                c0 = max(min(p[1] for p in allp) - 60, 0); c1 = min(max(p[1] for p in allp) + 60, n - 1)
                mcp = MCP_Geometric(cost[r0:r1 + 1, c0:c1 + 1])
                costs, _ = mcp.find_costs([(p[0] - r0, p[1] - c0) for p in st])
                e = min(en, key=lambda p: costs[p[0] - r0, p[1] - c0])
                for r, c in mcp.traceback((e[0] - r0, e[1] - c0)):
                    net[r + r0, c + c0] = True
    net = ndimage.binary_dilation(net, iterations=2)
    return _skeleton_lines(net, R, h)


# ------------------------------------------------------------------ streets
def street_items(roads, frame, naip4=None, meta=None, shift=(0.0, 0.0)):
    """[{line, width, kind, surface, curb, centre_line}] from OSM streets (local metres)."""
    out = []
    for f in roads.get("features", []):
        hw = f["properties"].get("highway")
        if hw not in ROAD_WIDTH:
            continue
        ln = np.array([frame.to_local(lo, la) for lo, la in f["geometry"]["coordinates"]])
        if len(ln) < 2:
            continue
        surface = "Road_Asphalt"
        if naip4 is not None:
            k = max(2, int(np.sum(np.linalg.norm(np.diff(ln, axis=0), axis=1)) / 3))
            t = np.linspace(0, 1, k)
            seg = np.interp(t, np.linspace(0, 1, len(ln)), ln[:, 0]), np.interp(t, np.linspace(0, 1, len(ln)), ln[:, 1])
            c = np.clip(((seg[0] - shift[0] - meta["x0"]) / meta["res"]).astype(int), 0, naip4.shape[1] - 1)
            r = np.clip(((meta["z1"] - (seg[1] - shift[1])) / meta["res"]).astype(int), 0, naip4.shape[0] - 1)
            px = naip4[r, c].astype(float) / 255
            if np.median(px[:, :3].mean(1)) > 0.52:
                surface = "Road_Concrete"
        out.append({"line": ln, "width": ROAD_WIDTH[hw], "kind": hw, "surface": surface,
                    "id": f"osm-{f['properties'].get('osm_id')}", "name": f["properties"].get("name"),
                    "curb": hw in ("residential", "tertiary", "secondary", "living_street"),
                    "centre_line": hw in ("tertiary", "secondary", "primary", "trunk")})
    return out


# ------------------------------------------------------------------ grading
def _resample(line, step):
    line = np.asarray(line, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return line[:1], d[:1]
    s = np.linspace(0, d[-1], max(2, int(d[-1] / step) + 1))
    return np.c_[np.interp(s, d, line[:, 0]), np.interp(s, d, line[:, 1])], s


def grade_paths(dem, items, water=None, protect=None):
    """Level each path/road cross-section and smooth its long profile, in place.
    items: [{line, width, smooth_m}]. water/protect: boolean grids left untouched."""
    from scipy import ndimage
    a = dem.data; n = a.shape[0]; res = dem.res_m; h = dem.frame.size_m / 2
    target = np.full(a.shape, np.nan, np.float32); weight = np.zeros(a.shape, np.float32)
    for it in items:
        pts, s = _resample(it["line"], res * 2)
        if len(pts) < 2:
            continue
        zc = dem.sample(pts[:, 0], pts[:, 1])
        sig = max(1.0, it.get("smooth_m", 6.0) / (res * 2))
        zc = ndimage.gaussian_filter1d(zc, sig, mode="nearest")
        half, sh = it["width"] / 2 + 0.3, it.get("shoulder_m", 2.0)
        x0, z0 = pts.min(0) - half - sh; x1, z1 = pts.max(0) + half + sh
        c0, c1 = max(int((x0 + h) / res), 0), min(int((x1 + h) / res) + 1, n)
        r0, r1 = max(int((h - z1) / res), 0), min(int((h - z0) / res) + 1, n)
        if c1 <= c0 or r1 <= r0:
            continue
        gx, gz = np.meshgrid(-h + (np.arange(c0, c1) + 0.5) * res, h - (np.arange(r0, r1) + 0.5) * res)
        from scipy.spatial import cKDTree
        d, idx = cKDTree(pts).query(np.c_[gx.ravel(), gz.ravel()])
        d = d.reshape(gx.shape); zt = zc[idx].reshape(gx.shape)
        w = np.clip(1 - (d - half) / sh, 0, 1)
        w = w * w * (3 - 2 * w)
        sub_w = weight[r0:r1, c0:c1]; sub_t = target[r0:r1, c0:c1]
        better = w > sub_w
        sub_t[better] = zt[better]; sub_w[better] = w[better]
    ok = weight > 0
    if water is not None:
        ok &= ~water
    if protect is not None:
        ok &= ~protect
    a[ok] = a[ok] * (1 - weight[ok]) + target[ok] * weight[ok]
    return int(ok.sum())


# ------------------------------------------------------------------ meshes
class _Mesh:
    def __init__(self):
        self.V, self.UV, self.T, self.M = [], [], [], []

    def strip(self, L, R_, uvL, uvR, mat):
        """Ribbon between two polylines (same length) of 3D points."""
        b = len(self.V)
        for i in range(len(L)):
            self.V += [L[i], R_[i]]; self.UV += [uvL[i], uvR[i]]
        for i in range(len(L) - 1):
            p = b + 2 * i
            self.T += [(p, p + 2, p + 1), (p + 1, p + 2, p + 3)]; self.M += [mat, mat]


def _offsets(pts):
    d = np.gradient(pts, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    return np.c_[-d[:, 1], d[:, 0]]          # left normal


def path_meshes(cart_lines, streets, dem, base_elev, water=None, bridges=None):
    """Meshes in the internal frame (x east, y up, z north)."""
    mk = _Mesh()
    bridges = bridges or []
    bzones = None
    if bridges:
        import shapely
        from shapely.geometry import LineString
        bzones = shapely.union_all([LineString(b["line"]).buffer(b["width"] / 2 + 0.6, cap_style="flat") for b in bridges])
        shapely.prepare(bzones)
    res = dem.res_m; h = dem.frame.size_m / 2; n = dem.data.shape[0]

    def wet(p):
        if water is None:
            return np.zeros(len(p), bool)
        c = np.clip(((p[:, 0] + h) / res).astype(int), 0, n - 1); r = np.clip(((h - p[:, 1]) / res).astype(int), 0, n - 1)
        return water[r, c]

    def ribbon(line, width, y_lift, mat, step=1.5, u_scale=4.0, offset=0.0, bridge=False):
        pts, s = _resample(line, step)
        if len(pts) < 2:
            return
        nrm = _offsets(pts)
        y = dem.sample(pts[:, 0], pts[:, 1]) - base_elev
        if bridge:
            wm = wet(pts)
            if wm.any() and (~wm).any():
                # deck spans the water at a straight grade between the banks, 0.6 m above them
                dry = np.nonzero(~wm)[0]
                y = np.where(wm, np.interp(np.arange(len(pts)), dry, y[dry]) + 0.6, y)
        L = pts + nrm * (offset + width / 2); Rr = pts + nrm * (offset - width / 2)
        yl = y + y_lift
        mk.strip([np.array([p[0], yy, p[1]]) for p, yy in zip(L, yl)], [np.array([p[0], yy, p[1]]) for p, yy in zip(Rr, yl)],
                 [(0, si / u_scale) for si in s], [(width / u_scale, si / u_scale) for si in s], mat)
        return pts, y, nrm, s

    def outside_bridges(line):
        """Split a line into the runs that are not on an owner-drawn bridge."""
        if bzones is None:
            return [np.asarray(line, float)]
        import shapely
        pts = _resample(line, 0.5)[0]
        ok = ~shapely.contains_xy(bzones, pts[:, 0], pts[:, 1])
        runs, cur = [], []
        for q, k in zip(pts, ok):
            if k:
                cur.append(q)
            else:
                if len(cur) > 1:
                    runs.append(np.array(cur))
                cur = []
        if len(cur) > 1:
            runs.append(np.array(cur))
        return runs

    cart_lines = [run for ln in cart_lines for run in outside_bridges(ln)]
    for ln in cart_lines:
        r = ribbon(ln, CART_WIDTH, 0.05, "Path_Concrete", bridge=True)
        if r is None:
            continue
        pts, y, nrm, s = r
        wm = wet(pts)
        if wm.any():                                   # bridge rails over water
            idx = np.nonzero(wm)[0]
            i0, i1 = max(idx.min() - 2, 0), min(idx.max() + 3, len(pts))
            for side in (1, -1):
                q = pts[i0:i1] + nrm[i0:i1] * side * (CART_WIDTH / 2 + 0.05)
                lo = y[i0:i1] + 0.05; hi = lo + 0.9
                mk.strip([np.array([p[0], a, p[1]]) for p, a in zip(q, lo)], [np.array([p[0], a, p[1]]) for p, a in zip(q, hi)],
                         [(si / 2, 0) for si in s[i0:i1]], [(si / 2, 0.5) for si in s[i0:i1]], "Bridge_Rail")
    for b in bridges:
        st = bridge_style(b["note"], b["carries"])
        pts, s, y = bridge_profile(dem, base_elev, b["line"], water, low=st.get("low", False))
        if len(pts) < 2:
            continue
        nrm = _offsets(pts); hw = b["width"] / 2
        th = 0.35 if b["carries"] != "road" else 0.6                       # deck thickness
        L3 = lambda off, yy: [np.array([p[0] + n[0] * off, a, p[1] + n[1] * off]) for p, n, a in zip(pts, nrm, yy)]  # noqa: E731
        u = [(0, si / 2.0) for si in s]; u2 = [(b["width"] / 2.0, si / 2.0) for si in s]
        mk.strip(L3(hw, y + 0.02), L3(-hw, y + 0.02), u, u2, st["deck"])              # deck
        for side in (1, -1):
            mk.strip(L3(side * hw, y + 0.02), L3(side * hw, y - th), [(si / 2, 0) for si in s],
                     [(si / 2, th / 2) for si in s], st["beam"])                     # deck edge / beam
            if st["rail_h"] > 0:
                rh = st["rail_h"]
                if st["solid_rail"]:
                    mk.strip(L3(side * (hw + 0.1), y), L3(side * (hw + 0.1), y + rh), [(si / 2, 0) for si in s],
                             [(si / 2, rh / 2) for si in s], st["rail"])
                else:                                                         # top rail + mid rail + posts
                    for hh in (rh, rh * 0.5):
                        mk.strip(L3(side * (hw + 0.05), y + hh - 0.08), L3(side * (hw + 0.05), y + hh),
                                 [(si / 2, 0) for si in s], [(si / 2, 0.05) for si in s], st["rail"])
                    for i in range(0, len(pts), max(1, int(1.8 / 0.75))):
                        p, n = pts[i], nrm[i]; c = p + n * side * (hw + 0.05); dd = np.array([-n[1], n[0]]) * 0.06
                        q0, q1 = c - dd, c + dd
                        mk.strip([np.array([q0[0], y[i], q0[1]]), np.array([q0[0], y[i] + rh, q0[1]])],
                                 [np.array([q1[0], y[i], q1[1]]), np.array([q1[0], y[i] + rh, q1[1]])],
                                 [(0, 0), (0, 0.5)], [(0.05, 0), (0.05, 0.5)], st["rail"])
        for i in (0, len(pts) - 1):                                           # abutments into each bank
            p, n = pts[i], nrm[i]
            d = np.array([-n[1], n[0]]) * (1 if i == 0 else -1) * 0.8
            a0, a1 = p + n * (hw + 0.2), p - n * (hw + 0.2)
            yb = y[i] - th - 1.5
            for q0, q1 in ((a0, a1), (a0 + d, a1 + d)):
                mk.strip([np.array([q0[0], yb, q0[1]]), np.array([q0[0], y[i] - th, q0[1]])],
                         [np.array([q1[0], yb, q1[1]]), np.array([q1[0], y[i] - th, q1[1]])],
                         [(0, 0), (0, 0.6)], [(1.2, 0), (1.2, 0.6)], st["beam"])
    from scipy.spatial import cKDTree
    rank = {k: i for i, k in enumerate(["service", "living_street", "road", "unclassified", "residential", "tertiary",
                                        "secondary", "primary", "trunk", "motorway"])}
    allp = [(_resample(st["line"], 2.0)[0], st["width"], i) for i, st in enumerate(streets)]
    cart_pts = [(_resample(ln, 1.0)[0], CART_WIDTH) for ln in cart_lines]
    for i, st in enumerate(streets):
        # bigger roads sit a hair higher so they win where streets overlap at junctions
        r = ribbon(st["line"], st["width"], 0.04 + 0.003 * rank.get(st["kind"], 0), st["surface"], step=2.0, u_scale=8.0,
                   bridge=True)
        if r is None:
            continue
        pts, y, nrm, s = r
        wm = wet(pts)
        if wm.any():                                   # road bridge: concrete parapets
            idx = np.nonzero(wm)[0]
            i0, i1 = max(idx.min() - 2, 0), min(idx.max() + 3, len(pts))
            for side in (1, -1):
                q = pts[i0:i1] + nrm[i0:i1] * side * (st["width"] / 2 + 0.3)
                lo = y[i0:i1]; hi = lo + 1.0
                mk.strip([np.array([p[0], a, p[1]]) for p, a in zip(q, lo)], [np.array([p[0], a, p[1]]) for p, a in zip(q, hi)],
                         [(si / 2, 0) for si in s[i0:i1]], [(si / 2, 0.5) for si in s[i0:i1]], "Road_Curb")
        if st["curb"]:
            pts = _resample(st["line"], 1.0)[0]
            others = [(p, w) for p, w, j in allp if j != i] + cart_pts   # curb cuts at cart path crossings
            ok = np.ones(len(pts), bool)
            for p, w in others:                      # no curb across a joining street
                d, _ = cKDTree(p).query(pts)
                ok &= d > w / 2 + 1.5
            runs, cur = [], []
            for q, k in zip(pts, ok):
                if k:
                    cur.append(q)
                elif len(cur) > 1:
                    runs.append(cur); cur = []
                else:
                    cur = []
            if len(cur) > 1:
                runs.append(cur)
            for run in runs:
                for side in (1, -1):
                    ribbon(np.array(run), 0.45, 0.12, "Road_Curb", step=1.0, u_scale=2.0, offset=side * (st["width"] / 2 + 0.22),
                           bridge=True)
        if st["centre_line"]:
            ribbon(st["line"], 0.12, 0.05 + 0.003 * rank.get(st["kind"], 0), "Road_Line", step=2.0, u_scale=6.0, bridge=True)
    return np.array(mk.V, float), np.array(mk.UV, float), np.array(mk.T, int), mk.M


def path_textures(out_dir: Path) -> dict:
    from PIL import Image, ImageDraw, ImageFilter
    from scipy import ndimage
    rng = np.random.default_rng(3)
    out_dir.mkdir(parents=True, exist_ok=True)

    def noise(size, sc, amp):
        a = ndimage.gaussian_filter(rng.normal(0, 1, (size, size)), sc)
        return (a - a.mean()) / (a.std() + 1e-9) * amp

    def surf(base, size=512, fine=6, coarse=6):
        a = base[0] + noise(size, 0.8, fine) + noise(size, 30, coarse)
        return np.stack([a, a - (base[0] - base[1]), a - (base[0] - base[2])], -1)
    tex = {}
    # cart path concrete: 2.5 m wide, scored joints every 3 m (u across 0..0.625 of 4 m tile)
    a = surf((196, 194, 188)); im = Image.fromarray(a.clip(0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
    for k in range(4):
        y = int(k * 512 / 4 * (4 / 3)) % 512
        dr.line([0, y, 512, y], fill=(150, 148, 144), width=2)
    tex["Path_Concrete"] = im
    tex["Road_Asphalt"] = Image.fromarray((surf((78, 78, 80), fine=10, coarse=5)).clip(0, 255).astype(np.uint8))
    a = surf((204, 202, 196)); im = Image.fromarray(a.clip(0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
    for k in range(2):
        dr.line([0, k * 256, 512, k * 256], fill=(160, 158, 152), width=2)       # panels every 4 m (8 m tile)
    dr.line([256, 0, 256, 512], fill=(160, 158, 152), width=2)
    tex["Road_Concrete"] = im
    tex["Road_Curb"] = Image.fromarray(surf((214, 212, 206), size=128).clip(0, 255).astype(np.uint8))
    im = Image.new("RGB", (64, 512), (78, 78, 80)); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 64, 512 * 3 / 6], fill=(226, 186, 52))                   # dashed yellow: 3 m on, 3 m off
    tex["Road_Line"] = im
    tex["Bridge_Rail"] = Image.fromarray(surf((120, 92, 64), size=128, fine=12, coarse=8).clip(0, 255).astype(np.uint8))
    a = surf((132, 100, 70), size=512, fine=10, coarse=6); im = Image.fromarray(a.clip(0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
    for k in range(0, 512, 37):                                                   # deck boards across the span
        dr.line([0, k, 512, k], fill=(80, 60, 42), width=3)
    tex["Bridge_Wood"] = im
    a = surf((176, 164, 140), size=256, fine=10, coarse=8); im = Image.fromarray(a.clip(0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
    for k in range(0, 256, 32):
        dr.line([0, k, 256, k], fill=(120, 110, 96), width=2)
        for j in range((k // 32) % 2 * 24, 256, 48):
            dr.line([j, k, j, k + 32], fill=(120, 110, 96), width=2)
    tex["Bridge_Stone"] = im
    tex["Bridge_Steel"] = Image.fromarray(surf((70, 74, 78), size=128, fine=4, coarse=3).clip(0, 255).astype(np.uint8))
    files = {}
    for k, im in tex.items():
        files[k] = f"{k}.png"; im.filter(ImageFilter.SMOOTH).save(out_dir / files[k])
    return files


def reviewed_lines(draft: dict, edits: list[dict], only_mine: bool = False):
    """Draft lines (page frame) overlaid with the owner's saved edits: an edit replaces the
    draft line with the same id, `deleted` removes it, new ids are added. With only_mine,
    untouched draft cart paths are dropped."""
    lines = {l["id"]: l for l in draft.get("lines", []) if not (only_mine and l["kind"] == "cart")}
    for e in edits:
        if not e.get("id"):
            continue
        if e.get("deleted"):
            lines.pop(e["id"], None)
        elif len(e.get("pts", [])) >= 2:
            lines[e["id"]] = e
    return list(lines.values())


def lines_from_review(lines, shift, surface_of=None, with_bridges=False):
    """Reviewed lines -> (cart lines in the LiDAR frame, street items[, bridges]). Hand-drawn
    lines are rounded a little at the corners so paths curve the way they're poured."""
    cart, streets, bridges = [], [], []
    for l in lines:
        pts = np.asarray(l["pts"], float)
        if l["kind"] == "bridge":
            carries = l.get("carries", "cart")
            q = pts if carries == "road" else pts + np.asarray(shift)
            bridges.append({"id": l["id"], "line": _chaikin(q, 2) if len(q) > 2 else q, "carries": carries,
                            "width": float(l.get("width") or {"cart": 3.0, "path": 2.0, "road": 8.0}.get(carries, 3.0)),
                            "note": (l.get("note") or "").strip()})
            continue
        if l["kind"] == "cart":
            q = pts + np.asarray(shift)                     # page/photo -> LiDAR frame, like traced shapes
            cart.append(_chaikin(q, 2) if (l.get("method") == "manual" or l.get("edited")) and len(q) > 2 else q)
        else:
            road = l.get("road") or l.get("highway") or "residential"
            streets.append({"line": pts, "width": float(l.get("width") or ROAD_WIDTH.get(road, 8.0)), "kind": road,
                            "surface": l.get("surface") or "Road_Asphalt", "id": l["id"], "name": l.get("name"),
                            "curb": road in ("residential", "tertiary", "secondary", "living_street"),
                            "centre_line": road in ("tertiary", "secondary", "primary", "trunk")})
    return (cart, streets, bridges) if with_bridges else (cart, streets)


def bridge_style(note: str, carries: str) -> dict:
    """Deck / rail / beam finishes from the owner's note (keywords), with sensible defaults."""
    n = (note or "").lower()
    if any(k in n for k in ("stone", "rock", "limestone", "masonry")):
        st = {"deck": "Path_Concrete", "rail": "Bridge_Stone", "beam": "Bridge_Stone", "rail_h": 0.75, "solid_rail": True}
    elif any(k in n for k in ("steel", "metal", "iron")):
        st = {"deck": "Bridge_Wood" if "wood" in n else "Path_Concrete", "rail": "Bridge_Steel", "beam": "Bridge_Steel",
              "rail_h": 1.05, "solid_rail": False}
    elif any(k in n for k in ("concrete", "culvert")) or carries == "road":
        st = {"deck": "Road_Asphalt" if carries == "road" else "Path_Concrete", "rail": "Road_Curb", "beam": "Road_Curb",
              "rail_h": 0.9, "solid_rail": True}
    else:                                                   # wood / timber (default for cart bridges)
        st = {"deck": "Bridge_Wood", "rail": "Bridge_Rail", "beam": "Bridge_Rail", "rail_h": 1.0, "solid_rail": False}
    if "no rail" in n or "without rail" in n:
        st["rail_h"] = 0.0
    if "low" in n:
        st["low"] = True
    return st


def bridge_profile(dem, base_elev, line, water=None, clearance=0.6, low=False):
    """Deck centreline (pts, s, y): ends on the graded ground at each bank, a gentle arch
    between, lifted to clear the ground beneath by `clearance`, smoothed so it never kinks."""
    from scipy import ndimage
    pts, s = _resample(line, 0.75)
    if len(pts) < 2:
        return pts, s, np.zeros(len(pts))
    g = dem.sample(pts[:, 0], pts[:, 1]) - base_elev
    L = s[-1]; t = s / max(L, 1e-6)
    y0, y1 = g[0] + 0.05, g[-1] + 0.05
    rise = 0.0 if low else float(np.clip(0.035 * L, 0.25, 1.4))
    y = y0 + (y1 - y0) * t + rise * np.sin(np.pi * t)
    inner = (t > 0.08) & (t < 0.92)
    need = np.where(inner, g + (0.25 if low else clearance), -1e9)
    base = y.copy()
    sig = max(1.0, 0.06 * len(y))
    for _ in range(4):                       # lift to clear the ground, then smooth, until both hold
        y = np.maximum(y, need)
        y = ndimage.gaussian_filter1d(y, sigma=sig, mode="nearest")
    y = np.maximum(y, need)
    y = ndimage.gaussian_filter1d(y, sigma=max(1.0, sig / 3), mode="nearest")
    # pin both ends to the banks with a smooth hand-off over the first/last 10 % of the span
    ss = lambda v: v * v * (3 - 2 * v)       # noqa: E731
    w = ss(np.clip(t / 0.1, 0, 1)) * ss(np.clip((1 - t) / 0.1, 0, 1))
    y = base * (1 - w) + y * w
    return pts, s, y


def draft_from_build(cart_lines, streets, shift) -> dict:
    """Lines for the review page (page/photo frame), from a build."""
    out = [{"id": f"c{i}", "kind": "cart", "pts": [[round(float(x - shift[0]), 2), round(float(z - shift[1]), 2)] for x, z in
                                                  np.asarray(ln)[::2].tolist() + [list(np.asarray(ln)[-1])]],
            "method": "traced"} for i, ln in enumerate(cart_lines)]
    for st in streets:
        out.append({"id": st.get("id") or f"r{len(out)}", "kind": "road", "road": st["kind"], "width": st["width"],
                    "surface": st["surface"], "name": st.get("name"), "method": "osm",
                    "pts": [[round(float(x), 2), round(float(z), 2)] for x, z in np.asarray(st["line"])]})
    return {"lines": out}


def write_paths(cart_lines, streets, dem, base_elev, out_dir: Path, water=None, shift=(0.0, 0.0), bridges=None) -> dict:
    import json as _json
    bridges = bridges or []
    (out_dir / "lines.json").write_text(_json.dumps(draft_from_build(cart_lines, streets, shift)))
    binfo = []
    for b in bridges:
        st = bridge_style(b["note"], b["carries"])
        pts, s, y = bridge_profile(dem, base_elev, b["line"], water, low=st.get("low", False))
        if len(pts) < 2:
            continue
        binfo.append({"id": b["id"], "note": b["note"], "carries": b["carries"], "width_m": b["width"],
                      "length_m": round(float(s[-1]), 1), "finish": {k: v for k, v in st.items() if k in ("deck", "rail", "beam")},
                      "ends": [[round(float(pts[0][0]), 2), round(float(y[0]), 2), round(float(pts[0][1]), 2)],
                               [round(float(pts[-1][0]), 2), round(float(y[-1]), 2), round(float(pts[-1][1]), 2)]],
                      "deck_rise_m": round(float(np.max(y - (y[0] + (y[-1] - y[0]) * s / max(s[-1], 1e-6)))), 2)})
    (out_dir / "bridges.json").write_text(_json.dumps({"bridges": binfo}, indent=1))
    files = path_textures(out_dir)
    with open(out_dir / "paths.mtl", "w") as f:
        for k, fn in files.items():
            f.write(f"newmtl {k}\nKd 1 1 1\nKs 0 0 0\nmap_Kd {fn}\n\n")
    V, UV, T, M = path_meshes(cart_lines, streets, dem, base_elev, water, bridges)
    if not len(T):
        return {"triangles": 0}
    vo = V * np.array([1.0, 1.0, -1.0])
    a, b, c = vo[T[:, 0]], vo[T[:, 1]], vo[T[:, 2]]
    nrm_ = np.cross(b - a, c - a)
    flip = (nrm_[:, 1] < 0) & (np.abs(nrm_[:, 1]) > 0.5 * np.linalg.norm(nrm_, axis=1))   # flat parts face up
    T = T.copy(); T[flip] = T[flip][:, [0, 2, 1]]
    with open(out_dir / "paths.obj", "w") as f:
        f.write("# Cart paths, streets, curbs, bridges: metres, right-handed Y-up, x east, z SOUTH\nmtllib paths.mtl\n")
        f.write("o Paths\ng Paths\n")
        np.savetxt(f, vo, fmt="v %.3f %.3f %.3f")
        np.savetxt(f, UV, fmt="vt %.4f %.4f")
        Ma = np.array(M)
        for m in sorted(set(M)):
            f.write(f"usemtl {m}\n")
            tt = T[Ma == m] + 1
            np.savetxt(f, np.column_stack([tt[:, 0], tt[:, 0], tt[:, 1], tt[:, 1], tt[:, 2], tt[:, 2]]), fmt="f %d/%d %d/%d %d/%d")
    lens = lambda ls: round(sum(float(np.sum(np.linalg.norm(np.diff(np.asarray(l), axis=0), axis=1))) for l in ls) / 1000, 2)  # noqa: E731
    return {"triangles": int(len(T)), "cart_path_km": lens(cart_lines), "street_km": lens([s["line"] for s in streets]),
            "bridges": binfo}
