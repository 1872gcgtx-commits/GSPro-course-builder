"""Draft course surfaces (greens, tees, fairways, bunkers) from imagery + LiDAR,
seeded by the traced hole lines (back tee -> aim points -> green centre).

Signals (all on the course's 0.5 m north-up grid):
  sand      bright, low-green-excess, treeless pixels, confirmed by a LiDAR bowl
  mown turf low image texture, green-excess, treeless, not sand/water/building
  greens    colour-seeded region grown from the green click on smooth turf,
            thresholds raised until the region "leaks" (area jumps) or exceeds a
            green's size; LiDAR pad lift is reported as a confidence cue
  tees      same growth from the back-tee click, limited to level ground (<4 %)
  fairways  turf matching the colour sampled along the hole line, inside a
            corridor around the line, minus green/tee/bunkers, cleaned morphologically
Anything that fails its size/shape checks falls back to a standard shape and is
marked `method: "fallback"` so the review map can flag it.

Output: a GeoJSON FeatureCollection (lon/lat) with properties
  golf, hole (scene number), nine, nine_hole, method, confidence, draft_id
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage

from .geo import CourseFrame
from .layout import HoleLine


class Rasters:
    def __init__(self, frame: CourseFrame, overlay_path, dem: np.ndarray, chm: np.ndarray | None,
                 water: np.ndarray | None = None, building: np.ndarray | None = None):
        self.frame = frame
        self.n = dem.shape[0]
        self.res = frame.size_m / self.n
        img = Image.open(overlay_path).convert("RGB")
        if img.size != (self.n, self.n):
            img = img.resize((self.n, self.n), Image.BILINEAR)
        rgb = np.asarray(img, dtype=np.float32)
        # the USGS NAIP mirror is JPEG-blocky: soften blocks before measuring colour
        self.rgb = ndimage.gaussian_filter(rgb, (1.2, 1.2, 0))
        R, G, B = self.rgb[..., 0], self.rgb[..., 1], self.rgb[..., 2]
        self.lum = (R + G + B) / 3
        self.exg = 2 * G - R - B
        mean = ndimage.uniform_filter(self.lum, 7)
        self.texture = np.sqrt(np.maximum(ndimage.uniform_filter(self.lum ** 2, 7) - mean ** 2, 0))
        self.dem = dem.astype(np.float32)
        gz, gx = np.gradient(ndimage.gaussian_filter(self.dem, 1.5), self.res)
        self.slope = np.hypot(gx, gz)
        self.detrend = self.dem - ndimage.gaussian_filter(self.dem, 8 / self.res)
        self.trees = (chm > 1.5) if chm is not None else np.zeros_like(self.dem, bool)
        self.water = water if water is not None and water.shape == self.dem.shape else np.zeros_like(self.dem, bool)
        self.building = building if building is not None and building.shape == self.dem.shape else np.zeros_like(self.dem, bool)
        blocked = self.trees | self.water | self.building
        self.lum_rel = self.lum - ndimage.gaussian_filter(self.lum, 15 / self.res)
        # sand on NAIP ~ RGB(190,188,178): bright, almost no green excess; turf exg is 40-60
        near_block = ndimage.binary_dilation(self.trees | self.building, iterations=3)
        sand_px = (self.lum > 160) & (self.exg < 22) & (self.lum_rel > 8) & ~blocked & ~near_block
        sand_px = ndimage.binary_opening(sand_px, iterations=1)
        self.sand = sand_px
        self.turf = (self.exg > 4) & (self.texture < 9) & ~blocked & ~sand_px
        self.no_bunker = ndimage.binary_dilation(self.water, iterations=int(15 / self.res))

    def exclude(self, mask):
        """Extra areas where bunkers can't be (e.g. the creek corridor from the layout)."""
        self.no_bunker |= mask

    def rc(self, x, z):
        h = self.frame.size_m / 2
        return int((h - z) / self.res), int((x + h) / self.res)

    def xz(self, r, c):
        h = self.frame.size_m / 2
        return -h + (c + 0.5) * self.res, h - (r + 0.5) * self.res


def _window(R: Rasters, x, z, half_m):
    r, c = R.rc(x, z)
    h = int(half_m / R.res)
    return slice(max(0, r - h), min(R.n, r + h)), slice(max(0, c - h), min(R.n, c + h)), (r, c)


def _grow(R: Rasters, x, z, half_m, allowed, area_range, seed_r_m=2.5, leak_ratio=1.8):
    """Colour-seeded region growing; returns (mask_in_window, window, area_m2) or None."""
    rs, cs, (r, c) = _window(R, x, z, half_m)
    rgb = R.rgb[rs, cs]
    al = allowed[rs, cs]
    lr, lc = r - rs.start, c - cs.start
    k = max(1, int(seed_r_m / R.res))
    seed = rgb[max(0, lr - k):lr + k + 1, max(0, lc - k):lc + k + 1].reshape(-1, 3)
    mu = np.median(seed, 0)
    dist = np.linalg.norm(rgb - mu, axis=-1)
    cell = R.res * R.res
    best, prev_area = None, None
    for t in (6, 8, 10, 12, 14, 17, 20, 24, 28):
        m = (dist < t) & al
        m = ndimage.binary_closing(m, iterations=2)
        lab, _ = ndimage.label(m)
        sl = lab[min(max(lr, 0), lab.shape[0] - 1), min(max(lc, 0), lab.shape[1] - 1)]
        if sl == 0:
            # seed pixel itself not inside: take the nearest component within 4 m
            d, idx = ndimage.distance_transform_edt(lab == 0, return_indices=True)
            if d[lr, lc] * R.res > 4:
                continue
            sl = lab[idx[0][lr, lc], idx[1][lr, lc]]
        reg = ndimage.binary_fill_holes(lab == sl)
        area = reg.sum() * cell
        if prev_area and area > prev_area * leak_ratio and prev_area >= area_range[0]:
            break                                   # leaked into fairway/rough: keep previous
        if area > area_range[1]:
            break
        if area >= area_range[0] and _solidity(reg) < 0.72:
            break                                   # tendrils = leaking along paths/fringes
        if area >= area_range[0]:
            best = reg
        prev_area = area
    if best is None:
        return None
    best = ndimage.binary_opening(best, iterations=2)
    lab, n = ndimage.label(best)
    if n > 1:
        sizes = ndimage.sum(best, lab, range(1, n + 1))
        best = lab == (1 + int(np.argmax(sizes)))
    return best, (rs, cs), best.sum() * cell


def _solidity(mask) -> float:
    from scipy.spatial import ConvexHull
    rr, cc = np.nonzero(mask)
    if len(rr) < 10:
        return 1.0
    pts = np.column_stack([rr, cc]).astype(float)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 1.0
    return float(len(rr) / max(hull.volume, 1.0))


def _polygon(R: Rasters, mask, win, tol_px):
    from skimage.measure import approximate_polygon, find_contours
    rs, cs = win
    padded = np.pad(mask.astype(float), 1)
    cs_ = find_contours(padded, 0.5)
    if not cs_:
        return None
    cont = max(cs_, key=len)
    cont = approximate_polygon(cont, tolerance=tol_px)
    ring = [R.xz(rs.start + p[0] - 1, cs.start + p[1] - 1) for p in cont]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring if len(ring) >= 4 else None


def _ellipse(cx, cz, ax, az, ang, n=28):
    ca, sa = math.cos(ang), math.sin(ang)
    pts = [(cx + ax * math.cos(t) * ca - az * math.sin(t) * sa,
            cz + ax * math.cos(t) * sa + az * math.sin(t) * ca)
           for t in (2 * math.pi * i / n for i in range(n))]
    return pts + [pts[0]]


def _rect(cx, cz, length, width, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    hl, hw = length / 2, width / 2
    pts = [(cx + dx * ca - dz * sa, cz + dx * sa + dz * ca) for dx, dz in
           ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))]
    return pts + [pts[0]]


def _rasterize(R: Rasters, ring, win):
    from PIL import ImageDraw
    rs, cs = win
    img = Image.new("L", (cs.stop - cs.start, rs.stop - rs.start), 0)
    pts = []
    for x, z in ring:
        r, c = R.rc(x, z)
        pts.append((c - cs.start, r - rs.start))
    ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.asarray(img, bool)


def draft_hole(R: Rasters, hole: HoleLine, nine: int | None, nine_hole: int | None) -> list[dict]:
    path = hole.path
    tee, green = path[0], path[-1]
    prev = path[-2]
    play_ang_in = math.atan2(green[1] - prev[1], green[0] - prev[0])
    play_ang_out = math.atan2(path[1][1] - tee[1], path[1][0] - tee[0])
    feats = []

    def add(kind, ring, method, conf, idx=0):
        feats.append({"kind": kind, "ring": ring, "method": method, "confidence": round(conf, 2),
                      "draft_id": f"h{hole.number:02d}-{kind}-{idx}"})

    # --- green
    allowed = R.turf | ((R.exg > 0) & ~R.trees & ~R.sand & ~R.water)
    g = _grow(R, *green, 40, allowed & (R.slope < 0.12), (180, 1150))
    green_mask_win = None
    if g:
        mask, win, area = g
        ring = _polygon(R, mask, win, 1.2)
        if ring:
            lift = float(np.median(R.detrend[win][mask]) - np.median(R.detrend[win][~mask]))
            conf = 0.55 + (0.2 if 300 <= area <= 1100 else 0) + (0.2 if lift > 0.08 else 0)
            add("green", ring, "detected", min(conf, 0.95))
            green_mask_win = (mask, win)
    if not feats:
        add("green", _ellipse(*green, 16, 12, play_ang_in), "fallback", 0.2)

    # --- back tee box: level LiDAR pad connected to the click
    tee_done = False
    rs, cs, (r, c) = _window(R, *tee, 28)
    if rs.stop - rs.start > 10 and cs.stop - cs.start > 10:
        lr, lc = r - rs.start, c - cs.start
        z = R.dem[rs, cs].astype(float)
        yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
        k = int(3 / R.res)
        near = (abs(yy - lr) <= k) & (abs(xx - lc) <= k)
        A = np.column_stack([np.ones(near.sum()), xx[near], yy[near]])
        coef, *_ = np.linalg.lstsq(A, z[near], rcond=None)
        resid = np.abs(z - (coef[0] + coef[1] * xx + coef[2] * yy))
        level = (resid < 0.12) & (R.slope[rs, cs] < 0.045) & ~R.trees[rs, cs] & ~R.sand[rs, cs]
        level = ndimage.binary_opening(level, iterations=2)
        lab, _ = ndimage.label(level)
        if 0 <= lr < lab.shape[0] and 0 <= lc < lab.shape[1]:
            sl = lab[lr, lc]
            if sl == 0:
                d, idx = ndimage.distance_transform_edt(lab == 0, return_indices=True)
                if d[lr, lc] * R.res <= 3:
                    sl = lab[idx[0][lr, lc], idx[1][lr, lc]]
            if sl:
                m = ndimage.binary_fill_holes(lab == sl)
                area = m.sum() * R.res * R.res
                if 70 <= area <= 900 and _solidity(m) > 0.75:
                    ring = _polygon(R, m, (rs, cs), 1.0)
                    if ring:
                        add("tee", ring, "detected", 0.7 if area >= 120 else 0.5)
                        tee_done = True
    if not tee_done:
        add("tee", _rect(*tee, 22, 11, play_ang_out), "fallback", 0.2)

    # --- bunkers near the green and along the hole
    span = math.dist(tee, green)
    cx, cz = (tee[0] + green[0]) / 2, (tee[1] + green[1]) / 2
    rs, cs, _ = _window(R, cx, cz, span / 2 + 70)
    win = (rs, cs)
    sand = R.sand[rs, cs] & (R.detrend[rs, cs] < 0.0) & ~R.no_bunker[rs, cs]
    sand = ndimage.binary_closing(sand, iterations=2)
    sand = ndimage.binary_fill_holes(sand)
    lab, nlab = ndimage.label(sand)
    from .layout import point_polyline_dist
    bunkers = 0
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        m = lab[sl] == i + 1
        area = m.sum() * R.res * R.res
        if not 15 <= area <= 700:
            continue
        if _solidity(m) < 0.55:
            continue
        bowl = float(np.median(R.detrend[rs, cs][sl][m]))
        if bowl > -0.06:
            continue                                # flat bright patch (path, roof, dirt) - not a bunker
        rr, cc = np.nonzero(m)
        x, z = R.xz(rs.start + sl[0].start + rr.mean(), cs.start + sl[1].start + cc.mean())
        dg = math.dist((x, z), green)
        if dg > 60 and point_polyline_dist((x, z), path) > 40:
            continue
        sub = np.zeros_like(sand)
        sub[sl] = m
        ring = _polygon(R, sub, win, 1.0)
        if ring:
            add("bunker", ring, "detected", 0.6 + (0.3 if bowl < -0.15 else 0), bunkers)
            bunkers += 1

    # --- fairway (par 4/5): mown width measured across the line every 10 m
    par = hole.par or 4
    if par >= 4 and span > 200:
        ring, conf = _fairway_profile(R, path, span)
        if ring:
            add("fairway", ring, "detected", conf)
        else:
            a = _along(path, 0.42)
            b = _along(path, max(0.5, 1 - 28 / max(span, 1)))
            add("fairway", _strip(a, b, 32), "fallback", 0.2)
    return feats


def _fairway_profile(R: Rasters, path, span):
    """Smooth fairway outline from the mown width left/right of the line."""
    L = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
    start = min(170.0, 0.4 * L)
    end = L - 26.0
    if end - start < 40:
        return None, 0
    stations = np.arange(start, end, 10.0)
    left, right, centres, normals, good = [], [], [], [], 0
    # fairway turf: mown, treeless, greener than the corridor's rough
    for s_ in stations:
        p = _along(path, s_ / L)
        q = _along(path, min(1.0, (s_ + 2) / L))
        dx, dz = q[0] - p[0], q[1] - p[1]
        n = math.hypot(dx, dz) or 1
        nx, nz = -dz / n, dx / n
        offs = np.arange(-45, 45.5, 0.5)
        xs, zs = p[0] + nx * offs, p[1] + nz * offs
        rr = np.clip(((R.frame.size_m / 2 - zs) / R.res).astype(int), 0, R.n - 1)
        cc = np.clip(((xs + R.frame.size_m / 2) / R.res).astype(int), 0, R.n - 1)
        turf = R.turf[rr, cc]
        exg = R.exg[rr, cc]
        ref = np.median(exg[turf]) if turf.any() else 0
        fw = turf & (exg >= ref - 2)
        fw = ndimage.binary_closing(fw, iterations=6)
        mid = len(offs) // 2
        # walk outwards from the line (or the nearest fairway cell within 12 m)
        idx = np.flatnonzero(fw)
        if len(idx) == 0 or np.min(np.abs(idx - mid)) * 0.5 > 12:
            left.append(np.nan); right.append(np.nan)
        else:
            c0 = idx[np.argmin(np.abs(idx - mid))]
            lo = c0
            while lo > 0 and fw[lo - 1]:
                lo -= 1
            hi = c0
            while hi < len(fw) - 1 and fw[hi + 1]:
                hi += 1
            left.append(offs[hi]); right.append(-offs[lo])
            good += 1
        centres.append(p); normals.append((nx, nz))
    left, right = np.array(left), np.array(right)
    if good < max(3, 0.4 * len(stations)):
        return None, 0
    def fill(a, default):
        ok = np.isfinite(a)
        a = np.interp(np.arange(len(a)), np.flatnonzero(ok), a[ok]) if ok.any() else np.full(len(a), default)
        a = np.clip(a, 10, 30)              # half-widths 10-30 m (fairways 20-60 m wide)
        return ndimage.gaussian_filter1d(a, 1.5)
    left, right = fill(left, 16), fill(right, 16)
    # taper the ends
    taper = np.ones(len(stations))
    taper[:2] = (0.55, 0.8)
    taper[-2:] = (0.85, 0.6)
    lpts = [(c[0] + n[0] * w * t, c[1] + n[1] * w * t) for c, n, w, t in zip(centres, normals, left, taper)]
    rpts = [(c[0] - n[0] * w * t, c[1] - n[1] * w * t) for c, n, w, t in zip(centres, normals, right, taper)]
    ring = lpts + rpts[::-1]
    ring.append(ring[0])
    return ring, round(0.4 + 0.4 * good / len(stations), 2)


def _along(path, f):
    seg = [math.dist(path[i], path[i + 1]) for i in range(len(path) - 1)]
    d, tot = f * sum(seg), 0.0
    for i, s in enumerate(seg):
        if tot + s >= d:
            u = (d - tot) / max(s, 1e-6)
            return (path[i][0] + u * (path[i + 1][0] - path[i][0]), path[i][1] + u * (path[i + 1][1] - path[i][1]))
        tot += s
    return path[-1]


def _strip(a, b, w):
    dx, dz = b[0] - a[0], b[1] - a[1]
    L = max(math.hypot(dx, dz), 1e-6)
    nx, nz = -dz / L * w / 2, dx / L * w / 2
    pts = [(a[0] + nx, a[1] + nz), (b[0] + nx, b[1] + nz), (b[0] - nx, b[1] - nz), (a[0] - nx, a[1] - nz)]
    return pts + [pts[0]]


def draft_course(R: Rasters, holes: list[HoleLine]) -> list[dict]:
    out = []
    for h in holes:
        nine = int(h.props["nine"]) if h.props.get("nine") else None
        nh = int(h.props["nine_hole"]) if h.props.get("nine_hole") else None
        for f in draft_hole(R, h, nine, nh):
            f.update({"hole": h.number, "nine": nine, "nine_hole": nh})
            out.append(f)
    return out


def to_geojson(feats: list[dict], frame: CourseFrame) -> dict:
    fc = []
    for f in feats:
        coords = [list(frame.to_lonlat(x, z)) for x, z in f["ring"]]
        props = {k: v for k, v in f.items() if k not in ("ring", "kind")}
        props["golf"] = f["kind"]
        props["source"] = "autodraft"
        fc.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [coords]}, "properties": props})
    return {"type": "FeatureCollection", "features": fc}
