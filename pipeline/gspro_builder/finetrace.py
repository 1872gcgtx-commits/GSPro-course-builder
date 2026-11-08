"""Fine tracing of greens and bunkers on the 0.3 m imagery, tuned on the owner's shapes.

Greens: the mowing line between green and collar shows as a faint closed edge in
the 0.3 m imagery. A marker-controlled watershed on the edge strength (inside
marker = disk at the owner's green click, outside markers = ring beyond the
search radius + trees + sand + water) settles on that edge, so natural shapes
(kidneys, peanuts, tiers) come out as they are.

Bunkers: a colour model (mean/covariance of R,G,B,NIR) learned from the owner's
bunkers ("tan = bunker"), confirmed by a LiDAR depression, shaped smooth/rounded.

`tune()` scores parameter choices against the owner's shapes (IoU for greens,
precision/recall for bunkers) and keeps the best, so every reviewed hole
improves the next draft.
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
from scipy import ndimage


class Hires:
    """Window access to the 0.3 m 4-band NAIP in course-frame metres."""

    def __init__(self, cfg):
        self.ext = json.loads((cfg.data_dir / "naip4.json").read_text())
        self.img = np.load(cfg.data_dir / "naip4.npy", mmap_mode="r")
        self.res = self.ext["res"]
        dem = np.load(cfg.data_dir / "dem.npy", mmap_mode="r")
        chm = np.load(cfg.data_dir / "chm.npy", mmap_mode="r")
        self.dem, self.chm = dem, chm
        self.dres = cfg.frame.size_m / dem.shape[0]
        self.half = cfg.frame.size_m / 2

    def window(self, x, z, r_m):
        r0 = int((self.ext["z1"] - (z + r_m)) / self.res)
        c0 = int((x - r_m - self.ext["x0"]) / self.res)
        n = int(2 * r_m / self.res)
        r0, c0 = max(r0, 0), max(c0, 0)
        w = np.asarray(self.img[r0:r0 + n, c0:c0 + n], np.float32)
        # LiDAR rasters resampled onto the same window
        rr, cc = np.mgrid[r0:r0 + w.shape[0], c0:c0 + w.shape[1]]
        xs = self.ext["x0"] + (cc + 0.5) * self.res
        zs = self.ext["z1"] - (rr + 0.5) * self.res
        dr = np.clip(((self.half - zs) / self.dres).astype(int), 0, self.dem.shape[0] - 1)
        dc = np.clip(((xs + self.half) / self.dres).astype(int), 0, self.dem.shape[1] - 1)
        r_min, r_max, c_min, c_max = dr.min(), dr.max() + 1, dc.min(), dc.max() + 1
        dem = np.asarray(self.dem[r_min:r_max, c_min:c_max], np.float32)
        chm = np.asarray(self.chm[r_min:r_max, c_min:c_max], np.float32)
        detr = dem - ndimage.gaussian_filter(dem, 8 / self.dres)
        return dict(img=w, r0=r0, c0=c0, chm=chm[dr - r_min, dc - c_min], detr=detr[dr - r_min, dc - c_min])

    def to_xz(self, win, r, c):
        return (self.ext["x0"] + (win["c0"] + c + 0.5) * self.res,
                self.ext["z1"] - (win["r0"] + r + 0.5) * self.res)

    def to_rc(self, win, x, z):
        return (int((self.ext["z1"] - z) / self.res) - win["r0"], int((x - self.ext["x0"]) / self.res) - win["c0"])


def raster_ring(shape, H: Hires, win, ring):
    from PIL import Image, ImageDraw
    img = Image.new("L", (shape[1], shape[0]), 0)
    pts = [H.to_rc(win, x, z)[::-1] for x, z in ring]
    ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.asarray(img, bool)


class BunkerModel:
    def __init__(self, mean, cov):
        self.mean, self.icov = mean, np.linalg.inv(cov + np.eye(4) * 4)

    def dist(self, img):
        d = img[..., :4] - self.mean
        return np.sqrt(np.einsum("...i,ij,...j->...", d, self.icov, d))

    @classmethod
    def from_shapes(cls, H, shapes):
        px = []
        for s in shapes:
            c = np.mean(s["ring"], axis=0)
            win = H.window(c[0], c[1], 25)
            m = ndimage.binary_erosion(raster_ring(win["img"].shape[:2], H, win, s["ring"]), iterations=3)
            if m.any():
                px.append(win["img"][m][:, :4])
        P = np.concatenate(px)
        return cls(P.mean(0), np.cov(P.T))


def trace_green(H: Hires, x, z, p, bunker_model=None):
    """Watershed green at click (x, z). p: dict(radius, sigma, edge_mix, min_area, max_area)."""
    from skimage.filters import sobel
    from skimage.segmentation import watershed
    win = H.window(x, z, p["radius"] + 6)
    img = win["img"]
    br = img[..., :3].mean(-1)
    nir = img[..., 3]
    s = p["sigma"] / H.res
    edge = sobel(ndimage.gaussian_filter(br, s)) + p["edge_mix"] * sobel(ndimage.gaussian_filter(nir, s))
    hh, ww = br.shape
    cr, cc = H.to_rc(win, x, z)
    yy, xx = np.mgrid[0:hh, 0:ww]
    dist = np.hypot(yy - cr, xx - cc) * H.res
    markers = np.zeros((hh, ww), np.int32)
    markers[dist <= 2.5] = 1
    outside = (dist >= p["radius"]) | (win["chm"] > 1.5)
    if bunker_model is not None:
        sand = (bunker_model.dist(img) < 3.5) & (dist > 4)
        outside |= ndimage.binary_opening(sand, iterations=2)
    markers[outside & (markers == 0)] = 2
    lab = watershed(edge, markers)
    m = ndimage.binary_fill_holes(lab == 1)
    m = ndimage.binary_opening(m, iterations=int(1.2 / H.res))
    l2, n = ndimage.label(m)
    if n == 0:
        return None, 0, None
    if l2[cr, cc] == 0:
        return None, 0, None
    m = l2 == l2[cr, cc]
    area = m.sum() * H.res ** 2
    if not p["min_area"] <= area <= p["max_area"]:
        return None, area, None
    m = ndimage.gaussian_filter(m.astype(np.float32), 0.8 / H.res) > 0.5
    return m, area, win


class GreenModel(BunkerModel):
    """Colour/NIR statistics of the owner's greens."""


def grow_green(H, x, z, gmodel, t, radius=34.0, area_rng=(150, 1400)):
    win = H.window(x, z, radius + 4)
    img = ndimage.gaussian_filter(win["img"], (2, 2, 0))
    d = gmodel.dist(img)
    hh, ww = d.shape
    cr, cc = H.to_rc(win, x, z)
    yy, xx = np.mgrid[0:hh, 0:ww]
    inside = (np.hypot(yy - cr, xx - cc) * H.res < radius) & (win["chm"] < 1.5)
    m = (d < t) & inside
    m = ndimage.binary_opening(m, iterations=int(1.0 / H.res))
    lab, n = ndimage.label(m)
    if n == 0 or not (0 <= cr < hh and 0 <= cc < ww):
        return None, 0, None
    k = lab[cr, cc]
    if not k:
        dd, ix = ndimage.distance_transform_edt(lab == 0, return_indices=True)
        if dd[cr, cc] * H.res > 4:
            return None, 0, None
        k = lab[ix[0][cr, cc], ix[1][cr, cc]]
    m = ndimage.binary_fill_holes(lab == k)
    area = m.sum() * H.res ** 2
    if not area_rng[0] <= area <= area_rng[1]:
        return None, area, None
    m = ndimage.gaussian_filter(m.astype(np.float32), 1.0 / H.res) > 0.5
    return m, area, win


def tune_grow(H, gmodel, owner_greens, clicks, log=print):
    best, best_sc = 3.0, -1
    for t in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        sc = []
        for key, ring in owner_greens.items():
            if key not in clicks:
                continue
            m, _, win = grow_green(H, *clicks[key], gmodel, t)
            sc.append(iou(H, win, m, ring) if m is not None else 0)
        v = float(np.mean(sc)) if sc else 0
        if v > best_sc:
            best, best_sc = t, v
    log(f"  green colour growth tuned: threshold {best}, mean IoU {best_sc:.2f}")
    return best, best_sc


def _contrast(H, win, m, sigma):
    from skimage.filters import sobel
    br = win["img"][..., :3].mean(-1)
    edge = sobel(ndimage.gaussian_filter(br, sigma / H.res))
    ring = m ^ ndimage.binary_erosion(m, iterations=2)
    inner = ndimage.binary_erosion(m, iterations=int(2 / H.res))
    return edge[ring].mean() / max(edge[inner].mean() if inner.any() else 1e-3, 1e-3)


def trace_green_combo(H, x, z, p, bunker_model, gmodel, gthr, ws_iou, grow_iou, median_area=400.0):
    """Candidates: watershed (several radii) + colour growth at several thresholds.
    Pick by mowing-line contrast x a size prior centred on the owner's median green
    (rule chosen by leave-one-out scoring against the owner's greens)."""
    cands = []
    m, a, w = trace_green_best(H, x, z, p, bunker_model)
    if m is not None:
        cands.append((m, a, w))
    for t in sorted({gthr, 2.0, 2.5, 3.0, 3.5}):
        m, a, w = grow_green(H, x, z, gmodel, t)
        if m is not None:
            cands.append((m, a, w))
    if not cands:
        return None, 0, None
    scored = [(_contrast(H, w, m, p["sigma"]) * math.exp(-abs(math.log(max(a, 1) / median_area))), m, a, w)
              for m, a, w in cands]
    _, m, a, w = max(scored, key=lambda t: t[0])
    return m, a, w


def trace_green_best(H, x, z, p, bunker_model=None, radii=(20.0, 24.0, 30.0, 36.0)):
    """Try several search radii; keep the outline sitting on the strongest mowing-line edge."""
    from skimage.filters import sobel
    best, best_score = (None, 0, None), -1
    for r in radii:
        m, area, win = trace_green(H, x, z, dict(p, radius=r), bunker_model)
        if m is None:
            continue
        br = win["img"][..., :3].mean(-1)
        edge = sobel(ndimage.gaussian_filter(br, p["sigma"] / H.res))
        ring = m ^ ndimage.binary_erosion(m, iterations=2)
        inner = ndimage.binary_erosion(m, iterations=int(2 / H.res))
        score = edge[ring].mean() / max(edge[inner].mean() if inner.any() else 1e-3, 1e-3)
        score *= min(1.0, area / 300.0)                 # don't reward tiny inner rings
        if score > best_score:
            best, best_score = (m, area, win), score
    return best


def mask_to_ring(H, win, m, n_vertices=24):
    from skimage.measure import find_contours
    from .learn import chaikin, resample_ring
    cs = find_contours(np.pad(m.astype(float), 1), 0.5)
    if not cs:
        return None
    cont = max(cs, key=len)
    pts = np.array([H.to_xz(win, r - 1, c - 1) for r, c in cont])
    pts = resample_ring(chaikin(resample_ring(pts, max(3 * n_vertices, 30)), 2), n_vertices)
    return [[round(float(a), 1), round(float(b), 1)] for a, b in pts]


def iou(H, win, m, ring):
    t = raster_ring(m.shape, H, win, ring)
    return float((m & t).sum() / max((m | t).sum(), 1))


GREEN_DEFAULTS = dict(radius=32.0, sigma=0.9, edge_mix=0.5, min_area=150.0, max_area=1400.0)


def tune_greens(H, owner_greens, clicks, bunker_model=None, log=print):
    """Grid-search watershed parameters against the owner's greens; returns best params + score."""
    grid = dict(radius=[24.0, 30.0, 36.0], sigma=[0.6, 0.9, 1.4], edge_mix=[0.0, 0.5, 1.0])
    best, best_score = dict(GREEN_DEFAULTS), -1
    for vals in itertools.product(*grid.values()):
        p = dict(GREEN_DEFAULTS, **dict(zip(grid.keys(), vals)))
        scores = []
        for key, ring in owner_greens.items():
            if key not in clicks:
                continue
            m, _, win = trace_green(H, *clicks[key], p, bunker_model)
            scores.append(iou(H, win, m, ring) if m is not None else 0.0)
        sc = float(np.mean(scores)) if scores else 0
        if sc > best_score:
            best, best_score = p, sc
    log(f"  green tracing tuned on {len(owner_greens)} owner greens: mean IoU {best_score:.2f} "
        f"(radius {best['radius']:.0f} m, sigma {best['sigma']}, nir mix {best['edge_mix']})")
    return best, best_score


def find_bunkers(H, near_x, near_z, model, thr, radius=45.0, exclude=None, bowl_max=0.02):
    """Bunkers around a green/fairway point: tan colour model + LiDAR depression."""
    win = H.window(near_x, near_z, radius)
    img = win["img"]
    br = img[..., :3].mean(-1)
    ndvi = (img[..., 3] - img[..., 0]) / (img[..., 3] + img[..., 0] + 1e-3)
    bright = (br > np.median(br) + 40) & (ndvi < 0.15)        # white/grey sand the owner hasn't drawn yet
    sand = ((model.dist(img) < thr) | bright) & (win["chm"] < 1.0)
    sand = ndimage.binary_opening(sand, iterations=2)
    sand = ndimage.binary_fill_holes(ndimage.binary_closing(sand, iterations=3))
    lab, n = ndimage.label(sand)
    out = []
    for j, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        m = lab[sl] == j + 1
        area = m.sum() * H.res ** 2
        if not 12 <= area <= 900:
            continue
        bowl = float(np.median(win["detr"][sl][m]))
        rr, cc = np.nonzero(m)
        ev = np.linalg.eigvalsh(np.cov(np.vstack([rr, cc]))) if len(rr) > 3 else [1, 1]
        if ev[0] / max(ev[1], 1e-6) < 0.09:          # sliver (path edge)
            continue
        if bowl > bowl_max:                          # bunkers sit lower than their surround
            continue
        perim = (m ^ ndimage.binary_erosion(m)).sum() * H.res
        if 2 * area / max(perim, 1e-3) < 3.5:       # mean width < 3.5 m: cart path / edge strip
            continue
        px = img[sl][m]
        if ((px[:, 2] - px[:, 0]) > 15).mean() > 0.1:   # any blue water in it: swimming pool area
            continue
        full = np.zeros(img.shape[:2], bool)
        full[sl] = m
        full = ndimage.gaussian_filter(full.astype(np.float32), 0.7 / H.res) > 0.5
        ring = mask_to_ring(H, win, full, 20 if area < 300 else 28)
        if ring and (exclude is None or not exclude(ring)):
            out.append({"ring": ring, "area": area, "bowl": bowl})
    return out


def tune_bunkers(H, model, owner_bunkers, green_points, log=print):
    """Pick the colour threshold that best reproduces the owner's bunkers near their greens."""
    best, best_f = (3.0, 0.02), -1
    for thr, bowl_max in itertools.product((1.5, 2.0, 2.5, 3.0, 4.0), (0.02, -0.02, -0.05, -0.08)):
        tp = fp = fn = 0
        for gx, gz in green_points:
            found = find_bunkers(H, gx, gz, model, thr, bowl_max=bowl_max)
            truth = [b for b in owner_bunkers if math.dist(np.mean(b["ring"], axis=0), (gx, gz)) < 45]
            used = set()
            for f in found:
                c = np.mean(f["ring"], axis=0)
                hit = next((i for i, b in enumerate(truth) if i not in used and
                            math.dist(c, np.mean(b["ring"], axis=0)) < 8), None)
                if hit is None:
                    fp += 1
                else:
                    tp += 1; used.add(hit)
            fn += len(truth) - len(used)
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        if f1 > best_f:
            best, best_f, stats = (thr, bowl_max), f1, (prec, rec)
    log(f"  bunker colour model tuned on {len(owner_bunkers)} owner bunkers: threshold/depth {best}, "
        f"precision {stats[0]:.2f}, recall {stats[1]:.2f}")
    return best, best_f
