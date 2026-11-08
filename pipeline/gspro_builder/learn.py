"""Draft v3: learn from the owner's finished holes, apply to the rest.

The owner's reviewed shapes (e.g. all of Creek 1) become the training labels:
  fairway / green / bunker / water polygons as drawn, "rough" = everything else
  inside those holes' corridors that isn't tree canopy, trees from LiDAR.
A random forest on 4-band NAIP + LiDAR features then labels the unreviewed
holes, so "tan = bunker", fairway mowing lines, green colour etc. are learned
from this course as the owner sees it. Shapes are post-processed to the owner's
style: smooth, rounded bunkers and greens (~20 vertices), one continuous smooth
fairway per hole, rectangular tee boxes.

Creeks come from the USGS National Hydrography Dataset (NHD) flowline, snapped
to the LiDAR channel bottom (thalweg), drawn as a narrow low-flow stream whose
water surface steps smoothly downhill.
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy import ndimage

from .autodraft import _along, _rect, _solidity, _strip
from .classify import (Grid, _fairway_continuous, _hole_partition, _line_mask, _smooth_mask, _sub_path,
                       build_features, refine_paths)

L3 = ["fairway", "green", "bunker", "water", "rough", "trees", "other"]


# --------------------------------------------------------------- geometry helpers
def chaikin(ring, iters=3):
    pts = np.asarray(ring, float)
    for _ in range(iters):
        q = 0.75 * pts + 0.25 * np.roll(pts, -1, 0)
        r = 0.25 * pts + 0.75 * np.roll(pts, -1, 0)
        pts = np.empty((len(pts) * 2, 2))
        pts[0::2], pts[1::2] = q, r
    return pts


def resample_ring(ring, n):
    pts = np.asarray(ring, float)
    seg = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n, endpoint=False)
    x = np.interp(t, cum, np.concatenate([pts[:, 0], pts[:1, 0]]))
    z = np.interp(t, cum, np.concatenate([pts[:, 1], pts[:1, 1]]))
    return np.column_stack([x, z])


def styled_ring(mask, G, n_vertices, smooth_px):
    """Mask -> smooth rounded polygon with ~n vertices (owner's style)."""
    from skimage.measure import find_contours
    if not mask.any():
        return None
    m = ndimage.gaussian_filter(mask.astype(np.float32), smooth_px)
    rr, cc = np.nonzero(m > 0.05)
    r0, r1 = max(rr.min() - 2, 0), rr.max() + 3
    c0, c1 = max(cc.min() - 2, 0), cc.max() + 3
    sub = np.pad(m[r0:r1, c0:c1], 1)
    cs = find_contours(sub, 0.5)
    if not cs:
        return None
    cont = max(cs, key=len)
    pts = np.array([G.xz(r0 + p[0] - 1, c0 + p[1] - 1) for p in cont])
    pts = resample_ring(pts, max(n_vertices * 3, 24))
    pts = chaikin(pts, 2)
    pts = resample_ring(pts, n_vertices)
    return [(round(float(x), 1), round(float(z), 1)) for x, z in pts]


def rasterize(G, rings):
    from PIL import Image, ImageDraw
    img = Image.new("L", (G.shape[1], G.shape[0]), 0)
    dr = ImageDraw.Draw(img)
    for ring in rings:
        dr.polygon([(G.rc(x, z)[1], G.rc(x, z)[0]) for x, z in ring], fill=1)
    return np.asarray(img, bool)


# --------------------------------------------------------------- creek from NHD
def nhd_creek(cfg, G, aux, names=("Salado Creek",), width_m=6.0, depth_m=0.3, log=print):
    """NHD flowline -> LiDAR thalweg -> narrow stream polygon + water-surface profile."""
    from skimage.graph import route_through_array
    nhd = json.loads((cfg.data_dir / "nhd.json").read_text())
    f = cfg.frame
    lines = []
    for ft in nhd["flowlines"]["features"]:
        p = ft["properties"]
        if (p.get("gnis_name") or p.get("GNIS_NAME")) not in names:
            continue
        geom = ft["geometry"]
        parts = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
        for part in parts:
            lines.append([f.to_local(lon, lat) for lon, lat in part])
    if not lines:
        return []
    # corridor around the NHD line(s); thalweg = least-cost path through the lowest ground
    corr = np.zeros(G.shape, bool)
    for ln in lines:
        corr |= _line_mask(G, ln, 30.0)
    dem = aux["dem"]
    low = dem - ndimage.minimum_filter(dem, size=int(40 / G.res))
    cost = np.where(corr, 1.0 + 25.0 * np.clip(low, 0, 3), 1e5).astype(np.float64)
    # endpoints: where the NHD lines leave the grid / are furthest apart
    pts = np.concatenate([np.asarray(ln) for ln in lines])
    inside = [(x, z) for x, z in pts if 0 <= G.rc(x, z)[0] < G.shape[0] and 0 <= G.rc(x, z)[1] < G.shape[1]]
    if len(inside) < 2:
        return []
    inside = np.array(inside)
    d = ((inside[:, None, :] - inside[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    a, b = G.rc(*inside[i]), G.rc(*inside[j])
    step = 2
    idx, _ = route_through_array(cost[::step, ::step], (a[0] // step, a[1] // step), (b[0] // step, b[1] // step),
                                 fully_connected=True, geometric=True)
    path = np.array([G.xz(r * step, c * step) for r, c in idx])
    # smooth the planform, resample every 2 m
    path = np.column_stack([ndimage.gaussian_filter1d(path[:, 0], 3, mode="nearest"),
                            ndimage.gaussian_filter1d(path[:, 1], 3, mode="nearest")])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, cum[-1], 2.0)
    path = np.column_stack([np.interp(t, cum, path[:, 0]), np.interp(t, cum, path[:, 1])])
    # bed elevation along the thalweg, flowing downhill
    bed = np.array([ndimage.minimum_filter(dem, size=7)[G.rc(x, z)] for x, z in path])
    if bed[0] < bed[-1]:
        path, bed = path[::-1], bed[::-1]
    bed = ndimage.gaussian_filter1d(bed, 6, mode="nearest")
    bed = np.minimum.accumulate(bed)                       # never flows uphill
    surface = bed + depth_m
    # narrow low-flow channel: buffer the thalweg, gently varying width
    tang = np.gradient(path, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    rng = np.random.default_rng(5)
    w = width_m / 2 * (1 + 0.25 * ndimage.gaussian_filter1d(rng.standard_normal(len(path)), 8))
    left, right = path + nrm * w[:, None], path - nrm * w[:, None]
    ring = np.concatenate([left, right[::-1]])
    ring = resample_ring(chaikin(ring, 1), min(400, max(40, len(path))))
    log(f"  NHD creek: {cum[-1]:.0f} m of {', '.join(names)}, water surface {surface[0]:.1f} -> {surface[-1]:.1f} m")
    return [{"kind": "water", "ring": [(round(float(x), 1), round(float(z), 1)) for x, z in ring],
             "method": "nhd", "confidence": 0.9, "stream": True,
             "profile": [[round(float(x), 1), round(float(z), 1), round(float(s), 2)] for (x, z), s in
                         zip(path[::5], surface[::5])]}]


# --------------------------------------------------------------- learn from reviewed holes
def train_from_owner(G, feats, aux, holes_train, shapes_train, log=print, seed=11):
    from sklearn.ensemble import RandomForestClassifier
    lab = np.full(G.shape, -1, np.int8)
    corridor = np.zeros(G.shape, bool)
    for h in holes_train:
        corridor |= _line_mask(G, _sub_path(h.path, 0, 1, 2.0), 55.0)
    by = {k: [s["ring"] for s in shapes_train if s["kind"] == k] for k in ("fairway", "green", "bunker", "water")}
    masks = {k: rasterize(G, v) if v else np.zeros(G.shape, bool) for k, v in by.items()}
    trees = aux["chm"] > 3
    drawn = np.zeros(G.shape, bool)
    for m in masks.values():
        drawn |= ndimage.binary_dilation(m, iterations=3)       # keep labels off the drawn edges
    lab[corridor & ~drawn & ~trees & (aux["chm"] < 0.8)] = L3.index("rough")
    lab[trees] = L3.index("trees")
    lab[aux["lb"]] = L3.index("other")
    for k in ("fairway", "green", "bunker", "water"):
        lab[ndimage.binary_erosion(masks[k], iterations=1)] = L3.index(k)
    rng = np.random.default_rng(seed)
    X, y = [], []
    flat = feats.reshape(-1, feats.shape[-1])
    for ci, name in enumerate(L3):
        idx = np.flatnonzero(lab.ravel() == ci)
        if len(idx) == 0:
            continue
        take = rng.choice(idx, min(len(idx), 30000), replace=False)
        X.append(flat[take]); y.append(np.full(len(take), ci))
        log(f"  owner labels {name:8s}: {len(idx):9,d} px")
    rf = RandomForestClassifier(n_estimators=80, max_depth=20, min_samples_leaf=2, n_jobs=-1,
                                class_weight="balanced", random_state=seed)
    rf.fit(np.concatenate(X), np.concatenate(y))
    return rf


def predict(rf, G, feats, region):
    H, W = G.shape
    prob = np.zeros((H * W, len(L3)), np.float32)
    prob[:, L3.index("other")] = 1
    idx = np.flatnonzero(region.ravel())
    flat = feats.reshape(-1, feats.shape[-1])
    for s in range(0, len(idx), 2_000_000):
        part = idx[s:s + 2_000_000]
        pp = rf.predict_proba(flat[part])
        prob[part] = 0
        for j, c in enumerate(rf.classes_):
            prob[part, c] = pp[:, j]
    prob = prob.reshape(H, W, len(L3))
    for j in range(len(L3)):
        prob[..., j] = ndimage.gaussian_filter(prob[..., j], 0.8)
    return prob


def draft_v3(cfg, holes, reviewed_shapes, train_nines=(1,), log=print):
    import copy
    G, feats, aux = build_features(cfg, log)
    tr_holes = [h for h in holes if int(h.props.get("nine", 0) or 0) in train_nines]
    todo = [h for h in holes if int(h.props.get("nine", 0) or 0) not in train_nines]
    tr_shapes = [s for s in reviewed_shapes if int(s.get("nine", 0)) in train_nines and s["kind"] != "tee"]
    rf = train_from_owner(G, feats, aux, tr_holes, tr_shapes, log)

    region = np.zeros(G.shape, bool)
    for h in todo:
        region |= _line_mask(G, _sub_path(h.path, 0, 1, 2.0), 110.0)
    prob = predict(rf, G, feats, region)
    I = {c: i for i, c in enumerate(L3)}
    cls = np.argmax(prob, -1)

    # dog-legs on the learned map (fairway turf cheapest)
    pseudo = np.zeros(prob.shape[:2] + (6,), np.float32)       # CLASSES order of classify.py
    pseudo[..., 0] = prob[..., I["fairway"]] + prob[..., I["green"]]
    pseudo[..., 1] = prob[..., I["rough"]]
    pseudo[..., 2] = prob[..., I["trees"]]
    pseudo[..., 3] = prob[..., I["bunker"]]
    pseudo[..., 4] = prob[..., I["water"]]
    pseudo[..., 5] = prob[..., I["other"]]
    paths = refine_paths(G, pseudo, aux, todo, log)
    todo = [copy.copy(h) for h in todo]
    for h in todo:
        if len(h.path) == 2:
            h.path = paths[h.number]
    part = _hole_partition(G, todo)

    out = []
    bunker_p, fair_p, green_p = prob[..., I["bunker"]], prob[..., I["fairway"]], prob[..., I["green"]]
    water_m = (cls == I["water"]) | ndimage.binary_opening(aux["lw"], iterations=2)
    for hi, h in enumerate(todo):
        nine, nh = int(h.props.get("nine", 0) or 0), int(h.props.get("nine_hole", 0) or 0)
        tee, green = h.path[0], h.path[-1]
        mine = part == hi

        def add(kind, ring, method, conf, **kw):
            out.append({"kind": kind, "ring": ring, "method": method, "confidence": conf,
                        "hole": h.number, "nine": nine, "nine_hole": nh, **kw})

        # green: learned green turf around the click, rounded
        gr, gc = G.rc(*green)
        gm = (green_p > 0.45) & mine
        gm = ndimage.binary_opening(gm, iterations=2)
        lab, _ = ndimage.label(gm)
        k = lab[gr, gc] if 0 <= gr < G.shape[0] and 0 <= gc < G.shape[1] else 0
        if not k and lab.any():
            dd, ix = ndimage.distance_transform_edt(lab == 0, return_indices=True)
            if dd[gr, gc] * G.res <= 8:
                k = lab[ix[0][gr, gc], ix[1][gr, gc]]
        gmask = np.zeros(G.shape, bool)
        if k:
            gmask = ndimage.binary_fill_holes(lab == k)
            area = gmask.sum() * G.res ** 2
            if not 150 <= area <= 1400:
                gmask[:] = False
        if gmask.any():
            add("green", styled_ring(gmask, G, 20, 3.0), "learned", 0.8)
        else:
            from .autodraft import _ellipse
            prev = h.path[-2]
            ang = math.atan2(green[1] - prev[1], green[0] - prev[0])
            ring = _ellipse(*green, 12, 9.5, ang)[:-1]
            add("green", [(round(x, 1), round(z, 1)) for x, z in ring], "fallback", 0.3)
            gmask = rasterize(G, [ring])

        # tee: owner style = small rectangle along the line of play at the back-tee click
        ang = math.atan2(h.path[1][1] - tee[1], h.path[1][0] - tee[0])
        ring = _rect(*tee, 14, 7, ang)[:-1]
        add("tee", [(round(x, 1), round(z, 1)) for x, z in ring], "learned", 0.6, tee_set="Blue")
        tmask = rasterize(G, [ring])

        # fairway: one continuous smooth shape on learned fairway turf
        fwmask = np.zeros(G.shape, bool)
        if (h.par or 4) >= 4:
            ok = (fair_p > 0.45) & mine & ~ndimage.binary_dilation(tmask, iterations=2)
            hazard = ndimage.binary_dilation(water_m, iterations=6)
            m = _fairway_continuous(G, h.path, ok, gmask, hazard=hazard)
            if m is not None and m.sum() * G.res ** 2 >= 800:
                add("fairway", styled_ring(m, G, 40, 4.0), "learned", 0.8)
                fwmask = m
            else:
                a, b = _along(h.path, 0.42), _along(h.path, 0.92)
                add("fairway", [(round(x, 1), round(z, 1)) for x, z in _strip(a, b, 30)[:-1]], "fallback", 0.2)
                fwmask = _line_mask(G, _sub_path(h.path, 0.42, 0.92), 15.0)

        # bunkers: learned tan sand near this hole's fairway/green, smooth & rounded
        near = (ndimage.distance_transform_edt(~fwmask) * G.res < 40) | (ndimage.distance_transform_edt(~gmask) * G.res < 35)
        bm = (bunker_p > 0.5) & mine & near & ~ndimage.binary_dilation(aux["lb"], iterations=4)
        bm = ndimage.binary_fill_holes(ndimage.binary_closing(ndimage.binary_opening(bm), iterations=2))
        lab, _ = ndimage.label(bm)
        for j, sl in enumerate(ndimage.find_objects(lab)):
            if sl is None:
                continue
            m = lab[sl] == j + 1
            area = m.sum() * G.res ** 2
            if not 12 <= area <= 900 or _solidity(m) < 0.5:
                continue
            fm = np.zeros(G.shape, bool)
            fm[sl] = m
            ring = styled_ring(fm, G, 20 if area < 300 else 28, 1.6)
            if ring:
                add("bunker", ring, "learned", 0.8)

    # ponds (not the creek) in the unreviewed area, rounded
    lab, _ = ndimage.label(water_m & region)
    from .layout import point_polyline_dist
    for j, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        m = lab[sl] == j + 1
        area = m.sum() * G.res ** 2
        if area < 200:
            continue
        fm = np.zeros(G.shape, bool)
        fm[sl] = m
        if _solidity(m) < 0.45:      # long ribbons = the creek, handled from NHD
            continue
        ring = styled_ring(fm, G, 36, 2.0)
        if not ring:
            continue
        cx, cz = np.mean(ring, axis=0)
        best = min(todo, key=lambda hh: point_polyline_dist((cx, cz), hh.path))
        out.append({"kind": "water", "ring": ring, "method": "learned", "confidence": 0.8, "hole": best.number,
                    "nine": int(best.props.get("nine", 0) or 0), "nine_hole": int(best.props.get("nine_hole", 0) or 0)})

    # the creek from NHD (attach to the nearest hole overall, it's shared)
    for c in nhd_creek(cfg, G, aux, log=log):
        cx, cz = np.mean(c["ring"], axis=0)
        best = min(holes, key=lambda hh: point_polyline_dist((cx, cz), hh.path))
        c.update({"hole": best.number, "nine": int(best.props.get("nine", 0) or 0),
                  "nine_hole": int(best.props.get("nine_hole", 0) or 0)})
        out.append(c)
    draft_v3.paths = {f"n{int(h.props.get('nine', 0) or 0)}h{int(h.props.get('nine_hole', 0) or 0)}":
                      [[round(x, 1), round(z, 1)] for x, z in h.path] for h in todo}
    log(f"  drafted {len(out)} shapes")
    return out


def creek_banks(cfg, G, aux, profile, max_half_m=30.0, log=print):
    """Bank-following creek polygon from a thalweg + water-surface profile.

    Water = ground below the local water surface, connected to the thalweg, plus
    LiDAR water returns and open water seen in NIR, inside the creek corridor.
    A 4 m thread along the thalweg guarantees one continuous shape.
    """
    from scipy.spatial import cKDTree
    P = np.asarray(profile, float)                        # x, z, surface (every ~10 m)
    # densify the thalweg to 1 m
    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, cum[-1], 1.0)
    th = np.column_stack([np.interp(t, cum, P[:, 0]), np.interp(t, cum, P[:, 1]), np.interp(t, cum, P[:, 2])])
    corridor = _line_mask(G, th[:, :2].tolist(), max_half_m)
    rr, cc = np.nonzero(corridor)
    xs = -G.frame.size_m / 2 + (cc + G.c0 + 0.5) * G.res
    zs = G.frame.size_m / 2 - (rr + G.r0 + 0.5) * G.res
    _, j = cKDTree(th[:, :2]).query(np.column_stack([xs, zs]))
    level = np.full(G.shape, -1e9, np.float32)
    level[rr, cc] = th[j, 2] + 0.2
    wet = corridor & (aux["dem"] <= level)
    wet |= corridor & ndimage.binary_opening(aux["lw"], iterations=1)
    wet |= corridor & (aux["ndvi"] < -0.02) & (aux["bri"] < 90) & (aux["chm"] < 1)
    thread = _line_mask(G, th[:, :2].tolist(), 2.0)
    wet = ndimage.binary_closing(wet | thread, iterations=3)
    wet = ndimage.binary_opening(wet, iterations=2) | thread
    lab, n = ndimage.label(wet)
    keep = np.unique(lab[thread])
    wet = np.isin(lab, keep[keep > 0])
    wet = ndimage.binary_fill_holes(wet)
    wet = _smooth_mask(wet, 3.0) | thread
    ring = styled_ring(wet, G, min(600, max(80, int(cum[-1] / 6))), 2.0)
    area = wet.sum() * G.res ** 2
    log(f"  creek banks: {area:,.0f} m2 over {cum[-1]:.0f} m (mean width {area / max(cum[-1], 1):.1f} m)")
    return ring
