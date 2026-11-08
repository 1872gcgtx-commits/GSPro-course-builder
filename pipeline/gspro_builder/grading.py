"""Course grading: turn raw LiDAR bare-earth into playable golf terrain.

Raw LiDAR is accurate to ~5-10 cm but noisy and blind under water. Each feature
type gets the treatment a real course architect/superintendent would expect:

  fairways  light de-noise (keeps the real roll), feathered into rough
  greens    shaped putting surface: smooth fit to the LiDAR (keeps the real tilt, crown/bowl
            and softened broad undulations, drops LiDAR bumps), eased toward level until the
            steepest 2 % of the green is <= green_max_slope (4 %), blended out through the collar
  tees      levelled pad: best-fit plane, cross-slope clamped to <= 1.5 %, optional raise
  bunkers   if LiDAR already shows the bowl it is kept (smoothed); otherwise carved to
            depth with a ramped face, plus a lip on the side facing the green
  water     ponds/lakes: one flat level from the shoreline. Creeks/rivers (tagged
            river/stream, or long and narrow): the level follows the banks
            downstream. Bed slopes to depth; banks raised so water can't leak.

Every change is blended into the surrounding ground over a feather distance, so
no cliffs or seams appear at feature edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .layout import Layout, Surface
from .raster import (cell_centers, feather_weight, fit_plane, grow_window, rasterize_instances,
                     slope_grid, smoothstep)
from .terrain import DEM


@dataclass
class GradingConfig:
    enabled: bool = True
    fairway_smooth_m: float = 0.5
    fairway_feather_m: float = 3.0
    green_smooth_m: float = 0.6
    green_feather_m: float = 2.0
    green_slope_warn: float = 0.03
    green_shape: bool = True           # shaped putting surface (False = smoothing only)
    green_max_slope: float = 0.04      # steepest 2 % of a green after shaping
    green_undulation: float = 0.5      # share of broad (> ~6 m) undulation kept on top of the fit
    green_blend_m: float = 4.0         # change fades out over this distance beyond the green edge
    tee_max_slope: float = 0.015
    tee_raise_m: float = 0.0
    tee_feather_m: float = 3.0
    bunker_mode: str = "auto"          # auto | carve | lidar
    bunker_depth_m: float = 0.6
    bunker_ramp_m: float = 2.0
    bunker_lip_m: float = 0.25
    bunker_lip_width_m: float = 1.5
    water_freeboard_m: float = 0.3
    water_depth_m: float = 1.8
    water_bed_slope: float = 0.25
    water_bank_m: float = 3.0
    stream_elongation: float = 6.0     # length/width above which water is treated as flowing
    stream_level_smooth_m: float = 25.0

    @classmethod
    def from_dict(cls, d: dict) -> "GradingConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class GradingResult:
    dem: DEM
    water_bodies: list[dict] = field(default_factory=list)   # {surface, level_m, flowing}
    water_level: np.ndarray | None = None                     # absolute water surface (NaN = dry)
    report: dict = field(default_factory=dict)


def fit_plane_grad(a, m, res):
    """(d/dx, d/dy) of the best-fit plane through a over mask m."""
    yy, xx = np.nonzero(m)
    A = np.c_[xx * res, yy * res, np.ones(len(xx))]
    c, *_ = np.linalg.lstsq(A, a[m].astype(np.float64), rcond=None)
    return c[0], c[1]


def _gauss(a: np.ndarray, sigma_m: float, res: float) -> np.ndarray:
    from scipy import ndimage
    return ndimage.gaussian_filter(a, max(sigma_m / res, 1e-3), mode="nearest")


def _smooth_kind(a, inst, sigma_m, feather_m, res):
    mask = inst > 0
    if not mask.any():
        return a
    w = feather_weight(mask, feather_m / res)
    return a * (1 - w) + _gauss(a, sigma_m, res) * w


def _shape_greens(a, inst, cfg, res):
    """Shaped putting surfaces (see module doc). Returns the new grid and per-green info."""
    from scipy import ndimage
    info = {}
    n = a.shape[0]
    pad = int(np.ceil(cfg.green_blend_m / res)) + 4
    for i, sl in enumerate(ndimage.find_objects(inst), start=1):
        if sl is None:
            continue
        r0, r1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, n)
        c0, c1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, n)
        m = inst[r0:r1, c0:c1] == i
        if m.sum() < 20:
            continue
        z = a[r0:r1, c0:c1]
        yy, xx = np.nonzero(m)
        X, Y = xx * res, yy * res
        mx, my = X.mean(), Y.mean()
        X, Y = X - mx, Y - my
        A = np.c_[np.ones_like(X), X, Y, X * X, X * Y, Y * Y]
        keep = np.ones(len(X), bool)
        for _ in range(3):                      # robust: ignore odd spots (sprinkler boxes, trees)
            c, *_ = np.linalg.lstsq(A[keep], z[m][keep], rcond=None)
            r = z[m] - A @ c
            keep = np.abs(r) < 2.5 * max(r[keep].std(), 0.01)
        gy, gx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
        GX, GY = gx * res - mx, gy * res - my
        fit = c[0] + c[1] * GX + c[2] * GY + c[3] * GX * GX + c[4] * GX * GY + c[5] * GY * GY
        resid = np.where(m, z - fit, 0.0)
        w = ndimage.gaussian_filter(m.astype(float), 3.0 / res)
        broad = ndimage.gaussian_filter(resid, 3.0 / res) / np.maximum(w, 1e-6)
        shaped = fit + cfg.green_undulation * broad
        # ease toward level until the steepest 2 % of the putting surface is within the limit
        mean = shaped[m].mean()
        k = 1.0
        for _ in range(12):
            t = mean + k * (shaped - mean)
            ty, tx = np.gradient(t, res)
            p98 = float(np.percentile(np.hypot(tx, ty)[m], 98))
            if p98 <= cfg.green_max_slope:
                break
            k *= max(0.5, cfg.green_max_slope / p98 * 0.98)
        t = mean + k * (shaped - mean)
        # blend: the change inside the green carries out through the collar and fades
        delta_in = np.where(m, t - z, 0.0)
        d, (ir, ic) = ndimage.distance_transform_edt(~m, return_indices=True)
        delta = delta_in[ir, ic] * (1 - smoothstep(d * res / cfg.green_blend_m))
        a[r0:r1, c0:c1] = z + delta
        info[i] = {"eased": round(1 - k, 2), "fit_rms_cm": round(float(r[keep].std()) * 100, 1)}
    return a, info


def grade(dem: DEM, layout: Layout, cfg: GradingConfig) -> GradingResult:
    from scipy import ndimage
    if not cfg.enabled:
        return GradingResult(dem, [], {"enabled": False})
    frame, res = dem.frame, dem.res_m
    n = dem.data.shape[0]
    a = dem.data.astype(np.float64).copy()
    by_kind = {k: [s for s in layout.surfaces if s.kind == k] for k in
               ("fairway", "green", "tee", "bunker", "water")}
    inst = {k: rasterize_instances(v, frame, n) for k, v in by_kind.items()}
    rep: dict = {"tees": [], "bunkers": [], "water": [], "greens": []}

    # 1. fairways and greens: de-noise, feathered
    a = _smooth_kind(a, inst["fairway"], cfg.fairway_smooth_m, cfg.fairway_feather_m, res)
    a = _smooth_kind(a, inst["green"], cfg.green_smooth_m, cfg.green_feather_m, res)
    shaped = {}
    if cfg.green_shape:
        a, shaped = _shape_greens(a, inst["green"], cfg, res)

    # 2. tees: levelled pads
    pad = int(np.ceil(cfg.tee_feather_m / res)) + 2
    for i, sl in enumerate(ndimage.find_objects(inst["tee"])):
        if sl is None:
            continue
        win = grow_window(sl, pad, n)
        m = inst["tee"][win] == i + 1
        x, z = cell_centers(frame, n, *win)
        y = a[win]
        c0, cx, cz = fit_plane(x[m], z[m], y[m])
        before = float(np.hypot(cx, cz))
        if before > cfg.tee_max_slope:
            cx, cz = cx * cfg.tee_max_slope / before, cz * cfg.tee_max_slope / before
        mx, mz, my = x[m].mean(), z[m].mean(), y[m].mean()
        plane = my + cfg.tee_raise_m + cx * (x - mx) + cz * (z - mz)
        w = feather_weight(m, cfg.tee_feather_m / res)
        a[win] = y * (1 - w) + plane * w
        s = by_kind["tee"][i]
        rep["tees"].append({"hole": s.hole, "slope_before_pct": round(before * 100, 2),
                            "slope_after_pct": round(float(np.hypot(cx, cz)) * 100, 2)})

    # 3. bunkers
    green_centroids = {s.hole: s.centroid for s in by_kind["green"] if s.hole is not None}
    pad = int(np.ceil((cfg.bunker_lip_width_m + cfg.bunker_ramp_m + 2) / res)) + 2
    for i, sl in enumerate(ndimage.find_objects(inst["bunker"])):
        if sl is None:
            continue
        s = by_kind["bunker"][i]
        win = grow_window(sl, pad, n)
        m = inst["bunker"][win] == i + 1
        x, z = cell_centers(frame, n, *win)
        y = a[win]
        d_out = ndimage.distance_transform_edt(~m) * res
        d_in = ndimage.distance_transform_edt(m) * res
        ring = (d_out > 0) & (d_out <= 1.5)
        c0, cx, cz = fit_plane(x[ring], z[ring], y[ring])
        rim = c0 + cx * x + cz * z
        existing = float(np.median((rim - y)[m]))
        carve = cfg.bunker_mode == "carve" or (cfg.bunker_mode == "auto" and existing < 0.4 * cfg.bunker_depth_m)
        if carve:
            depth = cfg.bunker_depth_m * smoothstep(d_in / cfg.bunker_ramp_m)
            y_new = np.where(m, np.minimum(y, rim - depth), y)
            # lip, strongest on the face toward the green
            gc = green_centroids.get(s.hole)
            if gc is not None:
                bc = s.centroid
                dx, dz = gc[0] - bc[0], gc[1] - bc[1]
                L = max(np.hypot(dx, dz), 1e-6)
                face = np.clip(((x - bc[0]) * dx + (z - bc[1]) * dz) / (L * np.maximum(np.hypot(x - bc[0], z - bc[1]), 1e-6)), 0, 1)
            else:
                face = np.full_like(x, 0.5)
            lip = cfg.bunker_lip_m * face * (1 - smoothstep(d_out / cfg.bunker_lip_width_m)) * (~m)
            y_new = y_new + lip
        else:   # keep the real shape, just take out the noise on the sand
            sm = _gauss(y, 0.4, res)
            y_new = np.where(m, sm, y)
        a[win] = y_new
        rep["bunkers"].append({"hole": s.hole, "mode": "carved" if carve else "lidar",
                               "depth_m": round(float(np.max((rim - a[win])[m])), 2)})

    # 4. water: level surface, sloped bed, raised banks
    water_bodies = []
    water_level = np.full(a.shape, np.nan, np.float32)
    pad = int(np.ceil((cfg.water_bank_m + 3) / res)) + 2
    for i, sl in enumerate(ndimage.find_objects(inst["water"])):
        if sl is None:
            continue
        s = by_kind["water"][i]
        win = grow_window(sl, pad, n)
        m = inst["water"][win] == i + 1
        y = a[win]
        d_out = ndimage.distance_transform_edt(~m) * res
        d_in = ndimage.distance_transform_edt(m) * res
        shore = (d_out > 0) & (d_out <= 3.0)
        half_w = max(float(d_in.max()), res)
        length = float(m.sum()) * res * res / (2 * half_w)
        tag = str(s.props.get("water", "") or s.props.get("waterway", "")).lower()
        flowing = tag in ("river", "stream", "canal", "riverbank") or length / (2 * half_w) > cfg.stream_elongation
        prof = s.props.get("profile")
        if prof:
            # explicit water-surface profile (NHD thalweg): nearest station's level, smoothed
            from scipy.spatial import cKDTree
            P = np.asarray(prof, float)
            xw, zw = cell_centers(frame, n, *win)
            _, j = cKDTree(P[:, :2]).query(np.column_stack([xw.ravel(), zw.ravel()]))
            level_f = ndimage.gaussian_filter(P[j, 2].reshape(xw.shape), 4 / res)
            flowing = True
        elif flowing:
            # level follows the low bank nearby, smoothed along the channel
            idx = ndimage.distance_transform_edt(~shore, return_distances=False, return_indices=True)
            near = y[tuple(idx)]
            near = ndimage.minimum_filter(near, size=max(3, int(10 / res)))
            level_f = ndimage.gaussian_filter(near, cfg.stream_level_smooth_m / res, mode="nearest") - cfg.water_freeboard_m
        else:
            level_f = np.full_like(y, float(np.percentile(y[shore], 10)) - cfg.water_freeboard_m)
        bed = level_f - np.minimum(cfg.water_depth_m, 0.3 + d_in * cfg.water_bed_slope)
        y_new = np.where(m, np.minimum(y, bed), y)
        bank = (~m) & (d_out <= cfg.water_bank_m)
        bank_min = level_f + 0.1 + 0.15 * (d_out / cfg.water_bank_m)
        y_new = np.where(bank, np.maximum(y_new, bank_min), y_new)
        a[win] = y_new
        wl = water_level[win]
        wl[m] = level_f[m]
        level = float(np.mean(level_f[m]))
        water_bodies.append({"surface": s, "level_m": level, "flowing": flowing})
        rep["water"].append({"hole": s.hole, "type": "stream" if flowing else "pond",
                             "level_m": round(level, 2),
                             "level_range_m": round(float(np.ptp(level_f[m])), 2),
                             "max_cut_m": round(float(np.max((y - y_new)[m])), 2),
                             "area_m2": round(float(m.sum()) * res * res)})

    graded = DEM(a.astype(np.float32), frame)

    # 5. green slope QA (on the graded surface)
    slope = slope_grid(graded.data.astype(np.float64), res)
    for i, s in enumerate(by_kind["green"]):
        m = inst["green"] == i + 1
        if not m.any():
            continue
        sv = slope[m]
        tilt = np.hypot(*fit_plane_grad(graded.data, m, res))
        rep["greens"].append({"hole": s.hole, "area_m2": round(float(m.sum()) * res * res),
                              "tilt_pct": round(float(tilt) * 100, 2),
                              "eased_toward_level": shaped.get(i + 1, {}).get("eased"),
                              "mean_slope_pct": round(float(sv.mean()) * 100, 2),
                              "max_slope_pct": round(float(np.percentile(sv, 98)) * 100, 2),
                              "pct_area_over_limit": round(float((sv > cfg.green_slope_warn).mean()) * 100, 1)})
    rep["max_change_m"] = round(float(np.abs(graded.data - dem.data).max()), 2)
    return GradingResult(graded, water_bodies, water_level=water_level, report=rep)
