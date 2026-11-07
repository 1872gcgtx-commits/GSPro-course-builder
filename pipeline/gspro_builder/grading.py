"""Course grading: turn raw LiDAR bare-earth into playable golf terrain.

Raw LiDAR is accurate to ~5-10 cm but noisy and blind under water. Each feature
type gets the treatment a real course architect/superintendent would expect:

  fairways  light de-noise (keeps the real roll), feathered into rough
  greens    de-noise that keeps contours; slope QA (putting surfaces > 3 % are flagged)
  tees      levelled pad: best-fit plane, cross-slope clamped to <= 1.5 %, optional raise
  bunkers   if LiDAR already shows the bowl it is kept (smoothed); otherwise carved to
            depth with a ramped face, plus a lip on the side facing the green
  water     flat water level from the shoreline, bed sloping to depth, banks
            raised so water cannot "leak" outside the outline

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

    @classmethod
    def from_dict(cls, d: dict) -> "GradingConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class GradingResult:
    dem: DEM
    water_bodies: list[dict] = field(default_factory=list)   # {surface, level_m (absolute)}
    report: dict = field(default_factory=dict)


def _gauss(a: np.ndarray, sigma_m: float, res: float) -> np.ndarray:
    from scipy import ndimage
    return ndimage.gaussian_filter(a, max(sigma_m / res, 1e-3), mode="nearest")


def _smooth_kind(a, inst, sigma_m, feather_m, res):
    mask = inst > 0
    if not mask.any():
        return a
    w = feather_weight(mask, feather_m / res)
    return a * (1 - w) + _gauss(a, sigma_m, res) * w


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

    # 4. water: flat level, sloped bed, raised banks
    water_bodies = []
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
        level = float(np.percentile(y[shore], 10)) - cfg.water_freeboard_m
        bed = level - np.minimum(cfg.water_depth_m, 0.3 + d_in * cfg.water_bed_slope)
        y_new = np.where(m, np.minimum(y, bed), y)
        bank = (~m) & (d_out <= cfg.water_bank_m)
        bank_min = level + 0.1 + 0.15 * (d_out / cfg.water_bank_m)
        y_new = np.where(bank, np.maximum(y_new, bank_min), y_new)
        a[win] = y_new
        water_bodies.append({"surface": s, "level_m": level})
        rep["water"].append({"hole": s.hole, "level_m": round(level, 2),
                             "area_m2": round(float(m.sum()) * res * res)})

    graded = DEM(a.astype(np.float32), frame)

    # 5. green slope QA (on the graded surface)
    slope = slope_grid(graded.data.astype(np.float64), res)
    for i, s in enumerate(by_kind["green"]):
        m = inst["green"] == i + 1
        if not m.any():
            continue
        sv = slope[m]
        rep["greens"].append({"hole": s.hole, "area_m2": round(float(m.sum()) * res * res),
                              "mean_slope_pct": round(float(sv.mean()) * 100, 2),
                              "max_slope_pct": round(float(np.percentile(sv, 98)) * 100, 2),
                              "pct_area_over_limit": round(float((sv > cfg.green_slope_warn).mean()) * 100, 1)})
    rep["max_change_m"] = round(float(np.abs(graded.data - dem.data).max()), 2)
    return GradingResult(graded, water_bodies, rep)
