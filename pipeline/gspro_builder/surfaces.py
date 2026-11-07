"""Rasterise the layout into Unity terrain layers (splat / control maps).

Mown-grass transitions are modelled the way courses are actually cut:
  fairway -> first cut (intermediate rough band) -> rough -> trees/native
  green   -> fringe/collar band -> (first cut / fairway / rough)
Band widths are in metres ([surfaces] in course.toml). Mown edges are blended
crisply, rough-to-native edges softly.

Outputs (north-up PNGs, splat_res px):
  surface_labels.png     8-bit index into TERRAIN_LAYERS
  splat_control{k}.png   RGBA weights for layers 4k..4k+3
  surface_preview.png    colour-coded preview
  surfaces.json          layer table (index, name, control map, channel, area)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .geo import CourseFrame
from .layout import SURFACE_ORDER, Layout

TERRAIN_LAYERS = ["rough", "trees", "first_cut", "fairway", "fringe", "tee", "green",
                  "bunker", "water", "cartpath"]

PREVIEW_COLORS = {
    "rough": (39, 132, 56), "trees": (22, 75, 32), "first_cut": (52, 170, 74),
    "fairway": (67, 229, 97), "fringe": (120, 214, 120), "tee": (160, 229, 184),
    "green": (188, 229, 164), "bunker": (229, 229, 170), "water": (0, 0, 192),
    "cartpath": (128, 128, 128),
}
# Blend sigma in metres: crisp for mown edges and hazards, soft for rough/native.
BLEND_M = {"rough": 1.2, "trees": 2.0, "first_cut": 0.3, "fairway": 0.25, "fringe": 0.2,
           "tee": 0.2, "green": 0.15, "bunker": 0.15, "water": 0.2, "cartpath": 0.15}
PATH_WIDTH_M = {"cartpath": 2.5}


@dataclass
class SurfaceConfig:
    fringe_width_m: float = 1.0
    first_cut_width_m: float = 2.0
    blend: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "SurfaceConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


L = {k: i for i, k in enumerate(TERRAIN_LAYERS)}


def rasterize_labels(layout: Layout, frame: CourseFrame, res: int, cfg: SurfaceConfig | None = None) -> np.ndarray:
    """Label raster (indices into TERRAIN_LAYERS), with fringe and first-cut bands."""
    from scipy import ndimage
    cfg = cfg or SurfaceConfig()
    img = Image.new("L", (res, res), L["rough"])
    draw = ImageDraw.Draw(img)
    px = lambda p: frame.local_to_pixel(p[0], p[1], res)  # noqa: E731
    for kind in SURFACE_ORDER:
        for s in (s for s in layout.surfaces if s.kind == kind):
            draw.polygon([px(p) for p in s.rings[0]], fill=L[kind])
            for inner in s.rings[1:]:
                draw.polygon([px(p) for p in inner], fill=L["rough"])
        for pkind, line in layout.paths:
            if pkind == kind and len(line) > 1:
                w = max(1, int(round(PATH_WIDTH_M.get(kind, 2.0) / frame.size_m * res)))
                draw.line([px(p) for p in line], fill=L[kind], width=w, joint="curve")
    lab = np.array(img, dtype=np.uint8)
    cell = frame.size_m / res

    def band(src: str, width_m: float, replaceable: tuple[str, ...], new: str) -> None:
        if width_m <= 0:
            return
        d = ndimage.distance_transform_edt(lab != L[src]) * cell
        repl = np.isin(lab, [L[k] for k in replaceable])
        lab[(d > 0) & (d <= width_m) & repl] = L[new]

    band("green", cfg.fringe_width_m, ("rough", "trees", "first_cut", "fairway"), "fringe")
    band("fairway", cfg.first_cut_width_m, ("rough", "trees"), "first_cut")
    return lab


def write_surfaces(layout: Layout, frame: CourseFrame, res: int, out_dir: Path,
                   cfg: SurfaceConfig | None = None) -> dict:
    from scipy import ndimage
    cfg = cfg or SurfaceConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = rasterize_labels(layout, frame, res, cfg)
    Image.fromarray(labels).save(out_dir / "surface_labels.png")
    cell = frame.size_m / res

    w = np.zeros((len(TERRAIN_LAYERS), res, res), np.float32)
    for i, k in enumerate(TERRAIN_LAYERS):
        ch = (labels == i).astype(np.float32)
        if cfg.blend and ch.any():
            ch = ndimage.gaussian_filter(ch, BLEND_M[k] / cell)
        w[i] = ch
    w /= np.maximum(w.sum(0, keepdims=True), 1e-6)
    w8 = np.round(w * 255).astype(np.uint8)
    n_maps = -(-len(TERRAIN_LAYERS) // 4)
    for k in range(n_maps):
        chans = np.zeros((4, res, res), np.uint8)
        part = w8[4 * k:4 * k + 4]
        chans[:len(part)] = part
        Image.fromarray(np.moveaxis(chans, 0, -1), "RGBA").save(out_dir / f"splat_control{k}.png")

    pal = np.array([PREVIEW_COLORS[k] for k in TERRAIN_LAYERS], dtype=np.uint8)
    Image.fromarray(pal[labels]).save(out_dir / "surface_preview.png")

    counts = np.bincount(labels.ravel(), minlength=len(TERRAIN_LAYERS))
    meta = {
        "resolution": res, "size_m": frame.size_m, "row_order": "north_to_south",
        "control_maps": n_maps,
        "layers": [{"index": i, "name": k, "control_map": f"splat_control{i // 4}.png",
                    "channel": "RGBA"[i % 4], "area_m2": round(float(counts[i]) * cell * cell, 1)}
                   for i, k in enumerate(TERRAIN_LAYERS)],
    }
    (out_dir / "surfaces.json").write_text(json.dumps(meta, indent=2))
    return meta
