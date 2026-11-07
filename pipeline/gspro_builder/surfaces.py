"""Rasterise the layout into Unity terrain splat (control) maps.

Outputs (all north-up PNGs, square, splat_res px):
  surface_labels.png    8-bit index into SURFACE_ORDER (0 = rough)
  splat_control0.png    RGBA weights for layers 0-3 (rough, trees, fairway, tee)
  splat_control1.png    RGBA weights for layers 4-7 (green, bunker, water, cartpath)
  surface_preview.png   colour-coded preview
  surfaces.json         layer table
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .geo import CourseFrame
from .layout import SURFACE_ORDER, Layout

PREVIEW_COLORS = {
    "rough": (39, 132, 56), "trees": (22, 75, 32), "fairway": (67, 229, 97),
    "tee": (160, 229, 184), "green": (188, 229, 164), "bunker": (229, 229, 170),
    "water": (0, 0, 192), "cartpath": (128, 128, 128),
}
PATH_WIDTH_M = {"cartpath": 2.5}


def rasterize_labels(layout: Layout, frame: CourseFrame, res: int) -> np.ndarray:
    img = Image.new("L", (res, res), SURFACE_ORDER.index("rough"))
    draw = ImageDraw.Draw(img)
    px = lambda p: frame.local_to_pixel(p[0], p[1], res)  # noqa: E731
    rough_idx = SURFACE_ORDER.index("rough")
    for kind in SURFACE_ORDER:
        idx = SURFACE_ORDER.index(kind)
        for s in (s for s in layout.surfaces if s.kind == kind):
            draw.polygon([px(p) for p in s.rings[0]], fill=idx)
            for inner in s.rings[1:]:            # islands inside a surface revert to rough
                draw.polygon([px(p) for p in inner], fill=rough_idx)
        for pkind, line in layout.paths:
            if pkind == kind and len(line) > 1:
                w = max(1, int(round(PATH_WIDTH_M.get(kind, 2.0) / frame.size_m * res)))
                draw.line([px(p) for p in line], fill=idx, width=w, joint="curve")
    return np.asarray(img, dtype=np.uint8)


def write_surfaces(layout: Layout, frame: CourseFrame, res: int, out_dir: Path,
                   blend_px: float = 1.0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = rasterize_labels(layout, frame, res)
    Image.fromarray(labels).save(out_dir / "surface_labels.png")

    weights = []
    for i in range(len(SURFACE_ORDER)):
        ch = Image.fromarray(((labels == i) * 255).astype(np.uint8))
        if blend_px > 0:
            ch = ch.filter(ImageFilter.GaussianBlur(blend_px))
        weights.append(np.asarray(ch, dtype=np.float32))
    w = np.stack(weights)
    w /= np.maximum(w.sum(0, keepdims=True), 1e-6)
    w8 = np.round(w * 255).astype(np.uint8)
    for k in range(2):
        Image.fromarray(np.moveaxis(w8[4 * k:4 * k + 4], 0, -1), "RGBA").save(out_dir / f"splat_control{k}.png")

    pal = np.array([PREVIEW_COLORS[k] for k in SURFACE_ORDER], dtype=np.uint8)
    Image.fromarray(pal[labels]).save(out_dir / "surface_preview.png")

    counts = np.bincount(labels.ravel(), minlength=len(SURFACE_ORDER))
    cell_m2 = (frame.size_m / res) ** 2
    meta = {
        "resolution": res, "size_m": frame.size_m, "row_order": "north_to_south",
        "layers": [{"index": i, "name": k, "control_map": f"splat_control{i // 4}.png",
                    "channel": "RGBA"[i % 4], "area_m2": round(float(counts[i]) * cell_m2, 1)}
                   for i, k in enumerate(SURFACE_ORDER)],
    }
    (out_dir / "surfaces.json").write_text(json.dumps(meta, indent=2))
    return meta
