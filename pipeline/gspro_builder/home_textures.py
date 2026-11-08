"""Procedural textures for the homes (no external assets).

Walls tile every 3 m (one window bay per tile, one storey per tile height):
limestone ashlar, rustic stone, tan / red brick, stucco. Roofs tile every 4 m along
the eave: asphalt shingles (per roof colour) and standing-seam metal. Plus front door,
garage door, concrete (driveways), white trim and foundation stone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PX = 512                      # pixels per tile
M = PX / 3.0                  # pixels per metre on 3 m wall tiles


def _noise(size, scale, rng, amp=1.0):
    from scipy import ndimage
    a = ndimage.gaussian_filter(rng.normal(0, 1, (size, size)), scale)
    return (a - a.mean()) / (a.std() + 1e-9) * amp


def _jit(c, rng, s=10):
    return tuple(int(np.clip(v + rng.normal(0, s), 0, 255)) for v in c)


def _window(dr, rng, x0=None):
    """1.2 m x 1.5 m double-hung window, sill at 0.9 m, white frame, centred in the bay."""
    w, h, sill = 1.2 * M, 1.5 * M, 0.9 * M
    cx = PX / 2 if x0 is None else x0
    l, r, b, t = cx - w / 2, cx + w / 2, PX - sill, PX - sill - h
    dr.rectangle([l - 9, t - 9, r + 9, b + 12], fill=(236, 234, 228))          # trim + sill
    for k in range(2):                                                         # two sashes
        y0 = t + k * h / 2; y1 = y0 + h / 2
        for j in range(18):                                                    # sky reflection gradient
            yy = y0 + (y1 - y0) * j / 18
            g = int(70 + 60 * (1 - j / 18))
            dr.rectangle([l, yy, r, yy + (y1 - y0) / 18 + 1], fill=(g - 20, g, g + 18))
    dr.line([l, (t + b) / 2, r, (t + b) / 2], fill=(240, 240, 236), width=7)
    for f in (1 / 3, 2 / 3):                                                   # muntins
        dr.line([l + w * f, t, l + w * f, b], fill=(240, 240, 236), width=3)
    dr.line([l, t + h / 4, r, t + h / 4], fill=(240, 240, 236), width=3)
    dr.line([l, t + 3 * h / 4, r, t + 3 * h / 4], fill=(240, 240, 236), width=3)


def wall(kind: str, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (PX, PX)); dr = ImageDraw.Draw(im)
    if kind == "Home_Wall_Limestone":           # cream ashlar, random block lengths
        base, mortar = (214, 202, 176), (190, 180, 160)
        dr.rectangle([0, 0, PX, PX], fill=mortar)
        y, row = 0, 0
        while y < PX:
            hgt = int(rng.choice([0.2, 0.3, 0.3]) * M)
            x = -int(rng.uniform(0, 0.6) * M)
            while x < PX:
                L = int(rng.uniform(0.4, 1.1) * M)
                dr.rectangle([x + 2, y + 2, x + L - 2, y + hgt - 2], fill=_jit(base, rng, 9))
                x += L
            y += hgt; row += 1
    elif kind == "Home_Wall_Stone":             # rustic random stones
        dr.rectangle([0, 0, PX, PX], fill=(150, 140, 125))
        for _ in range(420):
            cx, cy = rng.uniform(0, PX), rng.uniform(0, PX)
            rx, ry = rng.uniform(0.12, 0.3) * M, rng.uniform(0.08, 0.18) * M
            c = _jit(rng.choice([(196, 176, 146), (176, 160, 136), (210, 196, 170), (160, 140, 118)]), rng, 8)
            dr.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=c)
    elif kind in ("Home_Wall_BrickTan", "Home_Wall_BrickRed"):
        base = (186, 150, 112) if kind.endswith("Tan") else (150, 78, 60)
        mortar = (205, 198, 186)
        dr.rectangle([0, 0, PX, PX], fill=mortar)
        bh, bl = 0.076 * M, 0.23 * M
        for r in range(int(PX / bh) + 1):
            off = (r % 2) * bl / 2
            x = -off
            while x < PX:
                dr.rectangle([x + 1, r * bh + 1, x + bl - 2, r * bh + bh - 2], fill=_jit(base, rng, 12))
                x += bl
    else:                                       # stucco
        a = 222 + _noise(PX, 3, rng, 6) + _noise(PX, 20, rng, 5)
        arr = np.stack([a, a - 3, a - 10], -1).clip(0, 255).astype(np.uint8)
        im = Image.fromarray(arr); dr = ImageDraw.Draw(im)
    _window(dr, rng)
    return im.filter(ImageFilter.SMOOTH)


ROOF_COLOURS = {"Home_Roof_Charcoal": (64, 64, 66), "Home_Roof_Grey": (118, 116, 112), "Home_Roof_Brown": (96, 78, 62),
                "Home_Roof_Tan": (150, 134, 108), "Home_Roof_Terracotta": (150, 86, 62),
                "Home_Roof_MetalLight": (196, 198, 196), "Home_Roof_MetalGreen": (70, 96, 80)}


def roof(kind: str, seed: int = 0) -> Image.Image:
    """u along the eave, v up the slope (tile = 4 m x 4 m)."""
    rng = np.random.default_rng(seed)
    base = ROOF_COLOURS[kind]
    m = PX / 4.0
    im = Image.new("RGB", (PX, PX), base); dr = ImageDraw.Draw(im)
    if "Metal" in kind:                         # standing seam: ribs every 0.45 m
        for i in range(int(4 / 0.45) + 1):
            x = i * 0.45 * m
            dr.rectangle([x, 0, x + 4, PX], fill=tuple(min(255, v + 30) for v in base))
            dr.rectangle([x + 5, 0, x + 8, PX], fill=tuple(max(0, v - 25) for v in base))
        arr = np.asarray(im).astype(float) + _noise(PX, 30, rng, 4)[..., None]
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    exp, tab = 0.14 * m, 0.33 * m               # 3-tab shingles: 14 cm courses
    for r in range(int(PX / exp) + 1):
        off = (r % 2) * tab / 2 + rng.uniform(-3, 3)
        x = -off
        y = PX - (r + 1) * exp
        while x < PX:
            dr.rectangle([x + 1, y, x + tab - 1, y + exp - 2], fill=_jit(base, rng, 10))
            x += tab
        dr.line([0, y + exp - 1, PX, y + exp - 1], fill=tuple(max(0, v - 35) for v in base), width=2)
    arr = np.asarray(im).astype(float) + _noise(PX, 1, rng, 6)[..., None] + _noise(PX, 40, rng, 5)[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def front_door() -> Image.Image:
    """1.0 m x 2.2 m panel door with a glass sidelight frame (texture 256 x 512)."""
    im = Image.new("RGB", (256, 512), (236, 234, 228)); dr = ImageDraw.Draw(im)
    dr.rectangle([22, 22, 234, 512], fill=(96, 62, 40))
    for y0, y1 in ((60, 230), (270, 470)):
        for x0, x1 in ((48, 118), (138, 208)):
            dr.rectangle([x0, y0, x1, y1], outline=(70, 44, 28), width=6)
    dr.ellipse([196, 270, 212, 286], fill=(200, 170, 90))
    return im


def garage_door() -> Image.Image:
    """4.9 m x 2.1 m sectional door, 4 rows of raised panels (texture 512 x 256)."""
    im = Image.new("RGB", (512, 256), (232, 230, 224)); dr = ImageDraw.Draw(im)
    dr.rectangle([12, 14, 500, 256], fill=(228, 226, 220))
    for r in range(4):
        y = 14 + r * 60
        dr.line([12, y, 500, y], fill=(170, 168, 160), width=3)
        for c in range(8):
            x = 18 + c * 60
            dr.rectangle([x + 6, y + 10, x + 54, y + 50], outline=(196, 194, 186), width=3)
    return im


def concrete(seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    a = 190 + _noise(256, 1.2, rng, 6) + _noise(256, 25, rng, 6)
    im = Image.fromarray(np.stack([a, a - 2, a - 6], -1).clip(0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
    dr.line([0, 128, 256, 128], fill=(150, 148, 144), width=2)             # control joints every 2 m (tile 4 m)
    dr.line([128, 0, 128, 256], fill=(150, 148, 144), width=2)
    return im


def solid(c, size=64) -> Image.Image:
    return Image.new("RGB", (size, size), c)


def write_textures(out_dir: Path, walls, roofs) -> dict:
    """PNG per finish; returns {material: texture file}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for i, w in enumerate(walls):
        files[w] = f"{w}.png"; wall(w, i).save(out_dir / files[w])
    for i, r in enumerate(roofs):
        files[r] = f"{r}.png"; roof(r, i).save(out_dir / files[r])
    extra = {"Home_FrontDoor": front_door(), "Home_GarageDoor": garage_door(), "Home_Concrete": concrete(),
             "Home_Trim": solid((238, 236, 230)), "Home_Foundation": wall("Home_Wall_Stone", 99).crop((0, 0, 512, 128))}
    for k, im in extra.items():
        files[k] = f"{k}.png"; im.save(out_dir / files[k])
    return files
