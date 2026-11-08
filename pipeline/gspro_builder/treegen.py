"""Procedural tree models for the local species (no external assets needed).

Each species is grown from a branching skeleton with its own habit:

  live_oak      short trunk, 3-5 heavy limbs spreading near-horizontal, crown ~2x wider
                than tall, tips turning up; dense small dark leaves
  ashe_juniper  2-3 stems from the ground, many short branches all the way up, ragged
                cone of dense blue-green scale foliage
  cedar_elm     clear trunk to ~1/3, upright limbs (vase), oval crown, small leaves
  pecan         tall clear trunk, few big ascending limbs, open broad crown, compound leaves
  bald_cypress  one straight trunk to the top, short level branches shrinking upward
                (spire), feathery foliage
  mesquite      2-3 crooked leaning stems, wide flat airy crown of fine leaflets
  hackberry     like elm but irregular and more open
  brush / juniper_shrub  small multi-stem shrubs

Geometry: tapered 6-sided branch tubes (bark) + leaf clusters made of crossed cards
with an alpha leaf texture drawn per species. Models are 10 m tall; scale per instance.
Writes OBJ + MTL + PNG textures for Unity, and a compact JSON for the web preview.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# habit parameters
SPEC = {
    #               trunk  limbs  limb_ang  len   droop  levels kids  ang2  crooked leaf_m clusters colour (hsl) stems
    "live_oak":     (0.20, (4, 6), (55, 75), 0.62, 0.30, 4, (2, 3), (25, 45), 0.25, 3.2, 220, (0.27, 0.40, 0.30), 1),
    "ashe_juniper": (0.03, (16, 22), (50, 78), 0.46, 0.05, 2, (2, 3), (20, 40), 0.30, 2.4, 260, (0.38, 0.30, 0.27), 2),
    "cedar_elm":    (0.34, (4, 5), (28, 42), 0.52, 0.10, 4, (2, 3), (22, 40), 0.20, 2.8, 210, (0.25, 0.42, 0.36), 1),
    "pecan":        (0.42, (3, 5), (32, 50), 0.58, 0.12, 4, (2, 3), (25, 40), 0.18, 3.2, 200, (0.22, 0.45, 0.38), 1),
    "bald_cypress": (0.08, (20, 26), (78, 92), 0.20, 0.10, 2, (2, 3), (25, 40), 0.10, 2.0, 240, (0.26, 0.38, 0.33), 1),
    "mesquite":     (0.18, (3, 5), (55, 72), 0.62, 0.15, 4, (2, 3), (25, 50), 0.55, 2.6, 170, (0.20, 0.32, 0.42), 3),
    "hackberry":    (0.30, (4, 5), (32, 55), 0.55, 0.18, 4, (2, 3), (25, 50), 0.40, 2.8, 180, (0.25, 0.36, 0.36), 1),
    "brush":        (0.02, (6, 8), (35, 65), 0.55, 0.10, 2, (2, 3), (25, 45), 0.40, 2.6, 90, (0.24, 0.34, 0.36), 3),
    "juniper_shrub": (0.02, (8, 11), (40, 70), 0.45, 0.05, 2, (2, 3), (20, 40), 0.30, 2.4, 110, (0.38, 0.30, 0.27), 3),
}
CONICAL = {"ashe_juniper", "bald_cypress", "juniper_shrub"}


def _rot(v, axis, ang):
    """Rotate vector v about unit axis by ang radians (Rodrigues)."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * np.dot(axis, v) * (1 - math.cos(ang))


def _perp(v):
    a = np.array([1.0, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 0, 1.0])
    p = np.cross(v, a)
    return p / np.linalg.norm(p)


def skeleton(species: str, seed: int = 0, height: float = 10.0):
    """Branch segments [(p0, p1, r0, r1, level)] and leaf cluster points [(pos, size)]."""
    (trunk, limbs, lang, llen, droop, levels, kids, ang2, crook, leaf_m, nclus, _, stems) = SPEC[species]
    rng = np.random.default_rng(seed)
    segs, tips = [], []
    up = np.array([0.0, 1.0, 0.0])
    conical = species in CONICAL

    def grow(p, d, length, r, level, steps=3):
        """A branch as a few bent segments; recurse into children."""
        pts = [p]
        dd = d.copy()
        for s in range(steps):
            # crookedness and droop (live oak limbs sag then turn up at the tips)
            dd = _rot(dd, _perp(dd), rng.normal(0, crook * 0.35))
            sag = droop * (1 if s < steps - 1 else -0.8) if level <= 1 else droop * 0.3
            dd = dd - up * sag * 0.25
            dd /= np.linalg.norm(dd)
            pts.append(pts[-1] + dd * length / steps)
        rr = np.linspace(r, r * 0.55, steps + 1)
        for i in range(steps):
            segs.append((pts[i], pts[i + 1], rr[i], rr[i + 1], level))
        end, rend = pts[-1], rr[-1]
        if level >= levels:
            tips.append((end, leaf_m * rng.uniform(0.8, 1.25)))
            # a few extra clusters along the last branch for fuller crowns
            for t in (0.4, 0.7):
                q = pts[0] + (end - pts[0]) * t
                tips.append((q + rng.normal(0, 0.25, 3), leaf_m * rng.uniform(0.6, 0.9)))
            return
        n = int(rng.integers(kids[0], kids[1] + 1))
        for k in range(n):
            az = 2 * math.pi * (k / n) + rng.uniform(-0.6, 0.6)
            nd = _rot(dd, _perp(dd), math.radians(rng.uniform(*ang2)))
            nd = _rot(nd, dd, az)
            if conical:
                nd = nd + up * 0.15
            nd /= np.linalg.norm(nd)
            grow(end, nd, length * rng.uniform(0.55, 0.75), rend * (1 / n) ** 0.45, level + 1, steps=2)

    for st in range(stems):
        base = np.array([rng.normal(0, 0.25), 0.0, rng.normal(0, 0.25)]) if stems > 1 else np.zeros(3)
        lean = _rot(up, _perp(up), math.radians(rng.uniform(4, 18) if stems > 1 else rng.uniform(0, 4)))
        lean = _rot(lean, up, rng.uniform(0, 2 * math.pi))
        r0 = 0.045 * height / math.sqrt(stems)
        if conical:
            # central leader to the top, branches all along it, shorter going up
            top = base + lean * height * rng.uniform(0.92, 1.0)
            nseg = 8
            for i in range(nseg):
                a = base + (top - base) * i / nseg; b = base + (top - base) * (i + 1) / nseg
                segs.append((a, b, r0 * (1 - i / nseg) + 0.02, r0 * (1 - (i + 1) / nseg) + 0.02, 0))
            nl = int(rng.integers(*limbs)) // stems + 3
            for i in range(nl):
                t = trunk + (1 - trunk) * (i + rng.uniform(0, 1)) / nl
                p = base + (top - base) * t
                ang = math.radians(rng.uniform(*lang))
                d = _rot(lean, _perp(lean), ang)
                d = _rot(d, lean, rng.uniform(0, 2 * math.pi))
                L = height * llen * (1.15 - t) * rng.uniform(0.8, 1.2)
                grow(p, d, L, r0 * 0.35 * (1.1 - t), 1, steps=2)
                tips.append((p + d * L * 0.35, leaf_m * rng.uniform(0.8, 1.1)))
            for i in range(10):                            # foliage hugging the leader
                t = trunk + (1 - trunk) * i / 10
                tips.append((base + (top - base) * t + rng.normal(0, 0.2, 3), leaf_m * (1.15 - t) * 0.9))
            tips.append((top, leaf_m * 0.7))
        else:
            fork = base + lean * height * trunk * rng.uniform(0.85, 1.15)
            segs.append((base, fork, r0, r0 * 0.8, 0))
            n = int(rng.integers(limbs[0], limbs[1] + 1))
            for k in range(n):
                az = 2 * math.pi * (k / n) + rng.uniform(-0.5, 0.5)
                ang = math.radians(rng.uniform(*lang))
                d = _rot(up, np.array([1.0, 0, 0]), ang)
                d = _rot(d, up, az)
                L = height * llen * rng.uniform(0.75, 1.1) * (0.6 if stems > 1 else 1)
                grow(fork, d, L, r0 * 0.8 * (1 / n) ** 0.4, 1)
    # scale to the requested height (top of leaves)
    allp = np.array([s[1] for s in segs] + [t[0] for t in tips])
    top = allp[:, 1].max() + leaf_m * 0.5
    k = height / max(top, 1e-6)
    segs = [(a * k, b * k, r0_ * k, r1_ * k, lv) for a, b, r0_, r1_, lv in segs]
    tips = [(p * k, s * k) for p, s in tips]
    # thin out to the target number of leaf clusters
    if len(tips) > nclus:
        idx = rng.choice(len(tips), nclus, replace=False)
        tips = [tips[i] for i in sorted(idx)]
    return segs, tips


def mesh(species: str, seed: int = 0, height: float = 10.0, sides: int = 6):
    """Bark and leaf meshes: dict(bark=(V, UV, F), leaves=(V, UV, F))."""
    segs, tips = skeleton(species, seed, height)
    rng = np.random.default_rng(seed + 99)
    V, UV, F = [], [], []
    for a, b, r0, r1, lv in segs:
        if lv >= 4 or r1 < 0.012:
            continue                                  # twigs are carried by the leaf cards
        d = b - a; L = np.linalg.norm(d)
        if L < 1e-4:
            continue
        d /= L; u = _perp(d); w = np.cross(d, u)
        base = len(V)
        for j, (p, r) in enumerate(((a, r0), (b, r1))):
            for i in range(sides):
                t = 2 * math.pi * i / sides
                V.append(p + r * (math.cos(t) * u + math.sin(t) * w))
                UV.append((i / sides, j * L / 2.0))
        for i in range(sides):
            i2 = (i + 1) % sides
            F += [(base + i, base + i2, base + sides + i2), (base + i, base + sides + i2, base + sides + i)]
    bark = (np.array(V), np.array(UV), np.array(F, int))
    V, UV, F = [], [], []
    for p, s in tips:
        # 3 crossed cards per cluster, random orientation, 4 atlas cells
        for c in range(3):
            n = rng.normal(0, 1, 3); n[1] *= 0.6; n /= np.linalg.norm(n)
            u = _perp(n); w = np.cross(n, u)
            cell = int(rng.integers(0, 4)); cu, cv = (cell % 2) * 0.5, (cell // 2) * 0.5
            base = len(V)
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                V.append(p + (u * du + w * dv) * s * 0.5)
                UV.append((cu + (du + 1) * 0.25, cv + (dv + 1) * 0.25))
            F += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    leaves = (np.array(V), np.array(UV), np.array(F, int))
    return {"bark": bark, "leaves": leaves}


def leaf_texture(species: str, size: int = 512, seed: int = 0):
    """RGBA atlas (2x2 cells) of leaf clusters drawn for the species."""
    from PIL import Image, ImageDraw, ImageFilter
    rng = np.random.default_rng(seed)
    h0, s0, l0 = SPEC[species][11]
    import colorsys
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    cell = size // 2

    def col(dl=0.0):
        r, g, b = colorsys.hls_to_rgb(h0 + rng.normal(0, 0.012), np.clip(l0 + dl + rng.normal(0, 0.05), 0.05, 0.8),
                                      np.clip(s0 + rng.normal(0, 0.05), 0, 1))
        return int(r * 255), int(g * 255), int(b * 255), 255

    for ci in range(4):
        ox, oy = (ci % 2) * cell, (ci // 2) * cell
        cx, cy = ox + cell / 2, oy + cell / 2
        R = cell * 0.46
        if species in ("ashe_juniper", "juniper_shrub", "bald_cypress"):
            # scale / feathery foliage: many short sprays radiating from twigs
            for _ in range(70):
                a = rng.uniform(0, 2 * math.pi); r = R * math.sqrt(rng.uniform(0.05, 1))
                x, y = cx + r * math.cos(a), cy + r * math.sin(a)
                ang = rng.uniform(0, 2 * math.pi); ln = rng.uniform(10, 24) * size / 512
                for k in range(7 if species == "bald_cypress" else 5):
                    t = k / 6
                    px, py = x + math.cos(ang) * ln * t, y + math.sin(ang) * ln * t
                    w = (5 if species == "bald_cypress" else 3.2) * size / 512
                    side = 1 if k % 2 else -1
                    dr.line([(px, py), (px + math.cos(ang + side * 1.1) * w * 1.8, py + math.sin(ang + side * 1.1) * w * 1.8)],
                            fill=col(-0.04 + 0.08 * t), width=max(1, int(w * 0.8)))
                dr.line([(x, y), (x + math.cos(ang) * ln, y + math.sin(ang) * ln)], fill=col(-0.08), width=max(1, int(2 * size / 512)))
        else:
            # broadleaf: twigs then small leaves (pecan/mesquite: leaflets along a rachis)
            for _ in range(10):
                a = rng.uniform(0, 2 * math.pi)
                dr.line([(cx, cy), (cx + R * 0.8 * math.cos(a), cy + R * 0.8 * math.sin(a))], fill=(70, 55, 40, 255),
                        width=max(1, int(2 * size / 512)))
            nleaf = {"live_oak": 260, "cedar_elm": 220, "hackberry": 160, "pecan": 60, "mesquite": 90, "brush": 160}[species]
            for _ in range(nleaf):
                a = rng.uniform(0, 2 * math.pi); r = R * math.sqrt(rng.uniform(0.02, 1))
                x, y = cx + r * math.cos(a), cy + r * math.sin(a)
                ang = rng.uniform(0, math.pi)
                shade = -0.10 * (r / R) + rng.normal(0, 0.03)       # darker inside, lit edges
                if species in ("pecan", "mesquite"):
                    n = 7 if species == "pecan" else 9
                    ll = (36 if species == "pecan" else 26) * size / 512
                    for k in range(n):
                        t = k / (n - 1)
                        px, py = x + math.cos(ang) * ll * t, y + math.sin(ang) * ll * t
                        for side in (-1, 1):
                            lx, ly = px + math.cos(ang + side * 1.2) * ll * 0.18, py + math.sin(ang + side * 1.2) * ll * 0.18
                            e = (5 if species == "pecan" else 2.5) * size / 512
                            dr.ellipse([lx - e, ly - e * 0.5, lx + e, ly + e * 0.5], fill=col(shade))
                else:
                    e = {"live_oak": 7, "cedar_elm": 6, "hackberry": 8, "brush": 6}[species] * size / 512
                    pts = [(x + math.cos(ang) * e * 1.4 * c_ - math.sin(ang) * e * 0.7 * s_,
                            y + math.sin(ang) * e * 1.4 * c_ + math.cos(ang) * e * 0.7 * s_)
                           for c_, s_ in ((1, 0), (0.3, 1), (-1, 0), (0.3, -1))]
                    dr.polygon(pts, fill=col(shade))
    a = im.split()[3].filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    im.putalpha(a)
    return im


def bark_texture(size: int = 256, seed: int = 0):
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (size, size // 4))
    a = np.repeat(a, 4, axis=1)
    from scipy import ndimage
    a = ndimage.gaussian_filter(a, (8, 1.2))
    a = (a - a.min()) / (a.max() - a.min())
    base = np.array([62, 52, 43], float)
    img = np.clip(base[None, None] * (0.6 + 0.7 * a[..., None]), 0, 255).astype(np.uint8)
    return Image.fromarray(img).filter(ImageFilter.SMOOTH)


def write_models(out_dir: Path, species=None, variants: int = 3) -> dict:
    """OBJ/MTL/PNG per species x variant for Unity, and models.json for the web preview."""
    out_dir.mkdir(parents=True, exist_ok=True)
    species = species or list(SPEC)
    bark_texture().save(out_dir / "bark.png")
    index, web = {}, {}
    for sp in species:
        leaf_texture(sp).save(out_dir / f"leaves_{sp}.png")
        mtl = out_dir / f"{sp}.mtl"
        mtl.write_text(f"newmtl Bark\nKd 1 1 1\nmap_Kd bark.png\n\nnewmtl Leaves_{sp}\nKd 1 1 1\nmap_Kd leaves_{sp}.png\n"
                       f"map_d leaves_{sp}.png\n")
        index[sp] = []
        for v in range(variants):
            m = mesh(sp, seed=v * 17 + len(sp))
            name = f"{sp}_{v}"
            with open(out_dir / f"{name}.obj", "w") as f:
                f.write(f"# {sp} variant {v}: metres, 10 m tall, Y-up, right-handed\nmtllib {sp}.mtl\no {name}\n")
                base = 1
                for part, mat in (("bark", "Bark"), ("leaves", f"Leaves_{sp}")):
                    V, UV, F = m[part]
                    if not len(F):
                        continue
                    np.savetxt(f, V, fmt="v %.3f %.3f %.3f")
                    np.savetxt(f, UV, fmt="vt %.4f %.4f")
                    f.write(f"usemtl {mat}\n")
                    t = F + base
                    np.savetxt(f, np.column_stack([t[:, 0], t[:, 0], t[:, 1], t[:, 1], t[:, 2], t[:, 2]]), fmt="f %d/%d %d/%d %d/%d")
                    base += len(V)
            index[sp].append(f"{name}.obj")
            if v == 0 or sp in ("live_oak", "ashe_juniper", "cedar_elm"):
                web[name] = {p: {"v": np.round(m[p][0], 3).ravel().tolist(), "uv": np.round(m[p][1], 4).ravel().tolist(),
                                 "f": m[p][2].ravel().tolist()} for p in ("bark", "leaves")}
    (out_dir / "models.json").write_text(json.dumps({"species": index, "height_m": 10.0,
                                                     "note": "scale by tree height / 10; leaves use alpha cutout"}))
    return {"species": len(species), "variants": variants, "web": web}
