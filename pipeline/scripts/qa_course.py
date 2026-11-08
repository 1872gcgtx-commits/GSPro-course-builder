"""Final QA of the reviewed course: completeness, geometry, smoothness, overlaps, hazards.

Usage: python scripts/qa_course.py <course dir> [--fix out.json]
Reads draft_surfaces.json + surface_edits/ + routing_export/ and prints a report.
With --fix, writes the shapes after safe automatic fixes (overlap clipping,
spike removal, invalid-ring repair) to out.json as {id: shape}.
"""
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.measure import find_contours

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gspro_builder.learn import chaikin, resample_ring  # noqa: E402
from gspro_builder.routing import reviewed_shapes  # noqa: E402

root = Path(sys.argv[1])
fix_out = sys.argv[sys.argv.index('--fix') + 1] if '--fix' in sys.argv else None
draft = json.loads((root / 'draft_surfaces.json').read_text())
edits = []
for f in glob.glob(str(root / 'surface_edits/surfaces/*.json')):
    v = json.load(open(f)); v = v.get('data', v); v.setdefault('id', Path(f).stem); edits.append(v)
holes = {}
for f in glob.glob(str(root / 'routing_export/holes/*.json')):
    v = json.load(open(f)); v = v.get('data', v); holes[(v['nine'], v['hole'])] = v
shapes = reviewed_shapes(draft, edits)
edited_ids = {e['id'] for e in edits}
issues = defaultdict(list)


def area(r):
    r = np.asarray(r, float)
    return 0.5 * abs(np.dot(r[:, 0], np.roll(r[:, 1], -1)) - np.dot(r[:, 1], np.roll(r[:, 0], -1)))


def self_intersects(r):
    r = np.asarray(r, float); n = len(r)
    def ccw(a, b, c): return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    for i in range(n):
        a, b = r[i], r[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = r[j], r[(j + 1) % n]
            if ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d):
                return True
    return False


def spikes(r, deg=25):
    r = np.asarray(r, float); n = len(r); out = 0
    for i in range(n):
        a, b, c = r[i - 1], r[i], r[(i + 1) % n]
        u, v = a - b, c - b
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-6 or nv < 1e-6:
            out += 1; continue
        ang = math.degrees(math.acos(np.clip(np.dot(u, v) / (nu * nv), -1, 1)))
        if ang < deg:
            out += 1
    return out


def raster(rings, x0, z1, W, H, res=0.25):
    im = Image.new('L', (W, H), 0); dr = ImageDraw.Draw(im)
    for r in rings:
        dr.polygon([((x - x0) / res, (z1 - z) / res) for x, z in r], fill=1)
    return np.asarray(im, bool)


def frame_of(*rings, pad=3):
    xs = [p[0] for r in rings for p in r]; zs = [p[1] for r in rings for p in r]
    x0, z1 = min(xs) - pad, max(zs) + pad
    return x0, z1, int((max(xs) + pad - x0) / 0.25) + 1, int((z1 - min(zs) + pad) / 0.25) + 1


def overlap_m2(r1, r2):
    x0, z1, W, H = frame_of(r1, r2)
    return float((raster([r1], x0, z1, W, H) & raster([r2], x0, z1, W, H)).sum()) * 0.0625


def mask_to_ring(m, x0, z1, n_min=20, step=4.0):
    c = max(find_contours(np.pad(m.astype(float), 1), 0.5), key=len)
    pts = np.array([(x0 + (cc - 1 + 0.5) * 0.25, z1 - (rr - 1 + 0.5) * 0.25) for rr, cc in c])
    per = float(np.sum(np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)))
    pts = resample_ring(chaikin(resample_ring(pts, max(120, int(per / 2))), 2), max(n_min, int(per / step)))
    return [[round(float(a), 1), round(float(b), 1)] for a, b in pts]


def cut(ring, cutters, gap_px=2, stop_at=None):
    """ring minus cutters (+gap); keeps the largest piece, smooth outline.
    stop_at=(green ring, unit direction of play): a fairway that runs through the green
    also loses everything past the green's centre, so it ends at the front of the green."""
    x0, z1, W, H = frame_of(ring, *cutters)
    m = raster([ring], x0, z1, W, H) & ~ndimage.binary_dilation(raster(cutters, x0, z1, W, H), iterations=gap_px)
    if stop_at is not None:
        gc, d = np.mean(stop_at[0], axis=0), np.asarray(stop_at[1], float)
        rr, cc = np.mgrid[0:H, 0:W]
        along = (x0 + (cc + 0.5) * 0.25 - gc[0]) * d[0] + (z1 - (rr + 0.5) * 0.25 - gc[1]) * d[1]
        m &= along < 0
    lab, n = ndimage.label(m)
    if n == 0:
        return ring
    m = lab == 1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1))))
    m = ndimage.gaussian_filter(m.astype(float), 1.5) > 0.5
    return mask_to_ring(m, x0, z1, n_min=max(20, len(ring)))


def repair(ring):
    """Rebuild a ring from its filled raster: removes self-intersections and spikes."""
    x0, z1, W, H = frame_of(ring)
    m = ndimage.binary_fill_holes(raster([ring], x0, z1, W, H))
    m = ndimage.binary_opening(m, iterations=2)
    m = ndimage.gaussian_filter(m.astype(float), 1.5) > 0.5
    lab, n = ndimage.label(m)
    if n == 0:
        return ring
    m = lab == 1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1))))
    return mask_to_ring(m, x0, z1, n_min=max(16, len(ring) // 2))


by_hole = defaultdict(list)
for s in shapes:
    by_hole[(s['nine'], s['hole'])].append(s)
size_ok = {'green': (150, 1600), 'bunker': (8, 1200), 'tee': (25, 1200), 'fairway': (300, 60000)}
fixed = {}

# ---- completeness per hole
for k, h in sorted(holes.items()):
    kinds = Counter(s['kind'] for s in by_hole[k])
    par = h.get('par')
    if kinds['green'] != 1:
        issues['completeness'].append(f"Creek {k[0]} #{k[1]}: {kinds['green']} greens")
    if kinds['tee'] < 1:
        issues['completeness'].append(f"Creek {k[0]} #{k[1]}: no tee box")
    if par in (4, 5) and kinds['fairway'] < 1:
        issues['completeness'].append(f"Creek {k[0]} #{k[1]} (par {par}): no fairway")
    sets = Counter(s.get('tee_set') for s in by_hole[k] if s['kind'] == 'tee')
    for t, c in sets.items():
        if t and c > 1:
            issues['tee sets'].append(f"Creek {k[0]} #{k[1]}: {c} tee boxes labelled {t}")

# ---- geometry, size, smoothness
for s in shapes:
    r = s['ring']; tag = f"Creek {s['nine']} #{s['hole']} {s['kind']} ({'yours' if s['id'] in edited_ids or str(s['id']).startswith('u') else 'draft'})"
    if len(r) < 3:
        issues['invalid'].append(tag + ': fewer than 3 points'); continue
    if s['kind'] in size_ok and not (size_ok[s['kind']][0] <= area(r) <= size_ok[s['kind']][1]):
        issues['size'].append(f"{tag}: {area(r):.0f} m2")
    if s.get('stream'):
        continue
    si = self_intersects(r)
    sp = spikes(r)
    if si or sp:
        issues['shape repaired' if fix_out else 'shape'].append(f"{tag}: {'self-crossing outline' if si else ''}{' and ' if si and sp else ''}{f'{sp} sharp spikes' if sp else ''}")
        if fix_out:
            fixed[s['id']] = dict(s, ring=repair(r))

cur = {s['id']: fixed.get(s['id'], s) for s in shapes}

# ---- shapes tagged to the wrong hole (e.g. drawn while another hole was selected):
# far (>150 m) from their own line but on (<30 m) another hole's line -> move them
def dist_line(p, pts):
    best = 1e9
    for a, c in zip(pts[:-1], pts[1:]):
        a, c = np.array(a), np.array(c); ab = c - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
        best = min(best, np.linalg.norm(a + ab * t - p))
    return best
lines = {k: [(p['x'], p['z']) for p in h['points']] for k, h in holes.items()}
for s in shapes:
    k = (s['nine'], s['hole'])
    if s.get('stream') or k not in lines:
        continue
    R = np.asarray(cur[s['id']]['ring'], float)
    own = min(dist_line(p, lines[k]) for p in R[::max(1, len(R) // 30)])
    near = min(lines, key=lambda kk: dist_line(R.mean(0), lines[kk]))
    if near != k and own > 150 and dist_line(R.mean(0), lines[near]) < 30:
        issues['hole tag fixed' if fix_out else 'hole tag'].append(
            f"{s['kind']} {s['id']} tagged Creek {k[0]} #{k[1]} ({own:.0f} m away) sits on Creek {near[0]} #{near[1]}")
        if fix_out:
            cur[s['id']] = fixed[s['id']] = dict(cur[s['id']], nine=near[0], hole=near[1])
shapes = [cur[s['id']] for s in shapes]

# ---- one bunker per patch of sand: overlapping bunkers (often the same sand drafted for two
# neighbouring holes) are merged into one shape, owned by the hole whose green is closest
bunkers = [s for s in shapes if s['kind'] == 'bunker']
parent = {s['id']: s['id'] for s in bunkers}
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
for i, a in enumerate(bunkers):
    for b in bunkers[i + 1:]:
        if np.linalg.norm(np.mean(a['ring'], 0) - np.mean(b['ring'], 0)) < 60 and overlap_m2(a['ring'], b['ring']) > 1:
            parent[find(a['id'])] = find(b['id'])
groups = defaultdict(list)
for s in bunkers:
    groups[find(s['id'])].append(s)
green_c = {(s['nine'], s['hole']): np.mean(s['ring'], 0) for s in shapes if s['kind'] == 'green'}
for grp in groups.values():
    if len(grp) < 2:
        continue
    keep = max(grp, key=lambda s: (s['id'] in edited_ids or str(s['id']).startswith('u'), area(s['ring'])))
    x0, z1, W, H = frame_of(*[s['ring'] for s in grp])
    m = ndimage.gaussian_filter(raster([s['ring'] for s in grp], x0, z1, W, H).astype(float), 1.5) > 0.5
    c = np.mean(np.vstack([s['ring'] for s in grp]), 0)
    owner = min({(s['nine'], s['hole']) for s in grp}, key=lambda k: np.linalg.norm(green_c.get(k, c + 1e4) - c))
    holes_txt = ', '.join(sorted({f"C{s['nine']}#{s['hole']}" for s in grp}))
    issues['duplicate bunkers merged' if fix_out else 'duplicate bunkers'].append(f"{len(grp)} overlapping bunkers ({holes_txt}) -> one, on Creek {owner[0]} #{owner[1]}")
    if fix_out:
        cur[keep['id']] = fixed[keep['id']] = dict(keep, nine=owner[0], hole=owner[1], ring=mask_to_ring(m, x0, z1, n_min=len(keep['ring'])))
        for s in grp:
            if s['id'] != keep['id']:
                cur.pop(s['id'], None); fixed[s['id']] = dict(s, deleted=True)
# ---- one continuous fairway: pieces of the same hole's fairway that touch (<1.5 m apart)
# are joined into one smooth shape; pieces split by a creek or rough stay separate
fw = defaultdict(list)
for s in cur.values():
    if s['kind'] == 'fairway':
        fw[(s['nine'], s['hole'])].append(s)
for k, items in fw.items():
    merged = True
    while merged and len(items) > 1:
        merged = False
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                x0, z1, W, H = frame_of(a['ring'], b['ring'])
                ma, mb = raster([a['ring']], x0, z1, W, H), raster([b['ring']], x0, z1, W, H)
                if not (ndimage.binary_dilation(ma, iterations=3) & mb).any():
                    continue
                keep, drop = (a, b) if area(a['ring']) >= area(b['ring']) else (b, a)
                m = ndimage.binary_closing(np.pad(ma | mb, 12), iterations=8)[12:-12, 12:-12]
                m = ndimage.gaussian_filter(m.astype(float), 1.5) > 0.5
                issues['fairway pieces joined' if fix_out else 'fairway in pieces'].append(f"Creek {k[0]} #{k[1]}: 2 touching fairway pieces -> one")
                new = dict(keep, ring=mask_to_ring(m, x0, z1, n_min=len(keep['ring'])))
                if fix_out:
                    cur[keep['id']] = fixed[keep['id']] = new
                    cur.pop(drop['id'], None); fixed[drop['id']] = dict(drop, deleted=True)
                items = [new if x is keep else x for x in items if x is not drop]
                merged = True
                break
            if merged:
                break
by_hole = defaultdict(list)
for s in cur.values():
    by_hole[(s['nine'], s['hole'])].append(s)

# ---- overlaps (greens never overlap bunkers; fairways stop at greens and bunkers; tees clear of sand)
for k, items in by_hole.items():
    ids = [s['id'] for s in items]
    g = [cur[i] for i in ids if cur[i]['kind'] == 'green']
    b = [cur[i] for i in ids if cur[i]['kind'] == 'bunker']
    f = [cur[i] for i in ids if cur[i]['kind'] == 'fairway']
    t = [cur[i] for i in ids if cur[i]['kind'] == 'tee']
    for gg in g:
        hits = [bb['ring'] for bb in b if overlap_m2(gg['ring'], bb['ring']) > 1]
        if hits:
            issues['overlap fixed' if fix_out else 'overlap'].append(f"Creek {k[0]} #{k[1]}: green overlaps {len(hits)} bunker(s)")
            if fix_out:
                cur[gg['id']] = fixed[gg['id']] = dict(gg, ring=cut(gg['ring'], hits))
    for ff in f:
        # bunkers/tees wholly inside a fairway are island bunkers or forward tees (both paint
        # over fairway): fine; a few m2 of edge contact is smoothing, not an overlap
        ov = [(x, overlap_m2(ff['ring'], x['ring'])) for x in g + b + t]
        cutters = [x['ring'] for x, o in ov if o > 5 and not (x['kind'] in ('bunker', 'tee') and o > 0.9 * area(x['ring']))]
        through = [x for x, o in ov if x['kind'] == 'green' and o > 0.5 * area(x['ring'])]
        stop = None
        if through and k in holes:
            pts = np.array([(p['x'], p['z']) for p in holes[k]['points']], float)
            d = pts[-1] - pts[-2]; stop = (through[0]['ring'], d / max(np.linalg.norm(d), 1e-6))
        if cutters:
            what = 'runs through the green' if through else f"overlaps {len(cutters)} green/bunker/tee shape(s)"
            issues['overlap fixed' if fix_out else 'overlap'].append(f"Creek {k[0]} #{k[1]}: fairway {what}")
            if fix_out:
                cur[ff['id']] = fixed[ff['id']] = dict(ff, ring=cut(ff['ring'], cutters, gap_px=4, stop_at=stop))

# ---- hazards: bunkers in the creek, shapes in water
creek = next((s for s in shapes if s.get('stream')), None)
waters = [s for s in shapes if s['kind'] == 'water']
for s in shapes:
    if s['kind'] in ('bunker', 'green', 'tee', 'fairway'):
        for w in waters:
            ov = overlap_m2(s['ring'], w['ring'])
            if ov > 0.25 * area(s['ring']):
                issues['in water'].append(f"Creek {s['nine']} #{s['hole']} {s['kind']}: {ov:.0f} m2 inside water")

# ---- hole assignment: each green/tee should be nearest its own routing line
def dist_line(p, pts):
    best = 1e9
    for a, c in zip(pts[:-1], pts[1:]):
        a, c = np.array(a), np.array(c); ab = c - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
        best = min(best, np.linalg.norm(a + ab * t - p))
    return best
lines = {k: [(p['x'], p['z']) for p in h['points']] for k, h in holes.items()}
for s in shapes:
    if s['kind'] not in ('green', 'tee'):
        continue
    c = np.mean(s['ring'], axis=0)
    own = dist_line(c, lines[(s['nine'], s['hole'])]) if (s['nine'], s['hole']) in lines else 1e9
    near_k = min(lines, key=lambda k: dist_line(c, lines[k]))
    if near_k != (s['nine'], s['hole']) and own > 25:
        issues['hole assignment'].append(f"Creek {s['nine']} #{s['hole']} {s['kind']} sits on Creek {near_k[0]} #{near_k[1]}'s line ({own:.0f} m from its own)")

# ---- creek profile
if creek:
    prof = np.array(creek['profile'])
    if np.any(np.diff(prof[:, 2]) > 0.01):
        issues['creek'].append('water surface rises somewhere downstream')

print(f"holes: {len(holes)}  shapes: {len(shapes)}  " + ", ".join(f"{k} {v}" for k, v in Counter(s['kind'] for s in shapes).items()))
for cat, items in issues.items():
    print(f"\n[{cat}] {len(items)}")
    for it in items[:40]:
        print('  -', it)
if not issues:
    print('no issues')
if fix_out:
    Path(fix_out).write_text(json.dumps(fixed))
    print(f"\n{len(fixed)} shapes fixed -> {fix_out}")
