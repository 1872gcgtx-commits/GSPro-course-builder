"""3D web preview of a built course (terrain, painted surfaces, water, LiDAR trees, holes).

Usage: python scripts/make_preview.py <course dir> <out dir>
Writes data.json + tex.jpg; serve them next to pipeline/preview/index.html.
"""
import sys, json, gzip, base64, numpy as np
from pathlib import Path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.spatial import cKDTree
from gspro_builder.config import load_config
C = sys.argv[1].rstrip('/') + '/'
O = C + 'output/'
OUT = sys.argv[2]
cfg = load_config(C)
rep = json.load(open(O + 'build_report.json'))['terrain']
N, S, rng = rep['resolution'], rep['size_m'], rep['height_range_m']
H = np.fromfile(O + 'terrain/heightmap_4097.raw', '<u2').reshape(N, N)[::-1].astype(np.float32) / 65535 * rng  # north-up
b = np.array([cfg.frame.to_local(lo, la) for lo, la in cfg.boundary])
pad = 120
x0, x1 = b[:, 0].min() - pad, b[:, 0].max() + pad
z0, z1 = b[:, 1].min() - pad, b[:, 1].max() + pad
STEP = 2.5
nx, nz = int((x1 - x0) / STEP) + 1, int((z1 - z0) / STEP) + 1
xs = x0 + np.arange(nx) * STEP; zs = z1 - np.arange(nz) * STEP          # rows north->south
r = S / (N - 1)
cols = (xs + S / 2) / r; rows = (S / 2 - zs) / r
Hs = ndimage.map_coordinates(H, np.meshgrid(rows, cols, indexing='ij'), order=1)
hmin = float(Hs.min())
hq = np.round((Hs - hmin) * 100).astype('<u2')                        # cm

# ---- water surface grid from water.json
w = json.load(open(O + 'water.json'))
V = np.array([(v['x'], v['z'], v['y']) for bd in w['bodies'] for v in bd['vertices']])
L = np.asarray(Image.open(O + 'surfaces/surface_labels.png'))
lr = S / L.shape[0]
LX, LZ = np.meshgrid(xs, zs)
lab = L[np.clip(((S / 2 - LZ) / lr).astype(int), 0, L.shape[0] - 1), np.clip(((LX + S / 2) / lr).astype(int), 0, L.shape[1] - 1)]
wm = ndimage.binary_dilation(lab == 8, iterations=1)
tree = cKDTree(V[:, :2])
wl = np.zeros_like(Hs)
d, i = tree.query(np.c_[LX[wm], LZ[wm]])
wl[wm] = np.where(d < 15, V[i, 2], 0)
wq = np.where(wl > 0, np.round((wl - hmin) * 100 + 1), 0).clip(0, 65535).astype('<u2')

# ---- texture: aerial detail under game colours
TEX_W = 4096
TEX_H = int(TEX_W * (z1 - z0) / (x1 - x0))
txs = np.linspace(x0, x1, TEX_W); tzs = np.linspace(z1, z0, TEX_H)
TX, TZ = np.meshgrid(txs, tzs)
ov = Image.open(O + 'overlay.jpg'); orr = S / ov.size[0]
c0, c1 = int((x0 + S / 2) / orr), int((x1 + S / 2) / orr); r0, r1 = int((S / 2 - z1) / orr), int((S / 2 - z0) / orr)
aer = np.asarray(ov.crop((c0, r0, c1, r1)).resize((TEX_W, TEX_H), Image.LANCZOS), np.float32) / 255
tl = L[np.clip(((S / 2 - TZ) / lr).astype(int), 0, L.shape[0] - 1), np.clip(((TX + S / 2) / lr).astype(int), 0, L.shape[1] - 1)]
lum = aer.mean(2, keepdims=True)
detail = (lum / max(float(np.median(lum)), 1e-3)) ** 0.6
pal = {0: (96, 128, 62), 1: (52, 78, 40), 2: (92, 150, 62), 3: (104, 170, 70), 4: (110, 182, 84),
       5: (112, 178, 82), 6: (122, 196, 92), 7: (232, 216, 172), 8: (44, 92, 118), 9: (176, 172, 160)}
col = np.zeros((TEX_H, TEX_W, 3), np.float32)
for k, c in pal.items():
    col[tl == k] = np.array(c, np.float32) / 255
# mowing stripes on fairways and greens (along x every 8 m alternating)
stripe = ((np.floor((TX * 0.8 + TZ * 0.6) / 7.0)) % 2 == 0)
for k, amp in ((3, 0.07), (6, 0.04), (5, 0.04)):
    m = tl == k
    col[m] *= np.where(stripe[m], 1 + amp, 1 - amp)[:, None]
# rough keeps more of the photo (dry grass, paths, houses); course surfaces stay clean
mix = np.where(np.isin(tl, [0, 1]), 0.55, 0.12)[..., None]
out = col * (1 - mix) * np.clip(detail, 0.7, 1.3) + aer * mix
# soft edges between classes
img = Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
img.save(OUT + '/tex.jpg', quality=80)

# ---- trees inside crop
t = np.genfromtxt(O + 'trees.csv', delimiter=',', names=True, dtype=None, encoding=None)
k = (t['x'] > x0) & (t['x'] < x1) & (t['z'] > z0) & (t['z'] < z1)
t = t[k]
tb = np.zeros(len(t), dtype=[('x', '<i2'), ('z', '<i2'), ('h', 'u1'), ('r', 'u1'), ('k', 'u1')])
tb['x'] = np.round(t['x'] * 10 / 2).astype('<i2')   # 0.2 m units
tb['z'] = np.round(t['z'] * 10 / 2).astype('<i2')
tb['h'] = np.clip(np.round(t['height_m'] * 8), 0, 255)
tb['r'] = np.clip(np.round(t['crown_radius_m'] * 20), 0, 255)
SPEC = ['live_oak', 'ashe_juniper', 'cedar_elm', 'pecan', 'bald_cypress', 'mesquite', 'hackberry', 'brush', 'juniper_shrub']
if 'species' in t.dtype.names:
    tb['k'] = [SPEC.index(v) if v in SPEC else (1 if ty == 'conifer' else 2) for v, ty in zip(t['species'], t['type'])]
else:
    tb['k'] = np.where(t['type'] == 'conifer', 1, 2)
# understory shrubs (shrubs.csv) appended with their own kinds
sh_path = Path(O) / 'shrubs.csv'
if sh_path.exists():
    sh = np.genfromtxt(sh_path, delimiter=',', names=True, dtype=None, encoding=None)
    sh = np.atleast_1d(sh)
    k2 = (sh['x'] > x0) & (sh['x'] < x1) & (sh['z'] > z0) & (sh['z'] < z1)
    sh = sh[k2]
    sb = np.zeros(len(sh), dtype=tb.dtype)
    sb['x'] = np.round(sh['x'] * 5).astype('<i2'); sb['z'] = np.round(sh['z'] * 5).astype('<i2')
    sb['h'] = np.clip(np.round(sh['height_m'] * 8), 0, 255); sb['r'] = np.clip(np.round(sh['height_m'] * 0.6 * 20), 0, 255)
    sb['k'] = [SPEC.index(v) for v in sh['kind']]
    tb = np.concatenate([tb, sb])

# ---- holes
h = json.load(open(O + 'holes.json'))
names = {1: 'Creek 1', 2: 'Creek 2', 3: 'Creek 3'}
holes = [{'n': x['nine'], 'h': x['nine_hole'], 'par': x['par'], 'yds': x['tees'][0]['yards'],
          'tee': [x['tees'][0]['position'][c] for c in 'xyz'], 'green': [x['green_center'][c] for c in 'xyz'],
          'pin': [x['pin'][c] for c in 'xyz'], 'aims': [[a[c] for c in 'xyz'] for a in x.get('aim_points', [])],
          'elev': x['elevation_change_m']} for x in h['holes']]
pack = lambda a: base64.b64encode(gzip.compress(a.tobytes(), 9)).decode()
meta = {'species': SPEC, 'nx': nx, 'nz': nz, 'step': STEP, 'x0': x0, 'z1': z1, 'hmin': hmin, 'holes': holes,
        'routings': json.load(open(O + 'routings/routings.json')) if True else None}
meta['bin'] = {'h': pack(hq), 'w': pack(wq), 't': pack(tb)}

# ---- tree models (procedural, per species) + leaf textures for the 3D trees
from gspro_builder.treegen import write_models
tm = write_models(Path(OUT) / 'tree_models', variants=3)
meta['treeModels'] = tm['web']
import shutil
for f in (Path(OUT) / 'tree_models').glob('*.png'):
    shutil.copy(f, Path(OUT) / f.name)

# ---- homes (output/homes/homes.obj): vertices + triangles + a finish index per vertex
hobj = Path(O) / 'homes' / 'homes.obj'
if hobj.exists():
    import re
    mtl = {}
    for blk in (Path(O) / 'homes' / 'homes.mtl').read_text().split('newmtl ')[1:]:
        name = blk.split()[0]; kd = re.search(r'Kd ([\d.]+) ([\d.]+) ([\d.]+)', blk)
        mtl[name] = [float(kd.group(i)) for i in (1, 2, 3)]
    names = list(mtl)
    vs, tris, vm, cur = [], [], {}, None
    for line in open(hobj):
        if line.startswith('v '):
            _, a, b, c = line.split(); vs.append((float(a), float(b), -float(c)))   # back to z north
        elif line.startswith('usemtl'):
            cur = names.index(line.split()[1])
        elif line.startswith('f '):
            t = [int(x.split('/')[0]) - 1 for x in line.split()[1:]]
            tris.append(t)
            for i in t:
                vm[i] = cur
    V = np.array(vs); T = np.array(tris, '<u4')
    keep = (V[:, 0] > x0) & (V[:, 0] < x1) & (V[:, 2] > z0) & (V[:, 2] < z1)
    T = T[keep[T].all(1)]
    hv = np.zeros(len(V), dtype=[('x', '<i2'), ('y', '<u2'), ('z', '<i2'), ('m', 'u1')])
    hv['x'] = np.round(V[:, 0] * 10).astype('<i2'); hv['z'] = np.round(V[:, 2] * 10).astype('<i2')
    hv['y'] = np.round((V[:, 1] - hmin + 5) * 100).clip(0, 65535).astype('<u2')
    hv['m'] = [vm.get(i, 0) for i in range(len(V))]
    meta['bin']['hv'] = pack(hv); meta['bin']['ht'] = pack(T)
    meta['homeMaterials'] = [mtl[n] for n in names]
    print(f"homes: {len(V)} vertices, {len(T)} triangles")
json.dump(meta, open(OUT + '/data.json', 'w'), separators=(',', ':'))
print(f"{nx}x{nz} grid, {len(tb)} trees -> {OUT}/data.json, {OUT}/tex.jpg")
