"""Relearn from the owner's fixes and redraft greens/bunkers on untouched holes."""
import sys, json, glob, math, collections, time
sys.path.insert(0, '/home/user/GSPro-course-builder/pipeline')
import numpy as np
from gspro_builder.config import load_config
from gspro_builder.layout import load_layout, point_polyline_dist
from gspro_builder.routing import reviewed_shapes
from gspro_builder.finetrace import (Hires, BunkerModel, GreenModel, tune_grow, trace_green_combo, tune_greens, tune_bunkers, trace_green, trace_green_best, mask_to_ring,
                                     find_bunkers)

SP = '/tmp/claude-0/-home-user-GSPro-course-builder/d528de22-f6e3-584f-b86d-eddfb6062aa0/scratchpad/'
root = '/home/user/GSPro-course-builder/courses/mill-creek-salado/'
t0 = time.time()
cfg = load_config(root)
import subprocess
doc = json.loads(subprocess.check_output(['git', 'show', 'HEAD:courses/mill-creek-salado/draft_surfaces.json'],
                                         cwd='/home/user/GSPro-course-builder'))
edits = []
for f in glob.glob(root + 'surface_edits/surfaces/*.json'):
    v = json.load(open(f)); v = v.get('data', v); v.setdefault('id', f.split('/')[-1][:-5]); edits.append(v)
reviewed = {f.split('/')[-1][:-5] for f in glob.glob(root + 'surface_edits/review/*.json')
            if (json.load(open(f)).get('data', json.load(open(f)))).get('reviewed')}
final = reviewed_shapes(doc, edits)
touched = {(e['nine'], e['hole']) for e in edits if 'nine' in e}
key = lambda n, h: f"n{n}h{h}"
clicks = {}
for f in glob.glob(root + 'routing_export/holes/*.json'):
    v = json.load(open(f)); v = v.get('data', v)
    clicks[key(v['nine'], v['hole'])] = (v['points'][-1]['x'], v['points'][-1]['z'])

# ---- training material: shapes on checked holes + any owner-drawn/edited shape
owner = [s for s in final if key(s['nine'], s['hole']) in reviewed or s.get('edited') or str(s['id']).startswith('u')]
owner_greens = {key(s['nine'], s['hole']): s['ring'] for s in owner if s['kind'] == 'green'}
owner_bunkers = [s for s in owner if s['kind'] == 'bunker']
print(f"learning from {len(owner_greens)} greens and {len(owner_bunkers)} bunkers you drew/checked")
H = Hires(cfg)
bmodel = BunkerModel.from_shapes(H, owner_bunkers)
gparams, gscore = tune_greens(H, owner_greens, clicks, bmodel)
gmodel = GreenModel.from_shapes(H, [{'ring': r} for r in owner_greens.values()])
gthr, grow_score = tune_grow(H, gmodel, owner_greens, clicks)
def _area(r): return 0.5 * abs(sum(r[i][0] * r[(i + 1) % len(r)][1] - r[(i + 1) % len(r)][0] * r[i][1] for i in range(len(r))))
median_green = float(np.median([_area(r) for r in owner_greens.values()]))
print(f"  your median green: {median_green:.0f} m2")
# creek corridor (NHD stream shape) = no bunkers
from gspro_builder.course_meshes import points_in_ring as _pir
creek = next((s for s in doc['shapes'] if s.get('stream')), None)
creek_ring = np.array(creek['ring']) if creek else None
def in_creek(pts):
    if creek_ring is None: return False
    from gspro_builder.finetrace import _dist_to_ring
    return _pir(np.asarray(pts), creek_ring).mean() > 0.2 or _dist_to_ring(np.asarray(pts), creek_ring) < 10
gkeys = list(owner_greens)
fw_by = {i: [s['ring'] for s in final if s['kind'] == 'fairway' and key(s['nine'], s['hole']) == k] for i, k in enumerate(gkeys)}
bthr, _ = tune_bunkers(H, bmodel, owner_bunkers, [np.mean(owner_greens[k], axis=0) for k in gkeys],
                       greens=[owner_greens[k] for k in gkeys], fairways=fw_by, hazard=in_creek)

# ---- redraft greens + bunkers on holes the owner hasn't touched
lay = load_layout(cfg.data_dir / 'osm.geojson', cfg.frame)
cart = [ln for k, ln in lay.paths if k == 'cartpath' and len(ln) > 1]
def on_cart(ring): return any(min(point_polyline_dist(tuple(p), ln) for p in ring) < 3 for ln in cart)
new_shapes, changed = [], collections.Counter()
for s in doc['shapes']:
    k = (s['nine'], s['hole'])
    if k in touched or key(*k) in reviewed or s['kind'] not in ('green', 'bunker'):
        new_shapes.append(s); continue
    # greens/bunkers on untouched holes get redrafted below
todo = sorted({(s['nine'], s['hole']) for s in doc['shapes']} - touched - {tuple(map(int, r[1:].split('h'))) for r in reviewed})
for n, h in todo:
    kk = key(n, h)
    if kk not in clicks:
        continue
    m, area, win = trace_green_combo(H, *clicks[kk], gparams, bmodel, gmodel, gthr, gscore, grow_score, median_green)
    old_green = next((s for s in doc['shapes'] if s['nine'] == n and s['hole'] == h and s['kind'] == 'green'), None)
    if m is not None:
        ring = mask_to_ring(H, win, m, 24)
        new_shapes.append({"id": f"v4-{kk}-green-0", "nine": n, "hole": h, "kind": "green", "method": "traced",
                           "conf": round(gscore, 2), "ring": ring})
        changed['green traced'] += 1
    elif old_green:
        new_shapes.append(old_green); changed['green kept'] += 1
    gx, gz = clicks[kk]
    gring = new_shapes[-1]['ring'] if new_shapes and new_shapes[-1]['kind'] == 'green' and new_shapes[-1]['hole'] == h and new_shapes[-1]['nine'] == n else None
    fws = [s['ring'] for s in doc['shapes'] if s['nine'] == n and s['hole'] == h and s['kind'] == 'fairway']
    for i, b in enumerate(find_bunkers(H, gx, gz, bmodel, bthr[0], exclude=on_cart, bowl_max=bthr[1],
                                       green=gring, fairways=fws, hazard=in_creek)):
        new_shapes.append({"id": f"v4-{kk}-bunker-{i}", "nine": n, "hole": h, "kind": "bunker", "method": "traced",
                           "conf": 0.8 if b['bowl'] < -0.08 else 0.6, "ring": b['ring']})
        changed['bunkers'] += 1
    # keep fairway-side bunkers from the previous draft that are >45 m from the green
    for s in doc['shapes']:
        if s['nine'] == n and s['hole'] == h and s['kind'] == 'bunker' and math.dist(np.mean(s['ring'], axis=0), (gx, gz)) > 45:
            R = np.array(s['ring']); A = 0.5 * abs(np.dot(R[:, 0], np.roll(R[:, 1], -1)) - np.dot(R[:, 1], np.roll(R[:, 0], -1)))
            P = np.sum(np.linalg.norm(np.roll(R, -1, 0) - R, axis=1))
            ev = np.linalg.eigvalsh(np.cov((R - R.mean(0)).T))
            if 2 * A / max(P, 1e-3) < 3.5 or ev[0] / max(ev[1], 1e-6) < 0.12 or on_cart(s['ring']):
                changed['old bunkers dropped (path/sliver)'] += 1; continue
            new_shapes.append(s); changed['fairway bunkers kept'] += 1
# trim fairways back from the greens (1.5 m) on redrafted holes
from PIL import Image, ImageDraw
from gspro_builder.learn import chaikin, resample_ring
from skimage.measure import find_contours
def trim(fw, green):
    xs=[p[0] for p in fw+green]; zs=[p[1] for p in fw+green]
    x0,z1=min(xs)-5,max(zs)+5; W=int((max(xs)+5-x0)/0.25); Hh=int((z1-min(zs)+5)/0.25)
    def ras(r):
        im=Image.new('L',(W,Hh),0); ImageDraw.Draw(im).polygon([((x-x0)/0.25,(z1-z)/0.25) for x,z in r],fill=1); return np.asarray(im,bool)
    from scipy import ndimage as ndi
    m=ras(fw)&~ndi.binary_dilation(ras(green),iterations=6)
    lab,n=ndi.label(m)
    if n==0: return fw
    m=lab==1+int(np.argmax(ndi.sum(m,lab,range(1,n+1))))
    c=max(find_contours(np.pad(m.astype(float),1),0.5),key=len)
    pts=np.array([(x0+(cc-1+0.5)*0.25, z1-(rr-1+0.5)*0.25) for rr,cc in c])
    per=float(np.sum(np.linalg.norm(np.diff(np.vstack([pts,pts[:1]]),axis=0),axis=1)))
    pts=resample_ring(chaikin(resample_ring(pts,max(120,int(per/2))),2),max(40,int(per/5)))
    return [[round(float(a),1),round(float(b),1)] for a,b in pts]
# a green never overlaps a bunker: cut detected bunkers (+0.5 m) out of redrafted greens
def cut_out(ring, holes_rings):
    xs=[p[0] for r in [ring]+holes_rings for p in r]; zs=[p[1] for r in [ring]+holes_rings for p in r]
    x0,z1=min(xs)-3,max(zs)+3; W=int((max(xs)+3-x0)/0.25); Hh=int((z1-min(zs)+3)/0.25)
    def ras(r):
        im=Image.new('L',(W,Hh),0); ImageDraw.Draw(im).polygon([((x-x0)/0.25,(z1-z)/0.25) for x,z in r],fill=1); return np.asarray(im,bool)
    from scipy import ndimage as ndi
    g=ras(ring); b=np.zeros_like(g)
    for r in holes_rings: b|=ras(r)
    if not (g&b).any(): return ring, False
    m=g&~ndi.binary_dilation(b,iterations=2)
    lab,n=ndi.label(m)
    if n==0: return ring, False
    m=lab==1+int(np.argmax(ndi.sum(m,lab,range(1,n+1))))
    m=ndi.gaussian_filter(m.astype(float),2)>0.5
    c=max(find_contours(np.pad(m.astype(float),1),0.5),key=len)
    pts=np.array([(x0+(cc-1+0.5)*0.25, z1-(rr-1+0.5)*0.25) for rr,cc in c])
    pts=resample_ring(chaikin(resample_ring(pts,120),2),28)
    return [[round(float(a),1),round(float(b),1)] for a,b in pts], True
cut=0
for s in new_shapes:
    k=(s['nine'],s['hole'])
    if s['kind']=='green' and k not in touched and key(*k) not in reviewed:
        bs=[b['ring'] for b in new_shapes if b['kind']=='bunker' and (b['nine'],b['hole'])==k]
        s['ring'], did = cut_out(s['ring'], bs)
        cut += did
print('greens with bunkers cut out:', cut)
greens={(s['nine'],s['hole']):s['ring'] for s in new_shapes if s['kind']=='green'}
for s in new_shapes:
    k=(s['nine'],s['hole'])
    if s['kind']=='fairway' and k not in touched and key(*k) not in reviewed:
        fc=np.mean(s['ring'],axis=0)
        for gk,gring in greens.items():
            gc=np.mean(gring,axis=0)
            if np.min(np.linalg.norm(np.array(s['ring'])-gc,axis=1))<60:
                s['ring']=trim(s['ring'],gring)
doc['shapes'] = new_shapes
doc['version'] = 5
doc['learned'] = {"greens_iou": round(gscore, 2), "green_params": gparams, "bunker_threshold": bthr,
                  "trained_on": {"greens": len(owner_greens), "bunkers": len(owner_bunkers)}}
for p in [SP + 'router/draft_surfaces.json', root + 'draft_surfaces.json']:
    json.dump(doc, open(p, 'w'), separators=(',', ':'))
print(dict(changed), 'holes redrafted:', len(todo), round(time.time() - t0), 's')

# ---- fairways on par 3 holes (mown approach between tee and green), any hole without one yet
from gspro_builder.classify import Grid, _fairway_continuous
from gspro_builder.learn import styled_ring, rasterize
from gspro_builder.layout import load_layout as _ll
lc = np.load(cfg.data_dir / 'landcover.npy')
meta = json.loads((cfg.data_dir / 'landcover.json').read_text())
n_full = int(round(cfg.frame.size_m / meta['res']))
G = Grid(cfg.frame, n_full, meta['res'], meta['r0'], meta['r0'] + lc.shape[0], meta['c0'], meta['c0'] + lc.shape[1])
chm = np.load(cfg.data_dir / 'chm.npy', mmap_mode='r')[G.r0:G.r1, G.c0:G.c1]
holes_par = {}
for f in glob.glob(root + 'routing_export/holes/*.json'):
    v = json.load(open(f)); v = v.get('data', v)
    holes_par[(v['nine'], v['hole'])] = (v.get('par'), [(p['x'], p['z']) for p in v['points']])
added = 0
for (n, h), (par, path) in sorted(holes_par.items()):
    if par != 3 or any(s['kind'] == 'fairway' and s['nine'] == n and s['hole'] == h for s in doc['shapes']):
        continue
    green = next((s['ring'] for s in doc['shapes'] if s['kind'] == 'green' and s['nine'] == n and s['hole'] == h), None)
    tee = [s['ring'] for s in doc['shapes'] if s['kind'] == 'tee' and s['nine'] == n and s['hole'] == h]
    if green is None:
        continue
    gmask = rasterize(G, [green])
    tmask = rasterize(G, tee) if tee else np.zeros(G.shape, bool)
    from scipy import ndimage as _nd
    ok = (lc == 0) & (np.asarray(chm) < 1.0) & ~_nd.binary_dilation(tmask, iterations=3)
    m = _fairway_continuous(G, path, ok, gmask, min_start=25.0, min_run=8)
    if m is None:
        # second opinion that doesn't depend on colour: mown turf is smooth in the 0.3 m imagery
        xs = [q[0] for q in path]; zs = [q[1] for q in path]
        cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
        half = max(max(xs) - min(xs), max(zs) - min(zs)) / 2 + 40
        win = H.window(cx, cz, half)
        img = win['img']; br = img[..., :3].mean(-1)
        ndv = (img[..., 3] - img[..., 0]) / (img[..., 3] + img[..., 0] + 1e-3)
        mu = _nd.uniform_filter(br, 7); tex = np.sqrt(np.maximum(_nd.uniform_filter(br * br, 7) - mu * mu, 0))
        tex = _nd.uniform_filter(tex, 9)
        turf = (ndv > 0.05) & (win['chm'] < 1.0)
        smooth = turf & (tex < np.percentile(tex[turf], 35)) if turf.any() else turf
        Gh = Grid(cfg.frame, int(round(cfg.frame.size_m / H.res)), H.res,
                  int(round((cfg.frame.size_m / 2 - H.ext['z1']) / H.res)) + win['r0'],
                  int(round((cfg.frame.size_m / 2 - H.ext['z1']) / H.res)) + win['r0'] + img.shape[0],
                  int(round((H.ext['x0'] + cfg.frame.size_m / 2) / H.res)) + win['c0'],
                  int(round((H.ext['x0'] + cfg.frame.size_m / 2) / H.res)) + win['c0'] + img.shape[1])
        gm_h = rasterize(Gh, [green]); tm_h = rasterize(Gh, tee) if tee else np.zeros(Gh.shape, bool)
        okh = smooth & ~_nd.binary_dilation(tm_h, iterations=8)
        mh = _fairway_continuous(Gh, path, okh, gm_h, min_start=25.0, min_run=8)
        if mh is not None and mh.sum() * Gh.res ** 2 >= 400:
            ring = styled_ring(mh, Gh, 32, 5.0)
            if ring:
                doc['shapes'].append({"id": f"v5-{key(n, h)}-fairway-0", "nine": n, "hole": h, "kind": "fairway",
                                      "method": "par3", "conf": 0.5, "ring": [[x, z] for x, z in ring]})
                added += 1
                print(f"  par 3 fairway (smooth-turf test): Creek {n} #{h} ({mh.sum() * Gh.res ** 2:.0f} m2)")
        continue
    if m is None or m.sum() * G.res ** 2 < 400:
        continue
    ring = styled_ring(m, G, 32, 3.0)
    if ring:
        doc['shapes'].append({"id": f"v5-{key(n, h)}-fairway-0", "nine": n, "hole": h, "kind": "fairway",
                              "method": "par3", "conf": 0.6, "ring": [[x, z] for x, z in ring]})
        added += 1
        print(f"  par 3 fairway: Creek {n} #{h} ({m.sum() * G.res ** 2:.0f} m2)")
print('par 3 fairways added:', added)
for p in [SP + 'router/draft_surfaces.json', root + 'draft_surfaces.json']:
    json.dump(doc, open(p, 'w'), separators=(',', ':'))
