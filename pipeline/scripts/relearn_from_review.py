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
bthr, _ = tune_bunkers(H, bmodel, owner_bunkers, [np.mean(r, axis=0) for r in owner_greens.values()])

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
    m, area, win = trace_green_combo(H, *clicks[kk], gparams, bmodel, gmodel, gthr, gscore, grow_score)
    old_green = next((s for s in doc['shapes'] if s['nine'] == n and s['hole'] == h and s['kind'] == 'green'), None)
    if m is not None:
        ring = mask_to_ring(H, win, m, 24)
        new_shapes.append({"id": f"v4-{kk}-green-0", "nine": n, "hole": h, "kind": "green", "method": "traced",
                           "conf": round(gscore, 2), "ring": ring})
        changed['green traced'] += 1
    elif old_green:
        new_shapes.append(old_green); changed['green kept'] += 1
    gx, gz = clicks[kk]
    for i, b in enumerate(find_bunkers(H, gx, gz, bmodel, bthr[0], exclude=on_cart, bowl_max=bthr[1])):
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
