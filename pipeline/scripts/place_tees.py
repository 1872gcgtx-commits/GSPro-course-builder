"""Place tee boxes for every tee set from the scorecard (see gspro_builder/teeplace.py).

Usage: python scripts/place_tees.py   (Mill Creek paths; adds dashed draft tees for review)
"""
import sys,json,glob,numpy as np
sys.path.insert(0,'/home/user/GSPro-course-builder/pipeline')
from gspro_builder.config import load_config
from gspro_builder.terrain import DEM
from gspro_builder.routing import reviewed_shapes
from gspro_builder.teeplace import place_tees
R='/home/user/GSPro-course-builder/courses/mill-creek-salado/'
SP='/tmp/claude-0/-home-user-GSPro-course-builder/d528de22-f6e3-584f-b86d-eddfb6062aa0/scratchpad/'
cfg=load_config(R)
sc=json.load(open(R+'scorecard.json'))
holes={}; trac={}
for f in glob.glob(R+'routing_export/holes/*.json'):
    v=json.load(open(f)); v=v.get('data',v); holes[(v['nine'],v['hole'])]=[(p['x'],p['z']) for p in v['points']]; trac[(v['nine'],v['hole'])]=v.get('yards')
# card: TXGA Blue/White for Creek 1-2 (holes 1-18 = nine 1 then nine 2)
card={}
gk_red={(512,1):368,(183,2):136,(340,3):289}           # Greenskeeper Red where its hole = today's hole (same Blue yardage + number)
for hh in sc['holes']:
    n=1 if hh['number']<=9 else 2; h=(hh['number']-1)%9+1
    y=hh['yards']; blue=y.get('Blue'); white=y.get('White')
    red=gk_red.get((blue,hh['number'])) or (round(blue*0.78) if blue else None)
    card[(n,h)]={'Blue':blue,'White':white,'Red':red}
for h in range(1,10):                                   # Creek 3: no card -> from the traced Blue length
    b=trac.get((3,h)); card[(3,h)]={'Blue':b,'White':round(b*0.931) if b else None,'Red':round(b*0.78) if b else None}
d=json.load(open(R+'draft_surfaces.json')); E=[]
for f in glob.glob(R+'surface_edits/surfaces/*.json'):
    v=json.load(open(f)); v=v.get('data',v); v.setdefault('id',f.split('/')[-1][:-5]); E.append(v)
S=reviewed_shapes(d,E)
existing={}
for s in S:
    if s['kind']=='tee': existing.setdefault((s['nine'],s['hole']),set()).add(s.get('tee_set') or 'Blue')
dem=DEM(np.load(cfg.data_dir/'dem.npy'),cfg.frame)
new=place_tees(holes,card,existing,dem,cfg.layout_shift_m)
est={(n,h) for n in (1,2,3) for h in range(1,10)}
for t in new:
    t['card_source']='TXGA Feb 2025' if (t['nine'] in (1,2) and t['tee_set'] in ('Blue','White')) else ('Greenskeeper' if t['tee_set']=='Red' and t['nine']==1 and t['hole']<=3 else 'estimated')
print(len(new),'new tee boxes;', sum(t['snapped_to_pad'] for t in new),'snapped to a LiDAR tee pad')
import collections; print(collections.Counter((t['nine'],t['tee_set']) for t in new))
json.dump({'new':new,'card':{f'{k[0]}-{k[1]}':v for k,v in card.items()}},open(SP+'tees_new.json','w'))
