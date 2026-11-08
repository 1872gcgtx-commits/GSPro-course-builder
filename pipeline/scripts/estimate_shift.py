"""Measure the offset between the aerial photo and the LiDAR terrain.

Usage: python scripts/estimate_shift.py <course dir>

Greens and tees are flat pads in the LiDAR. Shapes traced on the aerial (NAIP is
only accurate to several metres) can sit a little off those pads. This tries shifts
of up to +/-5 m and reports the one that puts the outlines on the flattest ground,
with a split-half check. Put the result in course.toml:

    [layout]
    shift_m = [east, north]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gspro_builder.config import load_config  # noqa: E402
from gspro_builder.routing import reviewed_shapes  # noqa: E402

root = Path(sys.argv[1])
cfg = load_config(root)
D = np.load(cfg.data_dir / 'dem.npy').astype(float)
n = D.shape[0]; size = cfg.frame.size_m; res = size / n
draft = json.loads((root / 'draft_surfaces.json').read_text())
edits = []
for f in glob.glob(str(root / 'surface_edits/surfaces/*.json')):
    v = json.load(open(f)); edits.append(v.get('data', v))
pads = [s for s in reviewed_shapes(draft, edits) if s['kind'] in ('green', 'tee')]
a = ndimage.gaussian_filter(D, 1.0 / res)
gz, gx = np.gradient(a, res)
slope = np.minimum(np.hypot(gx, gz), 0.3)


def score(ring, dx, dz, h=40):
    r = np.asarray(ring, float); cx, cz = r.mean(0)
    c0 = int((cx - h + size / 2) / res); r0 = int((size / 2 - (cz + h)) / res); w = int(2 * h / res)
    im = Image.new('L', (w, w), 0)
    ImageDraw.Draw(im).polygon([((x + dx - cx + h) / res, (cz + h - z - dz) / res) for x, z in r], fill=1)
    return slope[r0:r0 + w, c0:c0 + w][np.asarray(im, bool)].mean()


grid = np.arange(-5, 5.01, 0.5)


def best(items):
    T = np.array([[np.mean([score(s['ring'], dx, dz) for s in items]) for dx in grid] for dz in grid])
    i, j = np.unravel_index(T.argmin(), T.shape)
    return grid[j], grid[i], T[i, j], T[len(grid) // 2, len(grid) // 2]


dx, dz, s1, s0 = best(pads)
h1, h2 = best(pads[::2]), best(pads[1::2])
print(f"{len(pads)} greens/tees: best shift {dx:+.1f} m east, {dz:+.1f} m north "
      f"(mean pad slope {s0 * 100:.2f}% -> {s1 * 100:.2f}%)")
print(f"split-half check: ({h1[0]:+.1f}, {h1[1]:+.1f}) and ({h2[0]:+.1f}, {h2[1]:+.1f})")
if abs(h1[0] - h2[0]) > 1.5 or abs(h1[1] - h2[1]) > 1.5:
    print("halves disagree - no reliable single offset; leave shift_m at [0, 0]")
else:
    print(f"\n[layout]\nshift_m = [{dx:.1f}, {dz:.1f}]")
