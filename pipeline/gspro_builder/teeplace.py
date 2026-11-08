"""Tee boxes for every tee set from the scorecard (no need to trace each one).

For a tee set with card yardage Y on a hole, the box goes on the line of play at Y yards
from the centre of the green (measured back along the owner's routing line, the way
card yardages are measured). It then snaps to the real tee pad if the LiDAR shows one
nearby: tee pads are flat, slightly raised, smooth ground, so the search scores
flatness, smoothness and height above the surroundings within +/-12 m across and +/-8 m
along the line. The box is a rectangle square to the line of play, sized by tee set.
"""
from __future__ import annotations

import math

import numpy as np

BOX = {"Black": (12.0, 7.0), "Blue": (11.0, 7.0), "White": (9.0, 6.0), "Gold": (8.0, 6.0), "Red": (7.0, 5.0),
       "Green": (6.0, 5.0)}


def point_back_from_green(line, dist_m):
    """Point (and direction of play there) dist_m back along the line from its last point."""
    pts = np.asarray(line, float)[::-1]                    # green -> tee
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    if dist_m >= cum[-1]:                                   # longer than the line: extend past the back tee
        d = (pts[-1] - pts[-2]) / max(seg[-1], 1e-6)
        return pts[-1] + d * (dist_m - cum[-1]), -d
    i = int(np.searchsorted(cum, dist_m) - 1); i = max(0, min(i, len(seg) - 1))
    t = (dist_m - cum[i]) / max(seg[i], 1e-6)
    d = (pts[i + 1] - pts[i]) / max(seg[i], 1e-6)
    return pts[i] + d * (dist_m - cum[i]) if t >= 0 else pts[i], -d


def snap_to_pad(dem, x, z, play_dir, search_across=12.0, search_along=8.0):
    """Best tee-pad centre near (x, z) from the LiDAR: flat, smooth, a little raised."""
    res = dem.res_m; n = dem.data.shape[0]; h = dem.frame.size_m / 2
    from scipy import ndimage
    r0 = int((h - z) / res); c0 = int((x + h) / res); w = int(25 / res)
    if r0 - w < 0 or c0 - w < 0 or r0 + w >= n or c0 + w >= n:
        return (x, z), 0.0
    a = dem.data[r0 - w:r0 + w + 1, c0 - w:c0 + w + 1].astype(float)
    gy, gx = np.gradient(a, res)
    slope = ndimage.uniform_filter(np.hypot(gx, gy), int(4 / res) | 1)
    rough = ndimage.generic_filter(a, np.std, size=int(3 / res) | 1) if a.size < 20000 else ndimage.uniform_filter(
        np.abs(a - ndimage.uniform_filter(a, int(3 / res) | 1)), int(3 / res) | 1)
    raised = ndimage.uniform_filter(a, int(5 / res) | 1) - ndimage.uniform_filter(a, int(20 / res) | 1)
    score = -slope * 20 - rough * 8 + np.clip(raised, -0.5, 1.0)
    yy, xx = np.mgrid[-w:w + 1, -w:w + 1]
    X = xx * res; Z = -yy * res
    across = np.array([-play_dir[1], play_dir[0]])
    u = X * play_dir[0] + Z * play_dir[1]; v = X * across[0] + Z * across[1]
    ok = (np.abs(u) <= search_along) & (np.abs(v) <= search_across)
    score = np.where(ok, score - 0.02 * np.hypot(u, v), -1e9)   # prefer the card point when it's a tie
    i = np.unravel_index(np.argmax(score), score.shape)
    best = score[i]; base = score[w, w]
    if best - base < 0.15:                                     # no clearly better pad: keep the card point
        return (x, z), 0.0
    return (x + X[i], z + Z[i]), float(best - base)


def tee_box(center, play_dir, size):
    L, W = size
    d = np.asarray(play_dir, float); a = np.array([-d[1], d[0]])
    c = np.asarray(center, float)
    return [list(map(lambda v: round(float(v), 2), p)) for p in
            (c - d * L / 2 - a * W / 2, c + d * L / 2 - a * W / 2, c + d * L / 2 + a * W / 2, c - d * L / 2 + a * W / 2)]


def place_tees(holes, card, existing, dem=None, shift=(0.0, 0.0)):
    """holes: {(nine, hole): line [(x, z), ...] (page frame, tee -> green)}
    card: {(nine, hole): {tee_set: yards}}; existing: {(nine, hole): set(tee_set)}.
    Returns new tee shapes (page frame) with tee_set and yardage notes."""
    out = []
    for (n, h), line in sorted(holes.items()):
        for ts, yds in (card.get((n, h)) or {}).items():
            if not yds or ts in existing.get((n, h), set()):
                continue
            p, d = point_back_from_green(line, yds * 0.9144)
            snapped, gain = p, 0.0
            if dem is not None:                                # LiDAR frame = page + shift
                q, gain = snap_to_pad(dem, p[0] + shift[0], p[1] + shift[1], d)
                snapped = (q[0] - shift[0], q[1] - shift[1])
            out.append({"id": f"t-n{n}h{h}-{ts.lower()}", "nine": n, "hole": h, "kind": "tee", "tee_set": ts,
                        "method": "fallback", "conf": 0.6, "card_yards": int(yds),
                        "snapped_to_pad": gain > 0, "ring": tee_box(snapped, d, BOX.get(ts, (8.0, 6.0)))})
    return out
