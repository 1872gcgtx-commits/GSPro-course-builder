"""Inkscape SVG of the course layout for the OPCD toolchain.

Format: 1 mm = 1 m, north up, page = the terrain square. One Inkscape layer per
hole (Hole01..Hole18) plus Hole99 for course-wide features, plus an `image`
layer with the aerial overlay (linked, locked) for tracing in Inkscape.

The fill colours are what the OPCD SVG converter keys on to decide the surface
type. The defaults below follow the community osm2svg4opcd palette; if the
current OPCD style guide differs, override them in course.toml:
    [svg.colors]
    green = "#BCE5A4"
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from .geo import CourseFrame
from .layout import SURFACE_ORDER, Layout

OPCD_COLORS = {
    "green": "#BCE5A4", "fairway": "#43E561", "tee": "#A0E5B8", "bunker": "#E5E5AA",
    "water": "#0000C0", "trees": "#164B20", "rough": "#278438", "cartpath": "#808080",
}
COURSE_WIDE = 99


def _d(rings, frame: CourseFrame) -> str:
    h = frame.size_m / 2
    parts = []
    for ring in rings:
        pts = [(x + h, h - z) for x, z in ring]
        if pts[0] == pts[-1]:
            pts = pts[:-1]
        parts.append("M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in pts) + " Z")
    return " ".join(parts)


def write_svg(layout: Layout, frame: CourseFrame, path: Path, overlay: str | None = None,
              colors: dict | None = None) -> Path:
    colors = {**OPCD_COLORS, **(colors or {})}
    size = frame.size_m
    h = size / 2
    by_hole: dict[int, list[str]] = {}
    order = {k: i for i, k in enumerate(SURFACE_ORDER)}
    for n, s in enumerate(sorted(layout.surfaces, key=lambda s: order[s.kind])):
        hole = s.hole if s.hole is not None else COURSE_WIDE
        by_hole.setdefault(hole, []).append(
            f'    <path id="h{hole:02d}_{s.kind}_{n}" inkscape:label="{s.kind}" '
            f'style="fill:{colors[s.kind]};fill-rule:evenodd;stroke:none" d="{_d(s.rings, frame)}"/>')
    for n, (kind, line) in enumerate(layout.paths):
        if kind not in colors:
            continue
        d = "M " + " L ".join(f"{x + h:.3f},{h - z:.3f}" for x, z in line)
        by_hole.setdefault(COURSE_WIDE, []).append(
            f'    <path id="path_{kind}_{n}" inkscape:label="{kind}" '
            f'style="fill:none;stroke:{colors[kind]};stroke-width:2.5" d="{d}"/>')

    out = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"',
        '     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"',
        '     xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"',
        f'     width="{size:g}mm" height="{size:g}mm" viewBox="0 0 {size:g} {size:g}" version="1.1">',
        f'  <!-- 1 mm = 1 m, north up, centre lon/lat {frame.center_lon:.7f},{frame.center_lat:.7f}, '
        f'UTM EPSG:{frame.epsg} -->',
    ]
    if overlay:
        out += [
            '  <g inkscape:groupmode="layer" id="layer_image" inkscape:label="image" sodipodi:insensitive="true">',
            f'    <image xlink:href="{escape(overlay)}" x="0" y="0" width="{size:g}" height="{size:g}" '
            'preserveAspectRatio="none"/>',
            '  </g>',
        ]
    for hole in sorted(by_hole):
        label = f"Hole{hole:02d}"
        out.append(f'  <g inkscape:groupmode="layer" id="layer_{label}" inkscape:label="{label}">')
        out += by_hole[hole]
        out.append('  </g>')
    out.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    return path
