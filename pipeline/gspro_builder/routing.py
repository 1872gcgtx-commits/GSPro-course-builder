"""Multi-nine facilities (e.g. 27 holes = three nines) and their 18-hole routings.

All holes live in ONE scene (one terrain, one set of meshes). Each hole gets a
course-wide number: nine 1 -> 1-9, nine 2 -> 10-18, nine 3 -> 19-27, which is
what the SVG layers (Hole01..Hole27) and mesh names (Hole19_Green ...) use.

Each routing (front nine + back nine) is then written as its own playable
course definition, renumbered 1-18:
  output/routings/<course-1-2>/holes.json, greenkeeper.csv
In OPCD/GreenKeeper each routing is built as its own course from the same scene,
so players pick "Course 1-2", "Course 2-3" or "Course 1-3" in GSPro.

course.toml:
  [nines]
  1 = "Creek 1"
  2 = "Creek 2"
  3 = "Creek 3"

  [[routing]]            # optional; default = every pair in order (1-2, 2-3, 1-3)
  name = "Course 1-2"
  front = 1
  back = 2
"""
from __future__ import annotations

import copy
import json
import re
from itertools import combinations
from pathlib import Path

from .geo import CourseFrame


def global_number(nine: int, hole: int) -> int:
    return (nine - 1) * 9 + hole


def default_routings(nines: list[int]) -> list[dict]:
    return [{"name": f"Course {a}-{b}", "front": a, "back": b} for a, b in combinations(sorted(nines), 2)]


def tracer_to_geojson(docs: list[dict], frame: CourseFrame) -> dict:
    """Hole lines traced in the routing page (local metres) -> golf=hole GeoJSON."""
    feats = []
    for d in sorted(docs, key=lambda d: (d["nine"], d["hole"])):
        pts = d.get("points") or []
        if len(pts) < 2:
            continue
        coords = [list(frame.to_lonlat(p["x"], p["z"])) for p in pts]
        n = global_number(int(d["nine"]), int(d["hole"]))
        feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                      "properties": {"golf": "hole", "ref": str(n), "nine": int(d["nine"]),
                                     "nine_hole": int(d["hole"]), "par": str(d.get("par", "")),
                                     "name": f"Nine {d['nine']} hole {d['hole']}", "source": "routing-tracer"}})
    return {"type": "FeatureCollection", "features": feats}


def merge_hole_lines(features_path: Path, hole_fc: dict) -> int:
    """Replace the golf=hole lines in features.geojson with the traced ones; keep everything else."""
    existing = json.loads(features_path.read_text()) if features_path.exists() else {"type": "FeatureCollection", "features": []}
    keep = [f for f in existing["features"] if (f.get("properties") or {}).get("golf") != "hole"]
    existing["features"] = keep + hole_fc["features"]
    features_path.write_text(json.dumps(existing, indent=1))
    return len(hole_fc["features"])


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def write_routings(holes: list[dict], routings: list[dict], nine_names: dict, out_dir: Path,
                   course_name: str) -> list[dict]:
    """Split the 27-hole hole list into one 18-hole definition per routing."""
    from .holes import write_holes
    by_num = {h["number"]: h for h in holes}
    summary = []
    for r in routings:
        seq = []
        for leg, nine in (("front", r["front"]), ("back", r["back"])):
            for k in range(1, 10):
                src = by_num.get(global_number(nine, k))
                if src is None:
                    continue
                h = copy.deepcopy(src)
                h["scene_hole"] = src["number"]
                h["number"] = k if leg == "front" else 9 + k
                h["name"] = f"{nine_names.get(nine, f'Nine {nine}')} {k}"
                seq.append(h)
        d = out_dir / "routings" / _slug(r["name"])
        write_holes(seq, f"{course_name} - {r['name']}", d)
        summary.append({"name": r["name"], "front": nine_names.get(r["front"], r["front"]),
                        "back": nine_names.get(r["back"], r["back"]), "holes": len(seq),
                        "par": sum(h["par"] for h in seq), "dir": str(d.relative_to(out_dir)),
                        "complete": len(seq) == 18})
    (out_dir / "routings" / "routings.json").write_text(json.dumps(summary, indent=2))
    return summary


TEE_ORDER = ["Black", "Blue", "White", "Gold", "Red", "Green"]


def reviewed_shapes(draft: dict, edits: list[dict]) -> list[dict]:
    """Draft shapes from the review page overlaid with the owner's saved edits
    (an edit replaces the draft shape with the same id; `deleted` removes it)."""
    shapes = {s["id"]: s for s in draft.get("shapes", [])}
    # shapes the owner edited from an earlier draft win over the new draft for that hole + type
    edited_ids = {e.get("id") for e in edits}
    own = {(e.get("nine"), e.get("hole"), e.get("kind")) for e in edits
           if not e.get("deleted") and e.get("id") and e["id"] not in shapes and not str(e["id"]).startswith("u")}
    shapes = {k: v for k, v in shapes.items()
              if k in edited_ids or (v.get("nine"), v.get("hole"), v.get("kind")) not in own}
    for e in edits:
        sid = e.get("id")
        if not sid:
            continue
        if e.get("deleted"):
            shapes.pop(sid, None)
        else:
            shapes[sid] = e
    return list(shapes.values())


def surfaces_to_geojson(shapes: list[dict], frame: CourseFrame) -> dict:
    feats = []
    for s in shapes:
        ring = [list(frame.to_lonlat(x, z)) for x, z in s["ring"]]
        if len(ring) < 3:
            continue
        ring.append(ring[0])
        props = {"golf": s["kind"], "hole": str(global_number(int(s["nine"]), int(s["hole"]))),
                 "nine": int(s["nine"]), "nine_hole": int(s["hole"]), "source": "review",
                 "method": s.get("method"), "edited": bool(s.get("edited")), "review_id": s["id"]}
        if s.get("tee_set"):
            props["tee_set"] = s["tee_set"]
        if s.get("profile"):                       # NHD creek: water surface along the thalweg
            props["profile"] = s["profile"]
            props["water"] = "stream"
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": props})
    return {"type": "FeatureCollection", "features": feats}


def merge_surfaces(features_path: Path, fc: dict) -> int:
    """Replace previously imported review/autodraft surfaces in features.geojson."""
    existing = json.loads(features_path.read_text()) if features_path.exists() else {"type": "FeatureCollection", "features": []}
    keep = [f for f in existing["features"] if (f.get("properties") or {}).get("source") not in ("review", "autodraft")]
    existing["features"] = keep + fc["features"]
    features_path.write_text(json.dumps(existing, indent=1))
    return len(fc["features"])
