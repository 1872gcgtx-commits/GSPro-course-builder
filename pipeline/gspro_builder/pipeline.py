"""Pipeline orchestration: fetch -> build.

fetch : downloads elevation (LiDAR), imagery and OSM layout into courses/<slug>/data/
build : turns data/ into courses/<slug>/output/ (all offline, re-runnable)

output/
  terrain/heightmap_<N>.raw     graded terrain, 16-bit LE, row 0 = south
  terrain/terrain.json          size, elevations, conventions
  terrain/terrain.obj           graded terrain mesh (Blender/OPCD)
  terrain/*_preview.png         hillshade previews (raw vs graded)
  surfaces/splat_control*.png   10 terrain layers incl. fringe and first cut
  course_surfaces.obj           Hole##_Green/Tee/Fairway/Bunker + water planes, conformed
  water.json                    water bodies (level + mesh) for the Unity importer
  trees.csv                     individual trees (LiDAR) or scattered planting
  homes/homes.obj, .mtl, .json  houses around the course (footprints + LiDAR heights, hipped roofs)
  tree_models/<species>_<n>.obj procedural tree models per species (bark + alpha leaf cards)
  overlay.jpg, <slug>_opcd.svg  tracing / OPCD SVG job input
  holes.json, greenkeeper.csv   tees, pins (front/middle/back), aim points, yardage
  build_report.json             grading + QA report and warnings
"""
from __future__ import annotations

import json
import shutil

import numpy as np

from .config import CourseConfig
from .course_meshes import build_meshes, write_obj as write_surface_obj, write_water_json
from .foliage import FoliageConfig, detect_trees, scatter_trees, write_shrubs_csv, write_trees_csv
from .grading import GradingConfig, grade
from .holes import build_holes, write_holes
from .homes import HomesConfig, build_homes, homes_mask_filter, write_homes
from .species import SPECIES, SpeciesConfig, assign_species, detect_shrubs, refine_trees
from .layout import load_layout
from .mesh import write_obj
from .raster import rasterize_instances
from .surfaces import SurfaceConfig, write_surfaces
from .svg_opcd import write_svg
from .terrain import DEM, _hillshade_png, write_heightmap


def log(msg: str) -> None:
    print(f"[gspro-builder] {msg}", flush=True)


def fetch(cfg: CourseConfig, what: tuple[str, ...] = ("dem", "imagery", "layout", "buildings")) -> None:
    d = cfg.data_dir
    d.mkdir(parents=True, exist_ok=True)
    if "dem" in what:
        if cfg.dem_source in ("lidar_ept", "lidar_files"):
            from .sources.lidar import read_ept, read_las_files, save_lidar
            if cfg.dem_source == "lidar_ept":
                log(f"streaming USGS 3DEP LiDAR point cloud ({cfg.frame.size_m:g} m tile @ {cfg.dem_res_m:g} m)")
                prod = read_ept(cfg.frame, cfg.dem_res_m, cfg.lidar_ept_url, log=log)
            else:
                log(f"reading LiDAR files {cfg.lidar_files}")
                prod = read_las_files(cfg.lidar_files, cfg.root, cfg.frame, cfg.dem_res_m,
                                      cfg.lidar_crs, cfg.lidar_z_units)
            save_lidar(prod, d)
            log(f"  ground coverage {float(prod['ground_coverage']) * 100:.0f}% of cells")
        else:
            from .sources.elevation import fetch_usgs_3dep, fill_nodata, read_dem_file
            if cfg.dem_source == "file":
                log(f"reading DEM {cfg.dem_file}")
                dem = read_dem_file(str(cfg.root / cfg.dem_file), cfg.frame, cfg.dem_res_m)
            else:
                log(f"downloading USGS 3DEP DEM ({cfg.frame.size_m:g} m @ {cfg.dem_res_m:g} m)")
                dem = fetch_usgs_3dep(cfg.frame, cfg.dem_res_m)
            np.save(d / "dem.npy", fill_nodata(dem))
            for stale in ("chm.npy", "lidar_masks.npz"):
                (d / stale).unlink(missing_ok=True)
    if "imagery" in what and cfg.imagery_source == "naip":
        from PIL import Image
        from .sources.imagery import fetch_naip
        log("downloading USDA NAIP imagery")
        Image.fromarray(fetch_naip(cfg.frame, cfg.imagery_res_m)).save(d / "overlay.jpg", quality=92)
    if "layout" in what and cfg.layout_source == "osm":
        from .sources.osm import fetch_osm
        log("downloading OpenStreetMap golf features")
        gj = fetch_osm(cfg.frame)
        (d / "osm.geojson").write_text(json.dumps(gj))
        log(f"  {len(gj['features'])} features")
    if "buildings" in what and cfg.homes.get("enabled", True):
        from .sources.buildings import fetch_ms_buildings
        log("downloading building footprints (Microsoft Global ML Building Footprints)")
        try:
            gj = fetch_ms_buildings(cfg.frame, log=log)
            (d / "buildings").mkdir(exist_ok=True)
            (d / "buildings" / "ms_buildings.geojson").write_text(json.dumps(gj))
            log(f"  {len(gj['features'])} footprints")
        except Exception as e:   # homes are optional scenery
            log(f"  building footprints unavailable ({e}); homes will be skipped")


def _layout_geojson(cfg: CourseConfig) -> dict:
    feats = []
    osm = cfg.data_dir / "osm.geojson"
    if cfg.layout_source == "osm" and osm.exists():
        feats += json.loads(osm.read_text())["features"]
    local = cfg.root / cfg.layout_file
    if local.exists():   # hand-traced features add to / override OSM
        extra = json.loads(local.read_text())["features"]
        drop = {f["properties"].get("replaces") for f in extra if f.get("properties")} - {None}
        feats = [f for f in feats if f["properties"].get("osm_id") not in drop] + extra
    if not feats:
        raise FileNotFoundError(f"No layout: run `fetch` (OSM) or create {local}")
    return {"type": "FeatureCollection", "features": feats}


def build(cfg: CourseConfig) -> dict:
    out = cfg.out_dir
    (out / "terrain").mkdir(parents=True, exist_ok=True)
    data = cfg.data_dir
    if not (data / "dem.npy").exists():
        raise FileNotFoundError(f"{data / 'dem.npy'} missing - run `fetch` first")
    raw_dem = DEM(np.load(data / "dem.npy"), cfg.frame)
    chm = np.load(data / "chm.npy") if (data / "chm.npy").exists() else None
    masks = np.load(data / "lidar_masks.npz") if (data / "lidar_masks.npz").exists() else None
    warnings: list[str] = []
    if raw_dem.res_m > 0.75:
        warnings.append(f"DEM resolution {raw_dem.res_m:.2f} m - greens/bunkers need <= 0.5 m (use LiDAR source)")

    layout = load_layout(_layout_geojson(cfg), cfg.frame)
    log(f"layout: {len(layout.holes)} holes, {len(layout.surfaces)} surfaces")
    if any(cfg.layout_shift_m):
        from .layout import shift_traced
        n = shift_traced(layout, *cfg.layout_shift_m)
        log(f"layout: {n} traced shapes moved {cfg.layout_shift_m[0]:+.1f} m E, {cfg.layout_shift_m[1]:+.1f} m N onto the LiDAR")

    log("grading: fairways, greens, tees, bunkers, water")
    g = grade(raw_dem, layout, GradingConfig.from_dict(cfg.grading))
    dem = g.dem

    log(f"terrain -> {cfg.heightmap_res}x{cfg.heightmap_res} RAW "
        f"({cfg.frame.size_m / (cfg.heightmap_res - 1):.2f} m spacing)")
    tinfo = write_heightmap(dem, cfg.heightmap_res, out / "terrain")
    base = tinfo.min_elev_m
    _hillshade_png(raw_dem.data, raw_dem.res_m, out / "terrain" / "raw_preview.png")
    write_obj(dem, out / "terrain" / "terrain.obj", base)

    smeta = write_surfaces(layout, cfg.frame, cfg.splat_res, out / "surfaces",
                           SurfaceConfig.from_dict(cfg.surfaces))

    log("course surface meshes")
    objs = build_meshes(layout, dem, base, g.water_bodies, cfg.mesh_offset_m, g.water_level)
    write_surface_obj(objs, out / "course_surfaces.obj", cfg.frame.size_m)
    write_water_json(objs, g.water_bodies, base, out / "water.json")

    fcfg_d = dict(cfg.foliage)
    source = fcfg_d.pop("source", "auto")
    fcfg = FoliageConfig.from_dict(fcfg_d)
    trees: list[dict] = []
    if fcfg.enabled and source != "none":
        if source in ("auto", "lidar") and chm is not None:
            log("detecting trees from LiDAR canopy")
            building = masks["building"] if masks is not None else None
            trees = detect_trees(chm, layout, dem, base, fcfg, building)
        elif source in ("auto", "scatter"):
            if source == "auto":
                warnings.append("no LiDAR canopy - trees scattered inside `trees` polygons")
            trees = scatter_trees(layout, dem, base, fcfg)
        else:
            warnings.append("foliage source 'lidar' requested but no LiDAR canopy (chm.npy)")
    if cfg.boundary:
        from .foliage import clip_to_boundary
        before = len(trees)
        ring = [cfg.frame.to_local(lon, lat) for lon, lat in cfg.boundary]
        trees = clip_to_boundary(trees, ring, fcfg.boundary_margin_m)
        log(f"trees: kept {len(trees)} of {before} inside the property outline (+{fcfg.boundary_margin_m:g} m)")
    # ---- trees up to the date of the aerial (gone / grown since the LiDAR), then species
    scfg = SpeciesConfig.from_dict(cfg.species)
    naip4 = np.load(data / "naip4.npy", mmap_mode="r") if (data / "naip4.npy").exists() else None
    naip4_meta = json.loads((data / "naip4.json").read_text()) if naip4 is not None else None
    tree_rep, shrubs = {}, []
    excl = wat = None
    ring_l = [cfg.frame.to_local(lon, lat) for lon, lat in cfg.boundary] if cfg.boundary else None
    if chm is not None:
        from .foliage import _exclusion
        from .raster import rasterize_instances
        excl = _exclusion(layout, cfg.frame, chm.shape[0], fcfg.keep_clear_m,
                          masks["building"] if masks is not None else None)
        wat = rasterize_instances([s_ for s_ in layout.surfaces if s_.kind == "water"], cfg.frame, chm.shape[0]) > 0
        if masks is not None:
            wat |= masks["water"]
        excl |= wat
    if scfg.enabled and chm is not None and trees and trees[0].get("source") == "lidar":
        trees, tree_rep = refine_trees(trees, chm, cfg.frame, layout, dem, base, naip4, naip4_meta,
                                       cfg.layout_shift_m, ring_l, fcfg.boundary_margin_m, excl, log=log)
    homes_rep = None
    hcfg = HomesConfig.from_dict(cfg.homes)
    fp_path = cfg.data_dir / "buildings" / "ms_buildings.geojson"
    if hcfg.enabled and fp_path.exists():
        from PIL import Image as _Image
        naip = np.asarray(_Image.open(data / "overlay.jpg")) if (data / "overlay.jpg").exists() else None
        ring = [cfg.frame.to_local(lon, lat) for lon, lat in cfg.boundary] if cfg.boundary else None
        homes, homes_rep = build_homes(json.loads(fp_path.read_text()), cfg.frame, dem, base, layout, hcfg, chm,
                                       masks["building"] if masks is not None else None, naip,
                                       cfg.layout_shift_m, ring, log=log,
                                       naip4=np.load(data / "naip4.npy", mmap_mode="r") if (data / "naip4.npy").exists() else None,
                                       naip4_meta=json.loads((data / "naip4.json").read_text()) if (data / "naip4.json").exists() else None)
        homes_rep.update(write_homes(homes, out / "homes", hcfg))
        before = len(trees)
        trees = homes_mask_filter(trees, homes)
        log(f"homes: {len(homes)} houses ({homes_rep['triangles']} triangles); "
            f"{before - len(trees)} trees removed from house lots")
    if scfg.enabled and chm is not None and trees:
        trees, counts = assign_species(trees, chm, cfg.frame, naip4, naip4_meta, cfg.layout_shift_m, wat, dem,
                                       seed=scfg.seed)
        tree_rep["species"] = counts
        log("trees by species: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
        if scfg.shrubs:
            sex = excl.copy()
            if ring_l:
                from .raster import rasterize_instances as _ri
                from .layout import Surface as _S
                zone_m = _ri([_S("x", [ring_l], None, {})], cfg.frame, chm.shape[0]) > 0
                from scipy import ndimage as _nd
                sex |= ~_nd.binary_dilation(zone_m, iterations=int(fcfg.boundary_margin_m / (cfg.frame.size_m / chm.shape[0])))
            shrubs = detect_shrubs(chm, cfg.frame, naip4, naip4_meta, cfg.layout_shift_m, sex, trees, dem, base, scfg)
            if hcfg.enabled and fp_path.exists():
                shrubs = homes_mask_filter(shrubs, homes)
            write_shrubs_csv(shrubs, out / "shrubs.csv")
            tree_rep["shrubs"] = len(shrubs)
            log(f"understory: {len(shrubs)} shrubs")
        from .treegen import write_models
        write_models(out / "tree_models")                     # procedural models per species for Unity
        (out / "tree_species.json").write_text(json.dumps(
            {"species": {k: {"name": v[1], "crown_base_share": v[0], "count": tree_rep["species"].get(k, 0)}
                         for k, v in SPECIES.items()},
             "note": "trees.csv `species` column; map each to a prefab (young/mature/old by size_class)"}, indent=1))
    write_trees_csv(trees, out / "trees.csv")
    log(f"trees: {len(trees)}")

    overlay = None
    if (data / "overlay.jpg").exists():
        shutil.copy(data / "overlay.jpg", out / "overlay.jpg")
        overlay = "overlay.jpg"
    write_svg(layout, cfg.frame, out / f"{cfg.slug}_opcd.svg", overlay, cfg.svg_colors)

    holes = build_holes(layout, dem, base, cfg.tee_sets)
    write_holes(holes, cfg.name, out)
    routing_summary = None
    nines = sorted({h["nine"] for h in holes if h.get("nine")})
    if nines:
        from .routing import default_routings, write_routings
        routes = cfg.routings or default_routings(nines)
        routing_summary = write_routings(holes, routes, cfg.nine_names, out, cfg.name)
        for r in routing_summary:
            log(f"routing {r['name']}: {r['front']} + {r['back']}, {r['holes']} holes, par {r['par']}")

    # ------------------------------------------------------------------ QA
    numbers = [h["number"] for h in holes]
    if len(numbers) not in (9, 18, 27, 36):
        warnings.append(f"{len(numbers)} holes found (expected 9, 18, 27 or 36)")
    for r in routing_summary or []:
        if not r["complete"]:
            warnings.append(f"{r['name']}: only {r['holes']} of 18 holes traced")
    for n in range(1, max(numbers, default=0) + 1):
        if n not in numbers:
            warnings.append(f"hole {n} missing a golf=hole line")
    for h in holes:
        if not any(s.kind == "green" and s.hole == h["number"] for s in layout.surfaces):
            warnings.append(f"hole {h['number']}: no green polygon")
        if not any(s.kind == "tee" and s.hole == h["number"] for s in layout.surfaces):
            warnings.append(f"hole {h['number']}: no tee polygon (tee pad not levelled)")
        if h["par_estimated"]:
            warnings.append(f"hole {h['number']}: par estimated from yardage ({h['par']})")
    gcfg = GradingConfig.from_dict(cfg.grading)
    for gr in g.report.get("greens", []):
        if gcfg.green_shape:
            # shaped greens are always puttable; flag ones the shaping had to flatten a lot
            # (outline probably runs onto a bank) or that still tilt hard overall
            if (gr.get("eased_toward_level") or 0) > 0.5:
                warnings.append(f"hole {gr['hole']} green: eased {gr['eased_toward_level']:.0%} toward level "
                                f"to be puttable - check the green outline sits on the putting surface")
            elif gr["tilt_pct"] > gcfg.green_max_slope * 100:
                warnings.append(f"hole {gr['hole']} green tilts {gr['tilt_pct']}% overall")
        elif gr["pct_area_over_limit"] > 25:
            warnings.append(f"hole {gr['hole']} green: {gr['pct_area_over_limit']}% steeper than "
                            f"{gcfg.green_slope_warn * 100:.0f}% - check LiDAR / layout alignment")
    if masks is not None:
        lw = masks["water"]
        layout_water = rasterize_instances([s for s in layout.surfaces if s.kind == "water"],
                                           cfg.frame, lw.shape[0]) > 0
        missing = lw & ~layout_water
        miss_m2 = float(missing.sum()) * (cfg.frame.size_m / lw.shape[0]) ** 2
        if miss_m2 > 200:
            warnings.append(f"LiDAR sees {miss_m2:.0f} m2 of water not in the layout - trace it as water_hazard")
    report = {"course": cfg.name, "terrain": tinfo.__dict__, "grading": g.report,
              "surfaces": smeta["layers"], "meshes": [o["name"] for o in objs],
              "trees": {"count": len(trees), **tree_rep,
                        "conifer": sum(t["type"] == "conifer" for t in trees),
                        "source": trees[0]["source"] if trees else None},
              "homes": homes_rep,
              "holes": len(holes), "par": sum(h["par"] for h in holes),
              "routings": routing_summary, "warnings": warnings}
    (out / "build_report.json").write_text(json.dumps(report, indent=2))
    for w in warnings:
        log(f"WARNING: {w}")
    log(f"done -> {out}")
    return report
