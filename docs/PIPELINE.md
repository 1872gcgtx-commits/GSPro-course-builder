# Building a local course for GSPro: full process, tools and access

## 1. Where GSPro course building is now (Nov 2026)

GSPro has moved to **Unity 6** and supports all three render pipelines (Built-in,
URP, HDRP). OPCD V4 courses are supported, and the OPCD V4 tools for Unity 6 are
in beta and rolling out to everyone. The OPCD pipeline still does the final step:

| Stage | Tool | Notes |
|---|---|---|
| LiDAR → graded terrain, surfaces, trees, holes | **this repo** (`pipeline/`) | automated |
| Layout trace / check | Inkscape or QGIS | OPCD SVG = one layer per hole |
| SVG → course meshes ("Clender" job) | OPCD web app (app.opcdgolf.com) | or use our `course_surfaces.obj` |
| Conform / cut bunkers & water / FBX | Blender 4.2 LTS + OPCD add-on | optional now that we conform meshes ourselves |
| Materials, grass, trees, lighting, **Build Course** | **Unity 6** + OPCD V4 base project (beta) | our Unity package runs inside this project |
| Tees, pins, hazards, OB, scorecard | GreenKeeper (OPCD V4) | fed from `holes.json` / `greenkeeper.csv` |
| Play/test | GSPro (opt into beta for V4 courses) | |

Older guides that call for Unity 2018.2/2018.4 describe the pre-Unity-6 toolchain.
Use the Unity 6 editor version that the OPCD V4 base project names; check the OPCD
Discord, since the beta is still moving.

**What stays on the official tools:** Build Course packages and encrypts the course
into the format GSPro loads. We don't reverse-engineer or bypass that, because it
would break GSPro's and OPCD's terms. Everything before that step is built here from
public data, so the OPCD project only needs materials and a final build.

## 2. Access checklist

| Need | How to get it | Cost / key |
|---|---|---|
| LiDAR point cloud | USGS 3DEP via the public EPT bucket on AWS (`usgs-lidar-public`), streamed automatically | free, public domain, no key |
| …or your own .laz | apps.nationalmap.gov/downloader, or a state LiDAR portal | free |
| Aerial imagery | USDA NAIP (`gis.apfo.usda.gov`) | free, public domain |
| Course layout | OpenStreetMap via Overpass; trace the gaps yourself | free, ODbL attribution |
| OPCD V4 tools (Unity 6 base project, GreenKeeper, Clender) | OPCD community/Discord, app.opcdgolf.com account, beta access | per OPCD terms |
| GSPro (+ beta opt-in) | gsprogolf.com | paid |
| Unity 6 | Unity Hub (use the exact 6000.x version OPCD names) | free Personal |
| Blender 4.2 LTS, Inkscape, QGIS | official sites | free |
| Windows PC, 32 GB RAM recommended | Build Course and GSPro are Windows-only | — |
| Club permission | Needed if you plan to share the course publicly | — |

Network hosts the pipeline uses: `s3-us-west-2.amazonaws.com` (LiDAR),
`raw.githubusercontent.com` (LiDAR coverage index), `elevation.nationalmap.gov`,
`gis.apfo.usda.gov`, `overpass-api.de`.

## 3. How each course element is produced

### Elevation
* The USGS 3DEP **point cloud** is used, not just the 1 m DEM. Ground returns
  (class 2) are gridded at **0.5 m**. Gaps under buildings, water and dense canopy
  are filled from the nearest ground and smoothed.
* Any LiDAR CRS works, including State Plane in US survey feet. Vertical units
  are read from the file, and you can override them with `lidar_z_units`.
* The heightmap is resampled with a **cubic spline**, so there are no facets
  between samples. A 4097 heightmap over 2000 m gives 0.49 m spacing.
* `build_report.json` warns when the DEM is coarser than 0.75 m.

### Grading (what makes it play like a golf course)
| Feature | Treatment | Defaults |
|---|---|---|
| Tee boxes | best-fit plane, cross-slope clamped, feathered into the surround | ≤ 1.5 %, 3 m feather |
| Greens | noise-removing smoothing that keeps the real contours; slope QA | σ 0.6 m; warns if > 25 % of a green is over 3 % |
| Fairways | light de-noise, feathered into rough | σ 0.5 m |
| Bunkers | `auto`: keep the LiDAR bowl if it's there, else carve. Carved bunkers get a ramped face and a lip on the **green-facing** side | 0.6 m deep, 2 m ramp, 0.25 m lip |
| Water | flat surface 0.3 m below the lowest 10 % of the shoreline; bed slopes to 1.8 m; banks raised so water can't spill outside the outline | |

Every operation blends into the surrounding ground, and the tests check for cliffs
and seams. Per-feature results go in the report: tee slope before and after, bunker
depth and mode, water level, and green slope statistics.

### Surfaces (fairway to rough)
Ten terrain layers: rough, trees/native, **first cut**, fairway, **fringe**, tee, green,
bunker, water, cart path. The first cut (2 m) wraps every fairway and the fringe
(1 m) rings every green. Mown edges blend crisply (0.15 to 0.3 m) and the
rough-to-native edge blends softly (1 to 2 m). Widths are set in `[surfaces]`.

### Course surface meshes
`course_surfaces.obj` contains one mesh per green, tee, fairway and bunker, named
`Hole##_Green`, `Hole##_Tee`, `Hole##_Fairway`, `Hole##_Bunker`, and so on, plus
`Hole##_WaterPlane`. The names follow the OPCD material-token convention. Meshes are
triangulated at 0.5 m (1 m on fairways) and draped on the graded terrain. They import
straight into Blender (north = +Y) and Unity 6. The Unity importer handles the OBJ
axis conversion.

### Foliage
Trees are detected individually from the LiDAR **canopy height model** (first returns
minus ground). Tree tops come from height-adaptive local maxima, and the crown radius
is measured in 8 directions. Each tree is classed as **deciduous or conifer** from its
crown shape. Trees are rejected on greens, tees, fairways, bunkers, water and roofs.
Without LiDAR canopy, trees are scattered at natural spacing inside the layout's
`trees` areas. The output is `trees.csv`. The Unity importer plants them as terrain
trees, using your prefabs or placeholders.

### Holes
* Tees per tee set, from back to forward tee pads.
* Aim points come from the hole line's dog-leg vertices.
* Yardage is measured along the line of play.
* Pins: front, middle and back positions are chosen on the flattest part of each green,
  at least 3.5 m from the edge and on slopes of 2.5 % or less.

## 4. Step-by-step for a real local course

1. `python -m gspro_builder new ../courses/my-club --lat 41.12345 --lon -81.54321 --size 2000`
2. `python -m gspro_builder fetch ../courses/my-club` downloads the LiDAR, NAIP and OSM data.
3. Open `data/osm.geojson` over `data/overlay.jpg` in QGIS. Trace whatever is missing
   into `features.geojson`: `golf=hole` lines (tee to green, with `ref` and `par`),
   `tee`, `fairway`, `green`, `bunker`, `water_hazard`, `rough`, `trees`, `cartpath`,
   `pin`. To override an OSM shape, add `"replaces": "way/123"` to your feature.
4. `python -m gspro_builder build ../courses/my-club`, then read every warning in
   `build_report.json`. The common fixes are a missing tee polygon, LiDAR water that
   isn't in the layout, or a green that's too steep (usually the outline is shifted
   off the real green).
5. **Unity 6** (inside the OPCD V4 base project, or a scratch project for preview):
   Package Manager ▸ Add from disk ▸ `unity/com.local.gspro-course-builder/package.json`.
   Then **GSPro Builder ▸ Import Course Output…** with the trees, water and surface
   meshes options on. Assign real grass and sand textures to the generated
   `.terrainlayer` assets, your tree prefabs, and the water shader. Adjust tees and
   pins, then use **Export Hole Markers…**.
6. In the OPCD project, set mesh materials, add grass detail, set lighting, run
   **Build Course**, then use GreenKeeper (tees, pins, aim points, hazards, OB,
   scorecard) and test in GSPro (beta).

## 5. Known gaps / roadmap

* No buildings or structures yet. The LiDAR building mask exists but isn't turned
  into meshes.
* No outer/backdrop terrain beyond the main tile.
* Surface outlines aren't classified from imagery. The layout comes from OSM plus
  your traces.
* The OPCD SVG colours and the OPCD V4 Unity 6 material-token names should be checked
  against the current beta docs. Colours can be overridden under `[svg.colors]`.
* The RAW orientation (south-first) and the OBJ axis conversions are unit-tested.
  Confirm them against a known landmark on your first real course.
