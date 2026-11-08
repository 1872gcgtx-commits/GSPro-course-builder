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

### Tree species and understory
The LiDAR tree list is brought up to the date of the aerial photo before species are
assigned. A canopy classifier is trained on the 4-band aerial, using the LiDAR crowns as
examples. With it, trees whose crown is gone from the aerial are dropped (most are lots
cleared for new streets). Canopy the LiDAR never saw becomes new trees, with height
estimated from crown size.

Each tree then gets a species from three kinds of evidence. Crown shape and colour are
compared with the course's own trees, not fixed thresholds:

- **Crown shape (LiDAR):** height, how wide versus tall, how peaked or flat-topped, how
  see-through.
- **Colour (aerial):** brightness, green and blue tint, NDVI.
- **Setting:** distance to water, height above it, surrounding canopy cover.

| Species | Where / what |
|---|---|
| bald_cypress | creek bank, ≥ 9 m, narrow and peaked |
| pecan | creek bottom, ≥ 12 m |
| mesquite | open ground, < 7.5 m, light and airy |
| ashe_juniper ("cedar") | narrow or peaked, dark blue-green; never in the creek bottom |
| live_oak | broad, flat-topped, dark and dense |
| cedar_elm | upright, mid green; favoured in the creek bottom |
| hackberry | irregular, see-through, mid green |

`trees.csv` keeps its original 7 columns and adds these: `species`, `species_conf`,
`crown_base_m` (where the branches start), `size_class` (young / mature / old within the
species), `tint_r/g/b` (the crown's colour in the aerial relative to its species) and
`rotation_deg`.

`shrubs.csv` holds the understory: vegetation 0.6–3 m tall, as `brush` or
`juniper_shrub`. `tree_species.json` lists the species and their counts.

These are rules, not field surveys. Expect the overall mix and the creek, upland and
open-ground pattern to be right, and individual trees to be sometimes wrong.

**Tree models:** `output/tree_models/` holds a procedural model for each species, in
3 variants (`<species>_<n>.obj` + `.mtl`, `leaves_<species>.png`, `bark.png`). Each model
is grown from a branching skeleton with that species' habit:

- live oak: short trunk, heavy limbs spreading near-horizontal
- Ashe juniper: several stems, a ragged cone of scale foliage
- cedar elm: vase-shaped, upright
- pecan: tall clear trunk, open crown
- bald cypress: a single straight spire
- mesquite: crooked multi-stem, flat airy crown

Each model is about 2k triangles: bark tubes plus crossed leaf cards with an alpha leaf
texture drawn for the species. Models are 10 m tall, and each tree is scaled to its own
height and crown width.

**Unity 6:** the importer has one prefab slot per species. Empty slots use the generated
models: copied into the project, combined into one mesh with bark and cutout leaf
materials (URP, HDRP or Built-in). Next comes your deciduous/conifer prefab, then a
simple placeholder. For the best look, drop a SpeedTree or OPCD tree pack model into a
species slot.
Each tree's tint becomes its instance colour, which drives the colour variation in
SpeedTree and terrain-tree shaders. Rotation comes from the CSV. Shrubs are planted as
small trees.

### Homes around the course
Footprints come from **Microsoft Global ML Building Footprints** (open data, ODbL;
`fetch --only buildings`), which include many houses built after the LiDAR was flown.
The footprint data still misses whole streets. To fill those gaps, a random forest is
trained on the 4-band aerial, using the known footprints as roof examples. Roofs it
finds that no footprint covers are added if they are house-shaped: at least 60 m², boxy,
and not long and narrow, which rules out roads, drives and walls. This also rejects
things simple rules mistake for roofs, such as pools and shadowed tree crowns.
Footprints are shifted onto the LiDAR first (best overlap with its building returns).
Each house gets:

- **Height:** the LiDAR roof height where the LiDAR saw that house, otherwise the
  footprint's own estimate, otherwise a single-storey default.
- **Roof:** a hipped roof at a 5:12 pitch, built from the footprint's inset outlines, so
  every roof plane is flat and hips and valleys meet exactly. The roof has a 0.5 m eave
  overhang, and large buildings get a flat-topped hip.
- **Walls:** from the lowest ground under the footprint up to the eave.
- **Finishes:** the roof finish is classed from the aerial photo (grey, charcoal, light
  metal, tan, brown, terracotta, green metal). The wall finish is drawn from the local
  palette: cream limestone, rustic stone, tan brick, red brick, white stucco.

Footprints on greens, fairways, tees, bunkers or water are dropped, and trees standing
inside a house lot are removed.

Each house is dressed for its street. Streets come from OpenStreetMap
(`fetch --only roads`). The wall facing the nearest street becomes the front:

- a front door with a concrete step;
- on most homes over 140 m², a 4.9 m garage door at one end of the front wall, with a
  5.5 m concrete driveway draped on the terrain out to the street;
- stone chimneys on about a third of the larger homes;
- on every house: a white fascia board along the eaves, a soffit under the overhang,
  and a stone foundation band.

Walls use a 6 m × 3 m texture: a single window and a pair, one storey per 3 m, so
two-storey homes get two rows of windows. Roof textures follow each roof face (shingle
courses run along the eave), as asphalt shingle or standing-seam metal. All textures are
drawn procedurally (`home_textures.py`), so there are no asset licences to worry about.

The build writes the following to `output/homes/`:

- `homes.obj` + `homes.mtl` + textures: one object per 250 m tile (`Homes_##_##`) for
  culling, with one sub-mesh per finish. Axes match `course_surfaces.obj`. About 200
  triangles per house.
- `homes.json`: one record per house (footprint, ground/eave/ridge heights, height
  source, finishes, front wall, garage, chimney).

**Unity 6:** the importer's **Homes** option copies `output/homes` into the project and
places it under the course root, turned like `course_surfaces.obj`. By hand:
1. Import `homes.obj`. Use Scale 1 and the same rotation as `course_surfaces.obj`. Set
   Normals to Calculate with a 30° smoothing angle, so roof planes stay crisp.
2. Remap the materials by name (`Home_Wall_*`, `Home_Roof_*`) to PBR materials. Use
   facade textures with windows for the walls, and shingle or standing-seam metal for
   the roofs.
3. Mark the tiles Static, enable occlusion culling, and add an LOD Group if needed.
   The whole neighbourhood is about 120k triangles.

### Holes
* Tees per tee set, from back to forward tee pads.
* Aim points come from the hole line's dog-leg vertices.
* Yardage is measured along the line of play.
* Pins: front, middle and back positions are chosen on the flattest part of each green,
  at least 3.5 m from the edge and on slopes of 2.5 % or less.

### 27-hole facilities (three nines, selectable routings)
* Trace every hole tee → aim points → green and tag it with its **nine** and **hole number**.
  The routing tracer page does this: imagery and LiDAR relief, with saves to a shared
  database. Then run `python -m gspro_builder import-routing <course> <export>`.
* All 27 holes are built into **one scene**, numbered nine 1 → 1-9, nine 2 → 10-18 and
  nine 3 → 19-27. Those numbers are used for the SVG layers `Hole01..Hole27` and the
  mesh names.
* Each routing gets its own playable definition, renumbered 1-18:
  `output/routings/course-1-2/`, `course-2-3/` and `course-1-3/`, each with
  `holes.json` and `greenkeeper.csv`. In OPCD, build one GSPro course per routing from
  the same scene, so players pick *Course 1-2*, *Course 2-3* or *Course 1-3*.
* Nine names and custom routings go in `course.toml` under `[nines]` and `[[routing]]`.

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
