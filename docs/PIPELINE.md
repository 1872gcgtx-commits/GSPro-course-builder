# Building a local course for GSPro: full process, tools and access

## 1. How a GSPro course actually gets built (current state)

GSPro does not take a Unity 6 project. Courses are produced with the community
**OPCD** toolchain, and the `.gspcrse` file GSPro loads is built inside the OPCD
Unity project:

| Stage | Tool | Version (as documented by OPCD users, Oct 2026) |
|---|---|---|
| Terrain + imagery prep | QGIS / this pipeline | any |
| Layout trace | Inkscape (SVG, one layer per hole) | 1.x |
| SVG → meshes ("Clender" job) | OPCD web app, app.opcdgolf.com | V4 |
| Conform meshes to terrain, cut bunkers/water, export FBX | Blender + OPCD add-on | **4.2.2+ LTS** |
| Materials, grass, trees, lighting, **Build Course → .gspcrse** | Unity + OPCD V4 BaseProject | **Unity 2018.2.8f1** |
| Tees, pins, hazards, OB, scorecard | GreenKeeper (OPCD) | V4 |
| Play/test | GSPro | current |

Key consequences:

* **The final build has to run in Unity 2018.2.8f1 with the OPCD base project,
  not Unity 6.** OPCD's Build Course step packages and encrypts the course
  (`AesDecrypt.dll`, Windows only) into the format GSPro reads. A course built
  in Unity 6 will not load in GSPro.
* **Unity 6 is useful as a preview and design studio.** This repo's Unity 6
  package imports the real terrain, surfaces and holes so you can walk the
  course, move tees and pins, and check yardages before the OPCD build.
* **We do not reverse-engineer or bypass GSPro's/OPCD's course encryption.**
  That would break their terms. Where a tool needs access we don't have, this
  repo builds our own equivalent from public data: the terrain, layout, hole
  data and OPCD-ready inputs below. The packaging step stays on the official
  tools. If GSPro or OPCD move to a newer Unity, change only the last stage.

Re-check the versions above in the OPCD Discord before each project. The Unity
and Blender versions have changed before.

## 2. Access checklist

| Need | How to get it | Cost/key |
|---|---|---|
| Elevation (bare-earth LiDAR DEM) | USGS 3DEP, `elevation.nationalmap.gov`; raw .laz from `apps.nationalmap.gov/downloader` | free, public domain, no key |
| Aerial imagery | USDA NAIP, `gis.apfo.usda.gov` | free, public domain, no key |
| Course layout | OpenStreetMap via Overpass, `overpass-api.de`; trace missing holes yourself | free, ODbL attribution |
| OPCD tools (Clender, Blender add-on, BaseProject, GreenKeeper) | Join the OPCD community/Discord and create an app.opcdgolf.com account | per OPCD's terms |
| GSPro | gsprogolf.com licence, for testing | paid |
| Unity 2018.2.8f1 + Unity 6 | Unity Hub → Install Editor → Archive (2018.2.8f1) | free Personal |
| Blender 4.2 LTS, Inkscape, QGIS | official sites | free |
| Windows PC | required by the OPCD Build Course step and GSPro | — |
| Permission from the course | Needed if you plan to **share** the course publicly. Private use of public data needs none, but most designers ask the club first. | — |

Outside the US, swap the sources. `[terrain] source = "file"` takes any DEM
GeoTIFF: Canada NRCan HRDEM, UK EA LiDAR, or state LiDAR portals.

## 3. What this repo automates

```
course.toml ──► fetch ──► data/dem.npy, overlay.jpg, osm.geojson
                         (+ your features.geojson edits)
            ──► build ──► output/
                 terrain/heightmap_<N>.raw   16-bit LE, row 0 = south (Flip Vertically OFF)
                 terrain/terrain.json        size, min/max elevation, height range
                 terrain/terrain.obj         2 m grid mesh, Y-up (Blender / OPCD conform)
                 surfaces/splat_control0/1   Unity splat maps (rough, trees, fairway, tee,
                                             green, bunker, water, cartpath)
                 <slug>_opcd.svg             Inkscape layers Hole01..HoleNN, Hole99, image
                 overlay.jpg                 aerial for tracing and the SVG image layer
                 holes.json, greenkeeper.csv tees per tee-set, pins, aim points, yards, par
                 build_report.json           warnings (missing greens, estimated par, ...)
```

Then in Unity 6: **GSPro Builder ▸ Import Course Output…** loads the course.
Edit markers, then use **GSPro Builder ▸ Export Hole Markers…**.

## 4. Step-by-step for a real local course

1. **Scaffold**
   `python -m gspro_builder new ../courses/my-club --lat 41.12345 --lon -81.54321 --size 2000`
   Pick the centre of the property and a size that covers every hole plus about 100 m.
2. **Fetch** with `python -m gspro_builder fetch ../courses/my-club`. This downloads the
   1 m DEM, NAIP imagery and the OSM golf features.
3. **Fix the layout.** Open `data/osm.geojson` and `data/overlay.jpg` in QGIS.
   Trace whatever is missing into `features.geojson` using the OSM tags:
   `golf=hole` (a line from tee to green, with `ref` and `par`), `tee`, `fairway`,
   `green`, `bunker`, `water_hazard`, `rough`, `cartpath`, `pin`. To replace an OSM
   shape, add `"replaces": "way/123"` to your feature. Set `[layout] source = "geojson"`
   to ignore OSM entirely.
4. **Build** with `python -m gspro_builder build ../courses/my-club`. Read the
   `build_report.json` warnings.
5. **Preview in Unity 6.** Add the package (Package Manager ▸ Add from disk ▸
   `unity/com.local.gspro-course-builder/package.json`), import the output, walk
   the holes, adjust tees, pins and aim points, then export.
6. **OPCD/GSPro build** (Windows):
   1. Open `<slug>_opcd.svg` in Inkscape and confirm the layers and colours match the
      current OPCD style guide. Upload it as an SVG job in the OPCD app.
   2. In Blender 4.2.2+ with the OPCD tools, open the returned `.blend`, import
      `terrain.obj` (or the OPCD terrain package), conform, run the bunker and water
      tools, then export FBX.
   3. In Unity 2018.2.8f1 with the OPCD V4 BaseProject: import the terrain from
      `heightmap_<N>.raw` (16-bit, Windows byte order, resolution N, size from
      terrain.json, Flip Vertically off). Import the FBX at (0,0,0), set mesh
      materials, add grass, plant trees, then Build Course.
   4. In GreenKeeper, enter the tees, pins, aim points and par from `greenkeeper.csv`
      or `holes_edited.json`. Set hazards and OB, then build and test in GSPro.

## 5. Known gaps / roadmap

* Trees aren't extracted yet. The next step is LiDAR canopy-height detection from .laz
  (PDAL) into `trees.csv` for OPCD's Arborist.
* There is no classification of surfaces from imagery yet. The layout comes from OSM
  plus hand tracing.
* Inner terrain only. A low-res outer/backdrop terrain is still to do.
* The OPCD SVG colour palette follows the community `osm2svg4opcd` styles. Verify it
  against the current OPCD guide and override it under `[svg.colors]` if needed.
* The RAW orientation convention (south-first) is unit-tested in this repo. Still
  confirm it against a known landmark on the first real course.
