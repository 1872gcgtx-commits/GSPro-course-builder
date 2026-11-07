# GSPro Course Builder

Builds local golf courses for GSPro (Unity 6 / OPCD V4) from public data. The
pipeline streams the USGS 3DEP LiDAR point cloud, USDA NAIP imagery and the
OpenStreetMap layout (plus your own traces) and produces:

* **Elevation**: a 0.5 m LiDAR ground model, graded into a 16-bit Unity heightmap
  and an OBJ terrain mesh
* **Grading**: level tee pads, de-noised greens with their contours kept, carved or
  kept bunkers with lips, flat water with banks and a sloped bed, all feathered so
  there are no seams
* **Surfaces**: 10 terrain layers with fairway → first cut → rough and
  green → fringe transitions
* **Course meshes**: `Hole##_Green/Tee/Fairway/Bunker` and water planes, conformed to
  the graded terrain
* **Foliage**: individual trees (deciduous or conifer, height and crown size) from the
  LiDAR canopy
* **Holes**: tees per tee set, front/middle/back pins on the flattest parts of each
  green, aim points, yardage
* an OPCD-style Inkscape SVG and a QA report

The **Unity 6 editor package** imports all of this, including terrain, trees, water,
meshes and hole markers. It works in Built-in, URP or HDRP, inside the OPCD V4 Unity 6
project. See [docs/PIPELINE.md](docs/PIPELINE.md) for the full process, the tools and
the access checklist.

## Quick start

```bash
cd pipeline
pip install -r requirements.txt

# offline demo (synthetic 3-hole course), no downloads
python -m gspro_builder demo ../courses/demo

# a real course
python -m gspro_builder new   ../courses/my-club --lat 41.12345 --lon -81.54321 --size 2000
python -m gspro_builder fetch ../courses/my-club
#   ...trace/fix holes in courses/my-club/features.geojson...
python -m gspro_builder build ../courses/my-club

python -m unittest discover -s tests
```

Unity 6: Package Manager ▸ **+** ▸ *Add package from disk* ▸
`unity/com.local.gspro-course-builder/package.json`, then
**GSPro Builder ▸ Import Course Output…** and pick `courses/<slug>/output`.

## Layout

```
pipeline/gspro_builder/   Python pipeline (numpy, scipy, Pillow, laspy, pyproj)
  sources/                LiDAR (USGS EPT / .laz), 3DEP DEM, NAIP, OpenStreetMap clients
  grading.py              tees, greens, fairways, bunkers, water
  terrain.py mesh.py      heightmap RAW / terrain OBJ
  surfaces.py             splat maps incl. fringe / first cut
  course_meshes.py        conformed surface meshes + water planes
  foliage.py              LiDAR tree detection
  svg_opcd.py             OPCD Inkscape SVG
  holes.py                tees/pins/aim points/yardage
unity/com.local.gspro-course-builder/   Unity 6 editor package (import + hole marker export)
docs/PIPELINE.md          end-to-end process, tools, access, roadmap
courses/<slug>/           one folder per course (data/ and output/ are git-ignored)
```

Data: USGS 3DEP and NAIP are public domain. OSM data is © OpenStreetMap
contributors (ODbL).
