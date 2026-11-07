# GSPro Course Builder

Builds local golf courses for GSPro from public data. The pipeline downloads real
elevation (USGS 3DEP LiDAR DEM), aerial imagery (USDA NAIP) and the course layout
(OpenStreetMap plus your own traces). From those it produces everything the GSPro/OPCD
toolchain needs:

* a 16-bit Unity terrain heightmap and an OBJ terrain mesh
* surface splat maps (fairway, green, tee, bunker, water, rough, trees, cart path)
* an OPCD-style Inkscape SVG with Hole01..HoleNN layers and an aerial image layer
* hole data: tees per tee-set, pins, aim points, yardages and par (`holes.json`, `greenkeeper.csv`)

It also includes a **Unity 6 editor package** that loads all of this into a scene so you
can preview and edit the course.

> **Important:** GSPro's course build still runs on the OPCD toolchain, which uses
> **Unity 2018.2.8f1**, Blender 4.2 and GreenKeeper. Unity 6 here is the preview and
> design environment. See [docs/PIPELINE.md](docs/PIPELINE.md) for the full process,
> the tools and the access checklist.

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
pipeline/gspro_builder/   Python pipeline (numpy, Pillow, requests; rasterio optional)
  sources/                USGS 3DEP, NAIP, OpenStreetMap/Overpass clients
  terrain.py mesh.py      heightmap RAW / OBJ
  surfaces.py             splat maps
  svg_opcd.py             OPCD Inkscape SVG
  holes.py                tees/pins/aim points/yardage
unity/com.local.gspro-course-builder/   Unity 6 editor package (import + hole marker export)
docs/PIPELINE.md          end-to-end process, tools, access, roadmap
courses/<slug>/           one folder per course (data/ and output/ are git-ignored)
```

Data: USGS 3DEP and NAIP are public domain. OSM data is © OpenStreetMap
contributors (ODbL).
