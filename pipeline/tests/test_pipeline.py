import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gspro_builder.config import load_config  # noqa: E402
from gspro_builder.demo import make_demo  # noqa: E402
from gspro_builder.geo import CourseFrame, lonlat_to_utm, utm_to_lonlat  # noqa: E402
from gspro_builder.layout import load_layout  # noqa: E402
from gspro_builder.pipeline import build  # noqa: E402
from gspro_builder.sources.elevation import fill_nodata  # noqa: E402
from gspro_builder.sources.osm import overpass_to_geojson  # noqa: E402
from gspro_builder.terrain import DEM, read_raw  # noqa: E402


class GeoTest(unittest.TestCase):
    def test_known_utm_point(self):
        x, y, zone = lonlat_to_utm(-81.0, 40.0)   # on zone 17's central meridian
        self.assertEqual(zone, 17)
        self.assertAlmostEqual(x, 500000.0, places=3)
        self.assertAlmostEqual(y, 4427757.22, delta=0.5)

    def test_round_trip(self):
        for lon, lat in [(-83.0, 40.0), (-122.4, 37.8), (151.2, -33.9), (-0.1, 51.5)]:
            x, y, z = lonlat_to_utm(lon, lat)
            lon2, lat2 = utm_to_lonlat(x, y, z, south=lat < 0)
            self.assertAlmostEqual(lon, lon2, places=7)
            self.assertAlmostEqual(lat, lat2, places=7)

    def test_local_frame(self):
        f = CourseFrame(-83.0, 40.0, 1000)
        lon, lat = f.to_lonlat(100.0, -250.0)
        x, z = f.to_local(lon, lat)
        self.assertAlmostEqual(x, 100.0, places=3)
        self.assertAlmostEqual(z, -250.0, places=3)
        self.assertEqual(f.local_to_pixel(-500, 500, 1000), (0.0, 0.0))  # NW corner


class TerrainTest(unittest.TestCase):
    def test_heightmap_row_order_is_south_first(self):
        f = CourseFrame(-83.0, 40.0, 100)
        north_up = np.tile(np.linspace(110, 100, 100)[:, None], (1, 100))  # higher in the north
        hm = DEM(north_up.astype(np.float32), f).heightmap(65)
        self.assertLess(hm[0].mean(), hm[-1].mean())

    def test_fill_nodata(self):
        a = np.full((5, 5), 10.0, dtype=np.float32)
        a[2, 2] = np.nan
        a[0, 0] = -9999
        b = fill_nodata(a)
        self.assertFalse(np.isnan(b).any())
        self.assertAlmostEqual(float(b[2, 2]), 10.0)
        self.assertAlmostEqual(float(b[0, 0]), 10.0)


class OsmTest(unittest.TestCase):
    def test_overpass_conversion(self):
        data = {"elements": [
            {"type": "way", "id": 1, "tags": {"golf": "green"},
             "geometry": [{"lon": 0, "lat": 0}, {"lon": 0.001, "lat": 0}, {"lon": 0.001, "lat": 0.001}, {"lon": 0, "lat": 0}]},
            {"type": "way", "id": 2, "tags": {"golf": "hole", "ref": "1", "par": "4"},
             "geometry": [{"lon": 0, "lat": 0}, {"lon": 0.003, "lat": 0.003}]},
            {"type": "way", "id": 3, "tags": {"natural": "water"},
             "geometry": [{"lon": 0, "lat": 0}, {"lon": 1, "lat": 0}, {"lon": 1, "lat": 1}, {"lon": 0, "lat": 0}]},
        ]}
        gj = overpass_to_geojson(data)
        kinds = [(f["properties"]["golf"], f["geometry"]["type"]) for f in gj["features"]]
        self.assertEqual(kinds, [("green", "Polygon"), ("hole", "LineString"), ("water_hazard", "Polygon")])


class DemoBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = make_demo(Path(cls.tmp.name) / "demo")
        cls.cfg = load_config(cls.root)
        cls.report = build(cls.cfg)
        cls.out = cls.cfg.out_dir

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outputs_exist(self):
        for rel in ["terrain/heightmap_1025.raw", "terrain/terrain.json", "terrain/terrain.obj",
                    "surfaces/splat_control0.png", "surfaces/splat_control1.png", "holes.json",
                    "greenkeeper.csv", "demo_opcd.svg", "overlay.jpg", "build_report.json"]:
            self.assertTrue((self.out / rel).exists(), rel)

    def test_raw_matches_metadata(self):
        tj = json.loads((self.out / "terrain/terrain.json").read_text())
        raw = read_raw(self.out / "terrain" / tj["raw_file"], tj["resolution"])
        self.assertEqual(raw.min(), 0)
        self.assertEqual(raw.max(), 65535)
        self.assertGreater(tj["height_range_m"], 10)

    def test_splat_weights_sum_to_one(self):
        a = np.asarray(Image.open(self.out / "surfaces/splat_control0.png"), dtype=int)
        b = np.asarray(Image.open(self.out / "surfaces/splat_control1.png"), dtype=int)
        total = a.sum(-1) + b.sum(-1)
        self.assertTrue(np.all(np.abs(total - 255) <= 4))

    def test_holes(self):
        holes = json.loads((self.out / "holes.json").read_text())["holes"]
        self.assertEqual([h["number"] for h in holes], [1, 2, 3])
        self.assertEqual([h["par"] for h in holes], [4, 3, 5])
        for h in holes:
            back = h["tees"][0]
            self.assertGreaterEqual(back["yards"], h["tees"][-1]["yards"])
            straight = math.dist((back["position"]["x"], back["position"]["z"]), (h["pin"]["x"], h["pin"]["z"])) * 1.0936
            self.assertGreaterEqual(back["yards"] + 1, straight)   # routed length >= straight line
        self.assertTrue(130 < holes[1]["tees"][-1]["yards"] < 230)  # par 3

    def test_svg_layers(self):
        svg = (self.out / "demo_opcd.svg").read_text()
        for layer in ["Hole01", "Hole02", "Hole03", "Hole99", "image"]:
            self.assertIn(f'inkscape:label="{layer}"', svg)
        self.assertIn('width="1200mm"', svg)

    def test_greens_assigned_to_holes(self):
        layout = load_layout(self.root / "features.geojson", self.cfg.frame)
        greens = sorted(s.hole for s in layout.surfaces if s.kind == "green")
        self.assertEqual(greens, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
