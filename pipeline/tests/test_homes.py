"""Homes: footprint -> house mesh with a hipped roof, outward walls, trees cleared from lots."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gspro_builder.geo import CourseFrame  # noqa: E402
from gspro_builder.homes import HomesConfig, build_homes, homes_mask_filter, write_homes  # noqa: E402
from gspro_builder.layout import Layout  # noqa: E402
from gspro_builder.terrain import DEM  # noqa: E402


class Homes(unittest.TestCase):
    def setUp(self):
        self.frame = CourseFrame(-97.5, 31.0, 400.0)
        self.dem = DEM(np.full((800, 800), 200.0, np.float32), self.frame)
        ring = [(0, 0), (20, 0), (20, 12), (8, 12), (8, 20), (0, 20)]          # L-shaped house
        coords = [list(self.frame.to_lonlat(x, z)) for x, z in ring]
        self.fp = {"features": [{"properties": {"height": 6.0},
                                 "geometry": {"type": "Polygon", "coordinates": [coords + [coords[0]]]}}]}

    def test_house(self):
        homes, rep = build_homes(self.fp, self.frame, self.dem, 200.0, Layout([], [], [], []), HomesConfig(),
                                 log=lambda *a: None)
        self.assertEqual(rep["count"], 1)
        h = homes[0]
        self.assertAlmostEqual(h["area_m2"], 304, delta=3)
        self.assertEqual(h["height_source"], "footprint")
        self.assertGreater(h["eave_y"] - h["ground_y"], 2.5)            # room for a storey
        self.assertAlmostEqual(h["ridge_y"] - h["ground_y"], 6.0, delta=0.4)
        self.assertGreater(h["ridge_y"], h["eave_y"] + 1.0)             # pitched roof
        with tempfile.TemporaryDirectory() as d:
            out = write_homes(homes, Path(d), HomesConfig())
            self.assertGreater(out["triangles"], 20)
            txt = (Path(d) / "homes.obj").read_text()
            self.assertIn("usemtl Home_Roof", txt)
            self.assertIn("usemtl Home_Wall", txt)
        trees = [{"x": 5.0, "z": 5.0}, {"x": 50.0, "z": 50.0}]
        self.assertEqual(len(homes_mask_filter(trees, homes)), 1)        # tree inside the lot removed


if __name__ == "__main__":
    unittest.main()
