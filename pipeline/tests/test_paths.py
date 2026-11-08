"""Cart paths and streets: cross-section levelled, profile smoothed, meshes draped, curbs trimmed."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gspro_builder.geo import CourseFrame  # noqa: E402
from gspro_builder.paths import grade_paths, path_meshes, write_paths  # noqa: E402
from gspro_builder.terrain import DEM  # noqa: E402


class Paths(unittest.TestCase):
    def setUp(self):
        self.frame = CourseFrame(-97.5, 31.0, 200.0)
        n = 400; x = (np.arange(n) + 0.5) * 0.5 - 100
        X, Z = np.meshgrid(x, -x)
        self.dem = DEM((100 + 0.08 * X + 0.3 * np.sin(Z / 3)).astype(np.float32), self.frame)   # cross-slope + bumps

    def test_grade_levels_cross_section(self):
        line = np.array([[-80.0, 0.0], [80.0, 0.0]])          # runs east, across the 8 % slope? no: along x
        line2 = np.array([[0.0, -80.0], [0.0, 80.0]])         # runs north: the 8 % slope is across it
        grade_paths(self.dem, [{"line": line2, "width": 2.5, "smooth_m": 6.0, "shoulder_m": 1.5}])
        z = np.linspace(-60, 60, 7)
        left = self.dem.sample(np.full(7, -1.0), z); right = self.dem.sample(np.full(7, 1.0), z)
        self.assertLess(np.abs(left - right).max(), 0.05)    # level across the path (was 0.16 m)
        V, UV, T, M = path_meshes([line], [{"line": line2, "width": 8.0, "kind": "residential", "surface": "Road_Asphalt",
                                             "curb": True, "centre_line": False}], self.dem, 90.0)
        self.assertIn("Path_Concrete", M); self.assertIn("Road_Curb", M)
        # curbs stop where the cart path crosses the street
        cv = V[np.array(T)[np.array(M) == "Road_Curb"].ravel()]
        self.assertGreater(np.abs(cv[:, 2]).min(), 2.0)
        with tempfile.TemporaryDirectory() as d:
            r = write_paths([line], [], self.dem, 90.0, Path(d))
            self.assertGreater(r["triangles"], 50)
            self.assertTrue((Path(d) / "Path_Concrete.png").exists())


if __name__ == "__main__":
    unittest.main()
