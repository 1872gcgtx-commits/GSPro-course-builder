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


class ReviewedPaths(unittest.TestCase):
    def test_merge(self):
        from gspro_builder.paths import lines_from_review, reviewed_lines
        draft = {"lines": [{"id": "c0", "kind": "cart", "pts": [[0, 0], [10, 0]]},
                           {"id": "c1", "kind": "cart", "pts": [[0, 5], [10, 5]]},
                           {"id": "osm-1", "kind": "road", "road": "residential", "width": 8, "pts": [[0, 20], [50, 20]]}]}
        edits = [{"id": "c0", "kind": "cart", "pts": [[0, 0], [5, 1], [10, 0]], "edited": True},
                 {"id": "osm-1", "deleted": True},
                 {"id": "uabc", "kind": "road", "road": "service", "width": 5, "pts": [[0, 30], [20, 30]], "method": "manual"}]
        got = {l["id"]: l for l in reviewed_lines(draft, edits)}
        self.assertEqual(set(got), {"c0", "c1", "uabc"})
        self.assertEqual(len(got["c0"]["pts"]), 3)
        self.assertEqual(set(l["id"] for l in reviewed_lines(draft, edits, only_mine=True)), {"c0", "uabc"})
        cart, streets = lines_from_review(list(got.values()), (1.0, -1.5))
        self.assertEqual(len(cart), 2); self.assertEqual(streets[0]["kind"], "service")
        self.assertAlmostEqual(cart[1][0][0], 1.0)                 # shifted into the LiDAR frame


class Bridges(unittest.TestCase):
    def test_profile_and_mesh(self):
        from gspro_builder.paths import bridge_profile, bridge_style, path_meshes
        frame = CourseFrame(-97.5, 31.0, 200.0)
        n = 400; x = (np.arange(n) + 0.5) * 0.5 - 100
        X, Z = np.meshgrid(x, -x)
        ground = 110 - 4.0 * np.exp(-(X / 6.0) ** 2) + 0.02 * X          # a creek 4 m deep, banks at different heights
        dem = DEM(ground.astype(np.float32), frame)
        line = np.array([[-15.0, 0.0], [15.0, 0.0]])
        pts, s, y = bridge_profile(dem, 100.0, line)
        g = dem.sample(pts[:, 0], pts[:, 1]) - 100.0
        self.assertAlmostEqual(y[0], g[0] + 0.05, places=2)                 # lands on the bank at each end
        self.assertAlmostEqual(y[-1], g[-1] + 0.05, places=2)
        mid = (s > 0.15 * s[-1]) & (s < 0.85 * s[-1])
        self.assertTrue(np.all(y[mid] >= g[mid] + 0.5))                    # clears the creek
        self.assertLess(np.abs(np.diff(y, 2)).max(), 0.05)                 # smooth: no kinks
        self.assertEqual(bridge_style("old stone arch", "cart")["rail"], "Bridge_Stone")
        self.assertEqual(bridge_style("", "cart")["deck"], "Bridge_Wood")
        V, UV, T, M = path_meshes([np.array([[-40.0, 0.0], [40.0, 0.0]])], [], dem, 100.0,
                                  bridges=[{"id": "b1", "line": line, "carries": "cart", "width": 3.0, "note": "wood deck"}])
        self.assertIn("Bridge_Wood", M)
        # the cart path stops at the bridge: no path concrete over the span
        pc = V[np.array(T)[np.array(M) == "Path_Concrete"].ravel()]
        self.assertGreater(np.abs(pc[:, 0]).min(), 14.0)


class TeePlacement(unittest.TestCase):
    def test_card_distance(self):
        from gspro_builder.teeplace import place_tees, point_back_from_green
        line = [(0.0, 0.0), (0.0, 300.0)]                 # tee at z=0, green at z=300
        p, d = point_back_from_green(line, 100.0)
        self.assertAlmostEqual(p[1], 200.0, places=3); self.assertAlmostEqual(d[1], 1.0)
        t = place_tees({(1, 1): line}, {(1, 1): {"White": 300, "Blue": 330}}, {(1, 1): {"Blue"}})
        self.assertEqual([x["tee_set"] for x in t], ["White"])           # existing Blue kept, White added
        c = np.mean(t[0]["ring"], axis=0)
        self.assertAlmostEqual(c[1], 300 - 300 * 0.9144, delta=0.1)      # 300 yd back from the green
