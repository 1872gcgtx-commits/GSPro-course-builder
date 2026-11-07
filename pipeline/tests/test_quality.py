"""Course-quality tests: elevations, grading, surface transitions, water, trees, meshes, LiDAR."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gspro_builder.config import load_config  # noqa: E402
from gspro_builder.demo import make_demo  # noqa: E402
from gspro_builder.geo import CourseFrame  # noqa: E402
from gspro_builder.grading import GradingConfig, grade  # noqa: E402
from gspro_builder.layout import load_layout  # noqa: E402
from gspro_builder.pipeline import build  # noqa: E402
from gspro_builder.raster import rasterize_instances, slope_grid  # noqa: E402
from gspro_builder.surfaces import TERRAIN_LAYERS  # noqa: E402
from gspro_builder.terrain import DEM  # noqa: E402


class DemoQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = make_demo(Path(cls.tmp.name) / "demo")
        cls.cfg = load_config(cls.root)
        cls.report = build(cls.cfg)
        cls.out = cls.cfg.out_dir
        cls.raw = DEM(np.load(cls.root / "data" / "dem.npy"), cls.cfg.frame)
        cls.layout = load_layout(cls.root / "features.geojson", cls.cfg.frame)
        cls.g = grade(cls.raw, cls.layout, GradingConfig())
        cls.n = cls.raw.data.shape[0]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _mask(self, kind):
        return rasterize_instances([s for s in self.layout.surfaces if s.kind == kind], self.cfg.frame, self.n) > 0

    def test_tees_are_level(self):
        a = self.g.dem.data.astype(float)
        sl = slope_grid(a, self.raw.res_m)
        from scipy import ndimage
        inner = ndimage.binary_erosion(self._mask("tee"), iterations=3)
        self.assertLess(np.percentile(sl[inner], 95), 0.02)       # <= 2 % everywhere on the pad
        for t in self.report["grading"]["tees"]:
            self.assertLessEqual(t["slope_after_pct"], 1.5 + 1e-6)

    def test_bunkers_are_below_surroundings(self):
        for b in self.report["grading"]["bunkers"]:
            self.assertGreaterEqual(b["depth_m"], 0.5)
        m = self._mask("bunker")
        diff = (self.raw.data - self.g.dem.data)[m]
        self.assertGreater(np.median(diff), 0.2)

    def test_water_is_flat_and_contained(self):
        body = self.g.water_bodies[0]
        level = body["level_m"]
        m = self._mask("water")
        self.assertTrue(np.all(self.g.dem.data[m] < level))            # bed below the surface
        from scipy import ndimage
        bank = ndimage.binary_dilation(m, iterations=4) & ~m
        self.assertTrue(np.all(self.g.dem.data[bank] > level))        # banks hold the water
        wj = json.loads((self.out / "water.json").read_text())
        ys = {v["y"] for v in wj["bodies"][0]["vertices"]}
        self.assertEqual(len(ys), 1)                                   # perfectly flat plane

    def test_greens_smoothed_not_flattened(self):
        m = self._mask("green")
        from scipy import ndimage
        inner = ndimage.binary_erosion(m, iterations=6)
        lap_raw = np.abs(ndimage.laplace(self.raw.data.astype(float)))[inner].mean()
        lap_new = np.abs(ndimage.laplace(self.g.dem.data.astype(float)))[inner].mean()
        self.assertLess(lap_new, lap_raw * 0.3)                         # LiDAR noise removed
        for gr in self.report["grading"]["greens"]:
            self.assertTrue(0.5 < gr["mean_slope_pct"] < 3.0)          # contour kept, playable

    def test_no_seams_at_feature_edges(self):
        sl = slope_grid(self.g.dem.data.astype(float), self.raw.res_m)
        water = ndimage_dilate(self._mask("water"), 10)
        self.assertLess(np.percentile(sl[~water], 99.9), 0.6)          # nothing cliff-like outside ponds

    def test_fairway_to_rough_transition(self):
        lab = np.asarray(Image.open(self.out / "surfaces/surface_labels.png"))
        L = {k: i for i, k in enumerate(TERRAIN_LAYERS)}
        from scipy import ndimage
        fw = lab == L["fairway"]
        ring = ndimage.binary_dilation(fw, iterations=2) & ~fw
        vals = lab[ring]
        self.assertGreater((vals == L["first_cut"]).mean(), 0.7)      # fairway is wrapped in first cut
        gr = lab == L["green"]
        ring = ndimage.binary_dilation(gr, iterations=1) & ~gr
        self.assertGreater(np.isin(lab[ring], [L["fringe"], L["bunker"]]).mean(), 0.9)

    def test_trees_match_planted(self):
        planted = json.loads((self.root / "data" / "demo_trees.json").read_text())["planted"]
        with open(self.out / "trees.csv") as fh:
            rows = list(csv.DictReader(fh))
        self.assertLessEqual(abs(len(rows) - planted), max(3, planted * 0.05))
        hs = [float(r["height_m"]) for r in rows]
        self.assertTrue(all(12 < h < 23 for h in hs))
        # no tree on a playing surface
        f = self.cfg.frame
        play = rasterize_instances([s for s in self.layout.surfaces if s.kind in ("green", "tee", "fairway", "bunker", "water")], f, 1200) > 0
        for r in rows:
            c, rr = f.local_to_pixel(float(r["x"]), float(r["z"]), 1200)
            self.assertFalse(play[int(rr), int(c)])

    def test_surface_meshes_conform_and_face_up(self):
        txt = (self.out / "course_surfaces.obj").read_text().splitlines()
        names = [l[2:] for l in txt if l.startswith("o ")]
        for want in ("Hole01_Green", "Hole01_Tee", "Hole01_Fairway", "Hole01_Bunker", "Hole02_WaterPlane"):
            self.assertIn(want, names)
        v = np.array([[float(x) for x in l.split()[1:]] for l in txt if l.startswith("v ")])
        faces = np.array([[int(p.split("/")[0]) - 1 for p in l.split()[1:]] for l in txt if l.startswith("f ")])
        a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
        ny = np.cross(b - a, c - a)[:, 1]
        self.assertTrue(np.all(ny > 0))                                 # every triangle faces up
        base = json.loads((self.out / "terrain/terrain.json").read_text())["min_elev_m"]
        dem = DEM(np.load(self.root / "data" / "dem.npy"), self.cfg.frame)
        g = grade(dem, self.layout, GradingConfig()).dem
        idx = np.random.default_rng(0).choice(len(v), 500, replace=False)
        terrain_y = g.sample_smooth(v[idx, 0], -v[idx, 2]) - base        # OBJ z is south
        err = np.abs(v[idx, 1] - terrain_y)
        self.assertLess(np.median(err), 0.02)

    def test_pins_on_flat_green(self):
        holes = json.loads((self.out / "holes.json").read_text())["holes"]
        for h in holes:
            names = [p["name"] for p in h["pin_positions"]]
            self.assertEqual(names, ["front", "middle", "back"])
            for p in h["pin_positions"]:
                self.assertLessEqual(p["slope_pct"], 2.5)


def ndimage_dilate(m, it):
    from scipy import ndimage
    return ndimage.binary_dilation(m, iterations=it)


class LidarFiles(unittest.TestCase):
    def test_state_plane_feet_laz(self):
        import laspy
        from pyproj import CRS, Transformer
        frame = CourseFrame(-83.0, 40.0, 200)
        crs = CRS.from_epsg(3735)   # NAD83 / Ohio South (ftUS)
        tr = Transformer.from_crs(frame.epsg, crs, always_xy=True)
        xmin, ymin, xmax, ymax = frame.utm_bbox
        rng = np.random.default_rng(0)
        n = 200_000
        ux, uy = rng.uniform(xmin, xmax, n), rng.uniform(ymin, ymax, n)
        ground_m = 250 + 0.02 * (ux - xmin)                           # 2 % slope to the east
        cls = np.full(n, 2, np.uint8)
        z_m = ground_m.copy()
        tree = (ux - xmin - 100) ** 2 + (uy - ymin - 100) ** 2 < 16  # 4 m crown at the centre
        cls[tree & (rng.random(n) < 0.7)] = 5
        z_m[cls == 5] += 18 - np.hypot(ux - xmin - 100, uy - ymin - 100)[cls == 5]
        sx, sy = tr.transform(ux, uy)
        hdr = laspy.LasHeader(point_format=3, version="1.4")
        hdr.add_crs(crs)
        hdr.scales = [0.01, 0.01, 0.01]
        hdr.offsets = [float(np.min(sx)), float(np.min(sy)), 0]
        las = laspy.LasData(hdr)
        las.x, las.y, las.z = sx, sy, z_m / 0.3048006096          # feet
        las.classification = cls
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "laz").mkdir()
            las.write(Path(d) / "laz" / "tile.laz")
            from gspro_builder.sources.lidar import read_las_files
            prod = read_las_files(["laz/*.laz"], Path(d), frame, 1.0)
        dtm, chm = prod["dtm"], prod["chm"]
        expect = 250 + 0.02 * (np.arange(200) + 0.5)
        self.assertLess(np.abs(dtm[100, 20:180] - expect[20:180]).max(), 0.1)   # metres, right orientation
        self.assertGreater(chm[100, 100], 15)
        self.assertLess(chm[20, 20], 0.5)


if __name__ == "__main__":
    unittest.main()
