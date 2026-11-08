"""Procedural species tree models: structure, size, textures, files."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gspro_builder.treegen import SPEC, leaf_texture, mesh, write_models  # noqa: E402


class TreeGen(unittest.TestCase):
    def test_shapes(self):
        w = {}
        for sp in SPEC:
            m = mesh(sp, seed=1)
            V = np.vstack([m["bark"][0], m["leaves"][0]])
            self.assertAlmostEqual(V[:, 1].max(), 10.0, delta=0.6, msg=sp)        # 10 m model
            self.assertGreater(len(m["leaves"][2]), 100, sp)                       # leafy
            self.assertGreater(len(m["bark"][2]), 30, sp)                          # trunk and limbs
            w[sp] = np.ptp(V[:, [0, 2]], axis=0).max()
        self.assertGreater(w["live_oak"], 1.4 * w["bald_cypress"])                # oak spreads, cypress is a spire
        self.assertGreater(w["live_oak"], w["cedar_elm"])

    def test_files(self):
        a = np.asarray(leaf_texture("live_oak", size=128))[..., 3]
        self.assertTrue(0.1 < (a > 0).mean() < 0.9)                              # leaves with gaps
        with tempfile.TemporaryDirectory() as d:
            r = write_models(Path(d), species=["live_oak", "ashe_juniper"], variants=2)
            self.assertEqual(r["variants"], 2)
            txt = (Path(d) / "live_oak_1.obj").read_text()
            self.assertIn("usemtl Leaves_live_oak", txt)
            self.assertTrue((Path(d) / "leaves_ashe_juniper.png").exists())


if __name__ == "__main__":
    unittest.main()
