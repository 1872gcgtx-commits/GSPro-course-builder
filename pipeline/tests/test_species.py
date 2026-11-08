"""Tree species rules: setting first (creek bank / bottom / open), then crown shape and colour."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gspro_builder.species import classify  # noqa: E402


def table(n=400, seed=0):
    r = np.random.default_rng(seed)
    return {"height": r.uniform(4, 9, n), "spread": r.uniform(0.3, 0.7, n), "peak": r.uniform(0.25, 0.35, n),
            "holes": r.uniform(0.35, 0.5, n), "solid": r.uniform(0.6, 0.9, n), "cover": r.uniform(0.2, 0.5, n),
            "lum": r.uniform(0.2, 0.28, n), "green": r.uniform(0.02, 0.06, n), "blue": r.uniform(0.0, 0.03, n),
            "ndvi": r.uniform(0.4, 0.55, n), "texture": r.uniform(0.01, 0.04, n),
            "water_dist": r.uniform(200, 600, n), "above_water": r.uniform(5, 15, n)}


class Species(unittest.TestCase):
    def test_settings(self):
        f = table()
        # tree 0: tall, narrow, peaked, on the creek bank -> bald cypress
        f["height"][0], f["water_dist"][0], f["above_water"][0], f["spread"][0], f["peak"][0] = 16, 8, 1.0, 0.25, 0.4
        # tree 1: tall in the creek bottom -> pecan
        f["height"][1], f["water_dist"][1], f["above_water"][1] = 20, 60, 2.5
        # tree 2: small, light, see-through, alone in a field -> mesquite
        f["height"][2], f["cover"][2], f["lum"][2], f["ndvi"][2], f["holes"][2] = 4.5, 0.05, 0.34, 0.3, 0.6
        # tree 3: broad, flat-topped, dark, dense -> live oak
        f["spread"][3], f["peak"][3], f["lum"][3], f["holes"][3] = 0.9, 0.2, 0.17, 0.25
        # tree 4: narrow, peaked, dark blue-green in upland woods -> Ashe juniper
        f["spread"][4], f["peak"][4], f["lum"][4], f["blue"][4], f["cover"][4] = 0.25, 0.42, 0.17, 0.05, 0.6
        sp, conf, _ = classify(f)
        self.assertEqual(list(sp[:5]), ["bald_cypress", "pecan", "mesquite", "live_oak", "ashe_juniper"])
        self.assertTrue(((conf >= 0) & (conf <= 1)).all())
        # no junipers or mesquite in the creek bottom
        f2 = table(seed=1); f2["water_dist"][:] = 40; f2["above_water"][:] = 2
        sp2, _, _ = classify(f2)
        self.assertFalse(np.isin(sp2, ["ashe_juniper", "mesquite"]).any())


if __name__ == "__main__":
    unittest.main()
