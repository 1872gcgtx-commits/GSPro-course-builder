import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gspro_builder.cli import main  # noqa: E402
from gspro_builder.demo import HOLES, make_demo  # noqa: E402


class RoutingTest(unittest.TestCase):
    def test_three_nines_make_three_routings(self):
        with tempfile.TemporaryDirectory() as d:
            root = make_demo(Path(d) / "demo")
            # The tracer page stores holes as {nine, hole, par, points:[{x,z}]}: put the demo's
            # three holes on three different nines (each nine's hole 1).
            docs = [{"nine": i + 1, "hole": 1, "par": par, "points": [{"x": x, "z": z} for x, z in path]}
                    for i, (_, par, _, path) in enumerate(HOLES)]
            exp = Path(d) / "export"
            exp.mkdir()
            for i, doc in enumerate(docs):
                (exp / f"n{i + 1}h1.json").write_text(json.dumps(doc))
            with open(root / "course.toml", "a") as f:
                f.write('\n[nines]\n1 = "Creek 1"\n2 = "Creek 2"\n3 = "Creek 3"\n')
            main(["import-routing", str(root), str(exp)])
            main(["build", str(root)])
            out = root / "output"
            summary = json.loads((out / "routings" / "routings.json").read_text())
            self.assertEqual([r["name"] for r in summary], ["Course 1-2", "Course 1-3", "Course 2-3"])
            c23 = json.loads((out / "routings" / "course-2-3" / "holes.json").read_text())["holes"]
            self.assertEqual([h["number"] for h in c23], [1, 10])              # front nine -> 1-9, back -> 10-18
            self.assertEqual([h["scene_hole"] for h in c23], [10, 19])         # scene numbering 1-27
            self.assertEqual(c23[0]["name"], "Creek 2 1")
            svg = (out / "demo_opcd.svg").read_text()
            for layer in ("Hole01", "Hole10", "Hole19"):
                self.assertIn(f'inkscape:label="{layer}"', svg)


if __name__ == "__main__":
    unittest.main()
