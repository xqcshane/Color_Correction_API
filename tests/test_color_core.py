from __future__ import annotations

import importlib.util
import math
import unittest

from api.color_core import ColorPoint, LocalWeightedRegression, delta_e_2000, lab_to_rgb, rgb_to_lab
from api.color_data import aggregate_pairs, parse_color_map_csv, parse_color_map_json


class ScalarColorTests(unittest.TestCase):
    def test_ciede2000_reference_pairs_match_js(self) -> None:
        self.assertAlmostEqual(delta_e_2000((50, 2.6772, -79.7751), (50, 0, -82.7485)), 2.0425, places=4)
        self.assertAlmostEqual(delta_e_2000((50, 3.1571, -77.2803), (50, 0, -82.7485)), 2.8615, places=4)

    def test_rgb_lab_round_trip(self) -> None:
        for rgb in ((0, 0, 0), (255, 255, 255), (14, 87, 201), (190, 42, 113)):
            converted = lab_to_rgb(rgb_to_lab(rgb))
            for actual, expected in zip(converted, rgb):
                self.assertLess(abs(actual - expected), 0.001)

    def test_local_weighted_regression_tracks_linear_print_response(self) -> None:
        points = []
        for c in (0, 50, 100):
            for m in (0, 50, 100):
                for y in (0, 100):
                    for k in (0, 100):
                        points.append(
                            ColorPoint(
                                c,
                                m,
                                y,
                                k,
                                100 - 0.20 * c - 0.15 * m - 0.10 * y - 0.40 * k,
                                0.40 * m - 0.20 * c,
                                0.40 * y - 0.10 * m,
                            )
                        )
        model = LocalWeightedRegression(points, neighbor_count=24)
        query = (35, 45, 20, 15)
        prediction = model.predict(query)
        expected = (100 - 0.20 * 35 - 0.15 * 45 - 0.10 * 20 - 0.40 * 15, 0.40 * 45 - 0.20 * 35, 0.40 * 20 - 0.10 * 45)
        self.assertLess(math.dist((prediction["L"], prediction["a"], prediction["b"]), expected), 0.35)
        self.assertGreaterEqual(prediction["nearestDistance"], 0)


class PairImportTests(unittest.TestCase):
    def test_csv_aliases_and_rejected_rows(self) -> None:
        csv_text = "\ufeff布上标签C (%),布上标签M (%),布上标签Y (%),布上标签K (%),平均仪器L*,平均仪器a*,平均仪器b*,布料名称,仪器\n10,20,30,40,51.2,3.4,-5.6,时尚数码牛奶丝,彩谱\nBAD,20,30,40,51,3,-5,时尚数码牛奶丝,彩谱\n"
        imported = parse_color_map_csv(csv_text, "milk.csv")
        self.assertEqual(len(imported.points), 1)
        self.assertEqual(len(imported.rejected), 1)
        point = imported.points[0]
        self.assertEqual(point.cmyk, (10, 20, 30, 40))
        self.assertEqual(point.meta["fabricName"], "时尚数码牛奶丝")
        self.assertEqual(point.meta["instrument"], "彩谱")
        self.assertEqual(imported.as_dict()["schema"], "cmyk-recipe-pairs/v1")

    def test_local_color_map_json_and_direct_api_schema(self) -> None:
        local_export = {
            "state": {
                "patches": [{"id": "p1", "c": 1, "m": 2, "y": 3, "k": 4, "code": "P1"}],
                "fabrics": [{"id": "f1", "name": "牛奶丝"}],
                "instruments": [{"id": "i1", "name": "彩谱"}],
                "records": [{"patchId": "p1", "fabricId": "f1", "instrumentId": "i1", "L": 90, "a": 2, "b": -1}],
            }
        }
        imported = parse_color_map_json(local_export, "export.json")
        self.assertEqual(imported.points[0].patch_code, "P1")
        self.assertEqual(imported.points[0].meta["fabricName"], "牛奶丝")
        direct = parse_color_map_json(imported.as_dict(), "canonical.json")
        self.assertEqual(direct.points[0].lab, (90, 2, -1))

    def test_duplicate_cmyk_is_weighted(self) -> None:
        points = [
            ColorPoint(10, 20, 30, 40, 50, 0, 0, count=1),
            ColorPoint(10, 20, 30, 40, 56, 3, -3, count=2),
        ]
        aggregated = aggregate_pairs(points)
        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0].lab, (54, 2, -2))
        self.assertEqual(aggregated[0].count, 3)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is an API runtime dependency")
class ArrayEngineTests(unittest.TestCase):
    def test_cpu_nearest_mapper_preserves_alpha_semantics(self) -> None:
        from api.color_array import Backend, NearestLutMapper
        import numpy as np

        points = [
            ColorPoint(0, 0, 0, 0, *rgb_to_lab((255, 255, 255))),
            ColorPoint(0, 0, 0, 100, *rgb_to_lab((0, 0, 0))),
        ]
        mapper = NearestLutMapper(points, rgb_bits=5, candidate_count=2, backend=Backend("cpu", np, "test"), batch_size=4096)
        rgba = np.asarray([[255, 255, 255, 255], [0, 0, 0, 128], [8, 9, 10, 0]], dtype=np.uint8)
        result = mapper.map_rgba(rgba)
        self.assertEqual(result["cmyk"].tolist(), [[0, 0, 0, 0], [0, 0, 0, 255], [0, 0, 0, 0]])
        self.assertEqual(result["preview"][:, 3].tolist(), [255, 128, 0])


if __name__ == "__main__":
    unittest.main()
