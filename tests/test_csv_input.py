from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from api.csv_input import parse_pairs_csv


class CsvInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "pairs.csv"

    def parse(self, content: str, method="nearest", encoding="utf-8-sig"):
        self.path.write_text(content, encoding=encoding)
        return parse_pairs_csv(self.path, method)

    def test_standard_columns_and_weighted_duplicates(self):
        points = self.parse("C,M,Y,K,L,a,b,count\n10,20,30,40,50,0,0,1\n10,20,30,40,56,3,-3,2\n")
        self.assertEqual(len(points), 1)
        self.assertEqual((points[0]["L"], points[0]["a"], points[0]["b"]), (54, 2, -2))
        self.assertEqual(points[0]["count"], 3)

    def test_original_chinese_export_and_gb18030(self):
        text = "布上标签C (%),布上标签M (%),布上标签Y (%),布上标签K (%),平均仪器L*,平均仪器a*,平均仪器b*,布料名称,仪器\n10,20,30,40,51.2,3.4,-5.6,牛奶丝,彩谱\n"
        for encoding in ("utf-8-sig", "gb18030"):
            with self.subTest(encoding=encoding):
                point = self.parse(text, encoding=encoding)[0]
                self.assertEqual(point["meta"]["fabricName"], "牛奶丝")
                self.assertEqual(point["meta"]["instrument"], "彩谱")
                self.assertEqual(point["c"], 10)

    def test_mixed_measurement_groups_are_rejected(self):
        for column in ("布料名称", "仪器", "批次", "测量组", "fabricName", "device", "session"):
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "多个"):
                self.parse(f"C,M,Y,K,L,a,b,{column}\n0,0,0,0,100,0,0,A\n0,0,0,100,0,0,0,B\n")

    def test_invalid_rows_are_never_silently_dropped(self):
        for invalid in ("nan,0,0,0,100,0,0", "101,0,0,0,100,0,0", "0,0,0,0,101,0,0", "0,0,0,0,50,inf,0", "0,0,0,0,50,0"):
            with self.subTest(row=invalid), self.assertRaisesRegex(ValueError, "第 3"):
                self.parse(f"C,M,Y,K,L,a,b\n0,0,0,0,100,0,0\n{invalid}\n")

    def test_invalid_counts_are_rejected(self):
        for value in ("0", "-1", "nan", "oops"):
            with self.subTest(count=value), self.assertRaises(ValueError):
                self.parse(f"C,M,Y,K,L,a,b,count\n0,0,0,0,100,0,0,{value}\n")

    def test_model_minimum_uses_distinct_recipes(self):
        with self.assertRaisesRegex(ValueError, "8"):
            self.parse("C,M,Y,K,L,a,b\n" + "0,0,0,0,100,0,0\n" * 8, method="model")
        content = "C,M,Y,K,L,a,b\n" + "".join(f"{i * 10},0,0,0,{100-i},0,0\n" for i in range(8))
        self.assertEqual(len(self.parse(content, method="model")), 8)

    def test_empty_and_duplicate_headers_are_rejected(self):
        for content in ("", "C,M,Y,K,L,a,b\n", "C,M,Y,K,L,a,b,L*\n0,0,0,0,100,0,0,100\n"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.parse(content)

    def test_method_validation(self):
        with self.assertRaisesRegex(ValueError, "method"):
            self.parse("C,M,Y,K,L,a,b\n0,0,0,0,100,0,0\n", method="unknown")


if __name__ == "__main__":
    unittest.main()
