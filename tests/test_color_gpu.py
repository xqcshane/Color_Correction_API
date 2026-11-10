from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from api.color_array import Backend, GpuUnavailableError, _bounded_batch_size, select_backend
from api.color_core import ColorPoint, delta_e_2000, lab_to_rgb, rgb_to_lab


class GpuBackendTests(unittest.TestCase):
    def test_no_cuda_device_fails_without_cpu_fallback(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0)))
        with patch.dict("sys.modules", {"cupy": fake}), self.assertRaises(GpuUnavailableError):
            select_backend()

    def test_broken_cuda_kernel_fails_without_cpu_fallback(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(
            Device=lambda: SimpleNamespace(id=0),
            runtime=SimpleNamespace(getDeviceCount=lambda: 1, getDeviceProperties=lambda _: {"name": b"NVIDIA GeForce RTX 3090"}),
        ))
        with patch.dict("sys.modules", {"cupy": fake}), patch("api.color_array.run_cupy_kernel_probe", side_effect=RuntimeError("NVRTC failed")):
            with self.assertRaisesRegex(GpuUnavailableError, "NVRTC failed"):
                select_backend()

    def test_3090_uses_detected_gpu_and_real_probe(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(
            Device=lambda: SimpleNamespace(id=0),
            runtime=SimpleNamespace(getDeviceCount=lambda: 1, getDeviceProperties=lambda _: {"name": b"NVIDIA GeForce RTX 3090"}),
        ))
        with patch.dict("sys.modules", {"cupy": fake}), patch("api.color_array.run_cupy_kernel_probe") as probe:
            backend = select_backend()
        probe.assert_called_once_with(fake)
        self.assertEqual(backend.name, "cuda")
        self.assertIn("3090", backend.device_name)

    def test_search_batch_shrinks_for_large_tables_and_low_free_vram(self):
        cpu = Backend("cpu", None, "test")
        self.assertEqual(_bounded_batch_size(cpu, 2048, 100), 2048)
        self.assertLessEqual(_bounded_batch_size(cpu, 2048, 1_000_000), 5)
        fake = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (16 * 1024**2, 24 * 1024**3))))
        gpu = Backend("cuda", fake, "3090")
        self.assertLess(_bounded_batch_size(gpu, 2048, 1_000), 2048)

    def test_ciede2000_reference(self):
        self.assertAlmostEqual(delta_e_2000((50, 2.6772, -79.7751), (50, 0, -82.7485)), 2.0425, places=4)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy runtime dependency")
class ArrayProcessingTests(unittest.TestCase):
    def test_fine_nearest_uses_measured_cmyk_and_ciede2000(self):
        from api.color_array import NearestLutMapper
        import numpy as np

        points = [ColorPoint(0, 0, 0, 0, *rgb_to_lab((255, 255, 255))), ColorPoint(0, 0, 0, 100, *rgb_to_lab((0, 0, 0)))]
        mapper = NearestLutMapper(points, rgb_bits=6, candidate_count=256, backend=select_backend(prefer_gpu=False), batch_size=2048)
        source = np.asarray([[245, 245, 245, 255], [10, 10, 10, 255]], dtype=np.uint8)
        output = mapper.map_rgba(source)
        self.assertEqual(output["cmyk"].tolist(), [[0, 0, 0, 0], [0, 0, 0, 255]])
        self.assertTrue(np.isfinite(output["stats"]["meanDeltaE"]))

    def test_explicit_cpu_backend_and_nearest_alpha(self):
        from api.color_array import NearestLutMapper
        import numpy as np

        backend = select_backend(prefer_gpu=False)
        self.assertEqual(backend.name, "cpu")
        points = [ColorPoint(0, 0, 0, 0, *rgb_to_lab((255, 255, 255))), ColorPoint(0, 0, 0, 100, *rgb_to_lab((0, 0, 0)))]
        mapper = NearestLutMapper(points, rgb_bits=4, backend=backend)
        rgba = np.asarray([[255, 255, 255, 255], [0, 0, 0, 128], [8, 9, 10, 0]], dtype=np.uint8)
        result = mapper.map_rgba(rgba)
        self.assertEqual(result["cmyk"].tolist(), [[0, 0, 0, 0], [0, 0, 0, 255], [0, 0, 0, 0]])
        self.assertEqual(result["preview"][:, 3].tolist(), [255, 128, 0])

    def test_inverse_mapper_returns_continuous_finite_cmyk(self):
        from api.color_array import InverseLutMapper
        import itertools
        import numpy as np

        points = [ColorPoint(c, m, y, k, 100 - .20*c - .15*m - .10*y - .40*k, .40*m - .20*c, .40*y - .10*m) for c, m, y, k in itertools.product((0, 100), repeat=4)]
        mapper = InverseLutMapper(points, grid_step=50, lut_size=3, backend=select_backend(prefer_gpu=False), batch_size=32)
        rgba = np.asarray([[120, 100, 80, 255], [80, 100, 120, 128], [30, 40, 50, 0]], dtype=np.uint8)
        result = mapper.map_rgba(rgba)
        self.assertEqual(result["cmyk"].shape, rgba.shape)
        self.assertTrue(np.isfinite(result["cmyk"]).all())
        self.assertEqual(result["preview"][:, 3].tolist(), [255, 128, 0])
        self.assertEqual(result["cmyk"][2].tolist(), [0, 0, 0, 0])
        self.assertEqual(result["stats"]["visiblePixels"], 2)

    def test_fine_inverse_predicts_an_intermediate_print_recipe(self):
        from api.color_array import InverseLutMapper
        import itertools
        import numpy as np

        def measured_lab(c, m, y, k):
            return (100 - .20*c - .15*m - .10*y - .40*k, .40*m - .20*c, .40*y - .10*m)

        points = [ColorPoint(c, m, y, k, *measured_lab(c, m, y, k)) for c, m, y, k in itertools.product((0, 100), repeat=4)]
        mapper = InverseLutMapper(points, grid_step=10, lut_size=13, neighbor_count=36, candidate_count=64, backend=select_backend(prefer_gpu=False), batch_size=1024)
        target_lab = measured_lab(35, 45, 20, 15)
        rgb = tuple(round(value) for value in lab_to_rgb(target_lab))
        output = mapper.map_rgba(bytes((*rgb, 255)), width=1, height=1)
        cmyk = output["cmyk"].reshape((1, 4))[0] / 2.55
        self.assertTrue(any(0 < value < 100 for value in cmyk))
        self.assertLess(delta_e_2000(target_lab, measured_lab(*cmyk)), 3.0)
        self.assertEqual(mapper.model_candidates, 14641)


if __name__ == "__main__":
    unittest.main()
