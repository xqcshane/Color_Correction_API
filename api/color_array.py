"""NumPy/CuPy array engine for full-image color mapping.

GPU scope
---------
* RGB -> Lab and CIEDE2000 batches.
* Nearest-measured RGB lookup-table construction and per-pixel lookup.
* Batched local weighted CMYK -> Lab regression.
* Inverse RGB -> CMYK LUT search/refinement and trilinear application.

The module requires CuPy/CUDA unless CPU execution is explicitly selected.
PDF decoding, layer compositing and file writing deliberately
remain outside this module; they are streaming API pipeline concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Iterable, Literal, Mapping, Sequence

from .color_core import ColorPoint, ensure_color_points
from .gpu_probe import concise_gpu_error, run_cupy_kernel_probe


@dataclass(frozen=True, slots=True)
class Backend:
    name: Literal["cuda", "cpu"]
    xp: Any
    device_name: str

    @property
    def is_gpu(self) -> bool:
        return self.name == "cuda"

    def to_host(self, value: Any) -> Any:
        if self.is_gpu:
            return self.xp.asnumpy(value)
        return value


class GpuUnavailableError(RuntimeError):
    """The requested GPU backend cannot execute real CUDA kernels."""


def select_backend(prefer_gpu: bool = True) -> Backend:
    """Require working CUDA, or explicitly select NumPy for local tests.

    A GPU request must never silently turn into a much slower CPU job.
    CuPy compiles kernels for the detected GPU (including RTX 3090/sm_86);
    no RTX 4090/sm_89 architecture is hard-coded.
    """

    if prefer_gpu:
        try:
            import cupy as cp  # type: ignore

            if cp.cuda.runtime.getDeviceCount() > 0:
                device_id = cp.cuda.Device().id
                properties = cp.cuda.runtime.getDeviceProperties(device_id)
                raw_name = properties.get("name", b"CUDA GPU")
                name = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
                # Device enumeration and allocation do not compile an NVRTC
                # kernel.  Exercise the same ufunc path as the color engine so
                # a broken CuPy include path fails before a job starts.
                run_cupy_kernel_probe(cp)
                return Backend("cuda", cp, name)
            raise RuntimeError("未检测到 CUDA 设备")
        except Exception as error:
            raise GpuUnavailableError(f"GPU 不可用：{concise_gpu_error(error)}") from error
    try:
        import numpy as np  # type: ignore
    except ImportError as error:
        raise RuntimeError("颜色数组引擎需要 numpy；GPU 加速另需 cupy-cuda12x") from error
    return Backend("cpu", np, "NumPy CPU")


def _bounded_batch_size(backend: Backend, requested: int, reference_count: int) -> int:
    """Bound pairwise search temporaries independently of CSV/PDF size.

    Broadcasting and argpartition create several float/int arrays per pair.
    A conservative 64-byte allowance keeps the live search working set near
    256 MiB, and smaller when another workload occupies most of the GPU.
    """

    budget = 256 * 1024**2
    if backend.is_gpu:
        try:
            free, _total = backend.xp.cuda.runtime.memGetInfo()
            budget = min(budget, max(1024**2, int(free) // 8))
        except (AttributeError, RuntimeError):
            pass
    return max(1, min(int(requested), budget // (max(1, reference_count) * 64)))


def rgb_to_lab_array(rgb: Any, xp: Any) -> Any:
    value = xp.asarray(rgb, dtype=xp.float32) / xp.float32(255.0)
    linear = xp.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883
    threshold = 216 / 24389
    coefficient = 24389 / 27
    fx = xp.where(x > threshold, xp.cbrt(x), (coefficient * x + 16) / 116)
    fy = xp.where(y > threshold, xp.cbrt(y), (coefficient * y + 16) / 116)
    fz = xp.where(z > threshold, xp.cbrt(z), (coefficient * z + 16) / 116)
    return xp.stack((116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)), axis=-1)


def lab_to_rgb_array(lab: Any, xp: Any) -> Any:
    value = xp.asarray(lab, dtype=xp.float32)
    L, a, b = value[..., 0], value[..., 1], value[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    threshold = 216 / 24389
    coefficient = 24389 / 27

    def inverse(channel: Any) -> Any:
        cube = channel**3
        return xp.where(cube > threshold, cube, (116 * channel - 16) / coefficient)

    x, y, z = 0.95047 * inverse(fx), inverse(fy), 1.08883 * inverse(fz)
    rgb = xp.stack(
        (
            x * 3.2404542 + y * -1.5371385 + z * -0.4985314,
            x * -0.9692660 + y * 1.8760108 + z * 0.0415560,
            x * 0.0556434 + y * -0.2040259 + z * 1.0572252,
        ),
        axis=-1,
    )
    gamma = xp.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * xp.maximum(rgb, 0) ** (1 / 2.4) - 0.055)
    return xp.clip(gamma * 255, 0, 255)


def delta_e_2000_array(lab1: Any, lab2: Any, xp: Any) -> Any:
    one, two = xp.asarray(lab1, dtype=xp.float32), xp.asarray(lab2, dtype=xp.float32)
    L1, a1, b1 = one[..., 0], one[..., 1], one[..., 2]
    L2, a2, b2 = two[..., 0], two[..., 1], two[..., 2]
    C1, C2 = xp.hypot(a1, b1), xp.hypot(a2, b2)
    c_bar = (C1 + C2) / 2
    G = 0.5 * (1 - xp.sqrt(c_bar**7 / (c_bar**7 + 25**7)))
    ap1, ap2 = (1 + G) * a1, (1 + G) * a2
    cp1, cp2 = xp.hypot(ap1, b1), xp.hypot(ap2, b2)
    hp1 = xp.mod(xp.degrees(xp.arctan2(b1, ap1)), 360)
    hp2 = xp.mod(xp.degrees(xp.arctan2(b2, ap2)), 360)
    dL, dC = L2 - L1, cp2 - cp1
    product = cp1 * cp2
    hue_difference = hp2 - hp1
    dh = xp.where(
        product == 0,
        0,
        xp.where(xp.abs(hue_difference) <= 180, hue_difference, xp.where(hp2 <= hp1, hue_difference + 360, hue_difference - 360)),
    )
    dH = 2 * xp.sqrt(product) * xp.sin(xp.radians(dh / 2))
    l_bar, c_prime_bar = (L1 + L2) / 2, (cp1 + cp2) / 2
    h_bar = xp.where(
        product == 0,
        hp1 + hp2,
        xp.where(
            xp.abs(hp1 - hp2) <= 180,
            (hp1 + hp2) / 2,
            xp.where(hp1 + hp2 < 360, (hp1 + hp2 + 360) / 2, (hp1 + hp2 - 360) / 2),
        ),
    )
    T = (
        1
        - 0.17 * xp.cos(xp.radians(h_bar - 30))
        + 0.24 * xp.cos(xp.radians(2 * h_bar))
        + 0.32 * xp.cos(xp.radians(3 * h_bar + 6))
        - 0.20 * xp.cos(xp.radians(4 * h_bar - 63))
    )
    d_theta = 30 * xp.exp(-(((h_bar - 275) / 25) ** 2))
    Rc = 2 * xp.sqrt(c_prime_bar**7 / (c_prime_bar**7 + 25**7))
    Sl = 1 + 0.015 * (l_bar - 50) ** 2 / xp.sqrt(20 + (l_bar - 50) ** 2)
    Sc = 1 + 0.045 * c_prime_bar
    Sh = 1 + 0.015 * c_prime_bar * T
    Rt = -xp.sin(xp.radians(2 * d_theta)) * Rc
    return xp.sqrt((dL / Sl) ** 2 + (dC / Sc) ** 2 + (dH / Sh) ** 2 + Rt * (dC / Sc) * (dH / Sh))


def _point_arrays(points: Iterable[ColorPoint | Mapping[str, Any]], xp: Any) -> tuple[list[ColorPoint], Any, Any]:
    values = ensure_color_points(points)
    if not values:
        raise ValueError("at least one measured CMYK/Lab pair is required")
    cmyk = xp.asarray([point.cmyk for point in values], dtype=xp.float32)
    lab = xp.asarray([point.lab for point in values], dtype=xp.float32)
    return values, cmyk, lab


def _uint8_input(value: Any, xp: Any) -> Any:
    """Accept bytes from the PDF engine without treating them as a scalar string."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return xp.frombuffer(value, dtype=xp.uint8)
    return xp.asarray(value, dtype=xp.uint8)


class NearestLutMapper:
    """Map every non-transparent RGB pixel to the nearest measured color."""

    def __init__(
        self,
        points: Iterable[ColorPoint | Mapping[str, Any]],
        *,
        rgb_bits: int = 5,
        candidate_count: int = 48,
        backend: Backend | None = None,
        batch_size: int = 2048,
    ):
        if not 4 <= rgb_bits <= 8:
            raise ValueError("rgb_bits must be between 4 and 8")
        self.backend = backend or select_backend()
        self.xp = self.backend.xp
        self.points, self.measured_cmyk, self.measured_lab = _point_arrays(points, self.xp)
        self.rgb_bits = rgb_bits
        self.levels = (1 << rgb_bits) - 1
        self.candidate_count = min(len(self.points), max(1, candidate_count))
        self.batch_size = _bounded_batch_size(self.backend, max(1, batch_size), len(self.points))
        self._build_lut()

    def _build_lut(self) -> None:
        xp, size = self.xp, self.levels + 1
        q = xp.arange(size, dtype=xp.float32)
        r, g, b = xp.meshgrid(q, q, q, indexing="ij")
        rgb = xp.stack((r.ravel(), g.ravel(), b.ravel()), axis=1) * (255 / self.levels)
        targets = rgb_to_lab_array(rgb, xp)
        indices, deltas = [], []
        for start in range(0, targets.shape[0], self.batch_size):
            target = targets[start : start + self.batch_size]
            squared = xp.sum((target[:, None, :] - self.measured_lab[None, :, :]) ** 2, axis=2)
            if self.candidate_count < len(self.points):
                shortlist = xp.argpartition(squared, self.candidate_count - 1, axis=1)[:, : self.candidate_count]
                candidates = self.measured_lab[shortlist]
                delta = delta_e_2000_array(target[:, None, :], candidates, xp)
                local = xp.argmin(delta, axis=1)
                rows = xp.arange(target.shape[0])
                indices.append(shortlist[rows, local])
                deltas.append(delta[rows, local])
            else:
                delta = delta_e_2000_array(target[:, None, :], self.measured_lab[None, :, :], xp)
                local = xp.argmin(delta, axis=1)
                rows = xp.arange(target.shape[0])
                indices.append(local)
                deltas.append(delta[rows, local])
        selected = xp.concatenate(indices)
        self.lut_cmyk = xp.rint(self.measured_cmyk[selected] * 2.55).clip(0, 255).astype(xp.uint8)
        self.lut_preview = xp.rint(lab_to_rgb_array(self.measured_lab[selected], xp)).clip(0, 255).astype(xp.uint8)
        self.lut_delta = xp.concatenate(deltas).astype(xp.float32)

    def map_rgba(self, rgba: Any, *, width: int | None = None, height: int | None = None) -> dict[str, Any]:
        xp = self.xp
        source = _uint8_input(rgba, xp)
        original_shape = source.shape
        if source.ndim == 1:
            if source.size % 4:
                raise ValueError("rgba length must be divisible by four")
            source = source.reshape((-1, 4))
        elif source.shape[-1] == 4:
            source = source.reshape((-1, 4))
        else:
            raise ValueError("rgba must be a flat array or have four channels")
        if width is not None and height is not None and source.shape[0] != width * height:
            raise ValueError("rgba length does not match width and height")
        shift = 8 - self.rgb_bits
        quantized = source[:, :3] >> shift
        keys = ((quantized[:, 0].astype(xp.int64) * (self.levels + 1) + quantized[:, 1]) * (self.levels + 1) + quantized[:, 2])
        visible = source[:, 3] > 0
        cmyk = xp.zeros_like(source)
        preview = source.copy()
        cmyk[visible] = self.lut_cmyk[keys[visible]]
        preview[visible, :3] = self.lut_preview[keys[visible]]
        deltas = self.lut_delta[keys[visible]]
        visible_count = int(self.backend.to_host(xp.sum(visible)))
        transparent_count = int(source.shape[0] - visible_count)
        stats = {
            "totalPixels": int(source.shape[0]),
            "visiblePixels": visible_count,
            "transparentPixels": transparent_count,
            "semiTransparentPixels": int(self.backend.to_host(xp.sum((source[:, 3] > 0) & (source[:, 3] < 255)))),
            "meanDeltaE": float(self.backend.to_host(xp.mean(deltas))) if visible_count else 0.0,
            "maxDeltaE": float(self.backend.to_host(xp.max(deltas))) if visible_count else 0.0,
            "outOfGamutPixels": int(self.backend.to_host(xp.sum(deltas > 4))) if visible_count else 0,
        }
        result_shape = original_shape if len(original_shape) > 1 else (source.shape[0] * 4,)
        return {
            "cmyk": self.backend.to_host(cmyk.reshape(result_shape)),
            "preview": self.backend.to_host(preview.reshape(result_shape)),
            "stats": stats,
            "backend": self.backend.name,
        }


class ArrayForwardModel:
    """Batched GPU/CPU form of the local weighted CMYK -> Lab regression."""

    def __init__(
        self,
        points: Iterable[ColorPoint | Mapping[str, Any]],
        *,
        neighbor_count: int = 36,
        backend: Backend | None = None,
    ):
        self.backend = backend or select_backend()
        self.xp = self.backend.xp
        self.points, cmyk, self.lab = _point_arrays(points, self.xp)
        if len(self.points) < 5:
            raise ValueError("at least 5 distinct CMYK/Lab pairs are required")
        self.cmyk = cmyk / 100
        self.neighbor_count = min(len(self.points), max(8, neighbor_count))

    def predict_batch(self, recipes: Any, *, batch_size: int = 2048) -> tuple[Any, Any]:
        xp = self.xp
        batch_size = _bounded_batch_size(self.backend, max(1, batch_size), len(self.points))
        values = xp.asarray(recipes, dtype=xp.float32).reshape((-1, 4)) / 100
        predictions, nearest_distances = [], []
        for start in range(0, values.shape[0], batch_size):
            query = values[start : start + batch_size]
            distance = xp.sum((self.cmyk[None, :, :] - query[:, None, :]) ** 2, axis=2)
            k = self.neighbor_count
            index = xp.argpartition(distance, k - 1, axis=1)[:, :k]
            rows = xp.arange(query.shape[0])[:, None]
            selected_distance = distance[rows, index]
            percentile = min(k - 1, math.floor(k * 0.7))
            bandwidth = xp.maximum(xp.sqrt(xp.partition(selected_distance, percentile, axis=1)[:, percentile]) * 1.15, 0.055)
            neighbor_cmyk = self.cmyk[index]
            features = xp.concatenate((xp.ones((query.shape[0], k, 1), dtype=xp.float32), neighbor_cmyk - query[:, None, :]), axis=2)
            weights = xp.exp(-selected_distance / (2 * bandwidth[:, None] ** 2))
            neighbor_lab = self.lab[index]
            matrix = xp.einsum("bni,bn,bnj->bij", features, weights, features)
            outputs = xp.einsum("bni,bn,bnc->bic", features, weights, neighbor_lab)
            diagonal = xp.arange(1, 5)
            matrix[:, diagonal, diagonal] += 0.0025
            matrix[:, 0, 0] += 1e-8
            solved = xp.linalg.solve(matrix, outputs)
            prediction = solved[:, 0, :]
            prediction[:, 0] = xp.clip(prediction[:, 0], 0, 100)
            prediction[:, 1:] = xp.clip(prediction[:, 1:], -160, 160)
            predictions.append(prediction)
            nearest_distances.append(xp.sqrt(xp.min(selected_distance, axis=1)))
        return xp.concatenate(predictions), xp.concatenate(nearest_distances)


class InverseLutMapper:
    """Continuous "predict new color" mapper for texture/image pixels."""

    def __init__(
        self,
        points: Iterable[ColorPoint | Mapping[str, Any]],
        *,
        grid_step: float = 12.5,
        lut_size: int = 9,
        neighbor_count: int = 36,
        candidate_count: int = 48,
        backend: Backend | None = None,
        batch_size: int = 1024,
    ):
        if not 0 < grid_step <= 50:
            raise ValueError("grid_step must be greater than zero and at most 50")
        if not 3 <= lut_size <= 65:
            raise ValueError("lut_size must be between 3 and 65")
        self.backend = backend or select_backend()
        self.xp = self.backend.xp
        self.forward = ArrayForwardModel(points, neighbor_count=neighbor_count, backend=self.backend)
        if len(self.forward.points) < 8:
            raise ValueError("at least 8 measured CMYK/Lab pairs are required")
        self.grid_step = float(grid_step)
        self.lut_size = int(lut_size)
        self.candidate_count = max(1, int(candidate_count))
        self.batch_size = max(1, int(batch_size))
        self._build_lut()

    def _closest_candidates(self, targets: Any, candidate_lab: Any, candidate_cmyk: Any) -> Any:
        xp = self.xp
        recipes = []
        count = min(self.candidate_count, int(candidate_lab.shape[0]))
        batch_size = _bounded_batch_size(self.backend, self.batch_size, int(candidate_lab.shape[0]))
        for start in range(0, targets.shape[0], batch_size):
            target = targets[start : start + batch_size]
            euclidean = xp.sum((target[:, None, :] - candidate_lab[None, :, :]) ** 2, axis=2)
            shortlist = xp.argpartition(euclidean, count - 1, axis=1)[:, :count]
            lab = candidate_lab[shortlist]
            delta = delta_e_2000_array(target[:, None, :], lab, xp)
            local = xp.argmin(delta, axis=1)
            rows = xp.arange(target.shape[0])
            recipes.append(candidate_cmyk[shortlist[rows, local]])
        return xp.concatenate(recipes)

    def _refine(self, targets: Any, recipes: Any) -> tuple[Any, Any, Any]:
        xp = self.xp
        prediction, nearest = self.forward.predict_batch(recipes, batch_size=self.batch_size)
        best_delta = delta_e_2000_array(targets, prediction, xp)
        steps = [value for value in (self.grid_step / 2, self.grid_step / 4, self.grid_step / 10) if value >= 1]
        for step in steps:
            for _round in range(2):
                improved_any = False
                for channel in range(4):
                    for direction in (-1, 1):
                        candidate = recipes.copy()
                        candidate[:, channel] = xp.clip(candidate[:, channel] + step * direction, 0, 100)
                        candidate_prediction, candidate_nearest = self.forward.predict_batch(candidate, batch_size=self.batch_size)
                        candidate_delta = delta_e_2000_array(targets, candidate_prediction, xp)
                        improved = candidate_delta + 1e-7 < best_delta
                        if bool(self.backend.to_host(xp.any(improved))):
                            recipes = xp.where(improved[:, None], candidate, recipes)
                            prediction = xp.where(improved[:, None], candidate_prediction, prediction)
                            nearest = xp.where(improved, candidate_nearest, nearest)
                            best_delta = xp.where(improved, candidate_delta, best_delta)
                            improved_any = True
                if not improved_any:
                    break
        return recipes, prediction, xp.stack((best_delta, nearest), axis=1)

    def _build_lut(self) -> None:
        xp = self.xp
        grid_values = []
        value = 0.0
        while value < 100:
            grid_values.append(value)
            value += self.grid_step
        if not grid_values or grid_values[-1] != 100:
            grid_values.append(100.0)
        candidate_cmyk = xp.asarray(list(itertools.product(grid_values, repeat=4)), dtype=xp.float32)
        candidate_lab, _ = self.forward.predict_batch(candidate_cmyk, batch_size=self.batch_size)

        axis = xp.linspace(0, 255, self.lut_size, dtype=xp.float32)
        r, g, b = xp.meshgrid(axis, axis, axis, indexing="ij")
        rgb_nodes = xp.stack((r.ravel(), g.ravel(), b.ravel()), axis=1)
        targets = rgb_to_lab_array(rgb_nodes, xp)
        initial = self._closest_candidates(targets, candidate_lab, candidate_cmyk)
        recipes, prediction, metrics = self._refine(targets, initial)
        shape = (self.lut_size, self.lut_size, self.lut_size)
        self.lut_cmyk = recipes.reshape((*shape, 4))
        self.lut_preview = lab_to_rgb_array(prediction, xp).reshape((*shape, 3))
        self.lut_delta = metrics[:, 0].reshape(shape)
        self.lut_distance = metrics[:, 1].reshape(shape)
        self.model_candidates = int(candidate_cmyk.shape[0])

    def _interpolate(self, array: Any, rgb: Any) -> Any:
        xp = self.xp
        scale = (self.lut_size - 1) / 255
        location = rgb.astype(xp.float32) * scale
        lower = xp.floor(location).astype(xp.int32)
        upper = xp.minimum(self.lut_size - 1, lower + 1)
        fraction = location - lower
        channels = 1 if array.ndim == 3 else array.shape[-1]
        result = xp.zeros((rgb.shape[0], channels), dtype=xp.float32)
        for x_upper, y_upper, z_upper in itertools.product((0, 1), repeat=3):
            ix = upper[:, 0] if x_upper else lower[:, 0]
            iy = upper[:, 1] if y_upper else lower[:, 1]
            iz = upper[:, 2] if z_upper else lower[:, 2]
            wx = fraction[:, 0] if x_upper else 1 - fraction[:, 0]
            wy = fraction[:, 1] if y_upper else 1 - fraction[:, 1]
            wz = fraction[:, 2] if z_upper else 1 - fraction[:, 2]
            sampled = array[ix, iy, iz]
            if channels == 1:
                sampled = sampled[:, None]
            result += sampled * (wx * wy * wz)[:, None]
        return result

    def map_rgba(self, rgba: Any, *, width: int | None = None, height: int | None = None) -> dict[str, Any]:
        xp = self.xp
        source = _uint8_input(rgba, xp)
        original_shape = source.shape
        if source.ndim == 1:
            if source.size % 4:
                raise ValueError("rgba length must be divisible by four")
            source = source.reshape((-1, 4))
        elif source.shape[-1] == 4:
            source = source.reshape((-1, 4))
        else:
            raise ValueError("rgba must be a flat array or have four channels")
        if width is not None and height is not None and source.shape[0] != width * height:
            raise ValueError("rgba length does not match width and height")
        visible = source[:, 3] > 0
        visible_rgb = source[visible, :3]
        recipe = self._interpolate(self.lut_cmyk, visible_rgb)
        preview_rgb = self._interpolate(self.lut_preview, visible_rgb)
        delta = self._interpolate(self.lut_delta, visible_rgb)[:, 0]
        distance = self._interpolate(self.lut_distance, visible_rgb)[:, 0]
        cmyk = xp.zeros_like(source)
        preview = source.copy()
        cmyk[visible] = xp.rint(recipe * 2.55).clip(0, 255).astype(xp.uint8)
        preview[visible, :3] = xp.rint(preview_rgb).clip(0, 255).astype(xp.uint8)
        visible_count = int(self.backend.to_host(xp.sum(visible)))
        stats = {
            "totalPixels": int(source.shape[0]),
            "visiblePixels": visible_count,
            "transparentPixels": int(source.shape[0] - visible_count),
            "semiTransparentPixels": int(self.backend.to_host(xp.sum((source[:, 3] > 0) & (source[:, 3] < 255)))),
            "meanDeltaE": float(self.backend.to_host(xp.mean(delta))) if visible_count else 0.0,
            "maxDeltaE": float(self.backend.to_host(xp.max(delta))) if visible_count else 0.0,
            "outOfGamutPixels": int(self.backend.to_host(xp.sum(delta > 4))) if visible_count else 0,
            "extrapolatedPixels": int(self.backend.to_host(xp.sum(distance > 0.18))) if visible_count else 0,
        }
        result_shape = original_shape if len(original_shape) > 1 else (source.shape[0] * 4,)
        return {
            "cmyk": self.backend.to_host(cmyk.reshape(result_shape)),
            "preview": self.backend.to_host(preview.reshape(result_shape)),
            "stats": stats,
            "backend": self.backend.name,
            "modelCandidates": self.model_candidates,
        }
