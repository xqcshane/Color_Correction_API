"""Color math shared by the GPU API.

The scalar implementation intentionally mirrors ``color-engine.js`` so that API
results can be regression-tested against the current browser implementation.
It has no third-party dependency; array/GPU acceleration lives in
``color_array.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Iterable, Mapping, Sequence


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


@dataclass(slots=True)
class ColorPoint:
    c: float
    m: float
    y: float
    k: float
    L: float
    a: float
    b: float
    count: float = 1.0
    patch_code: str = ""
    patch_label: str = ""
    source_file: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cmyk(self) -> tuple[float, float, float, float]:
        return self.c, self.m, self.y, self.k

    @property
    def lab(self) -> tuple[float, float, float]:
        return self.L, self.a, self.b

    def as_dict(self) -> dict[str, Any]:
        return {
            "c": self.c,
            "m": self.m,
            "y": self.y,
            "k": self.k,
            "L": self.L,
            "a": self.a,
            "b": self.b,
            "count": self.count,
            "patchCode": self.patch_code,
            "patchLabel": self.patch_label,
            "sourceFile": self.source_file,
            "meta": self.meta,
        }


def _lab_tuple(value: ColorPoint | Mapping[str, float] | Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, ColorPoint):
        return value.lab
    if isinstance(value, Mapping):
        return float(value["L"]), float(value["a"]), float(value["b"])
    return float(value[0]), float(value[1]), float(value[2])


def _cmyk_tuple(value: ColorPoint | Mapping[str, float] | Sequence[float]) -> tuple[float, float, float, float]:
    if isinstance(value, ColorPoint):
        return value.cmyk
    if isinstance(value, Mapping):
        return float(value["c"]), float(value["m"]), float(value["y"]), float(value["k"])
    return float(value[0]), float(value[1]), float(value[2]), float(value[3])


def rgb_to_lab(rgb: Mapping[str, float] | Sequence[float]) -> tuple[float, float, float]:
    if isinstance(rgb, Mapping):
        r, g, b = float(rgb["r"]), float(rgb["g"]), float(rgb["b"])
    else:
        r, g, b = map(float, rgb[:3])

    def linear(value: float) -> float:
        value /= 255.0
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = linear(r), linear(g), linear(b)
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883

    def f(value: float) -> float:
        return value ** (1.0 / 3.0) if value > 216 / 24389 else (24389 / 27 * value + 16) / 116

    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def lab_to_rgb(lab: ColorPoint | Mapping[str, float] | Sequence[float]) -> tuple[float, float, float]:
    L, a, b = _lab_tuple(lab)
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    def inverse(value: float) -> float:
        return value**3 if value**3 > 216 / 24389 else (116 * value - 16) / (24389 / 27)

    x, y, z = 0.95047 * inverse(fx), inverse(fy), 1.08883 * inverse(fz)
    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    blue = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    def gamma(value: float) -> float:
        return 255 * (12.92 * value if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055)

    return clamp(gamma(r), 0, 255), clamp(gamma(g), 0, 255), clamp(gamma(blue), 0, 255)


def delta_e_2000(
    lab1: ColorPoint | Mapping[str, float] | Sequence[float],
    lab2: ColorPoint | Mapping[str, float] | Sequence[float],
) -> float:
    """CIEDE2000 with kL = kC = kH = 1 (same equations as the JS app)."""

    L1, a1, b1 = _lab_tuple(lab1)
    L2, a2, b2 = _lab_tuple(lab2)
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    c_bar = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(c_bar**7 / (c_bar**7 + 25**7)))
    ap1, ap2 = (1 + G) * a1, (1 + G) * a2
    cp1, cp2 = math.hypot(ap1, b1), math.hypot(ap2, b2)

    def hue(a: float, b: float) -> float:
        if a == 0 and b == 0:
            return 0
        return math.degrees(math.atan2(b, a)) % 360

    hp1, hp2 = hue(ap1, b1), hue(ap2, b2)
    dL, dC = L2 - L1, cp2 - cp1
    if cp1 * cp2 == 0:
        dh = 0
    elif abs(hp2 - hp1) <= 180:
        dh = hp2 - hp1
    elif hp2 <= hp1:
        dh = hp2 - hp1 + 360
    else:
        dh = hp2 - hp1 - 360
    dH = 2 * math.sqrt(cp1 * cp2) * math.sin(math.radians(dh / 2))
    l_bar, c_prime_bar = (L1 + L2) / 2, (cp1 + cp2) / 2
    if cp1 * cp2 == 0:
        h_bar = hp1 + hp2
    elif abs(hp1 - hp2) <= 180:
        h_bar = (hp1 + hp2) / 2
    elif hp1 + hp2 < 360:
        h_bar = (hp1 + hp2 + 360) / 2
    else:
        h_bar = (hp1 + hp2 - 360) / 2
    T = (
        1
        - 0.17 * math.cos(math.radians(h_bar - 30))
        + 0.24 * math.cos(math.radians(2 * h_bar))
        + 0.32 * math.cos(math.radians(3 * h_bar + 6))
        - 0.20 * math.cos(math.radians(4 * h_bar - 63))
    )
    d_theta = 30 * math.exp(-(((h_bar - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(c_prime_bar**7 / (c_prime_bar**7 + 25**7))
    Sl = 1 + 0.015 * (l_bar - 50) ** 2 / math.sqrt(20 + (l_bar - 50) ** 2)
    Sc = 1 + 0.045 * c_prime_bar
    Sh = 1 + 0.015 * c_prime_bar * T
    Rt = -math.sin(math.radians(2 * d_theta)) * Rc
    return math.sqrt((dL / Sl) ** 2 + (dC / Sc) ** 2 + (dH / Sh) ** 2 + Rt * (dC / Sc) * (dH / Sh))


def find_nearest_lab(
    points: Sequence[ColorPoint], target: ColorPoint | Mapping[str, float] | Sequence[float], limit: int = 5
) -> list[tuple[ColorPoint, float]]:
    return sorted(((point, delta_e_2000(point, target)) for point in points), key=lambda item: item[1])[:limit]


def _solve_system(matrix: list[list[float]], outputs: list[list[float]]) -> list[list[float]] | None:
    n, channels = len(matrix), len(outputs[0])
    augmented = [matrix[index][:] + outputs[index][:] for index in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-10:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for cell in range(column, n + channels):
            augmented[column][cell] /= divisor
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            for cell in range(column, n + channels):
                augmented[row][cell] -= factor * augmented[column][cell]
    return [row[n:] for row in augmented]


class LocalWeightedRegression:
    """CMYK -> measured Lab local-linear model used by "predict new color".

    For every query it selects nearby measured CMYK recipes, applies a Gaussian
    kernel and solves a ridge-stabilised local linear regression.  This is an
    exact scalar port of ``createForwardModel`` in ``color-engine.js``.
    """

    def __init__(self, points: Sequence[ColorPoint], neighbor_count: int | None = None):
        if len(points) < 5:
            raise ValueError("at least 5 distinct CMYK/Lab pairs are required")
        self.points = list(points)
        default = round(math.sqrt(len(points)) * 1.15)
        requested = neighbor_count if neighbor_count is not None else default
        self.neighbor_count = int(clamp(requested, 12, 36))

    def _neighbors(self, cmyk: Sequence[float], exclude_index: int = -1) -> list[tuple[int, ColorPoint, float]]:
        result: list[tuple[int, ColorPoint, float]] = []
        for index, point in enumerate(self.points):
            if index == exclude_index:
                continue
            distance = sum(((value - query) / 100) ** 2 for value, query in zip(point.cmyk, cmyk))
            result.append((index, point, distance))
        count = min(self.neighbor_count, len(result))
        return sorted(result, key=lambda item: item[2])[:count]

    def predict(self, cmyk: ColorPoint | Mapping[str, float] | Sequence[float], exclude_index: int = -1) -> dict[str, float]:
        x = _cmyk_tuple(cmyk)
        neighbors = self._neighbors(x, exclude_index)
        if not neighbors:
            raise ValueError("no neighbors are available")
        percentile_index = min(len(neighbors) - 1, math.floor(len(neighbors) * 0.7))
        bandwidth = max(math.sqrt(neighbors[percentile_index][2]) * 1.15, 0.055)
        matrix = [[0.0] * 5 for _ in range(5)]
        outputs = [[0.0] * 3 for _ in range(5)]
        fallback = [0.0] * 3
        weight_sum = 0.0
        for _, point, distance in neighbors:
            weight = math.exp(-distance / (2 * bandwidth**2))
            features = [1.0, *((value - query) / 100 for value, query in zip(point.cmyk, x))]
            values = point.lab
            for row in range(5):
                for column in range(5):
                    matrix[row][column] += weight * features[row] * features[column]
                for channel in range(3):
                    outputs[row][channel] += weight * features[row] * values[channel]
            for channel in range(3):
                fallback[channel] += weight * values[channel]
            weight_sum += weight
        for index in range(1, 5):
            matrix[index][index] += 0.0025
        matrix[0][0] += 1e-8
        solved = _solve_system(matrix, outputs)
        predicted = solved[0] if solved is not None else [value / weight_sum for value in fallback]
        return {
            "L": clamp(predicted[0], 0, 100),
            "a": clamp(predicted[1], -160, 160),
            "b": clamp(predicted[2], -160, 160),
            "nearestDistance": math.sqrt(neighbors[0][2]),
            "neighbors": len(neighbors),
            "bandwidth": bandwidth,
        }

    def cross_validate(self, max_samples: int = 120) -> dict[str, float | int]:
        sample_count = min(max_samples, len(self.points))
        errors: list[float] = []
        for sample in range(sample_count):
            index = min(len(self.points) - 1, math.floor(sample * len(self.points) / sample_count))
            point = self.points[index]
            errors.append(delta_e_2000(point, self.predict(point.cmyk, index)))
        errors.sort()

        def quantile(q: float) -> float:
            return errors[min(len(errors) - 1, math.floor((len(errors) - 1) * q))] if errors else 0

        return {
            "n": len(errors),
            "rmse": math.sqrt(sum(value**2 for value in errors) / max(1, len(errors))),
            "median": quantile(0.5),
            "p90": quantile(0.9),
        }


def optimize_cmyk(
    model: LocalWeightedRegression,
    target: ColorPoint | Mapping[str, float] | Sequence[float],
    *,
    seed: int = 92821,
    random_candidates: int = 90,
) -> dict[str, Any]:
    """Search a continuous CMYK recipe, matching the browser's deterministic search."""

    rng = random.Random(seed)
    candidates = [list(point.cmyk) for point, _ in find_nearest_lab(model.points, target, 10)]
    candidates.extend(([0, 0, 0, 0], [100, 100, 100, 100], [0, 0, 0, 100]))
    candidates.extend([[rng.random() * 100 for _ in range(4)] for _ in range(random_candidates)])

    def evaluate(cmyk: Sequence[float]) -> dict[str, Any]:
        bounded = [clamp(value, 0, 100) for value in cmyk]
        prediction = model.predict(bounded)
        return {"cmyk": bounded, "prediction": prediction, "delta": delta_e_2000(prediction, target)}

    best = sorted((evaluate(value) for value in candidates), key=lambda item: item["delta"])[:10]
    for step in (20, 10, 5, 2, 1, 0.5):
        next_items: list[dict[str, Any]] = []
        for item in best:
            local = item
            improved, rounds = True, 0
            while improved and rounds < 3:
                improved, rounds = False, rounds + 1
                for channel in range(4):
                    for direction in (-1, 1):
                        cmyk = local["cmyk"][:]
                        cmyk[channel] = clamp(cmyk[channel] + step * direction, 0, 100)
                        result = evaluate(cmyk)
                        if result["delta"] + 1e-7 < local["delta"]:
                            local, improved = result, True
            next_items.append(local)
        best = sorted([*best, *next_items], key=lambda item: item["delta"])[:8]
    result = best[0]
    result["cmyk"] = [round(value, 1) for value in result["cmyk"]]
    result["prediction"] = model.predict(result["cmyk"])
    result["delta"] = delta_e_2000(result["prediction"], target)
    result["boundaryChannels"] = [index for index, value in enumerate(result["cmyk"]) if value <= 0.5 or value >= 99.5]
    result["extrapolation"] = result["prediction"]["nearestDistance"]
    return result


def ensure_color_points(points: Iterable[ColorPoint | Mapping[str, Any]]) -> list[ColorPoint]:
    result: list[ColorPoint] = []
    for value in points:
        if isinstance(value, ColorPoint):
            result.append(value)
            continue
        result.append(
            ColorPoint(
                c=float(value["c"]),
                m=float(value["m"]),
                y=float(value["y"]),
                k=float(value["k"]),
                L=float(value["L"]),
                a=float(value["a"]),
                b=float(value["b"]),
                count=float(value.get("count", 1)),
                patch_code=str(value.get("patchCode", value.get("patch_code", ""))),
                patch_label=str(value.get("patchLabel", value.get("patch_label", ""))),
                source_file=str(value.get("sourceFile", value.get("source_file", ""))),
                meta=dict(value.get("meta") or {}),
            )
        )
    return result
