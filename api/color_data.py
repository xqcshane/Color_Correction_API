"""Import local-color-map CSV/JSON exports into one canonical API structure."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass, field
import io
import json
import math
import re
from typing import Any, Iterable, Mapping

from .color_core import ColorPoint, delta_e_2000


@dataclass(slots=True)
class RejectedPair:
    row: int
    reason: str


@dataclass(slots=True)
class PairingImport:
    points: list[ColorPoint]
    rejected: list[RejectedPair] = field(default_factory=list)
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "cmyk-recipe-pairs/v1",
            "source": self.source,
            "points": [point.as_dict() for point in self.points],
            "rejected": [{"row": item.row, "reason": item.reason} for item in self.rejected],
        }


_HEADER_RE = re.compile(r"[\s%％*()（）_\-/\\]")


def _normalize_header(value: Any) -> str:
    return _HEADER_RE.sub("", str(value or "").strip().lower())


def _number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(str(value).strip().removesuffix("%"))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _meta(meta: Mapping[str, Any] | None = None) -> dict[str, str]:
    value = meta or {}

    def clean(name: str) -> str:
        return str(value.get(name, "") or "").strip()

    result = {
        name: clean(name)
        for name in (
            "fabricCode",
            "fabricName",
            "material",
            "manufacturer",
            "batch",
            "instrument",
            "instrumentSerial",
            "illuminant",
            "observer",
            "geometry",
            "specular",
            "aperture",
            "uv",
            "profile",
            "session",
        )
    }
    result["key"] = "|".join(
        (
            result["fabricCode"] or result["fabricName"] or result["material"],
            result["manufacturer"],
            result["batch"],
            result["instrument"],
            result["instrumentSerial"],
            result["illuminant"],
            result["observer"],
            result["geometry"],
            result["specular"],
            result["aperture"],
            result["uv"],
            result["profile"],
            result["session"],
        )
    )
    result["label"] = " · ".join(
        filter(
            None,
            (
                result["fabricName"] or result["fabricCode"] or result["material"] or "未命名布料",
                result["manufacturer"],
                result["instrument"],
                f"测量组 {result['session']}" if result["session"] else "",
            ),
        )
    )
    return result


def _valid(point: ColorPoint) -> bool:
    return (
        all(math.isfinite(value) and 0 <= value <= 100 for value in point.cmyk)
        and math.isfinite(point.L)
        and 0 <= point.L <= 100
        and math.isfinite(point.a)
        and math.isfinite(point.b)
        and math.isfinite(point.count)
        and point.count > 0
    )


def parse_color_map_csv(text: str | bytes, file_name: str = "CSV") -> PairingImport:
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(str(text).lstrip("\ufeff")))
    first = next((row for row in reader if any(str(value).strip() for value in row)), None)
    if first is None:
        raise ValueError("CSV 中没有可读取的数据行")
    headers = [_normalize_header(value) for value in first]
    nonempty_headers = [value for value in headers if value]
    if len(set(nonempty_headers)) != len(nonempty_headers):
        raise ValueError("CSV 存在重复的列名")
    points: list[ColorPoint] = []
    rejected: list[RejectedPair] = []

    for values in reader:
        offset = reader.line_num
        if not any(str(value).strip() for value in values):
            continue
        row = {header: values[index] if index < len(values) else "" for index, header in enumerate(headers)}

        def get(*aliases: str) -> str:
            for alias in aliases:
                value = row.get(_normalize_header(alias), "")
                if str(value).strip():
                    return value
            return ""

        raw = (
            _number(get("布上标签C (%)", "布上标签C", "标签C", "C (%)", "C")),
            _number(get("布上标签M (%)", "布上标签M", "标签M", "M (%)", "M")),
            _number(get("布上标签Y (%)", "布上标签Y", "标签Y", "Y (%)", "Y")),
            _number(get("布上标签K (%)", "布上标签K", "标签K", "K (%)", "K")),
            _number(get("平均仪器L*", "仪器L*", "实测L*", "Lab L", "L*", "L")),
            _number(get("平均仪器a*", "仪器a*", "实测a*", "Lab a", "a*", "a")),
            _number(get("平均仪器b*", "仪器b*", "实测b*", "Lab b", "b*", "b")),
        )
        if any(value is None for value in raw):
            rejected.append(RejectedPair(offset, "缺少有效的标签 CMYK 或实测 Lab"))
            continue
        count_raw = get("读数数量", "重复数量", "count")
        count = _number(count_raw) if count_raw else 1
        if count is None or count <= 0:
            rejected.append(RejectedPair(offset, "读数数量必须是大于 0 的有效数字"))
            continue
        point = ColorPoint(
            *[float(value) for value in raw],
            count=count,
            patch_code=str(get("色块编号", "色块", "编号") or f"行 {offset}").strip(),
            patch_label=str(get("标签色号", "色块名称", "标签") or "").strip(),
            source_file=file_name,
            meta=_meta(
                {
                    "instrument": get("仪器", "测量仪器", "instrument", "设备", "device"),
                    "instrumentSerial": get("仪器编号", "仪器序列号", "instrumentSerial"),
                    "illuminant": get("光源", "illuminant"),
                    "observer": get("观察者", "观察者角度", "observer"),
                    "geometry": get("几何结构", "geometry"),
                    "specular": get("镜面反射", "specular"),
                    "aperture": get("测量口径", "aperture"),
                    "uv": get("UV"),
                    "profile": get("CMYK转换设置", "profile"),
                    "fabricCode": get("布料编号", "fabricCode"),
                    "fabricName": get("布料名称", "fabricName"),
                    "material": get("材质", "material"),
                    "manufacturer": get("生产商家", "生产商", "商家", "manufacturer"),
                    "batch": get("批次", "batch"),
                    "session": get("测量组", "session"),
                }
            ),
        )
        if _valid(point):
            points.append(point)
        else:
            rejected.append(RejectedPair(offset, "缺少有效的标签 CMYK 或实测 Lab"))
    if not points:
        raise ValueError("没有找到同时包含标签 CMYK 与实测 Lab 的有效行")
    return PairingImport(points, rejected, file_name)


def _point_from_direct(value: Mapping[str, Any], index: int, file_name: str) -> ColorPoint | None:
    channels = tuple(_number(value.get(name)) for name in ("c", "m", "y", "k", "L", "a", "b"))
    if any(channel is None for channel in channels):
        return None
    point = ColorPoint(
        *[float(channel) for channel in channels],
        count=_number(value.get("count")) or 1,
        patch_code=str(value.get("patchCode", value.get("patch_code", f"记录 {index + 1}"))),
        patch_label=str(value.get("patchLabel", value.get("patch_label", ""))),
        source_file=file_name,
        meta=_meta(value.get("meta") if isinstance(value.get("meta"), Mapping) else {}),
    )
    return point if _valid(point) else None


def parse_color_map_json(text: str | bytes | Mapping[str, Any] | list[Any], file_name: str = "JSON") -> PairingImport:
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")
    try:
        data = json.loads(text) if isinstance(text, str) else text
    except json.JSONDecodeError as error:
        raise ValueError("JSON 文件格式无效") from error

    # The API's canonical schema and simple {points:[...]} exports.
    if isinstance(data, list):
        direct = data
    elif isinstance(data, Mapping) and isinstance(data.get("points"), list):
        direct = data["points"]
    else:
        direct = None
    if direct is not None:
        points, rejected = [], []
        for index, value in enumerate(direct):
            point = _point_from_direct(value, index, file_name) if isinstance(value, Mapping) else None
            if point is None:
                rejected.append(RejectedPair(index + 1, "记录不是有效的 Lab + 标签 CMYK 配对"))
            else:
                points.append(point)
        if not points:
            raise ValueError("JSON 中没有有效的 Lab + 标签 CMYK 配对")
        return PairingImport(points, rejected, file_name)

    state = data.get("state", data) if isinstance(data, Mapping) else None
    if not isinstance(state, Mapping) or not isinstance(state.get("records"), list):
        raise ValueError("JSON 中没有 local-color-map 的 records 数据")
    patches = {str(item.get("id")): item for item in state.get("patches", []) if isinstance(item, Mapping)}
    fabrics = {str(item.get("id")): item for item in state.get("fabrics", []) if isinstance(item, Mapping)}
    instruments = {str(item.get("id")): item for item in state.get("instruments", []) if isinstance(item, Mapping)}
    points, rejected = [], []
    for index, record in enumerate(state["records"]):
        if not isinstance(record, Mapping):
            rejected.append(RejectedPair(index + 1, "记录不是有效的 Lab + 标签 CMYK 配对"))
            continue
        patch = record.get("patchSnapshot") or patches.get(str(record.get("patchId"))) or {}
        fabric = record.get("fabricSnapshot") or fabrics.get(str(record.get("fabricId"))) or {}
        instrument = record.get("instrumentSnapshot") or instruments.get(str(record.get("instrumentId"))) or {}
        direct_record = {
            "c": patch.get("c"),
            "m": patch.get("m"),
            "y": patch.get("y"),
            "k": patch.get("k"),
            "L": record.get("L"),
            "a": record.get("a"),
            "b": record.get("b"),
            "patchCode": patch.get("code") or record.get("patchId") or f"记录 {index + 1}",
            "patchLabel": patch.get("label") or "",
            "meta": {
                "instrument": instrument.get("name"),
                "instrumentSerial": instrument.get("serial"),
                "illuminant": instrument.get("illuminant"),
                "observer": instrument.get("observer"),
                "geometry": instrument.get("geometry"),
                "specular": instrument.get("specular"),
                "aperture": instrument.get("aperture"),
                "uv": instrument.get("uv"),
                "profile": instrument.get("cmykProfile"),
                "fabricCode": fabric.get("code"),
                "fabricName": fabric.get("name"),
                "material": fabric.get("material"),
                "manufacturer": fabric.get("manufacturer"),
                "batch": fabric.get("batch"),
                "session": record.get("session"),
            },
        }
        point = _point_from_direct(direct_record, index, file_name)
        if point is None:
            rejected.append(RejectedPair(index + 1, "记录不是有效的 Lab + 标签 CMYK 配对"))
        else:
            points.append(point)
    if not points:
        raise ValueError("JSON 中没有有效的 Lab + 标签 CMYK 配对")
    return PairingImport(points, rejected, file_name)


def aggregate_pairs(points: Iterable[ColorPoint]) -> list[ColorPoint]:
    """Merge repeated readings with identical CMYK, weighted by reading count."""

    groups: dict[tuple[float, float, float, float], list[ColorPoint]] = defaultdict(list)
    for point in points:
        groups[tuple(round(value, 4) for value in point.cmyk)].append(point)
    result: list[ColorPoint] = []
    for group in groups.values():
        total = sum(point.count or 1 for point in group)
        mean = tuple(sum(getattr(point, channel) * (point.count or 1) for point in group) / total for channel in ("L", "a", "b"))
        spread = math.sqrt(sum(delta_e_2000(point, mean) ** 2 for point in group) / len(group)) if len(group) > 1 else 0
        first = group[0]
        result.append(
            ColorPoint(
                *first.cmyk,
                *mean,
                count=total,
                patch_code=" / ".join(dict.fromkeys(point.patch_code for point in group if point.patch_code)),
                patch_label=" / ".join(dict.fromkeys(point.patch_label for point in group if point.patch_label)),
                source_file=" / ".join(dict.fromkeys(point.source_file for point in group if point.source_file)),
                meta={**first.meta, "repeatSpread": spread},
            )
        )
    return result
