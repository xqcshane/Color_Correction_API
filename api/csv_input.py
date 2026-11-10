"""Strict CMYK–Lab CSV input for the standalone processing endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .color_data import aggregate_pairs, parse_color_map_csv


def parse_pairs_csv(path: str | Path, method: str) -> list[dict[str, Any]]:
    """Return one validated measurement group's canonical, aggregated points.

    CSV uses CMYK percentages (0–100), Lab L (0–100), and finite a/b.
    Standard seven-column files and the original local-color-map Chinese
    exports are supported. An API cannot ask the caller to select a data
    group interactively, so ambiguous groups and invalid rows are rejected.
    """

    if method not in {"nearest", "model"}:
        raise ValueError("method 必须是 nearest 或 model")
    source = Path(path)
    # UTF-8 is the normal export format; GB18030 also accepts Excel CSV
    # exports with Chinese headers. Never replace undecodable characters.
    try:
        text = source.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = source.read_text(encoding="gb18030")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV 编码须为 UTF-8 或 GB18030") from exc
    imported = parse_color_map_csv(text, source.name)
    if imported.rejected:
        rows = ", ".join(str(item.row) for item in imported.rejected[:12])
        suffix = " 等" if len(imported.rejected) > 12 else ""
        raise ValueError(f"CSV 第 {rows}{suffix} 行缺少有效 CMYK–Lab 配对；请修正后提交")
    for point in imported.points:
        if point.count <= 0:
            raise ValueError("CSV 读数数量必须大于 0")
    group_keys = {str(point.meta.get("key", "")) for point in imported.points}
    if len(group_keys) > 1:
        raise ValueError("CSV 包含多个布料、设备、批次或测量组；每次请求只能提交同一组数据")
    points = aggregate_pairs(imported.points)
    minimum = 8 if method == "model" else 1
    if len(points) < minimum:
        raise ValueError(f"当前算法至少需要 {minimum} 个不同 CMYK 的有效配对（重复读数已合并）")
    return [point.as_dict() for point in points]
