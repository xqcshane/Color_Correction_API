from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Any, Mapping

from .color_array import InverseLutMapper, NearestLutMapper, select_backend
from .color_core import ensure_color_points
from .color_data import aggregate_pairs
from .jobs import JobCancelled, JobContext
from .pdf_engine import PdfProcessingCancelled, TileResult, replace_pdf_layers


def _as_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    tobytes = getattr(value, "tobytes", None)
    if callable(tobytes):
        return tobytes()
    return bytes(value)


def _build_mapper(options: Mapping[str, Any], context: JobContext):
    raw_points = options.get("points")
    if not isinstance(raw_points, list):
        raise ValueError("任务缺少标准化的 CMYK–Lab 配对 points")
    points = aggregate_pairs(ensure_color_points(raw_points))
    method = str(options.get("method") or "").strip().lower()
    quality = str(options.get("quality") or "fine").strip().lower()
    backend = select_backend(prefer_gpu=not bool(options.get("forceCpu", False)))
    context.report(0.02, "建立颜色引擎", f"{backend.device_name} · {len(points)} 个实测配方")
    started = time.perf_counter()

    if method == "nearest":
        settings = (
            {"rgb_bits": 6, "candidate_count": min(len(points), 256), "batch_size": 2048}
            if quality == "fine"
            else {"rgb_bits": 5, "candidate_count": min(len(points), 96), "batch_size": 4096}
        )
        engine = NearestLutMapper(points, backend=backend, **settings)
        info = {
            "method": method,
            "quality": quality,
            "rgbBits": settings["rgb_bits"],
            "candidateCount": settings["candidate_count"],
            "measuredPoints": len(points),
        }
    elif method == "model":
        settings = (
            {"grid_step": 10.0, "lut_size": 13, "neighbor_count": 36, "candidate_count": 64}
            if quality == "fine"
            else {"grid_step": 12.5, "lut_size": 9, "neighbor_count": 36, "candidate_count": 48}
        )
        engine = InverseLutMapper(points, backend=backend, batch_size=1024, **settings)
        info = {
            "method": method,
            "quality": quality,
            "gridStep": settings["grid_step"],
            "lutSize": settings["lut_size"],
            "neighborCount": settings["neighbor_count"],
            "candidateCount": settings["candidate_count"],
            "modelCandidates": engine.model_candidates,
            "measuredPoints": len(points),
        }
    else:
        raise ValueError("method 必须是 nearest 或 model")

    info.update(
        {
            "backend": backend.name,
            "deviceName": backend.device_name,
            "buildMilliseconds": round((time.perf_counter() - started) * 1000),
            "algorithmVersion": "api-gpu-v1",
        }
    )

    def mapper(rgba: bytes, width: int, height: int, _tile_context: Any):
        context.check_cancelled()
        result = engine.map_rgba(rgba, width=width, height=height)
        stats = dict(result.get("stats") or {})
        visible = int(stats.get("visiblePixels") or 0)
        stats["deltaSum"] = float(stats.get("meanDeltaE") or 0) * visible
        return TileResult(
            cmyk=_as_bytes(result["cmyk"]),
            preview_rgba=_as_bytes(result["preview"]),
            stats=stats,
        )

    return mapper, info


def _normalise_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(stats)
    visible = float(result.get("visiblePixels") or 0)
    delta_sum = float(result.pop("deltaSum", 0) or 0)
    if delta_sum or visible:
        result["meanDeltaE"] = delta_sum / max(1.0, visible)
    for key, value in list(result.items()):
        if isinstance(value, float) and value.is_integer():
            result[key] = int(value)
    return result


def _job_progress(context: JobContext, detail: Mapping[str, Any]) -> None:
    progress = float(detail.get("progress") or 0)
    phase = str(detail.get("phase") or "处理文件")
    label = str(detail.get("label") or detail.get("layer") or "")
    context.report(0.12 + 0.83 * max(0.0, min(1.0, progress)), phase, label)


def _process_pdf(context: JobContext, mapper, engine_info: Mapping[str, Any]) -> Path:
    options = context.record.options
    pdf_options = options.get("pdf") if isinstance(options.get("pdf"), Mapping) else {}
    selected = pdf_options.get("processedLayerIds") or []
    dpi = float(pdf_options.get("dpi") or 0)
    if dpi <= 0:
        raise ValueError("PDF 任务缺少有效的原稿 DPI / px/mm")
    try:
        result = replace_pdf_layers(
            context.record.input_pdf,
            context.record.result_pdf,
            selected_layer_xrefs=selected,
            dpi=dpi,
            mapper=mapper,
            tile_size=int(pdf_options.get("tileSize") or 2048),
            bleed=2,
            preserve_native_density=bool(pdf_options.get("preserveNativeDensity", True)),
            cancelled=context.record.cancel_event.is_set,
            progress=lambda detail: _job_progress(context, detail),
        )
    except PdfProcessingCancelled as exc:
        raise JobCancelled(str(exc)) from exc

    context.check_cancelled()
    result["stats"] = _normalise_stats(result.get("stats") or {})
    result["info"] = dict(engine_info)
    result["sourceName"] = context.record.original_filename
    result["downloadUrl"] = f"/api/jobs/{context.record.id}/result"
    (context.record.directory / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return context.record.result_pdf


def process_job(context: JobContext) -> Path:
    mapper, engine_info = _build_mapper(context.record.options, context)
    return _process_pdf(context, mapper, engine_info)
