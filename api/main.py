from __future__ import annotations

import asyncio
import json
import math
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from . import __version__
from .capabilities import detect_capabilities
from .config import Settings
from .csv_input import parse_pairs_csv
from .jobs import JobManager, JobRecord
from .pdf_engine import inspect_pdf
from .processor import process_job


class BodyLimitMiddleware:
    """Count bytes before multipart parsing, including chunked uploads."""
    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope.get("headers", [])).get(b"content-length")
        if length:
            try:
                oversized = int(length) > self.max_bytes
            except ValueError:
                oversized = True
            if oversized:
                return await JSONResponse({"detail": "上传请求超过配置大小上限"}, status_code=413)(scope, receive, send)
        total = 0

        async def limited_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                scope["upload_limit_exceeded"] = True
                # MultipartParser closes all spooled temporary files for this
                # exception, including files uploaded before the limit is hit.
                raise MultiPartException("上传请求超过配置大小上限")
            return message

        await self.app(scope, limited_receive, send)


async def _save_upload(upload: UploadFile, path: Path, settings: Settings, max_bytes: int) -> int:
    size = 0
    with path.open("wb") as output:
        while chunk := await upload.read(settings.upload_chunk_bytes):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, f"文件超过上限 {max_bytes} bytes")
            await run_in_threadpool(output.write, chunk)
    if not size:
        raise HTTPException(422, "上传文件不能为空")
    return size


def _resolve_pdf_options(metadata: dict, names: list[str], dpi: float | None) -> dict:
    available: dict[str, list[str]] = {}
    for layer in metadata["layers"]:
        available.setdefault(layer["name"], []).append(layer["id"])
    selected = []
    for name in names:
        ids = available.get(name, [])
        if not ids:
            raise ValueError(f"找不到图层 {name!r}；可用层名：{list(available)}")
        if len(ids) != 1:
            raise ValueError(f"层名 {name!r} 重复，无法唯一选择；请在 PDF 中改为唯一名称")
        selected.extend(ids)
    actual_dpi = dpi if dpi is not None else metadata.get("detectedDpi")
    if actual_dpi is None or not math.isfinite(float(actual_dpi)) or float(actual_dpi) <= 0:
        raise ValueError("无法确定原稿密度（例如纯矢量 PDF），请显式设置 dpi")
    return {"processedLayerIds": selected, "dpi": float(actual_dpi),
            "preserveNativeDensity": dpi is None, "tileSize": 2048}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(os.getenv("CUPY_CACHE_DIR", str(settings.work_dir / "cupy-cache")))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUPY_CACHE_DIR", str(cache))
    jobs = JobManager(settings.work_dir, worker_count=1)
    slots = threading.BoundedSemaphore(settings.max_pending_jobs)

    @asynccontextmanager
    async def lifespan(app):
        app.state.compute = await run_in_threadpool(detect_capabilities, settings.force_cpu)
        yield
        jobs.shutdown()

    app = FastAPI(title="Color Correction API", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.settings, app.state.jobs = settings, jobs
    app.add_middleware(BodyLimitMiddleware,
                       max_bytes=settings.max_upload_bytes + settings.max_csv_bytes + 1024**2)

    @app.exception_handler(StarletteHTTPException)
    async def request_error(request, exc):
        if request.scope.get("upload_limit_exceeded"):
            return JSONResponse({"detail": "上传请求超过配置大小上限"}, status_code=413)
        return await http_exception_handler(request, exc)

    def compute():
        return getattr(app.state, "compute", None) or detect_capabilities(settings.force_cpu)

    def require_gpu():
        info = compute()
        if not settings.force_cpu and not info.gpu_available:
            raise HTTPException(503, f"GPU 不可用：{info.note}")
        return info

    def require_job(job_id: str) -> JobRecord:
        jobs.cleanup_expired()
        try:
            return jobs.require(job_id)
        except KeyError as exc:
            raise HTTPException(404, "任务不存在或已过保留期") from exc

    def result_response(record: JobRecord) -> FileResponse:
        if record.status != "succeeded" or not record.result_pdf.is_file():
            raise HTTPException(409, {"status": record.status, "error": record.error})
        stem = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", Path(record.original_filename).stem)[:120] or "result"
        return FileResponse(record.result_pdf, media_type="application/pdf",
                            filename=f"{stem}-{record.options.get('method', 'processed')}-CMYK.pdf",
                            headers={"X-Job-ID": record.id})

    @app.get("/api/health")
    def health():
        info = compute()
        return {"ok": True, "service": "color-correction-api", "version": __version__,
                "gpuReady": info.gpu_available, "gpuName": info.gpu_name,
                "backend": "cupy" if info.gpu_available else "numpy"}

    @app.get("/api/ready")
    def ready():
        require_gpu()
        return health()

    @app.get("/api/capabilities")
    def capabilities():
        return {**health(), "compute": compute().to_dict(), "quality": "fine",
                "limits": {"maxUploadBytes": settings.max_upload_bytes,
                           "maxCsvBytes": settings.max_csv_bytes,
                           "maxPendingJobs": settings.max_pending_jobs,
                           "workerCount": 1},
                "features": {"asyncJobs": True, "largeFileDiskSpooling": True,
                             "nativeImagePixelDensityPreserved": True}}

    multipart_schema = {"requestBody": {"required": True, "content": {"multipart/form-data": {
        "schema": {"type": "object", "required": ["pdf", "csv", "layers"], "properties": {
            "pdf": {"type": "string", "format": "binary"},
            "csv": {"type": "string", "format": "binary"},
            "layers": {"type": "string", "description": '需处理的层名 JSON 数组，例如 ["Artwork"]'},
            "dpi": {"type": "number", "exclusiveMinimum": 0,
                    "description": "可选输出 DPI；省略即保持原稿密度"}
        }}}}}}
    pdf_responses = {200: {"description": "处理后的 CMYK PDF", "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}}},
                     202: {"description": "异步任务；轮询 status_url 并下载 result_url"}}

    async def convert(request: Request, method: str, wait: bool):
        require_gpu()
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "任务队列已满，请稍后重试", headers={"Retry-After": "10"})
        record = None
        submitted = False
        try:
            async with request.form(max_files=2, max_fields=2, max_part_size=64 * 1024) as form:
                if set(form.keys()) - {"pdf", "csv", "layers", "dpi"}:
                    raise HTTPException(422, "仅支持 pdf、csv、layers、dpi 字段")
                if any(len(form.getlist(key)) != 1 for key in form):
                    raise HTTPException(422, "字段不能重复；多层请在 layers 中使用 JSON 数组")
                pdf, csv_file = form.get("pdf"), form.get("csv")
                if not isinstance(pdf, UploadFile) or not isinstance(csv_file, UploadFile):
                    raise HTTPException(422, "必须上传 pdf 和 csv 文件")
                try:
                    names = json.loads(str(form.get("layers", "")))
                    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names):
                        raise ValueError("layers 必须是非空的层名 JSON 数组")
                    if len(set(names)) != len(names):
                        raise ValueError("layers 中层名不能重复")
                    dpi = float(form["dpi"]) if "dpi" in form else None
                    if dpi is not None and (not math.isfinite(dpi) or dpi <= 0):
                        raise ValueError("dpi 必须是大于 0 的有限数字")
                except (ValueError, TypeError) as exc:
                    raise HTTPException(422, str(exc)) from exc
                record = jobs.reserve(original_filename=(pdf.filename or "input.pdf").replace("\\", "/").split("/")[-1],
                                      options={}, has_pairs=True)
                input_size = await _save_upload(pdf, record.input_pdf, settings, settings.max_upload_bytes)
                await _save_upload(csv_file, record.pairs_file, settings, settings.max_csv_bytes)
            try:
                points = await run_in_threadpool(parse_pairs_csv, record.pairs_file, method)
                metadata = await run_in_threadpool(inspect_pdf, record.input_pdf)
                pdf_options = _resolve_pdf_options(metadata, names, dpi)
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(422, str(exc)) from exc
            jobs.update(record.id, options={"method": method, "quality": "fine", "points": points,
                                           "forceCpu": settings.force_cpu, "pdf": pdf_options},
                        input_bytes=input_size)
            future = jobs.submit(record.id, process_job)
            if future is None:
                raise HTTPException(409, "任务在提交前已取消")
            future.add_done_callback(lambda _: slots.release())
            submitted = True
            if not wait:
                return JSONResponse({**record.public_dict(), "status_url": f"/api/jobs/{record.id}",
                                     "result_url": f"/api/jobs/{record.id}/result"}, status_code=202)
            while record.status in {"queued", "running", "uploading"}:
                if await request.is_disconnected():
                    jobs.cancel(record.id)
                    raise HTTPException(499, "客户端断开，已请求取消任务")
                await asyncio.sleep(0.2)
            if record.status != "succeeded":
                raise HTTPException(422, {"job_id": record.id, "status": record.status, "error": record.error})
            return result_response(record)
        finally:
            if not submitted:
                slots.release()
                if record is not None:
                    jobs.update(record.id, status="failed", stage="请求验证失败")
                    jobs.delete(record.id)

    @app.post("/api/nearest", openapi_extra=multipart_schema, responses=pdf_responses,
              summary="GPU 找最接近", response_class=FileResponse)
    async def nearest(request: Request, wait: bool = Query(True, description="false 返回异步任务（适合大 PDF）")):
        return await convert(request, "nearest", wait)

    @app.post("/api/predict", openapi_extra=multipart_schema, responses=pdf_responses,
              summary="GPU 推算新色", response_class=FileResponse)
    async def predict(request: Request, wait: bool = Query(True, description="false 返回异步任务（适合大 PDF）")):
        return await convert(request, "model", wait)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return require_job(job_id).public_dict()

    @app.get("/api/jobs/{job_id}/result")
    def get_result(job_id: str):
        return result_response(require_job(job_id))

    @app.post("/api/jobs/{job_id}/cancel", status_code=202)
    def cancel_job(job_id: str):
        return jobs.cancel(require_job(job_id).id).public_dict()

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        try:
            jobs.delete(require_job(job_id).id)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"deleted": True}

    return app


app = create_app()
