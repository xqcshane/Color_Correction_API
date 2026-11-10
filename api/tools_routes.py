"""CPU-only PDF/image tools, sharing output retention and download leases."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from .config import Settings
from .jobs import JobContext, JobManager, JobRecord
from .pdf_image_tools import place_image_on_pdf
from .qr_tools import generate_qr_png


TAG = "PDF 与二维码工具"
SaveUpload = Callable[[UploadFile, Path, Settings, int], Awaitable[int]]
ResultResponse = Callable[[JobRecord], FileResponse]


class Placement(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    page: int = Field(1, ge=1)
    x_mm: float = Field(ge=0)
    y_mm: float = Field(ge=0)
    width_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)
    layer_name: str = Field(min_length=1, max_length=200)

    @field_validator("layer_name")
    @classmethod
    def valid_layer(cls, value: str) -> str:
        if not value.strip() or any(ord(char) < 32 for char in value):
            raise ValueError("图层名称不能为空或包含控制字符")
        return value


class QrCodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=1, max_length=4096, description="要编码的 HTTP/HTTPS 链接；不访问该地址，原文写入二维码", examples=["https://example.com/"])


def _manifest(record: JobRecord, result: dict[str, Any]) -> None:
    result["downloadUrl"] = f"/api/jobs/{record.id}/result"
    temporary = record.directory / "result.json.part"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(record.directory / "result.json")


async def _await_result(request: Request, jobs: JobManager, record: JobRecord,
                        wait: bool, result_response: ResultResponse):
    if not wait:
        return JSONResponse({**record.public_dict(), "status_url": f"/api/jobs/{record.id}",
                             "result_url": f"/api/jobs/{record.id}/result"}, status_code=202)
    while record.status in {"uploading", "queued", "running"}:
        if await request.is_disconnected():
            jobs.cancel(record.id)
            raise HTTPException(499, "客户端断开，已请求取消任务")
        await asyncio.sleep(.1)
    if record.status != "succeeded":
        raise HTTPException(422, {"job_id": record.id, "status": record.status, "error": record.error})
    return await run_in_threadpool(result_response, record)


def create_tools_router(*, settings: Settings, jobs: JobManager,
                        slots: threading.BoundedSemaphore, save_upload: SaveUpload,
                        result_response: ResultResponse) -> APIRouter:
    router = APIRouter(prefix="/api/tools", tags=[TAG])
    placement_schema = {"requestBody": {"required": True, "content": {"multipart/form-data": {
        "schema": {"type": "object", "required": ["pdf", "image", "x_mm", "y_mm", "width_mm", "height_mm", "layer_name"],
            "properties": {
                "pdf": {"type": "string", "format": "binary", "description": "原始 PDF；不整页重绘，保留原稿图像密度"},
                "image": {"type": "string", "format": "binary", "description": "需要放置的图片，保留原像素与透明度"},
                "page": {"type": "integer", "minimum": 1, "default": 1, "description": "页码，从 1 开始"},
                "x_mm": {"type": "number", "minimum": 0, "example": 10, "description": "距显示页面左上角的横向毫米距离"},
                "y_mm": {"type": "number", "minimum": 0, "example": 10, "description": "距显示页面左上角的纵向毫米距离"},
                "width_mm": {"type": "number", "exclusiveMinimum": 0, "example": 30, "description": "放置后的实际宽度，毫米"},
                "height_mm": {"type": "number", "exclusiveMinimum": 0, "example": 30, "description": "放置后的实际高度，毫米"},
                "layer_name": {"type": "string", "minLength": 1, "maxLength": 200, "example": "QR_Code", "description": "新图层名称；每次新建独立 OCG，已有同名层也不合并"},
            }}}}}}
    pdf_responses = {200: {"description": "保持原稿密度的 PDF", "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}}},
                     202: {"description": "异步任务，可查询进度并下载结果"}}
    png_responses = {200: {"description": "透明背景 PNG 二维码", "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}},
                     202: {"description": "异步任务，可查询进度并下载结果"}}

    @router.post("/place-image", summary="将图片放入 PDF 独立图层", openapi_extra=placement_schema,
                 responses=pdf_responses, response_class=FileResponse)
    async def place_image(request: Request, wait: bool = Query(True, description="false 返回异步任务，适用于大 PDF")):
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "任务队列已满，请稍后重试", headers={"Retry-After": "10"})
        upload_lease = None
        record = None
        submitted = False
        try:
            upload_lease = await run_in_threadpool(jobs.begin_upload)
            async with request.form(max_files=2, max_fields=6, max_part_size=64 * 1024) as form:
                allowed = {"pdf", "image", "page", "x_mm", "y_mm", "width_mm", "height_mm", "layer_name"}
                if set(form) - allowed or any(len(form.getlist(key)) != 1 for key in form):
                    raise HTTPException(422, "字段不支持或重复；请按文档提交单个图片及一组位置参数")
                pdf, image = form.get("pdf"), form.get("image")
                if not isinstance(pdf, UploadFile) or not isinstance(image, UploadFile):
                    raise HTTPException(422, "必须上传 pdf 和 image 文件")
                try:
                    placement = Placement.model_validate({key: form[key] for key in form if key not in {"pdf", "image"}})
                except ValidationError as exc:
                    raise HTTPException(422, f"放置参数无效：{exc}") from exc
                filename = (pdf.filename or "input.pdf").replace("\\", "/").split("/")[-1]
                record = jobs.reserve(original_filename=filename,
                                      options={"method": "place-image", "placement": placement.model_dump()}, has_pairs=False)
                suffix = Path(image.filename or "image.png").suffix.lower()
                if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".gif"}:
                    suffix = ".img"
                image_path = record.directory / f"image{suffix}"
                size = await save_upload(pdf, record.input_pdf, settings, settings.max_upload_bytes)
                size += await save_upload(image, image_path, settings, settings.max_image_bytes)
            if await request.is_disconnected():
                raise HTTPException(499, "客户端断开，任务未进入处理队列")
            jobs.update(record.id, input_bytes=size)

            def process(context: JobContext) -> Path:
                context.report(.05, "放置图片", "保留原稿对象并新建独立图层")
                try:
                    result = place_image_on_pdf(context.record.input_pdf, image_path,
                        context.record.result_pdf, page_number=placement.page, x_mm=placement.x_mm,
                        y_mm=placement.y_mm, width_mm=placement.width_mm, height_mm=placement.height_mm,
                        layer_name=placement.layer_name, max_image_pixels=settings.max_image_pixels,
                        cancelled=context.record.cancel_event.is_set)
                except Exception:
                    context.check_cancelled()
                    raise
                context.check_cancelled()
                result["kind"] = "pdf"
                _manifest(context.record, result)
                return context.record.result_pdf

            future = jobs.submit(record.id, process, pool="tools")
            if future is None:
                raise HTTPException(409, "任务在提交前已取消")
            future.add_done_callback(lambda _: slots.release())
            submitted = True
            return await _await_result(request, jobs, record, wait, result_response)
        finally:
            if upload_lease is not None:
                upload_lease.release()
            if not submitted:
                slots.release()
                if record is not None:
                    jobs.update(record.id, status="failed", stage="请求验证失败")
                    jobs.delete(record.id)

    @router.post("/qrcode", summary="链接转透明 PNG 二维码", responses=png_responses,
                 response_class=FileResponse)
    async def qrcode(request: Request, payload: QrCodeRequest,
                     wait: bool = Query(True, description="false 返回异步任务")):
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "任务队列已满，请稍后重试", headers={"Retry-After": "10"})
        upload_lease = None
        record = None
        submitted = False
        try:
            upload_lease = await run_in_threadpool(jobs.begin_upload)
            try:
                input_bytes = len(payload.url.encode("utf-8"))
            except UnicodeError as exc:
                raise HTTPException(422, "链接字符编码无效，请使用有效的 Unicode 链接") from exc
            record = jobs.reserve(original_filename="qrcode.png", options={"method": "qrcode"}, has_pairs=False)
            jobs.update(record.id, result_pdf=record.directory / "result.png", input_bytes=input_bytes)
            if await request.is_disconnected():
                raise HTTPException(499, "客户端断开，任务未进入处理队列")

            def process(context: JobContext) -> Path:
                context.report(.1, "生成二维码", "生成带透明背景的 PNG 图片")
                result = generate_qr_png(payload.url, context.record.result_pdf)
                context.check_cancelled()
                _manifest(context.record, result)
                return context.record.result_pdf

            future = jobs.submit(record.id, process, pool="tools")
            if future is None:
                raise HTTPException(409, "任务在提交前已取消")
            future.add_done_callback(lambda _: slots.release())
            submitted = True
            return await _await_result(request, jobs, record, wait, result_response)
        finally:
            if upload_lease is not None:
                upload_lease.release()
            if not submitted:
                slots.release()
                if record is not None:
                    jobs.update(record.id, status="failed", stage="请求验证失败")
                    jobs.delete(record.id)

    return router
