from __future__ import annotations

import json
import os
import shutil
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal


JobStatus = Literal[
    "uploading", "queued", "running", "succeeded", "failed", "cancelled"
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class JobRecord:
    id: str
    status: JobStatus
    created_at: str
    updated_at: str
    directory: Path
    input_pdf: Path
    pairs_file: Path | None
    result_pdf: Path
    options: dict[str, Any]
    original_filename: str
    progress: float = 0.0
    stage: str = "等待上传"
    message: str = ""
    error: str | None = None
    input_bytes: int = 0
    result_bytes: int = 0
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def public_dict(self) -> dict[str, Any]:
        result_url = f"/api/jobs/{self.id}/result" if self.status == "succeeded" else None
        payload = {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": round(max(0.0, min(1.0, self.progress)), 6),
            "stage": self.stage,
            "message": self.message,
            "error": self.error,
            "input_bytes": self.input_bytes,
            "result_bytes": self.result_bytes,
            "original_filename": self.original_filename,
            "result_url": result_url,
        }
        result_manifest = self.directory / "result.json"
        if self.status == "succeeded" and result_manifest.is_file():
            try:
                payload["result"] = json.loads(result_manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                # A damaged optional result manifest must not hide the finished
                # file from the user. The download URL above remains valid.
                pass
        return payload


class JobContext:
    def __init__(self, manager: "JobManager", record: JobRecord):
        self.manager = manager
        self.record = record

    def check_cancelled(self) -> None:
        if self.record.cancel_event.is_set():
            raise JobCancelled("任务已由用户取消")

    def report(self, progress: float, stage: str, message: str = "") -> None:
        self.check_cancelled()
        self.manager.update(
            self.record.id,
            progress=max(0.0, min(1.0, float(progress))),
            stage=stage,
            message=message,
        )


Processor = Callable[[JobContext], Path | None]


class JobManager:
    def __init__(
        self,
        work_dir: Path,
        worker_count: int = 1,
        retention_hours: float | None = None,
    ):
        self.work_dir = work_dir
        self.jobs_dir = work_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, worker_count), thread_name_prefix="cmyk-job"
        )
        if retention_hours is None:
            try:
                retention_hours = float(os.getenv("CMYK_API_JOB_TTL_HOURS", "24"))
            except ValueError:
                retention_hours = 24.0
        self.retention = timedelta(hours=max(1.0, retention_hours))
        self._restore_and_cleanup()

    def reserve(
        self,
        *,
        original_filename: str,
        options: dict[str, Any],
        has_pairs: bool,
    ) -> JobRecord:
        self.cleanup_expired()
        job_id = uuid.uuid4().hex
        directory = self.jobs_dir / job_id
        directory.mkdir(parents=True, exist_ok=False)
        now = utc_now()
        record = JobRecord(
            id=job_id,
            status="uploading",
            created_at=now,
            updated_at=now,
            directory=directory,
            input_pdf=directory / "input.pdf",
            pairs_file=(directory / "pairs.dat") if has_pairs else None,
            result_pdf=directory / "result.pdf",
            options=options,
            original_filename=original_filename,
        )
        with self._lock:
            self._jobs[job_id] = record
        self._write_manifest(record)
        return record

    def submit(self, job_id: str, processor: Processor) -> Future[None] | None:
        with self._lock:
            record = self.require(job_id)
            if record.cancel_event.is_set():
                record.status = "cancelled"
                record.stage = "已取消"
                record.updated_at = utc_now()
                self._write_manifest(record)
                return
            record.status = "queued"
            record.stage = "排队中"
            record.updated_at = utc_now()
            self._write_manifest(record)
            self._futures[job_id] = self._executor.submit(
                self._run, record, processor
            )
            return self._futures[job_id]

    def _run(self, record: JobRecord, processor: Processor) -> None:
        with self._lock:
            if record.cancel_event.is_set():
                record.status = "cancelled"
                record.stage = "已取消"
                record.updated_at = utc_now()
                self._write_manifest(record)
                return
            record.status = "running"
            record.stage = "开始处理"
            record.updated_at = utc_now()
            self._write_manifest(record)

        try:
            context = JobContext(self, record)
            produced = processor(context)
            context.check_cancelled()
            if produced is not None and produced.resolve() != record.result_pdf.resolve():
                shutil.move(str(produced), str(record.result_pdf))
            if not record.result_pdf.is_file():
                raise RuntimeError("处理器没有生成 result.pdf")
            with self._lock:
                record.result_bytes = record.result_pdf.stat().st_size
                record.progress = 1.0
                record.status = "succeeded"
                record.stage = "已完成"
                record.message = "PDF 已生成，可以下载。"
                record.updated_at = utc_now()
                self._write_manifest(record)
        except JobCancelled as exc:
            with self._lock:
                record.status = "cancelled"
                record.stage = "已取消"
                record.message = str(exc)
                record.updated_at = utc_now()
                self._write_manifest(record)
        except Exception as exc:
            with self._lock:
                record.status = "failed"
                record.stage = "处理失败"
                record.error = str(exc)
                record.message = traceback.format_exc(limit=8)
                record.updated_at = utc_now()
                self._write_manifest(record)

    def require(self, job_id: str) -> JobRecord:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(f"找不到任务 {job_id}") from exc

    def list(self) -> list[JobRecord]:
        self.cleanup_expired()
        with self._lock:
            return sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)

    def update(self, job_id: str, **changes: Any) -> JobRecord:
        with self._lock:
            record = self.require(job_id)
            for key, value in changes.items():
                if not hasattr(record, key):
                    raise AttributeError(key)
                setattr(record, key, value)
            record.updated_at = utc_now()
            self._write_manifest(record)
            return record

    def cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self.require(job_id)
            if record.status in {"succeeded", "failed", "cancelled"}:
                return record
            record.cancel_event.set()
            future = self._futures.get(job_id)
            # If execution has already started, _run owns the terminal status.
            # Never allow a directory to be deleted while that future can use it.
            if record.status == "uploading" or (
                record.status == "queued" and future is not None and future.cancel()
            ):
                record.status = "cancelled"
                record.stage = "已取消"
                record.updated_at = utc_now()
                self._write_manifest(record)
        return record

    def delete(self, job_id: str) -> None:
        with self._lock:
            record = self.require(job_id)
            if record.status in {"uploading", "queued", "running"}:
                raise RuntimeError("运行中的任务不能删除；请先取消。")
            future = self._futures.get(job_id)
            if future is not None and not future.done():
                raise RuntimeError("任务仍在停止中，请稍后删除。")
            self._jobs.pop(job_id, None)
            self._futures.pop(job_id, None)
        shutil.rmtree(record.directory, ignore_errors=True)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def cleanup_expired(self) -> int:
        """Delete finished jobs after their download-retention window.

        Active jobs are never removed.  This is called at startup and whenever a
        new job is reserved or the job list is requested, so a long-running
        local service cannot accumulate results forever.
        """

        cutoff = datetime.now(timezone.utc) - self.retention
        expired: list[JobRecord] = []
        with self._lock:
            for record in self._jobs.values():
                if record.status in {"uploading", "queued", "running"}:
                    continue
                future = self._futures.get(record.id)
                if future is not None and not future.done():
                    continue
                if self._parse_timestamp(record.updated_at) < cutoff:
                    expired.append(record)
            for record in expired:
                self._jobs.pop(record.id, None)
                self._futures.pop(record.id, None)
        for record in expired:
            shutil.rmtree(record.directory, ignore_errors=True)
        return len(expired)

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return datetime.fromtimestamp(0, tz=timezone.utc)

    @staticmethod
    def _find_result_file(directory: Path) -> Path:
        preferred = ("result.pdf", "result.tif", "result.tiff")
        for name in preferred:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        for candidate in sorted(directory.glob("result.*")):
            if candidate.is_file() and candidate.name not in {
                "result.json",
                "result.json.part",
            }:
                return candidate
        return directory / "result.pdf"

    def _restore_and_cleanup(self) -> None:
        """Restore downloadable results and discard expired orphan directories.

        A process restart cannot resume an in-flight GPU task.  Such manifests
        are converted to a terminal failed state, making them safe for normal
        TTL cleanup and explicit deletion through the API.
        """

        cutoff = datetime.now(timezone.utc) - self.retention
        for directory in self.jobs_dir.iterdir():
            if not directory.is_dir():
                continue
            manifest = directory / "job.json"
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                job_id = str(payload.get("id") or directory.name)
                status = str(payload.get("status") or "failed")
                updated_at = str(payload.get("updated_at") or payload.get("created_at") or utc_now())
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                shutil.rmtree(directory, ignore_errors=True)
                continue

            if self._parse_timestamp(updated_at) < cutoff:
                shutil.rmtree(directory, ignore_errors=True)
                continue

            result_file = self._find_result_file(directory)
            now = utc_now()
            interrupted = status in {"uploading", "queued", "running"}
            if interrupted:
                status = "failed"
            if status == "succeeded" and not result_file.is_file():
                status = "failed"
                interrupted = True

            record = JobRecord(
                id=job_id,
                status=status if status in {"succeeded", "failed", "cancelled"} else "failed",  # type: ignore[arg-type]
                created_at=str(payload.get("created_at") or now),
                updated_at=now if interrupted else updated_at,
                directory=directory,
                input_pdf=directory / "input.pdf",
                pairs_file=None,
                result_pdf=result_file,
                options=dict(payload.get("options") or {}),
                original_filename=str(payload.get("original_filename") or "result"),
                progress=float(payload.get("progress") or 0.0),
                stage="服务重启后已中止" if interrupted else str(payload.get("stage") or ""),
                message=(
                    "本机服务重启，未完成任务无法恢复。"
                    if interrupted
                    else str(payload.get("message") or "")
                ),
                error=(
                    "本机服务重启，未完成任务无法恢复。"
                    if interrupted
                    else payload.get("error")
                ),
                input_bytes=int(payload.get("input_bytes") or 0),
                result_bytes=(
                    result_file.stat().st_size
                    if result_file.is_file()
                    else int(payload.get("result_bytes") or 0)
                ),
            )
            self._jobs[job_id] = record
            if interrupted:
                self._write_manifest(record)

    def _write_manifest(self, record: JobRecord) -> None:
        manifest = record.directory / "job.json"
        temporary = manifest.with_suffix(".json.part")
        payload = record.public_dict() | {"options": record.options}
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(manifest)
