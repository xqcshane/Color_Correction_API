from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import threading

from fastapi.testclient import TestClient
import pytest

from api.config import Settings
from api.jobs import JobManager
from api.main import create_app


def stale_timestamp():
    return (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()


def test_completed_pdf_remains_downloadable_after_restart(tmp_path):
    manager = JobManager(tmp_path)
    record = manager.reserve(original_filename="production.pdf", options={"method": "model"}, has_pairs=True)
    expected_pdf = b"%PDF-restored-download"
    record.result_pdf.write_bytes(expected_pdf)
    manager.update(record.id, status="succeeded", progress=1., result_bytes=len(expected_pdf))
    manager.shutdown()

    app = create_app(Settings(work_dir=tmp_path, force_cpu=True))
    with TestClient(app) as client:
        response = client.get(f"/api/jobs/{record.id}")
        assert response.status_code == 200
        assert response.json()["status"] == "succeeded"
        download = client.get(response.json()["result_url"])
        assert download.status_code == 200
        assert download.content == expected_pdf
        assert "model-CMYK.pdf" in download.headers["content-disposition"]


@pytest.mark.parametrize("status", ["uploading", "queued", "running"])
def test_interrupted_jobs_are_marked_failed_on_restart(tmp_path, status):
    manager = JobManager(tmp_path)
    record = manager.reserve(original_filename="interrupted.pdf", options={}, has_pairs=True)
    manager.update(record.id, status=status)
    manager.shutdown()
    restored = JobManager(tmp_path)
    try:
        recovered = restored.require(record.id)
        assert recovered.status == "failed"
        assert "重启" in recovered.error
        manifest = json.loads((record.directory / "job.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "failed"
        restored.delete(record.id)
        assert not record.directory.exists()
    finally:
        restored.shutdown()


def test_missing_successful_result_is_failed_on_restart(tmp_path):
    manager = JobManager(tmp_path)
    record = manager.reserve(original_filename="missing.pdf", options={}, has_pairs=False)
    manager.update(record.id, status="succeeded", progress=1.)
    manager.shutdown()
    restored = JobManager(tmp_path)
    try:
        assert restored.require(record.id).status == "failed"
    finally:
        restored.shutdown()


def test_expired_finished_jobs_are_removed_but_live_jobs_remain(tmp_path):
    manager = JobManager(tmp_path, retention_hours=1)
    finished = manager.reserve(original_filename="old.pdf", options={}, has_pairs=False)
    manager.update(finished.id, status="succeeded")
    active = manager.reserve(original_filename="active.pdf", options={}, has_pairs=False)
    finished.finished_at = stale_timestamp()
    active.updated_at = stale_timestamp()
    try:
        assert manager.cleanup_expired() == 1
        assert not finished.directory.exists()
        assert active.directory.exists()
        with pytest.raises(KeyError):
            manager.require(finished.id)
        assert manager.require(active.id).status == "uploading"
    finally:
        manager.shutdown()


def test_expired_results_are_cleaned_on_download_attempt(tmp_path):
    app = create_app(Settings(work_dir=tmp_path, force_cpu=True))
    with TestClient(app) as client:
        manager = app.state.jobs
        record = manager.reserve(original_filename="expired.pdf", options={}, has_pairs=False)
        record.result_pdf.write_bytes(b"old pdf")
        manager.update(record.id, status="succeeded")
        record.finished_at = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        response = client.get(f"/api/jobs/{record.id}/result")
        assert response.status_code == 404
        assert not record.directory.exists()


def test_expired_job_is_removed_on_restart(tmp_path):
    manager = JobManager(tmp_path, retention_hours=1)
    record = manager.reserve(original_filename="old.pdf", options={}, has_pairs=False)
    manager.update(record.id, status="succeeded")
    record.finished_at = stale_timestamp()
    manager._write_manifest(record)
    manager.shutdown()
    restored = JobManager(tmp_path, retention_hours=1)
    try:
        assert restored.list() == []
        assert not record.directory.exists()
    finally:
        restored.shutdown()


def test_cleanup_and_delete_wait_until_future_is_fully_done(tmp_path):
    manager = JobManager(tmp_path, retention_hours=1)
    record = manager.reserve(original_filename="stopping.pdf", options={}, has_pairs=False)
    manager.update(record.id, status="cancelled")
    record.finished_at = stale_timestamp()
    future = Future()
    manager._futures[record.id] = future
    try:
        with pytest.raises(RuntimeError, match="停止"):
            manager.delete(record.id)
        assert manager.cleanup_expired() == 0
        assert record.directory.exists()
        future.set_result(None)
        assert manager.cleanup_expired() == 1
        assert not record.directory.exists()
    finally:
        manager.shutdown()


def test_cancelled_queued_future_never_recreates_deleted_directory(tmp_path):
    manager = JobManager(tmp_path)
    started, release = threading.Event(), threading.Event()
    blocker = manager.reserve(original_filename="active.pdf", options={}, has_pairs=False)

    def blocking(context):
        started.set()
        assert release.wait(3)
        context.record.result_pdf.write_bytes(b"done")
        return context.record.result_pdf

    first_future = manager.submit(blocker.id, blocking)
    try:
        assert started.wait(2)
        queued = manager.reserve(original_filename="queued.pdf", options={}, has_pairs=False)
        called = threading.Event()

        def should_never_run(context):
            called.set()
            raise AssertionError("cancelled queued task executed")

        queued_future = manager.submit(queued.id, should_never_run)
        assert manager.cancel(queued.id).status == "cancelled"
        assert queued_future.cancelled()
        manager.delete(queued.id)
        release.set()
        first_future.result(timeout=3)
        assert not called.is_set()
        assert not queued.directory.exists()
    finally:
        release.set()
        manager.shutdown()


def test_finished_timestamp_is_immutable_under_metadata_updates(tmp_path):
    manager = JobManager(tmp_path, retention_hours=1)
    record = manager.reserve(original_filename="metadata.pdf", options={}, has_pairs=False)
    manager.update(record.id, status="succeeded")
    record.finished_at = stale_timestamp()
    finished_at = record.finished_at
    try:
        manager.update(record.id, message="metadata touched", finished_at=datetime.now(timezone.utc).isoformat())
        assert record.finished_at == finished_at
        assert manager._parse_timestamp(record.updated_at) > manager._parse_timestamp(finished_at)
        assert manager.cleanup_expired() == 1
        assert not record.directory.exists()
    finally:
        manager.shutdown()


def test_download_pins_protect_results_until_last_idempotent_release(tmp_path):
    manager = JobManager(tmp_path, retention_hours=1)
    record = manager.reserve(original_filename="downloading.pdf", options={}, has_pairs=False)
    record.result_pdf.write_bytes(b"download me")
    manager.update(record.id, status="succeeded")
    first = manager.pin(record.id)
    second = manager.pin(record.id)
    record.finished_at = stale_timestamp()
    try:
        assert first.record is record
        assert manager.cleanup_expired() == 0
        with pytest.raises(KeyError, match="保留期"):
            manager.pin(record.id)
        with pytest.raises(RuntimeError, match="下载"):
            manager.delete(record.id)
        first.release()
        first.release()
        assert manager.cleanup_expired() == 0
        assert record.result_pdf.read_bytes() == b"download me"
        second.release()
        assert manager.cleanup_expired() == 1
        assert not record.directory.exists()
    finally:
        first.release()
        second.release()
        manager.shutdown()


def test_job_pin_context_releases_after_exception(tmp_path):
    manager = JobManager(tmp_path)
    record = manager.reserve(original_filename="exception.pdf", options={}, has_pairs=False)
    manager.update(record.id, status="failed")
    try:
        with pytest.raises(ValueError):
            with manager.pin(record.id) as pinned:
                assert pinned is record
                raise ValueError("download failed")
        manager.delete(record.id)
        assert not record.directory.exists()
    finally:
        manager.shutdown()


@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
def test_worker_immediately_removes_inputs_and_intermediates(tmp_path, status):
    from api.jobs import JobCancelled
    manager = JobManager(tmp_path)
    record = manager.reserve(original_filename="large.pdf", options={}, has_pairs=True)
    record.input_pdf.write_bytes(b"large input")
    record.pairs_file.write_bytes(b"CSV data")

    def processor(context):
        context.record.result_pdf.write_bytes(b"PDF result")
        (context.record.directory / "result.json").write_text('{"dpi":36}', encoding="utf-8")
        intermediates = context.record.directory / "tiles"
        intermediates.mkdir()
        (intermediates / "pixels.raw").write_bytes(b"temporary pixels")
        (context.record.directory / "partial.tmp").write_bytes(b"partial")
        if status == "failed":
            raise RuntimeError("processing failed")
        if status == "cancelled":
            raise JobCancelled("cancelled")
        return context.record.result_pdf

    try:
        manager.submit(record.id, processor).result(timeout=3)
        assert record.status == status
        assert record.finished_at is not None
        remaining = {path.name for path in record.directory.iterdir()}
        assert remaining == ({"job.json", "result.json", "result.pdf"} if status == "succeeded" else {"job.json"})
    finally:
        manager.shutdown()


def test_cancelled_upload_keeps_input_until_uploader_acknowledges(tmp_path):
    manager = JobManager(tmp_path)
    record = manager.reserve(original_filename="still-uploading.pdf", options={}, has_pairs=True)
    record.input_pdf.write_bytes(b"active upload")
    try:
        cancelled = manager.cancel(record.id)
        assert cancelled.cancel_event.is_set()
        assert cancelled.status == "uploading"
        assert cancelled.finished_at is None
        assert manager.maintenance()["removed_artifacts"] == 0
        assert record.input_pdf.exists()
        processor_called = threading.Event()
        assert manager.submit(record.id, lambda _: processor_called.set()) is None
        assert not processor_called.is_set()
        assert record.status == "cancelled"
        assert record.finished_at is not None
        assert not record.input_pdf.exists()
    finally:
        manager.shutdown()


def test_restart_cleans_inputs_and_uses_original_finished_at(tmp_path):
    manager = JobManager(tmp_path, retention_hours=1)
    record = manager.reserve(original_filename="restored.pdf", options={}, has_pairs=True)
    record.input_pdf.write_bytes(b"input")
    record.pairs_file.write_bytes(b"CSV")
    record.result_pdf.write_bytes(b"PDF")
    manager.update(record.id, status="succeeded")
    record.finished_at = stale_timestamp()
    manager.update(record.id, message="recent metadata write")
    manager.shutdown()
    restarted = JobManager(tmp_path, retention_hours=1)
    try:
        assert restarted.list() == []
        assert not record.directory.exists()
    finally:
        restarted.shutdown()


def test_expired_directory_removal_failure_is_retried_without_api_access(tmp_path, monkeypatch):
    import api.jobs
    manager = JobManager(tmp_path, retention_hours=1)
    record = manager.reserve(original_filename="locked.pdf", options={}, has_pairs=False)
    record.result_pdf.write_bytes(b"expired PDF")
    manager.update(record.id, status="succeeded")
    record.finished_at = stale_timestamp()
    real_rmtree = api.jobs.shutil.rmtree
    failures = []

    def temporarily_locked(path, *args, **kwargs):
        if Path(path) == record.directory and not failures:
            failures.append(path)
            raise PermissionError("temporary download/filesystem lock")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(api.jobs.shutil, "rmtree", temporarily_locked)
    try:
        assert manager.cleanup_expired() == 1
        assert record.directory.exists()
        with pytest.raises(KeyError):
            manager.require(record.id)
        with pytest.raises(KeyError):
            manager.pin(record.id)
        manager.maintenance()
        assert not record.directory.exists()
        assert not manager._pending_deletions
    finally:
        manager.shutdown()
