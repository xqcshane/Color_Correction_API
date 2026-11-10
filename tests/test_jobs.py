from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
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
    finished.updated_at = active.updated_at = stale_timestamp()
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
        record.updated_at = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        response = client.get(f"/api/jobs/{record.id}/result")
        assert response.status_code == 404
        assert not record.directory.exists()


def test_expired_job_is_removed_on_restart(tmp_path):
    manager = JobManager(tmp_path, retention_hours=1)
    record = manager.reserve(original_filename="old.pdf", options={}, has_pairs=False)
    manager.update(record.id, status="succeeded")
    record.updated_at = stale_timestamp()
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
    record.updated_at = stale_timestamp()
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
