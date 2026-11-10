from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
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


@pytest.mark.parametrize("pool", ["gpu", "tools"])
def test_cancelled_queued_future_never_recreates_deleted_directory(tmp_path, pool):
    manager = JobManager(tmp_path)
    started, release = threading.Event(), threading.Event()
    blocker = manager.reserve(original_filename="active.pdf", options={}, has_pairs=False)

    def blocking(context):
        started.set()
        assert release.wait(3)
        context.record.result_pdf.write_bytes(b"done")
        return context.record.result_pdf

    first_future = manager.submit(blocker.id, blocking, pool=pool)
    try:
        assert started.wait(2)
        queued = manager.reserve(original_filename="queued.pdf", options={}, has_pairs=False)
        queued.input_pdf.write_bytes(b"queued upload")
        called = threading.Event()

        def should_never_run(context):
            called.set()
            raise AssertionError("cancelled queued task executed")

        queued_future = manager.submit(queued.id, should_never_run, pool=pool)
        assert manager.maintenance()["removed_artifacts"] == 0
        assert queued.input_pdf.read_bytes() == b"queued upload"
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


def test_cpu_tool_finishes_while_gpu_queue_is_blocked(tmp_path):
    manager = JobManager(tmp_path)
    started, release = threading.Event(), threading.Event()
    gpu = manager.reserve(original_filename="gpu.pdf", options={}, has_pairs=False)
    tool = manager.reserve(original_filename="code.png", options={"method": "qr"}, has_pairs=False)
    manager.update(tool.id, result_pdf=tool.directory / "result.png")

    def blocking_gpu(context):
        started.set()
        assert release.wait(5)
        context.record.result_pdf.write_bytes(b"GPU PDF")
        return context.record.result_pdf

    def cpu_tool(context):
        context.record.result_pdf.write_bytes(b"\x89PNG\r\nCPU result")
        return context.record.result_pdf

    gpu_future = manager.submit(gpu.id, blocking_gpu)
    try:
        assert started.wait(2)
        tool_future = manager.submit(tool.id, cpu_tool, pool="tools")
        tool_future.result(timeout=2)
        assert tool.status == "succeeded"
        assert "图片" in tool.message
        assert manager._futures[tool.id] is tool_future
        assert not gpu_future.done()
        assert tool.result_pdf.read_bytes() == b"\x89PNG\r\nCPU result"
    finally:
        release.set()
        gpu_future.result(timeout=3)
        manager.shutdown()


def test_unknown_worker_pool_is_rejected_without_changing_job(tmp_path):
    manager = JobManager(tmp_path)
    record = manager.reserve(original_filename="unknown.pdf", options={}, has_pairs=False)
    called = threading.Event()
    try:
        with pytest.raises(ValueError, match="gpu 或 tools"):
            manager.submit(record.id, lambda _: called.set(), pool="unknown")
        assert record.status == "uploading"
        assert record.id not in manager._futures
        assert not called.is_set()
    finally:
        manager.shutdown()


@pytest.mark.parametrize("pool,suffix", [("gpu", ".pdf"), ("tools", ".png")])
def test_cancel_during_final_file_check_wins_before_success_commit(tmp_path, monkeypatch, pool, suffix):
    manager = JobManager(tmp_path)
    record = manager.reserve(original_filename=f"source{suffix}", options={}, has_pairs=False)
    manager.update(record.id, result_pdf=record.directory / f"result{suffix}")
    record.input_pdf.write_bytes(b"active input")
    checked, release = threading.Event(), threading.Event()
    original_is_file = Path.is_file

    def pause_final_check(path):
        if path == record.result_pdf and not checked.is_set():
            checked.set()
            assert release.wait(5)
        return original_is_file(path)

    def process(context):
        context.record.result_pdf.write_bytes(b"completed processing")
        return context.record.result_pdf

    monkeypatch.setattr(Path, "is_file", pause_final_check)
    future = manager.submit(record.id, process, pool=pool)
    try:
        assert checked.wait(2)
        # Processing and the lock-free cancellation check have finished, but
        # the worker has not yet committed its terminal status.
        assert manager.cancel(record.id).status == "running"
        assert record.cancel_event.is_set()
        release.set()
        future.result(timeout=3)
        assert record.status == "cancelled"
        assert record.finished_at is not None
        assert not record.result_pdf.exists()
        assert not record.input_pdf.exists()
        manifest = json.loads((record.directory / "job.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "cancelled"
        assert manifest["result_url"] is None
    finally:
        release.set()
        future.result(timeout=3)
        manager.shutdown()


def test_png_success_cleanup_pin_and_expiry_use_same_24_hour_lifecycle(tmp_path):
    manager = JobManager(tmp_path, retention_hours=24)
    record = manager.reserve(original_filename="link.png", options={"method": "qr"}, has_pairs=True)
    manager.update(record.id, result_pdf=record.directory / "result.png")
    record.input_pdf.write_bytes(b"uploaded image")
    record.pairs_file.write_bytes(b"unused source")
    expected = b"\x89PNG\r\nretained result"

    def process(context):
        context.record.result_pdf.write_bytes(expected)
        (context.record.directory / "result.json").write_text('{"kind":"png"}', encoding="utf-8")
        (context.record.directory / "intermediate.tmp").write_bytes(b"temporary")
        return context.record.result_pdf

    lease = None
    try:
        manager.submit(record.id, process, pool="tools").result(timeout=3)
        assert {path.name for path in record.directory.iterdir()} == {"job.json", "result.json", "result.png"}
        record.finished_at = (datetime.now(timezone.utc) - timedelta(hours=23)).isoformat()
        assert manager.cleanup_expired() == 0
        lease = manager.pin(record.id)
        record.finished_at = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        finished_at = record.finished_at
        manager.update(record.id, message="download does not extend retention")
        assert record.finished_at == finished_at
        assert manager.cleanup_expired() == 0
        assert lease.record.result_pdf.read_bytes() == expected
        with pytest.raises(RuntimeError, match="下载"):
            manager.delete(record.id)
        with pytest.raises(KeyError, match="保留期"):
            manager.pin(record.id)
        lease.release()
        assert manager.cleanup_expired() == 1
        assert not record.directory.exists()
    finally:
        if lease is not None:
            lease.release()
        manager.shutdown()


def test_completed_png_is_restored_with_download_lease_after_restart(tmp_path):
    manager = JobManager(tmp_path)
    record = manager.reserve(original_filename="link.png", options={"method": "qr"}, has_pairs=False)
    manager.update(record.id, result_pdf=record.directory / "result.png")
    expected = b"\x89PNG\r\nrestart result"

    def process(context):
        context.record.result_pdf.write_bytes(expected)
        return context.record.result_pdf

    try:
        manager.submit(record.id, process, pool="tools").result(timeout=3)
        finished_at = record.finished_at
    finally:
        manager.shutdown()
    restored = JobManager(tmp_path)
    try:
        recovered = restored.require(record.id)
        assert recovered.status == "succeeded"
        assert recovered.result_pdf.name == "result.png"
        assert recovered.finished_at == finished_at
        assert recovered.options["method"] == "qr"
        with restored.pin(record.id) as pinned:
            assert pinned.result_pdf.read_bytes() == expected
            assert pinned.public_dict()["result_url"] == f"/api/jobs/{record.id}/result"
            assert restored.cleanup_terminal_files() == 0
        assert recovered.result_pdf.is_file()
    finally:
        restored.shutdown()


@pytest.mark.parametrize("suffix,label", [(".pdf", "PDF"), (".png", "图片"), (".zip", "结果文件")])
def test_result_message_matches_output_type(tmp_path, suffix, label):
    manager = JobManager(tmp_path)
    record = manager.reserve(original_filename="source", options={}, has_pairs=False)
    manager.update(record.id, result_pdf=record.directory / f"result{suffix}")

    def process(context):
        context.record.result_pdf.write_bytes(b"result")
        return context.record.result_pdf

    try:
        manager.submit(record.id, process, pool="tools").result(timeout=3)
        assert record.status == "succeeded"
        assert record.message.startswith(label)
        assert record.result_pdf.read_bytes() == b"result"
    finally:
        manager.shutdown()


@pytest.mark.parametrize("status", ["running", "cancelled"])
def test_live_cpu_future_prevents_idle_cache_and_input_cleanup(tmp_path, status):
    manager = JobManager(tmp_path)
    record = manager.reserve(original_filename="cpu.png", options={}, has_pairs=False)
    manager.update(record.id, result_pdf=record.directory / "result.png")
    record.input_pdf.write_bytes(b"active input")
    cache = tmp_path / "cupy-cache" / "old.cubin"
    spool = tmp_path / "tmp" / "old-spool.tmp"
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).timestamp()
    for path in (cache, spool):
        path.parent.mkdir()
        path.write_bytes(b"old cache")
        os.utime(path, (old, old))
    started, release = threading.Event(), threading.Event()

    def process(context):
        started.set()
        assert release.wait(5)
        context.record.result_pdf.write_bytes(b"PNG result")
        return context.record.result_pdf

    future = manager.submit(record.id, process, pool="tools")
    try:
        assert started.wait(2)
        # Even a terminal status must not make a still-running future idle.
        if status == "cancelled":
            manager.update(record.id, status="cancelled")
        result = manager.maintenance(cache_dir=cache.parent, temp_dir=spool.parent)
        assert result["removed_cache_files"] == result["removed_temp_files"] == 0
        assert cache.exists() and spool.exists()
        assert record.input_pdf.read_bytes() == b"active input"
        release.set()
        future.result(timeout=3)
        result = manager.maintenance(cache_dir=cache.parent, temp_dir=spool.parent)
        assert result["removed_cache_files"] == result["removed_temp_files"] == 1
        assert record.result_pdf.is_file()
    finally:
        release.set()
        future.result(timeout=3)
        manager.shutdown()


def test_shutdown_cancels_queued_tasks_in_both_worker_pools(tmp_path):
    manager = JobManager(tmp_path)
    release = threading.Event()
    running = []
    queued = []
    called = threading.Event()

    def never_run(_context):
        called.set()

    try:
        for pool in ("gpu", "tools"):
            started = threading.Event()

            def block(context, started=started):
                started.set()
                assert release.wait(5)
                context.record.result_pdf.write_bytes(b"result")
                return context.record.result_pdf

            first = manager.reserve(original_filename=f"{pool}.pdf", options={}, has_pairs=False)
            running.append(manager.submit(first.id, block, pool=pool))
            assert started.wait(2)
            second = manager.reserve(original_filename=f"{pool}-queued.pdf", options={}, has_pairs=False)
            queued.append(manager.submit(second.id, never_run, pool=pool))
        manager.shutdown()
        assert all(future.cancelled() for future in queued)
        release.set()
        for future in running:
            future.result(timeout=3)
        assert not called.is_set()
    finally:
        release.set()
        for future in running:
            future.result(timeout=3)
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
