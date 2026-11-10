from concurrent.futures import Future
import os
from pathlib import Path
import subprocess
import threading
import time

import pytest

from api.jobs import JobManager
from api.maintenance import cleanup_old_cache_files, cleanup_old_temp_files


def old_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"old")
    old = time.time() - 25 * 3600
    os.utime(path, (old, old))
    return path


def test_compiled_cache_removes_old_files_and_preserves_directories(tmp_path):
    root = tmp_path / "cache"
    old = old_file(root / "nested" / "old.cubin")
    fresh = root / "fresh.cubin"
    fresh.write_bytes(b"fresh")
    assert cleanup_old_cache_files(root) == 1
    assert not old.exists()
    assert fresh.exists()
    assert root.is_dir() and old.parent.is_dir()


def test_temp_cleanup_removes_old_files_and_empty_subdirs_only(tmp_path):
    root = tmp_path / "tmp"
    old = old_file(root / "old-subdir" / "spooled-upload.tmp")
    fresh = root / "fresh-subdir" / "fresh.tmp"
    fresh.parent.mkdir()
    fresh.write_bytes(b"fresh")
    old_mtime = time.time() - 25 * 3600
    os.utime(old.parent, (old_mtime, old_mtime))
    os.utime(root, (old_mtime, old_mtime))
    assert cleanup_old_temp_files(root) == (1, 1)
    assert root.is_dir()
    assert not old.parent.exists()
    assert fresh.read_bytes() == b"fresh"


def test_unreserved_upload_leases_protect_cache_and_spools(tmp_path):
    manager = JobManager(tmp_path / "output")
    cache = old_file(tmp_path / "cache" / "kernel.cubin")
    spool = old_file(tmp_path / "tmp" / "large-upload.tmp")
    first, second = manager.begin_upload(), manager.begin_upload()
    try:
        assert manager.list() == []  # Multipart parsing can precede reserve().
        assert manager.maintenance(cache_dir=cache.parent, temp_dir=spool.parent)["removed_cache_files"] == 0
        assert cache.exists() and spool.exists()
        first.release()
        first.release()
        manager.maintenance(cache_dir=cache.parent, temp_dir=spool.parent)
        assert cache.exists() and spool.exists()
        second.release()
        result = manager.maintenance(cache_dir=cache.parent, temp_dir=spool.parent)
        assert result["removed_cache_files"] == result["removed_temp_files"] == 1
        assert not cache.exists() and not spool.exists()
    finally:
        first.release()
        second.release()
        manager.shutdown()


@pytest.mark.parametrize("status", ["uploading", "queued", "running"])
def test_active_jobs_keep_inputs_and_disable_cache_maintenance(tmp_path, status):
    manager = JobManager(tmp_path / "output")
    record = manager.reserve(original_filename="active.pdf", options={}, has_pairs=True)
    record.input_pdf.write_bytes(b"input still in use")
    manager.update(record.id, status=status)
    cache = old_file(tmp_path / "cache" / "kernel.cubin")
    spool = old_file(tmp_path / "tmp" / "upload.tmp")
    try:
        result = manager.maintenance(cache_dir=cache.parent, temp_dir=spool.parent)
        assert result["expired_jobs"] == result["removed_artifacts"] == result["removed_cache_files"] == result["removed_temp_files"] == 0
        assert record.input_pdf.exists() and cache.exists() and spool.exists()
    finally:
        manager.shutdown()


def test_not_yet_finished_future_disables_cache_cleanup(tmp_path):
    manager = JobManager(tmp_path / "output")
    record = manager.reserve(original_filename="finishing.pdf", options={}, has_pairs=False)
    manager.update(record.id, status="failed")
    future = Future()
    manager._futures[record.id] = future
    cache = old_file(tmp_path / "cache" / "kernel.cubin")
    try:
        assert manager.maintenance(cache_dir=cache.parent)["removed_cache_files"] == 0
        assert cache.exists()
        future.set_result(None)
        assert manager.maintenance(cache_dir=cache.parent)["removed_cache_files"] == 1
    finally:
        manager.shutdown()


def test_new_upload_waits_until_idle_cleanup_releases_scheduling_lock(tmp_path, monkeypatch):
    import api.maintenance
    manager = JobManager(tmp_path / "output")
    entered, release, attempting_upload, upload_started = (threading.Event() for _ in range(4))
    leases = []

    def slow_cleanup(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return 0

    def start_upload():
        attempting_upload.set()
        leases.append(manager.begin_upload())
        upload_started.set()

    monkeypatch.setattr(api.maintenance, "cleanup_old_cache_files", slow_cleanup)
    cleaner = threading.Thread(target=lambda: manager.maintenance(cache_dir=tmp_path / "cache"))
    uploader = threading.Thread(target=start_upload)
    try:
        cleaner.start()
        assert entered.wait(2)
        uploader.start()
        assert attempting_upload.wait(2)
        assert not upload_started.wait(.05)
        release.set()
        cleaner.join(2)
        uploader.join(2)
        assert upload_started.is_set()
    finally:
        release.set()
        cleaner.join(3)
        if uploader.ident is not None:
            uploader.join(3)
        for lease in leases:
            lease.release()
        manager.shutdown()


def test_cleanup_rejects_roots_containing_or_inside_jobs(tmp_path):
    manager = JobManager(tmp_path / "output")
    record = manager.reserve(original_filename="keep.pdf", options={}, has_pairs=False)
    manager.update(record.id, status="succeeded")
    try:
        for root in (manager.work_dir, manager.jobs_dir, record.directory):
            with pytest.raises(ValueError, match="任务目录"):
                manager.maintenance(cache_dir=root)
            with pytest.raises(ValueError, match="任务目录"):
                manager.maintenance(temp_dir=root)
        assert record.directory.exists()
    finally:
        manager.shutdown()


def test_cache_cleanup_does_not_follow_links_or_touch_other_directories(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    outside = old_file(tmp_path / "outside" / "important.pdf")
    try:
        (root / "outside-link").symlink_to(outside.parent, target_is_directory=True)
        (root / "file-link").symlink_to(outside)
        (tmp_path / "cache-link").symlink_to(outside.parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Filesystem cannot create symlinks: {exc}")
    assert cleanup_old_cache_files(root) == 0
    assert cleanup_old_temp_files(tmp_path / "cache-link") == (0, 0)
    assert outside.read_bytes() == b"old"


def test_pinned_job_skips_artifact_cleanup_until_response_ends(tmp_path):
    manager = JobManager(tmp_path / "output")
    record = manager.reserve(original_filename="download.pdf", options={}, has_pairs=True)
    record.input_pdf.write_bytes(b"input")
    record.result_pdf.write_bytes(b"result")
    manager.update(record.id, status="succeeded")
    try:
        with manager.pin(record.id):
            assert manager.maintenance()["removed_artifacts"] == 0
            assert record.input_pdf.exists()
        assert manager.maintenance()["removed_artifacts"] == 1
        assert not record.input_pdf.exists()
        assert record.result_pdf.read_bytes() == b"result"
    finally:
        manager.shutdown()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_cache_cleanup_does_not_follow_windows_junctions(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    outside = old_file(tmp_path / "outside" / "important.pdf")
    link = root / "outside-junction"
    result = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(outside.parent)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    try:
        assert link.is_junction()
        assert cleanup_old_cache_files(root) == 0
        assert cleanup_old_temp_files(link) == (0, 0)
        assert outside.read_bytes() == b"old"
    finally:
        # Remove the junction itself with a non-recursive operation. Its target
        # is a separate test fixture, never traversed by cache cleanup.
        link.rmdir()
