from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import time

from fastapi.testclient import TestClient
import pytest

from api.config import Settings
from api.lifecycle import LeasedFileResponse, maintenance_loop, upload_temporary_directory
from api.main import create_app


def test_default_output_is_in_project_directory(monkeypatch):
    for name in ("CMYK_API_WORK_DIR", "CMYK_API_OUTPUT_DIR"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.work_dir == Path(__file__).resolve().parents[1] / "output"
    assert settings.job_ttl_hours == 24
    assert settings.cache_ttl_hours == 24
    assert settings.cleanup_interval_seconds == 3600


def test_storage_directory_override_and_legacy_work_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("CMYK_API_WORK_DIR", raising=False)
    monkeypatch.setenv("CMYK_API_OUTPUT_DIR", str(tmp_path / "custom-output"))
    assert Settings.from_env().work_dir == tmp_path / "custom-output"
    monkeypatch.setenv("CMYK_API_WORK_DIR", str(tmp_path / "container-data"))
    assert Settings.from_env().work_dir == tmp_path / "container-data"


@pytest.mark.parametrize("name,value", [("CMYK_API_JOB_TTL_HOURS", "nan"),
    ("CMYK_API_CACHE_TTL_HOURS", "0"), ("CMYK_API_CLEANUP_INTERVAL_SECONDS", "-1")])
def test_invalid_cleanup_configuration_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        Settings.from_env()


def test_multipart_spool_directory_is_restored_after_lifespan(tmp_path):
    original = tempfile.tempdir
    target = tmp_path / "output" / "tmp"
    with upload_temporary_directory(target):
        assert Path(tempfile.gettempdir()) == target
        with tempfile.TemporaryFile() as spool:
            spool.write(b"multipart bytes")
            assert spool.tell() == 15
    assert tempfile.tempdir == original


def test_cleanup_runs_without_requests_and_survives_one_failure(tmp_path):
    class Jobs:
        calls = 0
        def maintenance(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise OSError("transient filesystem error")

    jobs = Jobs()
    async def scenario():
        stop = asyncio.Event()
        task = asyncio.create_task(maintenance_loop(jobs, cache_dir=tmp_path / "cache",
            temp_dir=tmp_path / "tmp", cache_ttl_hours=24, interval_seconds=.01, stop=stop))
        try:
            for _ in range(200):
                if jobs.calls >= 3:
                    break
                await asyncio.sleep(.01)
            assert jobs.calls >= 3
        finally:
            stop.set()
            await task
    asyncio.run(scenario())


def test_completed_pdf_expires_periodically_with_no_http_requests(tmp_path):
    app = create_app(Settings(work_dir=tmp_path / "output", force_cpu=True,
                              cleanup_interval_seconds=.02))
    previous = tempfile.tempdir
    with TestClient(app):
        assert Path(tempfile.gettempdir()) == app.state.settings.work_dir / "tmp"
        jobs = app.state.jobs
        record = jobs.reserve(original_filename="input.pdf", options={}, has_pairs=True)
        record.result_pdf.write_bytes(b"%PDF-1.7 synthetic result")
        jobs.update(record.id, status="succeeded")
        record.finished_at = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        for _ in range(200):
            if not record.directory.exists():
                break
            time.sleep(.01)
        assert not record.directory.exists()
        with pytest.raises(KeyError):
            jobs.require(record.id)
    assert tempfile.tempdir == previous


@pytest.mark.parametrize("disconnect", [False, True])
def test_download_lease_is_released_on_success_or_client_disconnect(tmp_path, disconnect):
    pdf = tmp_path / "result.pdf"
    pdf.write_bytes(b"%PDF-1.7 synthetic result")
    class Lease:
        releases = 0
        def release(self):
            self.releases += 1
    lease = Lease()
    response = LeasedFileResponse(pdf, media_type="application/pdf", lease=lease)
    scope = {"type": "http", "method": "GET", "headers": [],
             "asgi": {"version": "3.0", "spec_version": "2.4"}}
    async def receive():
        return {"type": "http.disconnect"}
    async def send(message):
        if disconnect and message["type"] == "http.response.body":
            raise OSError("client closed download")
    async def scenario():
        if disconnect:
            with pytest.raises(Exception):
                await response(scope, receive, send)
        else:
            await response(scope, receive, send)
    asyncio.run(scenario())
    assert lease.releases == 1
