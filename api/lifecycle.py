"""Response leases and request-independent storage maintenance."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import logging
from pathlib import Path
import tempfile
from typing import Any

from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool


logger = logging.getLogger(__name__)


class LeasedFileResponse(FileResponse):
    def __init__(self, *args, lease: Any, **kwargs):
        self._lease = lease
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            lease.release()
            raise

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Background callbacks do not reliably run after a client disconnect.
            self._lease.release()


@contextmanager
def upload_temporary_directory(directory: Path):
    """Keep multipart spooling in output/tmp while this app is running."""
    directory.mkdir(parents=True, exist_ok=True)
    previous = tempfile.tempdir
    configured = str(directory.resolve())
    tempfile.tempdir = configured
    try:
        yield
    finally:
        if tempfile.tempdir == configured:
            tempfile.tempdir = previous


async def maintenance_loop(jobs, *, cache_dir: Path, temp_dir: Path,
                           cache_ttl_hours: float, interval_seconds: float,
                           stop: asyncio.Event) -> None:
    # Run once at startup, then on a fixed interval even with no API traffic.
    while not stop.is_set():
        try:
            await run_in_threadpool(jobs.maintenance, cache_dir=cache_dir,
                                    temp_dir=temp_dir, cache_ttl_hours=cache_ttl_hours)
        except Exception:
            logger.exception("Output cleanup failed; retrying at the next interval")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except TimeoutError:
            continue
