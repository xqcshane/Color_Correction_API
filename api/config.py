from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} 必须大于 0")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是大于 0 的有限数字")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    work_dir: Path
    max_upload_bytes: int = 64 * 1024**3
    max_csv_bytes: int = 64 * 1024**2
    upload_chunk_bytes: int = 8 * 1024**2
    max_pending_jobs: int = 4
    force_cpu: bool = False
    job_ttl_hours: float = 24.0
    cache_ttl_hours: float = 24.0
    cleanup_interval_seconds: int = 3600

    @classmethod
    def from_env(cls) -> "Settings":
        project_dir = Path(__file__).resolve().parent.parent
        output_dir = os.getenv("CMYK_API_OUTPUT_DIR", str(project_dir / "output"))
        return cls(
            work_dir=Path(os.getenv("CMYK_API_WORK_DIR", output_dir)).expanduser().resolve(),
            max_upload_bytes=_positive_int("CMYK_API_MAX_UPLOAD_GIB", 64) * 1024**3,
            max_csv_bytes=_positive_int("CMYK_API_MAX_CSV_MIB", 64) * 1024**2,
            upload_chunk_bytes=_positive_int("CMYK_API_UPLOAD_CHUNK_MIB", 8) * 1024**2,
            max_pending_jobs=_positive_int("CMYK_API_MAX_PENDING_JOBS", 4),
            force_cpu=os.getenv("CMYK_API_FORCE_CPU", "false").lower() in {"1", "true", "yes", "on"},
            job_ttl_hours=_positive_float("CMYK_API_JOB_TTL_HOURS", 24),
            cache_ttl_hours=_positive_float("CMYK_API_CACHE_TTL_HOURS", 24),
            cleanup_interval_seconds=_positive_int("CMYK_API_CLEANUP_INTERVAL_SECONDS", 3600),
        )
