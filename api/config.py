from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} 必须大于 0")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    work_dir: Path
    max_upload_bytes: int = 64 * 1024**3
    max_csv_bytes: int = 64 * 1024**2
    upload_chunk_bytes: int = 8 * 1024**2
    max_pending_jobs: int = 4
    force_cpu: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            work_dir=Path(os.getenv("CMYK_API_WORK_DIR", str(Path(__file__).resolve().parent.parent / "runtime"))).expanduser().resolve(),
            max_upload_bytes=_positive_int("CMYK_API_MAX_UPLOAD_GIB", 64) * 1024**3,
            max_csv_bytes=_positive_int("CMYK_API_MAX_CSV_MIB", 64) * 1024**2,
            upload_chunk_bytes=_positive_int("CMYK_API_UPLOAD_CHUNK_MIB", 8) * 1024**2,
            max_pending_jobs=_positive_int("CMYK_API_MAX_PENDING_JOBS", 4),
            force_cpu=os.getenv("CMYK_API_FORCE_CPU", "false").lower() in {"1", "true", "yes", "on"},
        )
