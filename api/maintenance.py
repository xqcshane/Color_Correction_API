"""Scoped age-based file cleanup; JobManager holds the scheduling/idle lock."""

from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path

_LOG = logging.getLogger(__name__)


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _cleanup_files(root_path: Path, *, retention_hours: float, remove_empty_dirs: bool) -> tuple[int, int]:
    if _is_link(root_path):
        return 0, 0
    root = root_path.resolve()
    if not root.is_dir():
        return 0, 0
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=retention_hours)).timestamp()
    removed = 0
    directories: list[tuple[Path, float]] = []
    for directory, names, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        safe_names = []
        for name in names:
            child = parent / name
            if _is_link(child) or not child.resolve().is_relative_to(root):
                continue
            safe_names.append(name)
            if remove_empty_dirs:
                try:
                    directories.append((child, child.stat().st_mtime))
                except OSError:
                    pass
        names[:] = safe_names
        for name in files:
            path = parent / name
            try:
                if _is_link(path) or not path.resolve().is_relative_to(root) or not path.is_file():
                    continue
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                _LOG.warning("Unable to remove cache file %s", path, exc_info=True)
    removed_dirs = 0
    for path, previous_mtime in reversed(directories):
        if previous_mtime >= cutoff or _is_link(path) or not path.resolve().is_relative_to(root):
            continue
        try:
            # rmdir only removes an empty directory. Never recursively remove
            # a temporary directory that might contain a fresh/active file.
            path.rmdir()
            removed_dirs += 1
        except OSError:
            pass
    return removed, removed_dirs


def cleanup_old_cache_files(cache_dir: Path, *, retention_hours: float = 24) -> int:
    """Delete old compiled files; preserve the cache root and subdirectories."""
    return _cleanup_files(cache_dir, retention_hours=retention_hours, remove_empty_dirs=False)[0]


def cleanup_old_temp_files(temp_dir: Path, *, retention_hours: float = 24) -> tuple[int, int]:
    """Delete old temporary files/empty subdirectories, preserving their root."""
    return _cleanup_files(temp_dir, retention_hours=retention_hours, remove_empty_dirs=True)
