"""Validate the output mount, then prepare only its four managed directories.

Executed by deploy/restart in a temporary root container. Never walks existing
job/cache contents and never recursively changes ownership.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile


DIRECTORIES = ("jobs", "tmp", "cupy-cache")


def output_source(config: dict) -> str:
    volumes = config["services"]["color-api"].get("volumes", [])
    matches = [item for item in volumes if item.get("target") == "/data"]
    if len(matches) != 1 or matches[0].get("type") != "bind":
        raise ValueError("color-api /data 必须配置为单个 bind mount")
    source = matches[0].get("source", "")
    if not isinstance(source, str) or not PurePosixPath(source).is_absolute():
        raise ValueError("输出 bind 源必须是绝对 Linux 路径")
    if any(char in source for char in "\r\n\x00,"):
        raise ValueError("输出路径不支持换行、NUL 或逗号")
    return source


def validate_layout(root: Path) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("输出根目录必须是真实目录")
    unexpected = [entry.name for entry in root.iterdir() if entry.name not in {*DIRECTORIES, ".gitkeep"}]
    if unexpected:
        raise ValueError(f"拒绝修改非 API 数据目录（含未知项目）：{unexpected[:8]}")
    targets = (root, *(root / name for name in DIRECTORIES))
    for path in targets:
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f"API 输出子目录必须是真实目录：{path.name}")
    return targets


def prepare(root: Path = Path("/data")) -> None:
    targets = validate_layout(root)
    for path in targets:
        path.mkdir(exist_ok=True)
        os.chown(path, 10001, 10001, follow_symlinks=False)
        path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IRWXU)


def describe(path: Path) -> str:
    try:
        info = path.lstat()
        return f"{path}: uid={info.st_uid} gid={info.st_gid} mode={stat.filemode(info.st_mode)}"
    except OSError as exc:
        return f"{path}: 无法读取目录属性：{exc}"


def verify(root: Path = Path("/data")) -> None:
    uid, gid = os.geteuid(), os.getegid()
    if uid != 10001:
        raise PermissionError(f"写入校验必须使用 API uid 10001，当前 uid={uid} gid={gid}")
    print(f"Output write check: process uid={uid} gid={gid}", flush=True)
    failed = []
    for path in (root, *(root / name for name in DIRECTORIES)):
        try:
            if path.is_symlink() or not path.is_dir():
                raise PermissionError("必须是已初始化的真实目录")
            # Probe a new unique file only. Existing uploads/results are never
            # opened or modified, and the probe is removed before returning.
            with tempfile.NamedTemporaryFile(dir=path, prefix=".colorapi-write-check-") as probe:
                probe.write(b"color-api storage write check\n")
                probe.flush()
                os.fsync(probe.fileno())
                probe.seek(0)
                if probe.read() != b"color-api storage write check\n":
                    raise OSError("写入校验内容不一致")
        except OSError as exc:
            failed.append(path)
            print(f"FAIL {describe(path)}; {exc}", file=sys.stderr, flush=True)
        else:
            print(f"OK   {describe(path)}", flush=True)
    if failed:
        raise PermissionError(
            f"API uid 10001 写入失败：{', '.join(str(path) for path in failed)}。"
            "请检查 bind 源路径、owner/mode 及宿主机 ACL/SELinux/NFS 挂载策略；不会自动修改这些策略。")


def main() -> None:
    if sys.argv[1:] == ["resolve"]:
        print(output_source(json.load(sys.stdin)))
    elif sys.argv[1:] == ["prepare"]:
        prepare()
    elif sys.argv[1:] == ["verify"]:
        verify()
    else:
        raise ValueError("Usage: prepare_storage.py resolve|prepare|verify")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        if sys.argv[1:] == ["prepare"]:
            for path in (Path("/data"), *(Path("/data") / name for name in DIRECTORIES)):
                print(describe(path), file=sys.stderr)
        raise SystemExit(f"Output preparation failed: {exc}") from exc
