"""Storage preparation tests use temporary directories and never perform chown."""
from __future__ import annotations

from pathlib import Path
import stat
from unittest.mock import Mock, call

import pytest

from scripts import prepare_storage


@pytest.fixture
def ownership(monkeypatch):
    mock = Mock()
    # os.chown does not exist on Windows. Production runs this helper in Linux.
    monkeypatch.setattr(prepare_storage.os, "chown", mock, raising=False)
    return mock


def managed_paths(root):
    return [root, root / "jobs", root / "tmp", root / "cupy-cache"]


def test_empty_output_creates_children_and_owns_exactly_four_top_level_directories(tmp_path, ownership):
    root = tmp_path / "output"
    root.mkdir()
    prepare_storage.prepare(root)
    targets = managed_paths(root)
    assert all(path.is_dir() for path in targets)
    assert {path.name for path in root.iterdir()} == {"jobs", "tmp", "cupy-cache"}
    assert ownership.call_args_list == [
        call(path, 10001, 10001, follow_symlinks=False) for path in targets]
    assert all(stat.S_IMODE(path.stat().st_mode) & stat.S_IRWXU == stat.S_IRWXU for path in targets)


def test_existing_nested_files_are_not_walked_chowned_or_modified(tmp_path, ownership, monkeypatch):
    root = tmp_path / "output"
    for name in prepare_storage.DIRECTORIES:
        (root / name / "existing" / "deep").mkdir(parents=True)
    files = [root / name / "existing" / "deep" / "old.data" for name in prepare_storage.DIRECTORIES]
    for path in files:
        path.write_bytes(b"existing data must remain unchanged\x00\xff")
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode) for path in files}
    original_iterdir = Path.iterdir
    visited = []

    def top_level_only(path):
        visited.append(path)
        assert path == root, "storage preparation traversed existing job/cache contents"
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", top_level_only)
    prepare_storage.prepare(root)
    assert visited == [root]
    assert [entry.args[0] for entry in ownership.call_args_list] == managed_paths(root)
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode) for path in files} == before


def test_prepare_is_idempotent_and_preserves_gitkeep(tmp_path, ownership):
    root = tmp_path / "output"
    root.mkdir()
    marker = root / ".gitkeep"
    marker.write_text("keep", encoding="utf-8")
    prepare_storage.prepare(root)
    first_layout = {path.name for path in root.iterdir()}
    ownership.reset_mock()
    prepare_storage.prepare(root)
    assert {path.name for path in root.iterdir()} == first_layout
    assert marker.read_text(encoding="utf-8") == "keep"
    assert ownership.call_args_list == [
        call(path, 10001, 10001, follow_symlinks=False) for path in managed_paths(root)]


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_unknown_layout_is_rejected_before_any_creation_or_ownership_change(tmp_path, ownership, kind):
    root = tmp_path / "output"
    root.mkdir()
    unrelated = root / "user-owned-data"
    if kind == "file":
        unrelated.write_bytes(b"keep")
    else:
        unrelated.mkdir()
    with pytest.raises(ValueError, match="非 API 数据目录"):
        prepare_storage.prepare(root)
    ownership.assert_not_called()
    assert list(root.iterdir()) == [unrelated]
    assert unrelated.is_file() if kind == "file" else unrelated.is_dir()


@pytest.mark.parametrize("name", prepare_storage.DIRECTORIES)
def test_managed_path_that_is_a_file_is_rejected_before_mutation(tmp_path, ownership, name):
    root = tmp_path / "output"
    root.mkdir()
    existing_file = root / name
    existing_file.write_bytes(b"not a directory")
    with pytest.raises(ValueError, match="子目录必须是真实目录"):
        prepare_storage.prepare(root)
    ownership.assert_not_called()
    assert list(root.iterdir()) == [existing_file]
    assert existing_file.read_bytes() == b"not a directory"


@pytest.mark.parametrize("name", [None, *prepare_storage.DIRECTORIES])
def test_root_and_managed_symlinks_are_rejected(tmp_path, ownership, monkeypatch, name):
    root = tmp_path / "output"
    target = tmp_path / "external-data"
    target.mkdir()
    sentinel = target / "untouched.data"
    sentinel.write_bytes(b"outside data")
    if name is not None:
        root.mkdir()
    link = root if name is None else root / name
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        # Windows may deny creation without Developer Mode/admin. Linux CI
        # uses real symlinks; this fallback still exercises rejection locally.
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda path: path == link or original(path))
    with pytest.raises(ValueError, match="真实目录"):
        prepare_storage.prepare(root)
    ownership.assert_not_called()
    assert sentinel.read_bytes() == b"outside data"
    assert list(target.iterdir()) == [sentinel]


def test_missing_output_root_is_not_created_or_owned_implicitly(tmp_path, ownership):
    root = tmp_path / "output"
    with pytest.raises(ValueError, match="根目录必须是真实目录"):
        prepare_storage.prepare(root)
    ownership.assert_not_called()
    assert not root.exists()


@pytest.fixture
def api_identity(monkeypatch):
    monkeypatch.setattr(prepare_storage.os, "geteuid", lambda: 10001, raising=False)
    monkeypatch.setattr(prepare_storage.os, "getegid", lambda: 10001, raising=False)


def test_verify_writes_all_four_directories_without_chown_or_leftover_probes(tmp_path, ownership, api_identity, capsys):
    root = tmp_path / "output"
    root.mkdir()
    prepare_storage.prepare(root)
    existing = root / "jobs" / "result.pdf"
    existing.write_bytes(b"existing result")
    ownership.reset_mock()
    prepare_storage.verify(root)
    ownership.assert_not_called()
    assert existing.read_bytes() == b"existing result"
    assert not list(root.rglob(".colorapi-write-check-*"))
    output = capsys.readouterr()
    assert output.out.count("OK   ") == 4
    assert "process uid=10001 gid=10001" in output.out
    assert not output.err


@pytest.mark.parametrize("name", [None, *prepare_storage.DIRECTORIES])
def test_verify_reports_owner_mode_and_path_when_any_directory_is_unwritable(
        tmp_path, ownership, api_identity, monkeypatch, capsys, name):
    root = tmp_path / "output"
    root.mkdir()
    prepare_storage.prepare(root)
    blocked = root if name is None else root / name
    original = prepare_storage.tempfile.NamedTemporaryFile

    def denied_probe(*args, **kwargs):
        if Path(kwargs["dir"]) == blocked:
            raise PermissionError(13, "Permission denied", str(blocked))
        return original(*args, **kwargs)

    monkeypatch.setattr(prepare_storage.tempfile, "NamedTemporaryFile", denied_probe)
    ownership.reset_mock()
    with pytest.raises(PermissionError, match="API uid 10001 写入失败"):
        prepare_storage.verify(root)
    ownership.assert_not_called()
    output = capsys.readouterr()
    assert str(blocked) in output.err
    assert "uid=" in output.err and "gid=" in output.err and "mode=" in output.err
    assert "Permission denied" in output.err
    assert output.out.count("OK   ") == 3
    assert not list(root.rglob(".colorapi-write-check-*"))


def test_verify_refuses_root_so_it_cannot_report_a_false_nonroot_success(tmp_path, ownership, monkeypatch):
    monkeypatch.setattr(prepare_storage.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(prepare_storage.os, "getegid", lambda: 0, raising=False)
    probe = Mock()
    monkeypatch.setattr(prepare_storage.tempfile, "NamedTemporaryFile", probe)
    with pytest.raises(PermissionError, match="必须使用 API uid 10001"):
        prepare_storage.verify(tmp_path)
    probe.assert_not_called()
    ownership.assert_not_called()
