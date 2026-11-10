"""Run real Bash against temporary fixtures; Git and Docker are shell fakes."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPOSITORY = Path(__file__).resolve().parent.parent
BASH = (str(Path("C:/Program Files/Git/bin/bash.exe")) if os.name == "nt"
        and Path("C:/Program Files/Git/bin/bash.exe").is_file() else shutil.which("bash"))
pytestmark = pytest.mark.skipif(not BASH, reason="Bash is required for deployment-script regression tests")


def write(path, content):
    path.write_text(content, encoding="utf-8", newline="\n")


@pytest.fixture
def fixture_project(tmp_path):
    (tmp_path / "scripts").mkdir()
    write(tmp_path / ".env", "# fixture only\n")
    write(tmp_path / ".env.example", "# fixture only\n")
    environment = tmp_path / "fake-commands.sh"
    write(environment, """#!/usr/bin/env bash
docker() { printf 'docker:%s\\n' "$*" >> "$FIXTURE_RECORD"; }
git() {
  case "${1:-}" in
    pull)
      printf 'pull\\n' >> "$FIXTURE_RECORD"
      mv -f -- "$FIXTURE_NEW_SCRIPT" "$FIXTURE_DEPLOY_SCRIPT" ;;
    status|rev-parse) return 0 ;;
    *) return 91 ;;
  esac
}
""")
    env = dict(os.environ, BASH_ENV=environment.as_posix(),
               FIXTURE_RECORD=(tmp_path / "record.txt").as_posix(),
               FIXTURE_NEW_SCRIPT=(tmp_path / "new.sh").as_posix(),
               FIXTURE_DEPLOY_SCRIPT=(tmp_path / "deploy.sh").as_posix())
    return tmp_path, env


def run_script(root, env, filename, *arguments):
    return subprocess.run([BASH, (root / filename).as_posix(), *arguments],
                          env=env, capture_output=True, text=True, timeout=15)


def install_replacement(root):
    write(root / "new.sh", """#!/usr/bin/env bash
set -eu
printf 'new-args:%s\\n' "$#" >> "$FIXTURE_RECORD"
printf 'new-prepare\\nnew-start\\n' >> "$FIXTURE_RECORD"
""")


def test_atomic_upgrade_without_reexec_reproduces_old_fd_skipping_new_preparation(fixture_project):
    root, env = fixture_project
    legacy = (REPOSITORY / "deploy.sh").read_text(encoding="utf-8")
    legacy = legacy.replace("  exec bash ./deploy.sh\n", "")
    legacy = legacy.replace("bash scripts/prepare-output.sh\n", "")
    write(root / "deploy.sh", legacy)
    install_replacement(root)
    result = run_script(root, env, "deploy.sh", "--pull")
    assert result.returncode == 0, result.stderr
    record = (root / "record.txt").read_text()
    assert "pull\n" in record
    assert "docker:compose up" in record
    assert "new-prepare" not in record
    assert "new-prepare" in (root / "deploy.sh").read_text()


def test_pull_reopens_updated_script_without_passing_pull_again(fixture_project):
    root, env = fixture_project
    write(root / "deploy.sh", (REPOSITORY / "deploy.sh").read_text(encoding="utf-8"))
    install_replacement(root)
    result = run_script(root, env, "deploy.sh", "--pull")
    assert result.returncode == 0, result.stderr
    record = (root / "record.txt").read_text()
    assert record.count("pull\n") == 1
    assert "new-args:0\nnew-prepare\nnew-start\n" in record
    assert "docker:compose build" not in record
    assert "docker:compose up" not in record


@pytest.mark.parametrize("filename", ["deploy.sh", "restart.sh"])
def test_preparation_failure_prevents_api_startup(fixture_project, filename):
    root, env = fixture_project
    write(root / filename, (REPOSITORY / filename).read_text(encoding="utf-8"))
    write(root / "scripts" / "prepare-output.sh", "printf 'prepare-failed\\n' >> \"$FIXTURE_RECORD\"\nexit 73\n")
    result = run_script(root, env, filename)
    assert result.returncode == 73
    record = (root / "record.txt").read_text()
    assert "prepare-failed" in record
    assert "docker:compose up" not in record


def bash_path(path):
    value = path.as_posix()
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


def test_output_helper_uses_api_uid_and_rejects_failed_write_check(fixture_project):
    root, env = fixture_project
    for filename in ("prepare-output.sh", "prepare_storage.py"):
        write(root / "scripts" / filename, (REPOSITORY / "scripts" / filename).read_text(encoding="utf-8"))
    env["FIXTURE_OUTPUT_SOURCE"] = bash_path(root / "output")
    write(root / "fake-commands.sh", """#!/usr/bin/env bash
docker() {
  if [[ "${1:-}" == compose ]]; then printf '{}'; return 0; fi
  local phase="${@: -1}" uid='' previous='' argument
  for argument in "$@"; do
    if [[ "$previous" == --user ]]; then uid="$argument"; fi
    previous="$argument"
  done
  if [[ "$phase" == resolve ]]; then cat >/dev/null; printf '%s\\n' "$FIXTURE_OUTPUT_SOURCE"; return 0; fi
  printf '%s uid=%s\\n' "$phase" "$uid" >> "$FIXTURE_RECORD"
  if [[ "$phase" == prepare ]]; then return 0; fi
  if [[ "$phase" == verify ]]; then return 17; fi
  return 92
}
""")
    result = run_script(root, env, "scripts/prepare-output.sh")
    assert result.returncode == 1
    assert (root / "record.txt").read_text().splitlines() == ["prepare uid=0", "verify uid=10001:10001"]
    assert "API uid 10001 cannot write" in result.stderr
    assert "startup was not attempted" in result.stderr
