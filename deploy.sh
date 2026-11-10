#!/usr/bin/env bash
# Build and deploy from this checkout. --pull updates the current tracked branch.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

usage() {
  printf 'Usage: bash deploy.sh [--pull]\n'
}

pull=false
case "${1:-}" in
  '') ;;
  --pull) pull=true ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then usage >&2; exit 2; fi

command -v docker >/dev/null || { printf 'Docker is required.\n' >&2; exit 1; }
docker compose version >/dev/null
docker info >/dev/null

if "$pull"; then
  command -v git >/dev/null || { printf 'Git is required for --pull.\n' >&2; exit 1; }
  git rev-parse --is-inside-work-tree >/dev/null
  if [[ -n "$(git status --porcelain)" ]]; then
    printf 'Commit or stash local changes before --pull.\n' >&2
    exit 1
  fi
  git pull --ff-only
fi

if [[ ! -f .env ]]; then
  cp -- .env.example .env
  printf 'Created .env with default settings (GPU 0, port 5522).\n'
fi

docker compose config --quiet
timeout="${HEALTH_TIMEOUT:-300}"
if [[ ! "$timeout" =~ ^[1-9][0-9]*$ ]]; then
  printf 'HEALTH_TIMEOUT must be a positive integer.\n' >&2
  exit 2
fi

docker compose build --pull color-api
if ! docker compose up -d --wait --wait-timeout "$timeout" color-api; then
  docker compose logs --tail=100 color-api >&2
  printf 'Deployment did not become ready. Check GPU / CUDA and the logs above.\n' >&2
  exit 1
fi
docker compose ps
printf 'Ready: http://localhost:5522/api/docs\n'
