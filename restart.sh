#!/usr/bin/env bash
# Recreate with the already-built image and current .env; keep persistent data.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if (( $# )); then
  printf 'Usage: bash restart.sh\n' >&2
  exit 2
fi
command -v docker >/dev/null || { printf 'Docker is required.\n' >&2; exit 1; }
docker compose version >/dev/null
docker info >/dev/null
docker compose config --quiet

timeout="${HEALTH_TIMEOUT:-300}"
if [[ ! "$timeout" =~ ^[1-9][0-9]*$ ]]; then
  printf 'HEALTH_TIMEOUT must be a positive integer.\n' >&2
  exit 2
fi
bash scripts/prepare-output.sh
if ! docker compose up -d --no-build --force-recreate --wait --wait-timeout "$timeout" color-api; then
  docker compose logs --tail=100 color-api >&2
  printf 'Restart did not become ready. Check GPU / CUDA and the logs above.\n' >&2
  exit 1
fi
docker compose ps
printf 'Ready: http://localhost:5522/docs\n'
