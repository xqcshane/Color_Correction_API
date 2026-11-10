#!/usr/bin/env bash
# Prepare only the configured API bind directory, using Python from the image.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

storage_helper="$(cat scripts/prepare_storage.py)"
storage_source="$(docker compose config --format json | docker run --rm -i \
  --user 0 --env NVIDIA_VISIBLE_DEVICES=void --entrypoint python \
  color-correction-api:local -c "$storage_helper" resolve)"
storage_canonical="$(realpath -m -- "$storage_source")"
storage_lexical="$(realpath -ms -- "$storage_source")"
storage_project="$(pwd -P)"
storage_home="$(realpath -m -- "${HOME:-/}")"

if [[ "$storage_canonical" != "$storage_lexical" ]]; then
  printf 'Output path must not contain symbolic links: %s\n' "$storage_source" >&2
  exit 1
fi
case "$storage_canonical" in
  /|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/media|/mnt|/opt|/proc|/root|/run|/sbin|/srv|/sys|/tmp|/usr|/var|/var/cache|/var/lib|/var/log)
    printf 'Refusing to use a system directory as API output: %s\n' "$storage_canonical" >&2
    exit 1 ;;
esac
if [[ "$storage_canonical" == "$storage_project" || "$storage_canonical" == "$storage_home" ]]; then
  printf 'Output directory must not be the repository root or home directory.\n' >&2
  exit 1
fi
if [[ -e "$storage_canonical" && ! -d "$storage_canonical" ]]; then
  printf 'Output path is not a directory: %s\n' "$storage_canonical" >&2
  exit 1
fi

mkdir -p -- "$storage_canonical"
printf 'Preparing API output: %s (container /data)\n' "$storage_canonical"
if ! docker run --rm --user 0 --env NVIDIA_VISIBLE_DEVICES=void --env TMPDIR=/tmp \
  --mount "type=bind,source=$storage_canonical,target=/data" --entrypoint python \
  color-correction-api:local -c "$storage_helper" prepare; then
  printf 'Output directory initialization failed; API startup was not attempted.\n' >&2
  exit 1
fi
if ! docker run --rm --user 10001:10001 --env NVIDIA_VISIBLE_DEVICES=void --env TMPDIR=/tmp \
  --mount "type=bind,source=$storage_canonical,target=/data" --entrypoint python \
  color-correction-api:local -c "$storage_helper" verify; then
  printf 'API uid 10001 cannot write the output bind mount; startup was not attempted.\n' >&2
  printf 'Inspect the directory owner/mode above and host mount/ACL policy.\n' >&2
  exit 1
fi
printf 'API output: %s (container /data; uid 10001)\n' "$storage_canonical"
