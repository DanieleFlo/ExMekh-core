#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
# The shared profile supplies Flash weights/projector, tiered cache and chunk.
# MTP is optional and can be requested with --mtp.
export DS4_MODEL_PROFILE=flash
exec "$REPO_DIR/start-ds4-server.sh" "$@"
