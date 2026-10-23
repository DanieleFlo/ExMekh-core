#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
# This entry point always selects Qwen27; Flash has its own launcher.
export DS4_MODEL_PROFILE=qwen27
# Preserve the agent launcher's existing Qwen27 defaults.
export DS4_CTX="${DS4_CTX:-26000}"
export DS4_KV_DISK_DIR="${DS4_KV_DISK_DIR-}"
# The agent's bundled image and embedding tools use this same server URL.
export DS4_SUITE_CONFIG="${DS4_SUITE_CONFIG:-config/native-suite.example.json}"
exec "$REPO_DIR/start-ds4-server.sh" "$@"
