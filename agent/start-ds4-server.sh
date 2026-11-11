#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
# A regular launch lets Core restore the profile selected in Agent.
if [[ $# -eq 0 ]]; then
    exec "$REPO_DIR/start-ds4-server.sh" --managed
fi
if [[ "$1" == --managed ]]; then
    exec "$REPO_DIR/start-ds4-server.sh" "$@"
fi
# Explicit native launches keep the Qwen27 defaults.
export DS4_MODEL_PROFILE=qwen27
# Keep CUDA defaults; the native Mac profile starts with a bounded LLM context.
if [[ "${DS4_BACKEND:-auto}" == metal || ( "${DS4_BACKEND:-auto}" == auto && "$(uname -s)" == Darwin ) ]]; then
    export DS4_CTX="${DS4_CTX:-512}"
    export DS4_SUITE_CONFIG="${DS4_SUITE_CONFIG-}"
else
    export DS4_CTX="${DS4_CTX:-26000}"
    export DS4_SUITE_CONFIG="${DS4_SUITE_CONFIG-config/native-suite.example.json}"
fi
export DS4_KV_DISK_DIR="${DS4_KV_DISK_DIR-}"
# The agent's bundled image and embedding tools use this same server URL.
exec "$REPO_DIR/start-ds4-server.sh" "$@"
