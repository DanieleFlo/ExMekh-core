#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
SERVER_PORT="${DS4_SERVER_PORT:-8080}"

cd "$REPO_DIR"

echo "Checking ds4-server build..."
make --no-print-directory -j2 ds4-server CUDA_ARCH="${CUDA_ARCH:-sm_86}"

vision_args=()
vision_model="${DS4_VISION_MODEL:-gguf/mmproj-Qwen3.8-27B-F16.gguf}"
if [[ -f "$vision_model" ]]; then
    vision_args=(--vision "$vision_model")
fi

echo "Starting ds4-server with MTP on http://0.0.0.0:${SERVER_PORT} ..."
echo "CUDA decode baseline: Qwen3.8 target + MTP, Q8_1-R8, split-K32 and GQA2"
exec ./ds4-server \
    --cuda \
    -m gguf/Qwen3.8-27B-UD-Q4_K_S.gguf \
    "${vision_args[@]}" \
    --mtp \
    --ctx 28000 \
    --host 0.0.0.0 \
    --port "$SERVER_PORT" \
    "$@"
