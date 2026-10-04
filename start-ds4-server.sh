#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SERVER_PORT="${DS4_SERVER_PORT:-8080}"
profile="${DS4_MODEL_PROFILE:-qwen27}"
cd "$SCRIPT_DIR"

profile_env=()
prefill_args=()
case "$profile" in
    qwen27)
        model="${DS4_MODEL:-gguf/Qwen3.8-27B-UD-Q4_K_S.gguf}"
        vision_model="${DS4_VISION_MODEL-gguf/mmproj-Qwen3.8-27B-F16.gguf}"
        context="${DS4_CTX:-22593}"
        ;;
    flash)
        model="${DS4_MODEL:-gguf/Qwen3.8-Flash-Next-Q2.gguf}"
        vision_model="${DS4_VISION_MODEL-gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf}"
        context="${DS4_CTX:-131072}"
        prefill_args=(--prefill-chunk "${DS4_PREFILL_CHUNK:-256}")
        # Explicit Flash selection opts into the single-device CUDA tiered path.
        # The engine validates family/backend/topology before activating it.
        profile_env=(
            "DS4_CUDA_QWEN4_TIERED_CACHE=${DS4_CUDA_QWEN4_TIERED_CACHE:-1}"
            "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-auto}"
            "DS4_CUDA_QWEN4_RAM_CACHE_MB=${DS4_CUDA_QWEN4_RAM_CACHE_MB:-auto}"
        )
        ;;
    *) echo "Unknown DS4_MODEL_PROFILE: $profile (use qwen27 or flash)" >&2; exit 2 ;;
esac
if [[ "$profile" == qwen27 && -n "${DS4_PREFILL_CHUNK:-}" ]]; then
    prefill_args=(--prefill-chunk "$DS4_PREFILL_CHUNK")
fi
vision_args=()
if [[ -n "$vision_model" && -f "$vision_model" ]]; then
    vision_args=(--vision "$vision_model")
fi
kv_args=()
kv_dir="${DS4_KV_DISK_DIR-${HOME}/.ds4/server-kv}"
if [[ -n "$kv_dir" ]]; then
    kv_args=(--kv-disk-dir "$kv_dir" --kv-disk-space-mb "${DS4_KV_DISK_SPACE_MB:-8192}")
fi
server_args=(--cuda -m "$model" "${vision_args[@]}" --ctx "$context"
    "${prefill_args[@]}" --host 0.0.0.0 --mtp --port "$SERVER_PORT" "${kv_args[@]}")

# Review the exact command without building, allocating a context or loading weights.
if [[ "${1:-}" == --print-command ]]; then
    shift
    printf '%q ' env "${profile_env[@]}" ./ds4-server "${server_args[@]}" "$@"
    printf '\n'
    exit 0
fi
[[ -f "$model" ]] || { echo "Model not found: $model" >&2; exit 2; }
echo "Checking ds4-server build..."
make --no-print-directory -j2 ds4-server CUDA_ARCH="${CUDA_ARCH:-native}"
echo "Starting $profile on http://0.0.0.0:${SERVER_PORT} (ctx=$context) ..."
exec env "${profile_env[@]}" ./ds4-server "${server_args[@]}" "$@"
