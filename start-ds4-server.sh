#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# A bare launch always restores managed Core. Native launches can use --manual.
start_managed() {
    if [[ -f "$SCRIPT_DIR/config/native-suite.env" ]]; then
        set -a
        source "$SCRIPT_DIR/config/native-suite.env"
        set +a
    fi
    exec python3 "$SCRIPT_DIR/tools/managed_server.py" "$@"
}
if [[ "${1:-}" == --managed ]]; then
    shift
    start_managed "$@"
fi
if [[ "${1:-}" == --manual ]]; then
    shift
elif [[ $# -eq 0 ]]; then
    start_managed
fi
SERVER_PORT="${DS4_SERVER_PORT:-8080}"
profile="${DS4_MODEL_PROFILE:-qwen27}"
backend="${DS4_BACKEND:-auto}"
if [[ "$backend" == auto ]]; then
    if [[ "$(uname -s)" == Darwin ]]; then backend=metal; else backend=cuda; fi
fi
case "$backend" in
    metal|cuda|cpu) ;;
    *) echo "Unknown DS4_BACKEND: $backend (use auto, metal, cuda, or cpu)" >&2; exit 2 ;;
esac
# Small native defaults fit the qualification profile; explicit capacities win.
if [[ "$backend" == cuda ]]; then
    default_mtp=1
else
    default_mtp=0
fi
mtp="${DS4_MTP:-$default_mtp}"
case "$mtp" in
    1) mtp_args=(--mtp) ;;
    0) mtp_args=(--no-mtp) ;;
    *) echo "Invalid DS4_MTP: $mtp (use 0 or 1)" >&2; exit 2 ;;
esac
cd "$SCRIPT_DIR"

server=ds4-server
suite_args=()
if [[ -n "${DS4_SUITE_CONFIG:-}" ]]; then
    # Keep native dependencies explicit and local to the suite build.
    if [[ -f config/native-suite.env ]]; then
        set -a
        source config/native-suite.env
        set +a
    fi
    server=ds4-server-suite
    suite_args=(--suite-config "$DS4_SUITE_CONFIG")
fi

profile_env=()
prefill_args=()
gguf_dir="${DS4_GGUF_DIR:-gguf}"
case "$profile" in
    qwen27)
        model="${DS4_MODEL:-$gguf_dir/Qwen3.8-27B-UD-Q4_K_S.gguf}"
        if [[ "$backend" == cuda ]]; then
            vision_model="${DS4_VISION_MODEL-$gguf_dir/mmproj-Qwen3.8-27B-F16.gguf}"
            context="${DS4_CTX:-22593}"
        else
            vision_model="${DS4_VISION_MODEL-}"
            context="${DS4_CTX:-512}"
            prefill_args=(--prefill-chunk "${DS4_PREFILL_CHUNK:-64}")
        fi
        ;;
    flash)
        model="${DS4_MODEL:-$gguf_dir/Qwen3.8-Flash-Next-Q2.gguf}"
        vision_model="${DS4_VISION_MODEL-}"
        context="${DS4_CTX:-512}"
        prefill_args=(--prefill-chunk "${DS4_PREFILL_CHUNK:-64}")
        if [[ "$backend" == cuda ]]; then
            vision_model="${DS4_VISION_MODEL-$gguf_dir/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf}"
            context="${DS4_CTX:-65536}"
            flash_runtime="${DS4_FLASH_PROFILE:-standard}"
            case "$flash_runtime" in
                standard)
                    while IFS='=' read -r name default; do
                        value="${!name:-$default}"
                        if [[ "$name" == DS4_PREFILL_CHUNK ]]; then
                            prefill_args=(--prefill-chunk "$value")
                        else
                            profile_env+=("$name=$value")
                        fi
                    done < config/flash-cuda-standard.env
                    ;;
                legacy)
                    prefill_args=(--prefill-chunk "${DS4_PREFILL_CHUNK:-256}")
                    profile_env=(
                        "DS4_CUDA_QWEN4_TIERED_CACHE=${DS4_CUDA_QWEN4_TIERED_CACHE:-1}"
                        "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-auto}"
                        "DS4_CUDA_QWEN4_RAM_CACHE_MB=${DS4_CUDA_QWEN4_RAM_CACHE_MB:-auto}"
                    )
                    ;;
                *) echo "Unknown DS4_FLASH_PROFILE: $flash_runtime (use standard or legacy)" >&2; exit 2 ;;
            esac
        fi
        ;;
    *) echo "Unknown DS4_MODEL_PROFILE: $profile (use qwen27 or flash)" >&2; exit 2 ;;
esac
if [[ "$profile" == qwen27 && -n "${DS4_PREFILL_CHUNK:-}" ]]; then
    prefill_args=(--prefill-chunk "$DS4_PREFILL_CHUNK")
fi
vision_args=()
if [[ -n "$vision_model" && ( -f "$vision_model" || -n "${DS4_VISION_MODEL+x}" ) ]]; then
    vision_args=(--vision "$vision_model")
fi
kv_args=()
kv_dir="${DS4_KV_DISK_DIR-${HOME}/.ds4/server-kv}"
if [[ -n "$kv_dir" ]]; then
    kv_args=(--kv-disk-dir "$kv_dir" --kv-disk-space-mb "${DS4_KV_DISK_SPACE_MB:-8192}")
fi
stream_args=()
case "${DS4_SSD_STREAMING:-0}" in
    0) ;;
    1)
        stream_args=(--ssd-streaming --ssd-streaming-cold)
        # Metal admission reserves graph/auxiliary memory and 2 GiB before
        # sizing the cache. Preserve explicit ceilings and CUDA defaults.
        if [[ -n "${DS4_SSD_CACHE_GB:-}" || "$backend" != metal ]]; then
            stream_args+=(--ssd-streaming-cache-experts "${DS4_SSD_CACHE_GB:-2}GB")
        fi
        ;;
    *) echo "Invalid DS4_SSD_STREAMING (use 0 or 1)" >&2; exit 2 ;;
esac
server_args=("--$backend" -m "$model" ${vision_args[@]+"${vision_args[@]}"} --ctx "$context"
    ${prefill_args[@]+"${prefill_args[@]}"} --host 0.0.0.0 "${mtp_args[@]}" --port "$SERVER_PORT" ${kv_args[@]+"${kv_args[@]}"}
    ${suite_args[@]+"${suite_args[@]}"} ${stream_args[@]+"${stream_args[@]}"})

# Review the exact command without building, allocating a context or loading weights.
if [[ "${1:-}" == --print-command ]]; then
    shift
    printf '%q ' env ${profile_env[@]+"${profile_env[@]}"} "./$server" "${server_args[@]}" "$@"
    printf '\n'
    exit 0
fi
[[ -f "$model" ]] || { echo "Model not found: $model" >&2; exit 2; }
if [[ ${#vision_args[@]} -gt 0 && ! -f "$vision_model" ]]; then
    echo "Vision model not found: $vision_model" >&2
    exit 2
fi
echo "Checking $server build..."
build_args=()
if [[ "$backend" == cuda ]]; then build_args=(CUDA_ARCH="${CUDA_ARCH:-native}"); fi
make --no-print-directory -j2 "$server" ${build_args[@]+"${build_args[@]}"}
echo "Starting $profile on http://0.0.0.0:${SERVER_PORT} (ctx=$context) ..."
exec env ${profile_env[@]+"${profile_env[@]}"} "./$server" "${server_args[@]}" "$@"
