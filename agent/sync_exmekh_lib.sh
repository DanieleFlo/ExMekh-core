#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
UV_BIN="${UV_BIN:-uv}"
if ! command -v "$UV_BIN" >/dev/null 2>&1; then
    if [[ "$UV_BIN" == "uv" && -x "$HOME/.local/bin/uv" ]]; then
        UV_BIN="$HOME/.local/bin/uv"
    else
        echo "ERROR: uv executable not found: $UV_BIN" >&2
        exit 127
    fi
fi
cd "$SCRIPT_DIR"
export UV_BIN
exec "$UV_BIN" run --no-project --python "$(<.python-version)" "$SCRIPT_DIR/src/exmekh_agent/sync.py" "$@"
