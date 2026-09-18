#!/usr/bin/env bash
# Compatibility wrapper. Stage order and checks live in src/pipeline.py.
set -Eeuo pipefail

if [[ $# -lt 1 || "$1" == "--help" || "$1" == "-h" ]]; then
    cat <<'USAGE'
Usage: bash run_all.sh <config-path> [--dry-run] [--stages <names...>]

Optional stages: --scale-audit, --backtest, --visualize
Environment: PYTHON_BIN, CUDA_VISIBLE_DEVICES, RUN_SCALE_AUDIT,
             RUN_BACKTEST, RUN_VISUALIZATION (each RUN_* is 0 or 1).
Set label.fixed_scale in JSON; LABEL_SCALE overrides are no longer accepted.
For all Python options: python scripts/run_pipeline.py --help
USAGE
    [[ $# -gt 0 ]] && exit 0 || exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 2
fi
if [[ -n "${LABEL_SCALE+x}" ]]; then
    echo "Set label.fixed_scale in the JSON configuration and unset LABEL_SCALE." >&2
    exit 2
fi
CONFIG="$1"
shift
EXTRA_ARGS=()
for name in RUN_SCALE_AUDIT RUN_BACKTEST RUN_VISUALIZATION; do
    value="${!name:-0}"
    if [[ "$value" != "0" && "$value" != "1" ]]; then
        echo "$name must be 0 or 1" >&2
        exit 2
    fi
    if [[ "$value" == "1" ]]; then
        case "$name" in
            RUN_SCALE_AUDIT) EXTRA_ARGS+=(--scale-audit) ;;
            RUN_BACKTEST) EXTRA_ARGS+=(--backtest) ;;
            RUN_VISUALIZATION) EXTRA_ARGS+=(--visualize) ;;
        esac
    fi
done
exec "$PYTHON_BIN" "$REPO_ROOT/scripts/run_pipeline.py" --config-path "$CONFIG" "${EXTRA_ARGS[@]}" "$@"
