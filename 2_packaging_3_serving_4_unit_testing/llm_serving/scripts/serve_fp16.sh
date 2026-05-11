#!/usr/bin/env bash
# Serve Qwen2.5-7B-Instruct in FP16 (full precision)
# Uses ~14GB VRAM on RTX 3080 16GB. Context capped at 4096 to fit in memory.
# Traces sent to HyperDX via OpenTelemetry.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

source "$PROJECT_DIR/.venv/bin/activate"

NVIDIA_LIBS="$PROJECT_DIR/.venv/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIBS/cu13/lib:$NVIDIA_LIBS/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

vllm serve Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --port 8000 \
    --otlp-traces-endpoint="https://in-otel.hyperdx.io/v1/traces"
