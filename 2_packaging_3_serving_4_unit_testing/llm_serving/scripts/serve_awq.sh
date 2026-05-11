#!/usr/bin/env bash
# Serve Qwen2.5-7B-Instruct-AWQ in INT4 (quantized)
# Uses ~4GB VRAM. Same model, 3.5x less memory.
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

vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --max-model-len 4096 \
    --port 8000 \
    --otlp-traces-endpoint="https://in-otel.hyperdx.io/v1/traces"
