# LLM Serving Demo: Qwen2.5-7B + vLLM

Quantization comparison (FP16 vs AWQ INT4) and LLM serving with vLLM's OpenAI-compatible API.

## Prerequisites

Install vLLM (separate from the project venv, as it bundles its own PyTorch + CUDA):

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

vLLM may require a newer CUDA runtime than what ships with PyTorch. If you see
`libcudart.so.13: cannot open shared object file`, install it:

```bash
uv pip install "nvidia-cuda-runtime>=13"
```

Install the project deps (openai client, rich for pretty output) on top:

```bash
uv pip install openai rich ruff pyright
```

**Note:** do not run `uv sync` after installing vLLM. It would remove vLLM since
it's not in `pyproject.toml` (vLLM requires the special `--torch-backend=auto` flag
that `uv sync` doesn't support).

## Setup

Copy `.env.example` to `.env` and fill in your HyperDX ingest key:

```bash
cp .env.example .env
```

## Quick Start

```bash
# 1. Serve the FP16 model (~14GB VRAM, traces sent to HyperDX)
bash scripts/serve_fp16.sh

# 2. In another terminal, test it
uv run python client/chat.py

# 3. Run benchmark
uv run python scripts/benchmark.py

# 4. Stop server (Ctrl+C), serve the AWQ variant (~4GB VRAM)
bash scripts/serve_awq.sh

# 5. Run benchmark again
uv run python scripts/benchmark.py

# 6. Compare results
uv run python scripts/compare.py
```

**Startup time:** first run takes ~10 min (model download + CUDA graph compilation).
Subsequent runs take ~2-3 min (cached).

**Available endpoints (while server is running):**

| URL | Description |
| --- | --- |
| http://localhost:8000/docs | Swagger UI (interactive API docs) |
| http://localhost:8000/health | Health check |
| http://localhost:8000/metrics | Prometheus metrics (tokens/sec, latency, queue depth) |
| http://localhost:8000/version | vLLM version |
| http://localhost:8000/v1/models | List loaded models |
| http://localhost:8000/v1/chat/completions | Chat endpoint (POST) |
| http://localhost:8000/v1/completions | Text completion endpoint (POST) |

## Project Structure

```
llm_serving/
├── scripts/
│   ├── serve_fp16.sh      # Serve Qwen2.5-7B in BF16 (full precision)
│   ├── serve_awq.sh       # Serve Qwen2.5-7B-AWQ in INT4 (quantized)
│   ├── benchmark.py       # Measure VRAM, tokens/sec, response time
│   └── compare.py         # Side-by-side FP16 vs AWQ comparison
├── client/
│   └── chat.py            # Simple OpenAI-compatible chat client
├── results/               # Benchmark outputs (auto-created)
├── .env.example           # HyperDX + OTLP config template
├── pyproject.toml
└── uv.lock
```

## What This Demonstrates

### Model Optimization: Quantization (FP16 vs AWQ INT4)

| Metric      | FP16 (BFloat16)           | AWQ (INT4)            |
| ----------- | ------------------------- | --------------------- |
| VRAM        | ~14GB                     | ~4GB                  |
| Precision   | 16-bit floating point     | 4-bit integer weights |
| Quality     | Full precision            | Near-identical output |
| Use case    | When VRAM is available    | When VRAM is limited  |

You don't quantize at serve time. You download a **pre-quantized checkpoint** from HuggingFace
(`Qwen/Qwen2.5-7B-Instruct-AWQ`). The quantization was done offline using AutoAWQ.

### Serialization Format: safetensors

Both the FP16 and AWQ models on HuggingFace use `.safetensors` files (not pickle).
This is the HuggingFace standard since 2023: same tensor data, no executable code path,
safe to load from untrusted sources.

### LLM Serving: vLLM

vLLM provides an OpenAI-compatible API out of the box:
- `POST /v1/chat/completions` (chat)
- `POST /v1/completions` (text completion)
- `GET /v1/models` (list loaded models)

No custom server code needed. Any OpenAI SDK client works directly.

Key vLLM features:
- **PagedAttention**: efficient KV-cache memory management
- **Continuous batching**: handles concurrent requests without queuing
- **OpenAI-compatible API**: drop-in replacement for OpenAI endpoints

### Observability: OpenTelemetry + HyperDX

vLLM has built-in OpenTelemetry tracing. The serve scripts auto-load `.env` and pass:
- `--otlp-traces-endpoint`: sends traces directly to HyperDX (no collector sidecar needed)
- `OTEL_EXPORTER_OTLP_HEADERS`: passes the HyperDX API key as an auth header
- `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`: uses HTTP/protobuf transport

Every inference request shows up as a trace in HyperDX with latency, token counts, and model info.

### Key Flags

- `--max-model-len 4096`: caps context window to fit in 16GB VRAM (default 32k is too large for FP16)
- `--dtype bfloat16`: explicitly sets FP16 precision
- `--quantization awq`: tells vLLM to use AWQ kernels for the quantized model. Tip: use `awq_marlin` instead for faster inference (vLLM auto-detects this but defaults to `awq` when explicitly set)
- `--otlp-traces-endpoint`: OTLP endpoint for sending traces to HyperDX

## Lessons Learned

- **Quantization is a packaging decision, not a serving decision**: pick the right checkpoint from HuggingFace, vLLM handles the rest
- **3.5x VRAM reduction with AWQ**: INT4 quantization dramatically reduces memory with near-identical output quality
- **safetensors is the standard**: every modern HF model ships with it, no pickle risk
- **vLLM replaces custom serving code**: no need for Litestar/FastAPI when serving LLMs, vLLM gives you a production-ready API
- **`--max-model-len` is critical**: without it, a 7B model's 32k default context won't fit in 16GB
- **vLLM has built-in OTLP tracing**: no custom instrumentation code, just pass `--otlp-traces-endpoint` and env vars
- **First start is slow**: vLLM compiles CUDA graphs on first run (5-10 min). Subsequent starts use the cached graphs and are much faster
- **vLLM pre-allocates KV cache**: even an AWQ model (~4GB weights) will claim most available VRAM for caching. This is intentional for throughput
- **CUDA runtime mismatch is common**: vLLM may need a newer `libcudart` than what PyTorch bundles. Fix with `uv pip install "nvidia-cuda-runtime>=13"`
- **Model weights live in `~/.cache/huggingface/hub/`**: shared across all HF tools, not inside the project directory
