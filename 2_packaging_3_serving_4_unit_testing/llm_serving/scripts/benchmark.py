"""Benchmark FP16 vs AWQ: VRAM usage, tokens/sec, and output quality.

Usage:
    1. Start vLLM with one of the serve scripts (serve_fp16.sh or serve_awq.sh)
    2. Run this script: uv run python scripts/benchmark.py
    3. Stop the server, start the other variant, run again
    4. Compare the saved results
"""

import json
import subprocess
import time
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.table import Table

console = Console()
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

PROMPTS = [
    "Explain gradient descent in 2 sentences.",
    "What is quantization in machine learning?",
    "Write a Python function that checks if a number is prime.",
]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def get_vram_usage_mb() -> float:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip().split("\n")[0])


def get_model_name() -> str:
    models = client.models.list()
    return models.data[0].id


def benchmark_prompt(model: str, prompt: str) -> dict:
    vram_before = get_vram_usage_mb()

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
    )
    elapsed = time.perf_counter() - start

    output = response.choices[0].message.content or ""
    total_tokens = response.usage.total_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

    return {
        "prompt": prompt,
        "output": output,
        "elapsed_sec": round(elapsed, 3),
        "total_tokens": total_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "vram_mb": vram_before,
    }


def run_benchmark() -> None:
    model = get_model_name()
    variant = "awq" if "AWQ" in model.upper() else "fp16"

    console.print(f"\n[bold]Benchmarking: {model}[/bold] ({variant})\n")

    results = []
    for prompt in PROMPTS:
        console.print(f"[dim]>>> {prompt}[/dim]")
        result = benchmark_prompt(model, prompt)
        results.append(result)
        tps = result["tokens_per_sec"]
        secs = result["elapsed_sec"]
        vram = result["vram_mb"]
        console.print(f"  {tps} tok/s | {secs}s | VRAM: {vram:.0f} MB")
        console.print(f"  [green]{result['output'][:120]}...[/green]\n")

    table = Table(title=f"Results: {variant.upper()}")
    table.add_column("Metric")
    table.add_column("Value")

    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_time = sum(r["elapsed_sec"] for r in results) / len(results)
    vram = results[0]["vram_mb"]

    table.add_row("Model", model)
    table.add_row("Variant", variant.upper())
    table.add_row("VRAM Usage", f"{vram:.0f} MB")
    table.add_row("Avg Tokens/sec", f"{avg_tps:.1f}")
    table.add_row("Avg Response Time", f"{avg_time:.2f}s")

    console.print(table)

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"benchmark_{variant}.json"
    with open(output_path, "w") as f:
        json.dump({"model": model, "variant": variant, "results": results}, f, indent=2)
    console.print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_benchmark()
