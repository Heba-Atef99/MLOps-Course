"""Compare benchmark results between FP16 and AWQ runs.

Usage: uv run python scripts/compare.py
"""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_results(variant: str) -> dict | None:
    path = RESULTS_DIR / f"benchmark_{variant}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compare() -> None:
    fp16 = load_results("fp16")
    awq = load_results("awq")

    if not fp16 or not awq:
        missing = []
        if not fp16:
            missing.append("fp16")
        if not awq:
            missing.append("awq")
        console.print(f"[red]Missing benchmark results for: {', '.join(missing)}[/red]")
        console.print("Run the benchmark for each variant first:")
        console.print("  1. bash scripts/serve_fp16.sh")
        console.print("     uv run python scripts/benchmark.py")
        console.print("  2. bash scripts/serve_awq.sh")
        console.print("     uv run python scripts/benchmark.py")
        return

    table = Table(title="FP16 vs AWQ Comparison")
    table.add_column("Metric")
    table.add_column("FP16", style="cyan")
    table.add_column("AWQ (INT4)", style="green")
    table.add_column("Difference", style="yellow")

    fp16_vram = fp16["results"][0]["vram_mb"]
    awq_vram = awq["results"][0]["vram_mb"]
    vram_reduction = ((fp16_vram - awq_vram) / fp16_vram) * 100

    fp16_tps = sum(r["tokens_per_sec"] for r in fp16["results"]) / len(fp16["results"])
    awq_tps = sum(r["tokens_per_sec"] for r in awq["results"]) / len(awq["results"])
    tps_change = ((awq_tps - fp16_tps) / fp16_tps) * 100

    fp16_time = sum(r["elapsed_sec"] for r in fp16["results"]) / len(fp16["results"])
    awq_time = sum(r["elapsed_sec"] for r in awq["results"]) / len(awq["results"])

    table.add_row("Model", fp16["model"], awq["model"], "")
    vram_diff = f"{vram_reduction:.1f}% less"
    table.add_row("VRAM", f"{fp16_vram:.0f} MB", f"{awq_vram:.0f} MB", vram_diff)
    tps_diff = f"{tps_change:+.1f}%"
    table.add_row("Avg Tokens/sec", f"{fp16_tps:.1f}", f"{awq_tps:.1f}", tps_diff)
    table.add_row("Avg Response Time", f"{fp16_time:.2f}s", f"{awq_time:.2f}s", "")

    console.print(table)

    console.print("\n[bold]Output Quality Comparison:[/bold]\n")
    for fp16_r, awq_r in zip(fp16["results"], awq["results"], strict=True):
        console.print(f"[dim]Prompt: {fp16_r['prompt']}[/dim]")
        console.print(f"  [cyan]FP16:[/cyan] {fp16_r['output'][:150]}")
        console.print(f"  [green]AWQ:[/green]  {awq_r['output'][:150]}")
        console.print()


if __name__ == "__main__":
    compare()
