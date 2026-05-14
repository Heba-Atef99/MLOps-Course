"""Generate CTR traffic scenarios with immediate feedback.

Usage:
  uv run python scripts/generate_traffic.py stable
  uv run python scripts/generate_traffic.py data-drift
  uv run python scripts/generate_traffic.py concept-drift
  uv run python scripts/generate_traffic.py all
"""

import argparse
import csv
import os
import random
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import requests

API_URL = "http://localhost:8000"
FeatureRanges = Mapping[str, tuple[int | float, int | float]]
DEFAULT_COUNT = 50
BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_BASELINE_PATH = Path(
    os.getenv("TRAINING_BASELINE_PATH", BASE_DIR / "data" / "training_baseline.csv")
)

STABLE_RANGES = {
    "hour_of_day": (8, 22),
    "device_type": (0, 2),
    "ad_position": (1, 5),
    "user_age": (18, 65),
    "session_duration_sec": (30, 1200),
    "page_views": (1, 30),
}

STABLE_FALLBACK_RANGES = {
    "hour_of_day": (0, 23),
    "device_type": (0, 2),
    "ad_position": (1, 5),
    "user_age": (18, 65),
    "session_duration_sec": (0, 1800),
    "page_views": (1, 50),
}

DATA_DRIFT_RANGES = {
    "hour_of_day": (0, 5),
    "device_type": (0, 0),
    "ad_position": (4, 5),
    "user_age": (55, 65),
    "session_duration_sec": (1200, 1800),
    "page_views": (1, 3),
}


def stable_click_probability(sample: dict[str, Any]) -> float:
    prob = 0.15
    prob -= (sample["ad_position"] - 1) * 0.025
    prob += 0.06 if sample["device_type"] == 1 else 0.0
    prob += 0.02 if sample["device_type"] == 2 else 0.0
    if 10 <= sample["hour_of_day"] <= 14 or 19 <= sample["hour_of_day"] <= 22:
        prob += 0.08
    prob += (sample["page_views"] / 50) * 0.06
    if 25 <= sample["user_age"] <= 34:
        prob += 0.04
    if sample["session_duration_sec"] > 1200:
        prob -= 0.03
    return max(0.02, min(prob, 0.90))


def concept_drift_click_probability(sample: dict[str, Any]) -> float:
    prob = 0.08
    prob += (sample["ad_position"] - 1) * 0.08
    prob += 0.25 if sample["device_type"] == 0 else 0.0
    if 0 <= sample["hour_of_day"] <= 8:
        prob += 0.25
    if sample["page_views"] <= 5:
        prob += 0.20
    if sample["session_duration_sec"] > 900:
        prob += 0.20
    if 45 <= sample["user_age"] <= 65:
        prob += 0.12
    return max(0.02, min(prob, 0.95))


def random_sample(ranges: FeatureRanges) -> dict[str, Any]:
    sample = {}
    for name, (lo, hi) in ranges.items():
        if name == "session_duration_sec":
            sample[name] = round(random.uniform(lo, hi), 1)
        else:
            sample[name] = random.randint(int(lo), int(hi))
    return sample


def load_training_baseline_rows() -> list[dict[str, Any]]:
    if not TRAINING_BASELINE_PATH.exists():
        return []

    with TRAINING_BASELINE_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            {
                "hour_of_day": int(row["hour_of_day"]),
                "device_type": int(row["device_type"]),
                "ad_position": int(row["ad_position"]),
                "user_age": int(row["user_age"]),
                "session_duration_sec": float(row["session_duration_sec"]),
                "page_views": int(row["page_views"]),
            }
            for row in reader
        ]


def sample_stable_row() -> dict[str, Any]:
    baseline_rows = load_training_baseline_rows()
    if baseline_rows:
        return random.choice(baseline_rows)
    return random_sample(STABLE_FALLBACK_RANGES)


def simulate_click(
    sample: dict[str, Any],
    probability_fn: Callable[[dict[str, Any]], float],
) -> bool:
    return random.random() < probability_fn(sample)


def send_predict(sample: dict[str, Any]) -> dict[str, Any] | None:
    resp = requests.post(f"{API_URL}/predict", json=sample, timeout=10)
    if resp.status_code not in (200, 201):
        print(f"    predict failed: {resp.status_code} {resp.text}")
        return None
    return resp.json()


def send_feedback(prediction_id: str, actual_click: bool) -> bool:
    resp = requests.post(
        f"{API_URL}/feedback",
        json={"prediction_id": prediction_id, "clicked": actual_click},
        timeout=10,
    )
    if resp.status_code not in (200, 201):
        print(f"    feedback failed: {resp.status_code} {resp.text}")
        return False
    return True


def generate(
    label: str,
    ranges: FeatureRanges | None,
    probability_fn: Callable[[dict[str, Any]], float],
    count: int,
    delay: float,
    sample_fn: Callable[[], dict[str, Any]] | None = None,
) -> None:
    print(f"\nSending {count} {label} requests (delay={delay}s)...")
    ok = 0
    correct = 0
    clicks = 0

    for i in range(count):
        sample = sample_fn() if sample_fn else random_sample(ranges or {})
        actual_click = simulate_click(sample, probability_fn)
        prediction = send_predict(sample)
        if prediction is None:
            time.sleep(delay)
            continue

        feedback_ok = send_feedback(prediction["prediction_id"], actual_click)
        if feedback_ok:
            ok += 1
            clicks += int(actual_click)
            correct += int(prediction["predicted_click"] == actual_click)

        if (i + 1) % 25 == 0:
            accuracy = correct / max(ok, 1)
            ctr = clicks / max(ok, 1)
            print(
                f"  {i + 1}/{count} sent ({ok} ok, acc={accuracy:.1%}, ctr={ctr:.1%})"
            )
        time.sleep(delay)

    accuracy = correct / max(ok, 1)
    ctr = clicks / max(ok, 1)
    print(f"Done: {ok}/{count} successful (acc={accuracy:.1%}, ctr={ctr:.1%})")


def generate_stable(count: int, delay: float) -> None:
    generate(
        label="stable",
        ranges=None,
        probability_fn=stable_click_probability,
        count=count,
        delay=delay,
        sample_fn=sample_stable_row,
    )


def generate_data_drift(count: int, delay: float) -> None:
    generate(
        label="data drift",
        ranges=DATA_DRIFT_RANGES,
        probability_fn=stable_click_probability,
        count=count,
        delay=delay,
    )


def generate_concept_drift(count: int, delay: float) -> None:
    generate(
        label="concept drift",
        ranges=STABLE_RANGES,
        probability_fn=concept_drift_click_probability,
        count=count,
        delay=delay,
        sample_fn=sample_stable_row,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CTR monitoring traffic.")
    parser.add_argument(
        "scenario",
        choices=["stable", "data-drift", "concept-drift", "all"],
        help="Traffic scenario to generate.",
    )
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("CTR Traffic Generator")
    print("=====================")

    if args.scenario == "stable":
        generate_stable(count=args.count or DEFAULT_COUNT, delay=args.delay)
    elif args.scenario == "data-drift":
        generate_data_drift(count=args.count or DEFAULT_COUNT, delay=args.delay)
    elif args.scenario == "concept-drift":
        generate_concept_drift(count=args.count or 100, delay=args.delay)
    else:
        generate_stable(count=args.count or DEFAULT_COUNT, delay=args.delay)
        generate_data_drift(count=args.count or DEFAULT_COUNT, delay=args.delay)
        generate_concept_drift(count=args.count or 100, delay=args.delay)

    print("\nTraffic generation complete.")
    print("Run 'uv run python scripts/compute_drift.py' to analyze drift.")


if __name__ == "__main__":
    main()
