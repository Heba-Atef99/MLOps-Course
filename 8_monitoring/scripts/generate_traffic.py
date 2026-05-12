"""Generate CTR traffic: predict + immediate feedback.

Sends normal traffic, then drifted traffic (late-night, different device mix).
Each prediction is followed by a feedback call with the simulated actual click.

Usage: uv run python scripts/generate_traffic.py
"""

import random
import time

import requests

API_URL = "http://localhost:8000"

NORMAL_RANGES = {
    "hour_of_day": (8, 22),
    "device_type": (0, 2),
    "ad_position": (1, 5),
    "user_age": (18, 65),
    "session_duration_sec": (30, 1200),
    "page_views": (1, 30),
}

DRIFTED_RANGES = {
    "hour_of_day": (0, 5),
    "device_type": (0, 0),
    "ad_position": (4, 5),
    "user_age": (55, 65),
    "session_duration_sec": (1200, 1800),
    "page_views": (1, 3),
}


def random_sample(ranges: dict[str, tuple[int | float, int | float]]) -> dict:
    sample = {}
    for k, (lo, hi) in ranges.items():
        if k in ("session_duration_sec",):
            sample[k] = round(random.uniform(lo, hi), 1)
        else:
            sample[k] = random.randint(int(lo), int(hi))
    return sample


def simulate_click(sample: dict, drifted: bool) -> bool:
    if drifted:
        return random.random() < 0.03
    prob = 0.15
    if sample["ad_position"] <= 2:
        prob += 0.05
    if 10 <= sample["hour_of_day"] <= 14:
        prob += 0.08
    if sample["device_type"] == 1:
        prob += 0.05
    return random.random() < prob


def send_predict_and_feedback(sample: dict, actual_click: bool) -> tuple[bool, bool]:
    resp = requests.post(f"{API_URL}/predict", json=sample, timeout=10)
    if resp.status_code not in (200, 201):
        return False, False

    prediction_id = resp.json()["prediction_id"]

    fb_resp = requests.post(
        f"{API_URL}/feedback",
        json={"prediction_id": prediction_id, "clicked": actual_click},
        timeout=10,
    )
    return True, fb_resp.status_code in (200, 201)


def generate(
    label: str,
    ranges: dict,
    count: int,
    drifted: bool,
    delay: float,
) -> None:
    print(f"\n  Sending {count} {label} requests (delay={delay}s)...")
    ok = 0
    for i in range(count):
        sample = random_sample(ranges)
        actual_click = simulate_click(sample, drifted=drifted)
        pred_ok, fb_ok = send_predict_and_feedback(sample, actual_click)
        if pred_ok and fb_ok:
            ok += 1
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{count} sent ({ok} ok)")
        time.sleep(delay)
    print(f"  Done: {ok}/{count} successful")


def main() -> None:
    print("CTR Traffic Generator")
    print("=====================")
    generate("normal", NORMAL_RANGES, count=100, drifted=False, delay=0.05)
    generate("drifted", DRIFTED_RANGES, count=50, drifted=True, delay=0.05)
    print("\nTraffic generation complete.")
    print("Run 'uv run python scripts/compute_drift.py' to analyze drift.")


if __name__ == "__main__":
    main()
