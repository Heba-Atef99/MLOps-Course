"""Compute drift: PSI on features, Page-Hinkley on error rate + confidence.

Page-Hinkley state (cumsum, min_cumsum, mean, n) is persisted to Axiom as
drift_ph_state events. Each run loads the last state, processes only new
hourly buckets since the last run, updates the state, and ingests it back.

Usage: uv run python scripts/compute_drift.py
"""

import json
import os
import sys
from datetime import UTC, datetime

import numpy as np
import requests
from axiom_py import Client as AxiomClient
from axiom_py.client import ContentEncoding, ContentType
from dotenv import load_dotenv

load_dotenv()

AXIOM_TOKEN = os.getenv("AXIOM_TOKEN")
AXIOM_ORG_ID = os.getenv("AXIOM_ORG_ID")
AXIOM_DATASET = os.getenv("AXIOM_DATASET", "mlops")
API_BASE = "https://api.axiom.co"

NUMERIC_FEATURES = {
    "feature_hour_of_day": [0, 4, 8, 12, 16, 20, 24],
    "feature_ad_position": [1, 2, 3, 4, 5, 6],
    "feature_user_age": [18, 25, 35, 45, 55, 65],
    "feature_session_duration_sec": [0, 300, 600, 900, 1200, 1500, 1800],
    "feature_page_views": [1, 5, 10, 20, 30, 50],
}

PH_DELTA = 0.005
PH_THRESHOLD = 50

CURRENT_WINDOW_MINUTES = int(os.getenv("DRIFT_WINDOW_MINUTES", "360"))
REFERENCE_LOOKBACK_DAYS = int(os.getenv("DRIFT_REFERENCE_DAYS", "30"))

HEADERS = {
    "Authorization": f"Bearer {AXIOM_TOKEN}",
    "X-Axiom-Org-Id": AXIOM_ORG_ID,
    "Content-Type": "application/json",
}


def query_axiom(apl: str) -> list[dict]:
    resp = requests.post(
        f"{API_BASE}/v1/datasets/_apl?format=tabular",
        headers=HEADERS,
        json={"apl": apl},
        timeout=30,
    )
    if not resp.ok:
        print(f"Query failed: {resp.status_code} {resp.text}", file=sys.stderr)
        return []

    data = resp.json()
    tables = data.get("tables", [])
    if not tables:
        return []

    table = tables[0]
    fields = [f["name"] for f in table.get("fields", [])]
    columns = table.get("columns", [])

    if not columns or not fields:
        return []

    n_rows = len(columns[0])
    return [
        {fields[j]: columns[j][i] for j in range(len(fields))} for i in range(n_rows)
    ]


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: list) -> float:
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)

    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)

    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-4, cur_pct)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 6)


# --- Page-Hinkley with persistent state ---


def load_ph_state(feature: str) -> dict | None:
    apl = (
        f"['{AXIOM_DATASET}'] | where event_type == 'drift_ph_state' "
        f"and feature == '{feature}' "
        f"| order by _time desc | limit 1"
    )
    rows = query_axiom(apl)
    if not rows:
        return None
    r = rows[0]
    return {
        "cumsum": float(r.get("cumsum", 0)),
        "min_cumsum": float(r.get("min_cumsum", 0)),
        "running_mean": float(r.get("running_mean", 0)),
        "n": int(r.get("n", 0)),
        "last_timestamp": r.get("last_timestamp", ""),
    }


def update_ph_incremental(
    state: dict | None,
    new_values: np.ndarray,
    delta: float = PH_DELTA,
    threshold: float = PH_THRESHOLD,
) -> dict:
    if state is None:
        state = {
            "cumsum": 0.0,
            "min_cumsum": 0.0,
            "running_mean": 0.0,
            "n": 0,
        }

    cumsum = state["cumsum"]
    min_cumsum = state["min_cumsum"]
    running_mean = state["running_mean"]
    n = state["n"]

    for x in new_values:
        n += 1
        running_mean += (float(x) - running_mean) / n
        cumsum += float(x) - running_mean - delta
        if cumsum < min_cumsum:
            min_cumsum = cumsum

    ph_value = round(cumsum - min_cumsum, 6)

    return {
        "cumsum": round(cumsum, 6),
        "min_cumsum": round(min_cumsum, 6),
        "running_mean": round(running_mean, 6),
        "n": n,
        "ph_value": ph_value,
        "drift_detected": ph_value > threshold,
    }


def fetch_prediction_data(time_filter: str) -> list[dict]:
    fields = ", ".join(list(NUMERIC_FEATURES.keys()) + ["confidence"])
    apl = (
        f"['{AXIOM_DATASET}'] | where event_type == 'prediction' "
        f"| {time_filter} "
        f"| project {fields}"
    )
    return query_axiom(apl)


def fetch_hourly_since(name: str, last_timestamp: str) -> list[dict]:
    ds = AXIOM_DATASET
    time_filter = (
        f"| where _time > todatetime('{last_timestamp}')"
        if last_timestamp
        else f"| where _time > ago({REFERENCE_LOOKBACK_DAYS}d)"
    )

    agg_map = {
        "error_rate": (
            f"['{ds}'] | where event_type == 'feedback' {time_filter} "
            f"| summarize val = 1.0 - avg(iff(correct, 1.0, 0.0)) "
            f"by bin(_time, 1h) | order by _time asc"
        ),
        "confidence": (
            f"['{ds}'] | where event_type == 'prediction' {time_filter} "
            f"| summarize val = avg(confidence) "
            f"by bin(_time, 1h) | order by _time asc"
        ),
        "click_through_rate": (
            f"['{ds}'] | where event_type == 'feedback' {time_filter} "
            f"| summarize val = avg(iff(actual_click, 1.0, 0.0)) "
            f"by bin(_time, 1h) | order by _time asc"
        ),
    }

    apl = agg_map.get(name, "")
    if not apl:
        return []
    return query_axiom(apl)


def ingest_events(events: list[dict]) -> None:
    try:
        client = AxiomClient()
        payload = json.dumps(events).encode("utf-8")
        client.ingest(
            AXIOM_DATASET,
            payload,
            ContentType.JSON,
            ContentEncoding.IDENTITY,
        )
        print(f"\n  Ingested {len(events)} events to '{AXIOM_DATASET}'")
    except Exception as e:
        print(f"\n  Failed to ingest: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    if not AXIOM_TOKEN or not AXIOM_ORG_ID:
        print(
            "Error: AXIOM_TOKEN and AXIOM_ORG_ID must be set",
            file=sys.stderr,
        )
        sys.exit(1)

    win = CURRENT_WINDOW_MINUTES
    ref = REFERENCE_LOOKBACK_DAYS
    print(f"Computing drift (reference: >{win}m ago, current: last {win}m)")

    ref_filter = f"where _time > ago({ref}d) and _time < ago({win}m)"
    cur_filter = f"where _time > ago({win}m)"

    ref_data = fetch_prediction_data(ref_filter)
    cur_data = fetch_prediction_data(cur_filter)

    now = datetime.now(UTC).isoformat()
    events: list[dict] = []

    if ref_data and cur_data:
        print(f"  Reference: {len(ref_data)} events, Current: {len(cur_data)} events")
        print("\n  PSI (features):")
        for feature, bins in NUMERIC_FEATURES.items():
            ref_vals = np.array(
                [r[feature] for r in ref_data if r.get(feature) is not None],
                dtype=float,
            )
            cur_vals = np.array(
                [r[feature] for r in cur_data if r.get(feature) is not None],
                dtype=float,
            )
            psi = compute_psi(ref_vals, cur_vals, bins)
            level = (
                "significant" if psi > 0.2 else "moderate" if psi > 0.1 else "stable"
            )
            print(f"    {feature}: {psi:.4f} ({level})")
            events.append(
                {
                    "_time": now,
                    "event_type": "drift_psi",
                    "feature": feature,
                    "psi": psi,
                    "level": level,
                    "ref_count": len(ref_vals),
                    "cur_count": len(cur_vals),
                }
            )
    else:
        print("  Insufficient data for PSI (need both reference and current)")

    print("\n  Page-Hinkley (incremental):")
    ph_signals = ["error_rate", "confidence", "click_through_rate"]
    for name in ph_signals:
        prev_state = load_ph_state(name)
        last_ts = prev_state["last_timestamp"] if prev_state else ""

        if prev_state:
            print(
                f"    {name}: loaded state (n={prev_state['n']}, "
                f"cumsum={prev_state['cumsum']:.4f})"
            )
        else:
            print(f"    {name}: no previous state, starting fresh")

        hourly = fetch_hourly_since(name, last_ts)

        if not hourly:
            print(f"    {name}: no new data since last run")
            if prev_state:
                ph_value = round(prev_state["cumsum"] - prev_state["min_cumsum"], 6)
                status = "DRIFT" if ph_value > PH_THRESHOLD else "stable"
                print(f"    {name}: PH={ph_value:.4f} ({status}) [unchanged]")
                events.append(
                    {
                        "_time": now,
                        "event_type": "drift_page_hinkley",
                        "feature": name,
                        "ph_value": ph_value,
                        "drift_detected": ph_value > PH_THRESHOLD,
                        "data_points": prev_state["n"],
                    }
                )
            continue

        new_values = np.array(
            [h["val"] for h in hourly if h.get("val") is not None],
            dtype=float,
        )
        new_timestamps = [
            h.get("_time", "") for h in hourly if h.get("val") is not None
        ]
        latest_ts = new_timestamps[-1] if new_timestamps else last_ts

        updated = update_ph_incremental(prev_state, new_values)
        status = "DRIFT" if updated["drift_detected"] else "stable"
        print(
            f"    {name}: +{len(new_values)} buckets, "
            f"PH={updated['ph_value']:.4f} ({status}), "
            f"n={updated['n']}"
        )

        events.append(
            {
                "_time": now,
                "event_type": "drift_page_hinkley",
                "feature": name,
                "ph_value": updated["ph_value"],
                "drift_detected": updated["drift_detected"],
                "data_points": updated["n"],
            }
        )

        events.append(
            {
                "_time": now,
                "event_type": "drift_ph_state",
                "feature": name,
                "cumsum": updated["cumsum"],
                "min_cumsum": updated["min_cumsum"],
                "running_mean": updated["running_mean"],
                "n": updated["n"],
                "last_timestamp": latest_ts,
            }
        )

    if events:
        psi_values = [e["psi"] for e in events if e["event_type"] == "drift_psi"]
        max_psi = max(psi_values) if psi_values else 0.0
        max_feature = next(
            (
                e["feature"]
                for e in events
                if e["event_type"] == "drift_psi" and e["psi"] == max_psi
            ),
            "none",
        )
        events.append(
            {
                "_time": now,
                "event_type": "drift_summary",
                "max_psi": max_psi,
                "max_psi_feature": max_feature,
                "any_ph_drift": any(
                    e.get("drift_detected", False)
                    for e in events
                    if e["event_type"] == "drift_page_hinkley"
                ),
            }
        )
        ingest_events(events)
    else:
        print("\n  No drift events to ingest.")


if __name__ == "__main__":
    main()
