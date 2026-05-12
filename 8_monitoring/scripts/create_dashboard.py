"""Create an Axiom dashboard for CTR model monitoring.

Usage: uv run python scripts/create_dashboard.py
"""

import os
import sys
import uuid

import requests
from dotenv import load_dotenv

load_dotenv()

AXIOM_TOKEN = os.getenv("AXIOM_TOKEN")
AXIOM_ORG_ID = os.getenv("AXIOM_ORG_ID")
AXIOM_DATASET = os.getenv("AXIOM_DATASET", "mlops")
API_BASE = "https://api.axiom.co"


def chart(name: str, apl: str, chart_type: str = "TimeSeries") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "type": chart_type,
        "query": {"apl": apl},
    }


def layout_item(chart_id: str, x: int, y: int, w: int, h: int) -> dict:
    return {"i": chart_id, "x": x, "y": y, "w": w, "h": h}


def build_dashboard() -> dict:
    ds = AXIOM_DATASET

    charts = [
        chart(
            "Error Rate Over Time (from feedback)",
            f"['{ds}'] | where event_type == 'feedback' "
            f"| summarize error_rate = round("
            f"1.0 - avg(iff(correct, 1.0, 0.0)), 4) "
            f"by bin_auto(_time)",
        ),
        chart(
            "Confidence Over Time",
            f"['{ds}'] | where event_type == 'prediction' "
            f"| summarize avg_conf = round(avg(confidence), 4), "
            f"p50 = round(percentile(confidence, 50), 4) "
            f"by bin_auto(_time)",
        ),
        chart(
            "Click-Through Rate (actual, from feedback)",
            f"['{ds}'] | where event_type == 'feedback' "
            f"| summarize ctr = round("
            f"avg(iff(actual_click, 1.0, 0.0)), 4) "
            f"by bin_auto(_time)",
        ),
        chart(
            "Prediction Latency (P95)",
            f"['{ds}'] | where event_type == 'prediction' "
            f"| summarize p95 = percentile(latency_seconds, 95) "
            f"by bin_auto(_time)",
        ),
        chart(
            "Request Volume by Status",
            f"['{ds}'] | where event_type == 'http_request' "
            f"| summarize count() by bin_auto(_time), status_code",
        ),
        chart(
            "Feature Means Over Time",
            f"['{ds}'] | where event_type == 'prediction' "
            f"| summarize "
            f"avg_hour = round(avg(feature_hour_of_day), 1), "
            f"avg_position = round(avg(feature_ad_position), 1), "
            f"avg_age = round(avg(feature_user_age), 1) "
            f"by bin_auto(_time)",
        ),
    ]

    layout = [
        layout_item(charts[0]["id"], x=0, y=0, w=4, h=4),
        layout_item(charts[1]["id"], x=4, y=0, w=4, h=4),
        layout_item(charts[2]["id"], x=8, y=0, w=4, h=4),
        layout_item(charts[3]["id"], x=0, y=4, w=4, h=4),
        layout_item(charts[4]["id"], x=4, y=4, w=4, h=4),
        layout_item(charts[5]["id"], x=8, y=4, w=4, h=4),
    ]

    return {
        "name": "CTR Model Monitoring",
        "description": "Error rate, confidence, CTR, latency, "
        "and feature distributions",
        "owner": "",
        "charts": charts,
        "layout": layout,
        "refreshTime": 60,
        "schemaVersion": 2,
        "timeWindowStart": "qr-now-24h",
        "timeWindowEnd": "qr-now",
    }


def main() -> None:
    if not AXIOM_TOKEN or not AXIOM_ORG_ID:
        print(
            "Error: AXIOM_TOKEN and AXIOM_ORG_ID must be set",
            file=sys.stderr,
        )
        sys.exit(1)

    dashboard = build_dashboard()
    dashboard_uid = "409eed9e-18e5-443e-a685-760acf18ecfc"

    resp = requests.post(
        f"{API_BASE}/v2/dashboards",
        headers={
            "Authorization": f"Bearer {AXIOM_TOKEN}",
            "X-Axiom-Org-Id": AXIOM_ORG_ID,
            "Content-Type": "application/json",
        },
        json={
            "dashboard": dashboard,
            "uid": dashboard_uid,
            "overwrite": True,
        },
        timeout=15,
    )

    if resp.ok:
        print("Dashboard created/updated: CTR Model Monitoring")
        print(f"UID: {dashboard_uid}")
    else:
        print(f"Error {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
