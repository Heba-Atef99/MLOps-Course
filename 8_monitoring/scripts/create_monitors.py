"""Create Axiom monitors (alerts) for CTR model monitoring.

Usage: uv run python scripts/create_monitors.py
"""

import os
import sys

try:
    from compute_drift import NUMERIC_FEATURES
except ModuleNotFoundError:
    from scripts.compute_drift import NUMERIC_FEATURES

import requests
from dotenv import load_dotenv

load_dotenv()

AXIOM_TOKEN = os.getenv("AXIOM_TOKEN")
AXIOM_ORG_ID = os.getenv("AXIOM_ORG_ID")
AXIOM_DATASET = os.getenv("AXIOM_DATASET", "mlops")
ALERT_EMAIL = os.getenv("ALERT_EMAIL")
API_BASE = "https://api.axiom.co"
PH_SIGNALS = ["error_rate", "confidence", "click_through_rate"]

HEADERS = {
    "Authorization": f"Bearer {AXIOM_TOKEN}",
    "X-Axiom-Org-Id": AXIOM_ORG_ID,
    "Content-Type": "application/json",
}


def build_monitors() -> list[dict]:
    ds = AXIOM_DATASET
    monitors = [
        {
            "name": "CTR: High Error Rate",
            "description": "Error rate from feedback exceeds 50%",
            "type": "Threshold",
            "aplQuery": (
                f"['{ds}'] | where event_type == 'feedback' "
                f"| summarize err = 1.0 - avg(iff(correct, 1.0, 0.0))"
            ),
            "columnName": "err",
            "operator": "Above",
            "threshold": 0.5,
            "intervalMinutes": 1,
            "rangeMinutes": 2,
            "alertOnNoData": False,
            "notifyByGroup": False,
        },
        {
            "name": "CTR: Low Confidence",
            "description": "Median prediction confidence drops below 0.6",
            "type": "Threshold",
            "aplQuery": (
                f"['{ds}'] | where event_type == 'prediction' "
                f"| summarize med_conf = percentile(confidence, 50)"
            ),
            "columnName": "med_conf",
            "operator": "Below",
            "threshold": 0.6,
            "intervalMinutes": 1,
            "rangeMinutes": 2,
            "alertOnNoData": False,
            "notifyByGroup": False,
        },
    ]

    for feature in NUMERIC_FEATURES:
        monitors.append(
            {
                "name": f"CTR: PSI {feature}",
                "description": (
                    f"PSI for {feature} exceeds 0.2 against the training baseline"
                ),
                "type": "Threshold",
                "aplQuery": (
                    f"['{ds}'] | where event_type == 'drift_psi' "
                    f"and feature == '{feature}' | summarize psi = max(psi)"
                ),
                "columnName": "psi",
                "operator": "Above",
                "threshold": 0.2,
                "intervalMinutes": 1,
                "rangeMinutes": 2,
                "alertOnNoData": False,
                "notifyByGroup": False,
            }
        )

    for signal in PH_SIGNALS:
        monitors.append(
            {
                "name": f"CTR: PH {signal}",
                "description": (
                    f"Page-Hinkley detected drift in {signal}"
                ),
                "type": "MatchEvent",
                "aplQuery": (
                    f"['{ds}'] | where event_type == 'drift_page_hinkley' "
                    f"and feature == '{signal}' and drift_detected == true"
                ),
                "intervalMinutes": 1,
                "rangeMinutes": 2,
                "alertOnNoData": False,
                "notifyByGroup": False,
            }
        )

    return monitors


def get_existing_monitors() -> set[str]:
    resp = requests.get(f"{API_BASE}/v2/monitors", headers=HEADERS, timeout=15)
    if not resp.ok:
        return set()
    return {m["name"] for m in resp.json()}


def get_or_create_notifier() -> str | None:
    if not ALERT_EMAIL:
        return None

    notifier_name = "CTR Monitoring Alerts"

    resp = requests.get(f"{API_BASE}/v2/notifiers", headers=HEADERS, timeout=15)
    if resp.ok:
        for n in resp.json():
            if n.get("name") == notifier_name:
                print(f"  Notifier exists: {n['id']}")
                return n["id"]

    resp = requests.post(
        f"{API_BASE}/v2/notifiers",
        headers=HEADERS,
        json={
            "name": notifier_name,
            "properties": {"email": {"emails": [ALERT_EMAIL]}},
        },
        timeout=15,
    )
    if resp.ok:
        nid = resp.json().get("id", "unknown")
        print(f"  Notifier created: {nid} -> {ALERT_EMAIL}")
        return nid

    print(
        f"  Notifier failed: {resp.status_code}: {resp.text}",
        file=sys.stderr,
    )
    return None


def main() -> None:
    if not AXIOM_TOKEN or not AXIOM_ORG_ID:
        print(
            "Error: AXIOM_TOKEN and AXIOM_ORG_ID must be set",
            file=sys.stderr,
        )
        sys.exit(1)

    notifier_id = get_or_create_notifier()
    existing = get_existing_monitors()
    monitors = build_monitors()
    created = skipped = failed = 0

    for monitor in monitors:
        if monitor["name"] in existing:
            print(f"  Skipped: {monitor['name']} (exists)")
            skipped += 1
            continue

        if notifier_id:
            monitor["notifierIds"] = [notifier_id]

        resp = requests.post(
            f"{API_BASE}/v2/monitors",
            headers=HEADERS,
            json=monitor,
            timeout=15,
        )

        if resp.ok:
            mid = resp.json().get("id", "unknown")
            print(f"  Created: {monitor['name']} (id: {mid})")
            created += 1
        else:
            print(
                f"  Failed: {monitor['name']}: {resp.status_code}: {resp.text}",
                file=sys.stderr,
            )
            failed += 1

    print(f"\nDone: {created} created, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
