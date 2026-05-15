"""Delete Axiom resources (dataset, dashboard, and monitors) for CTR monitoring.

Usage: uv run python scripts/drop_resources.py
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

AXIOM_TOKEN = os.getenv("AXIOM_TOKEN")
AXIOM_ORG_ID = os.getenv("AXIOM_ORG_ID")
AXIOM_DATASET = os.getenv("AXIOM_DATASET", "mlops")
API_BASE = "https://api.axiom.co"

HEADERS = {
    "Authorization": f"Bearer {AXIOM_TOKEN}",
    "X-Axiom-Org-Id": AXIOM_ORG_ID,
    "Content-Type": "application/json",
}

# The fixed UID from create_dashboard.py
DASHBOARD_UID = "409eed9e-18e5-443e-a685-760acf18ecfc"


def delete_dashboard() -> None:
    print(f"Deleting dashboard: {DASHBOARD_UID}...")
    resp = requests.delete(
        f"{API_BASE}/v2/dashboards/{DASHBOARD_UID}",
        headers=HEADERS,
        timeout=15,
    )
    if resp.status_code == 204 or resp.ok:
        print("  Successfully deleted dashboard.")
    elif resp.status_code == 404:
        print("  Dashboard not found (already deleted).")
    else:
        print(f"  Error deleting dashboard: {resp.status_code} {resp.text}")


def delete_dataset() -> None:
    print(f"Deleting dataset: {AXIOM_DATASET}...")
    # Dataset deletion is usually /v1/datasets/{name}
    resp = requests.delete(
        f"{API_BASE}/v1/datasets/{AXIOM_DATASET}",
        headers=HEADERS,
        timeout=15,
    )
    if resp.status_code == 204 or resp.ok:
        print(f"  Successfully deleted dataset '{AXIOM_DATASET}'.")
    elif resp.status_code == 404:
        print(f"  Dataset '{AXIOM_DATASET}' not found.")
    else:
        print(f"  Error deleting dataset: {resp.status_code} {resp.text}")


def delete_monitors() -> None:
    print("Listing monitors to delete CTR-related ones...")
    resp = requests.get(f"{API_BASE}/v2/monitors", headers=HEADERS, timeout=15)
    if not resp.ok:
        print(f"  Error listing monitors: {resp.status_code} {resp.text}")
        return

    monitors = resp.json()
    deleted_count = 0
    for m in monitors:
        if m.get("name", "").startswith("CTR: "):
            mid = m["id"]
            mname = m["name"]
            print(f"  Deleting monitor: {mname} ({mid})...")
            m_resp = requests.delete(
                f"{API_BASE}/v2/monitors/{mid}",
                headers=HEADERS,
                timeout=15,
            )
            if m_resp.ok:
                deleted_count += 1
            else:
                print(f"    Failed to delete {mname}: {m_resp.status_code}")

    print(f"  Deleted {deleted_count} monitors.")


def main() -> None:
    if not AXIOM_TOKEN or not AXIOM_ORG_ID:
        print(
            "Error: AXIOM_TOKEN and AXIOM_ORG_ID must be set",
            file=sys.stderr,
        )
        sys.exit(1)

    # Ask for confirmation if not forced
    if "--force" not in sys.argv:
        confirm = input(
            f"Delete dataset '{AXIOM_DATASET}', CTR dashboard & monitors? (y/N): "
        )
        if confirm.lower() != "y":
            print("Aborted.")
            return

    delete_dashboard()
    delete_monitors()
    delete_dataset()
    print("\nCleanup complete.")


if __name__ == "__main__":
    main()
