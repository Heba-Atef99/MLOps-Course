# CTR Monitoring Demo: Click-Through Prediction + Axiom

A monitoring demo built on top of a click-through rate prediction model. Predicts whether a user will click an ad, receives immediate feedback (clicked or not), and monitors model health via Axiom.

## Why Click-Through?

Unlike static datasets, CTR gives us **immediate feedback**: we predict "will the user click?", and seconds later we know if they did. This closes the feedback loop and lets us monitor actual model accuracy in real-time, not just proxy metrics.

## Quick Start

```bash
uv sync
uv run python training/train.py      # train CTR model + save training baseline
cp .env.example .env                 # fill in Axiom credentials
uv run uvicorn app.main:app --reload # start server
```

Swagger UI: http://localhost:8000/schema/swagger

## Endpoints

| Method | Path        | Description                                  |
| ------ | ----------- | -------------------------------------------- |
| GET    | `/`         | Welcome + endpoint list                      |
| GET    | `/health`   | Health check                                 |
| POST   | `/predict`  | Predict click-through, returns prediction_id |
| POST   | `/feedback` | Submit actual click result for a prediction  |

## The Feedback Loop

```
1. POST /predict   -->  prediction_id + predicted_click + confidence
2. (user clicks or doesn't)
3. POST /feedback   -->  prediction_id + actual clicked (true/false)
4. Server computes: was the prediction correct?
5. Both events sent to Axiom with full context
```

This is the "Immediate Feedback" pattern: the correct answer arrives seconds after the prediction.

## Project Structure

```
8_monitoring/
├── app/
│   ├── main.py              # Litestar app + Axiom middleware + /feedback endpoint
│   ├── model.py             # CTR model loading + prediction
│   ├── schemas.py           # Request/response dataclasses
│   └── axiom_client.py      # Axiom event ingestion helper
├── training/
│   └── train.py             # Train on synthetic CTR data (82% accuracy)
├── scripts/
│   ├── generate_traffic.py  # Send stable, data drift, or concept drift traffic
│   ├── compute_drift.py     # PSI vs training baseline, Page-Hinkley on signals
│   ├── create_dashboard.py  # Create/update Axiom dashboard (idempotent)
│   └── create_monitors.py   # Create Axiom alerts
├── tests/                   # 9 tests (predict, feedback, model, edge cases)
├── model/
├── data/
├── pyproject.toml
└── .env.example
```

## Drift Detection: PSI vs Page-Hinkley

We use two complementary techniques, each applied where it fits best:

### PSI (Population Stability Index) on input features

Compares the distribution of each production feature against the training baseline saved by `training/train.py` at `data/training_baseline.csv`. This catches cases where the model is receiving inputs that no longer look like the data it learned from.

Applied to: `hour_of_day`, `ad_position`, `user_age`, `session_duration_sec`, `page_views`

| PSI Value | Interpretation    |
| --------- | ----------------- |
| < 0.1     | Stable            |
| 0.1 - 0.2 | Moderate shift    |
| > 0.2     | Significant drift |

### Page-Hinkley on output/operational signals

Tracks a cumulative sum of deviations from the mean over time. Catches gradual trends that PSI's windowed comparison would miss.

Applied to:

- **Error rate** (from feedback): is the model getting more wrong over time?
- **Confidence**: is the model getting less sure?
- **Click-through rate** (actual): is user behavior changing?

### Why this split?

- PSI answers "did the input distribution change?" (shape changes, sudden shifts)
- Page-Hinkley answers "is this signal trending in one direction?" (gradual drift)
- PSI on confidence would be wrong: you care about a downward _trend_, not a distribution _shape change_
- Page-Hinkley on features would miss shape changes where the mean stays the same

## Running the Full Flow

```bash
# 1. Start the server
uv run uvicorn app.main:app --reload

# 2. Generate traffic with feedback
uv run python scripts/generate_traffic.py stable
uv run python scripts/generate_traffic.py data-drift
uv run python scripts/generate_traffic.py concept-drift

# 3. Compute drift metrics against the training baseline
uv run python scripts/compute_drift.py

# 4. Create/update Axiom dashboard (idempotent, same UID every time)
uv run python scripts/create_dashboard.py

# 5. Create Axiom monitors/alerts
uv run python scripts/create_monitors.py
```

Axiom dashboard: https://app.axiom.co/iti-ihxq/dashboards/409eed9e-18e5-443e-a685-760acf18ecfc

## Traffic Scenarios

The traffic generator has three scenarios:

| Scenario        | Command                                                | Expected detector          |
| --------------- | ------------------------------------------------------ | -------------------------- |
| Stable          | `uv run python scripts/generate_traffic.py stable`     | Training-like baseline     |
| Data drift      | `uv run python scripts/generate_traffic.py data-drift` | PSI on input features      |
| Concept drift   | `uv run python scripts/generate_traffic.py concept-drift` | Page-Hinkley on feedback signals |

Stable and data-drift scenarios default to 50 requests. Concept drift defaults to
100 requests so the feedback signal has enough points to move visibly.

Data drift shifts one input feature while keeping the original click behavior.
The demo uses `hour_of_day` because it has a PSI monitor and appears in the
dashboard feature summary, making the shift easy to see.

| Feature              | Normal         | Drifted          |
| -------------------- | -------------- | ---------------- |
| hour_of_day          | Training-like  | 0-5 (late night) |
| device_type          | Training-like  | Training-like    |
| ad_position          | Training-like  | Training-like    |
| user_age             | Training-like  | Training-like    |
| session_duration_sec | Training-like  | Training-like    |
| page_views           | Training-like  | Training-like    |

Stable traffic is sampled from the saved training baseline, so PSI should stay
quiet unless the training baseline itself changes.

Concept drift keeps the same feature distribution as stable traffic but changes
the click behavior for one relationship: `hour_of_day`. In this simulation,
early-day traffic becomes very likely to click while later-day traffic becomes
unlikely to click. That means the input distribution can still look normal while
feedback accuracy, CTR, and error behavior change, which is what Page-Hinkley is
meant to catch.

For the clearest Page-Hinkley demo, send stable traffic first so the detector has a normal feedback baseline, then send concept drift traffic.

## Axiom Monitors

| Monitor                 | Type       | Triggers when                            |
| ----------------------- | ---------- | ---------------------------------------- |
| CTR: High Error Rate    | Threshold  | Error rate from feedback > 50%           |
| CTR: Low Confidence     | Threshold  | Median confidence < 0.6                  |
| CTR: PSI feature_hour_of_day | Threshold | PSI for that feature > 0.2            |
| CTR: PSI feature_ad_position | Threshold | PSI for that feature > 0.2            |
| CTR: PSI feature_user_age | Threshold | PSI for that feature > 0.2              |
| CTR: PSI feature_session_duration_sec | Threshold | PSI for that feature > 0.2      |
| CTR: PSI feature_page_views | Threshold | PSI for that feature > 0.2             |
| CTR: PH error_rate      | MatchEvent | Page-Hinkley detects drift in error rate |
| CTR: PH confidence      | MatchEvent | Page-Hinkley detects drift in confidence |
| CTR: PH click_through_rate | MatchEvent | Page-Hinkley detects drift in CTR   |

For a quick classroom demo, the monitors run every 1 minute over a 2 minute window.

Alerts sent to email via Axiom notifier.

## Lessons Learned

- **PSI for features, Page-Hinkley for signals**: use the right tool for the right data
- **Immediate feedback closes the loop**: CTR lets us monitor actual accuracy, not just proxies
- **Dashboard scripts should be idempotent**: use a fixed UID + `overwrite: True` to avoid duplicates
- **Data drift is easy to simulate**: shift feature ranges and watch PSI light up
- **Concept drift can happen without feature drift**: keep input ranges stable, change feedback behavior, and watch Page-Hinkley light up
