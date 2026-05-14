# CTR Monitoring Demo: Click-Through Prediction + Axiom

A monitoring demo built on top of a click-through rate prediction model. Predicts whether a user will click an ad, receives immediate feedback (clicked or not), and monitors model health via Axiom.

## Why Click-Through?

Unlike static datasets, CTR gives us **immediate feedback**: we predict "will the user click?", and seconds later we know if they did. This closes the feedback loop and lets us monitor actual model accuracy in real-time, not just proxy metrics.

## Quick Start

```bash
uv sync
uv run python training/train.py     # train CTR model (synthetic data)
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
│   ├── generate_traffic.py  # Send normal + drifted traffic with feedback
│   ├── compute_drift.py     # PSI on features, Page-Hinkley on error rate
│   ├── create_dashboard.py  # Create/update Axiom dashboard (idempotent)
│   └── create_monitors.py   # Create Axiom alerts
├── tests/                   # 9 tests (predict, feedback, model, edge cases)
├── model/
├── pyproject.toml
└── .env.example
```

## Drift Detection: PSI vs Page-Hinkley

We use two complementary techniques, each applied where it fits best:

### PSI (Population Stability Index) on input features

Compares the distribution of a feature between a reference window (old data) and a current window (recent data). Catches sudden distribution shifts in model inputs.

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

# 2. Generate traffic (100 normal + 50 drifted, with feedback)
uv run python scripts/generate_traffic.py

# 3. Compute drift metrics (ingests results back to Axiom)
uv run python scripts/compute_drift.py

# 4. Create/update Axiom dashboard (idempotent, same UID every time)
uv run python scripts/create_dashboard.py

# 5. Create Axiom monitors/alerts
uv run python scripts/create_monitors.py
```

Axiom dashboard: https://app.axiom.co/iti-ihxq/dashboards/409eed9e-18e5-443e-a685-760acf18ecfc

## Drifted Traffic

The traffic generator simulates drift by shifting feature distributions:

| Feature              | Normal         | Drifted          |
| -------------------- | -------------- | ---------------- |
| hour_of_day          | 8-22 (daytime) | 0-5 (late night) |
| device_type          | 60% mobile     | 100% mobile      |
| ad_position          | 1-5            | 4-5 (bottom)     |
| user_age             | 18-65          | 55-65 (older)    |
| session_duration_sec | 30-1200        | 1200-1800        |
| page_views           | 1-30           | 1-3              |
| simulated CTR        | ~15-25%        | ~3%              |

This causes the model to encounter out-of-distribution inputs, lowering confidence and increasing error rate, which PSI and Page-Hinkley detect.

## Axiom Monitors

| Monitor                 | Type       | Triggers when                            |
| ----------------------- | ---------- | ---------------------------------------- |
| CTR: High Error Rate    | Threshold  | Error rate from feedback > 50%           |
| CTR: Low Confidence     | Threshold  | Median confidence < 0.6                  |
| CTR: PSI Drift Detected | Threshold  | Max PSI across features > 0.2            |
| CTR: Page-Hinkley Drift | MatchEvent | Page-Hinkley detects drift in any signal |

Alerts sent to email via Axiom notifier.

## Lessons Learned

- **PSI for features, Page-Hinkley for signals**: use the right tool for the right data
- **Immediate feedback closes the loop**: CTR lets us monitor actual accuracy, not just proxies
- **Dashboard scripts should be idempotent**: use a fixed UID + `overwrite: True` to avoid duplicates
- **Drifted traffic is easy to simulate**: shift feature ranges and watch PSI light up
- **Page-Hinkley needs time**: it tracks cumulative deviation over hours/days, not minutes. In production with weeks of data, it catches slow model degradation that PSI misses
