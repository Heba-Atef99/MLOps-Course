import numpy as np
import pytest

from scripts import compute_drift


def test_compute_psi_zero_for_identical_distributions():
    values = np.array([0, 1, 2, 3, 4, 5])

    psi = compute_drift.compute_psi(values, values, [0, 2, 4, 6])

    assert psi == 0.0


def test_compute_psi_detects_shifted_distribution():
    reference = np.array([0, 0, 1, 1, 2, 2])
    current = np.array([4, 4, 5, 5, 5, 5])

    psi = compute_drift.compute_psi(reference, current, [0, 2, 4, 6])

    assert psi > 0.2


def test_update_ph_incremental_continues_from_stored_state():
    initial = compute_drift.update_ph_incremental(None, np.array([0.0, 0.0, 0.0]))

    updated = compute_drift.update_ph_incremental(initial, np.array([1.0, 1.0, 1.0]))

    assert updated["n"] == 6
    assert updated["running_mean"] == 0.5
    assert updated["ph_value"] > initial["ph_value"]


def test_update_ph_incremental_flags_drift_when_threshold_crossed():
    values = np.array([0.0] * 100 + [1.0] * 400)

    state = compute_drift.update_ph_incremental(None, values, threshold=50)

    assert state["drift_detected"] is True
    assert state["ph_value"] > 50


def test_load_training_baseline_reads_feature_mapping(tmp_path, monkeypatch):
    baseline = tmp_path / "training_baseline.csv"
    baseline.write_text(
        "hour_of_day,device_type,ad_position,user_age,session_duration_sec,page_views\n"
        "10,1,2,28,450.0,12\n"
        "21,0,5,60,1200.0,3\n"
    )
    monkeypatch.setattr(compute_drift, "TRAINING_BASELINE_PATH", str(baseline))

    data = compute_drift.load_training_baseline()

    assert set(data) == set(compute_drift.BASELINE_COLUMNS)
    assert data["feature_hour_of_day"].tolist() == [10.0, 21.0]
    assert data["feature_session_duration_sec"].tolist() == [450.0, 1200.0]


def test_load_training_baseline_exits_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        compute_drift, "TRAINING_BASELINE_PATH", str(tmp_path / "missing.csv")
    )

    with pytest.raises(SystemExit):
        compute_drift.load_training_baseline()


def test_query_axiom_converts_tabular_response(monkeypatch):
    class Response:
        ok = True

        @staticmethod
        def json():
            return {
                "tables": [
                    {
                        "fields": [{"name": "_time"}, {"name": "val"}],
                        "columns": [["t1", "t2"], [0.1, 0.2]],
                    }
                ]
            }

    monkeypatch.setattr(compute_drift.requests, "post", lambda *_, **__: Response())

    rows = compute_drift.query_axiom("fake apl")

    assert rows == [{"_time": "t1", "val": 0.1}, {"_time": "t2", "val": 0.2}]


def test_query_axiom_returns_empty_for_failed_or_empty_response(monkeypatch):
    class FailedResponse:
        ok = False
        status_code = 401
        text = "unauthorized"

    monkeypatch.setattr(
        compute_drift.requests, "post", lambda *_, **__: FailedResponse()
    )
    assert compute_drift.query_axiom("fake apl") == []

    class EmptyResponse:
        ok = True

        @staticmethod
        def json():
            return {"tables": []}

    monkeypatch.setattr(
        compute_drift.requests, "post", lambda *_, **__: EmptyResponse()
    )
    assert compute_drift.query_axiom("fake apl") == []


def test_compute_psi_returns_zero_for_missing_values():
    assert compute_drift.compute_psi(np.array([]), np.array([1]), [0, 2]) == 0.0


def test_load_ph_state_restores_persisted_values(monkeypatch):
    monkeypatch.setattr(
        compute_drift,
        "query_axiom",
        lambda _: [
            {
                "feature": "error_rate",
                "cumsum": "1.5",
                "min_cumsum": "-0.5",
                "running_mean": "0.25",
                "n": "10",
                "last_timestamp": "2026-01-01T00:00:00Z",
            }
        ],
    )

    state = compute_drift.load_ph_state("error_rate")

    assert state == {
        "cumsum": 1.5,
        "min_cumsum": -0.5,
        "running_mean": 0.25,
        "n": 10,
        "last_timestamp": "2026-01-01T00:00:00Z",
    }


def test_load_ph_state_skips_rows_for_other_features(monkeypatch):
    monkeypatch.setattr(
        compute_drift,
        "query_axiom",
        lambda _: [
            {
                "feature": "confidence",
                "cumsum": "1.5",
                "min_cumsum": "-0.5",
                "running_mean": "0.25",
                "n": "10",
                "last_timestamp": "2026-01-01T00:00:00Z",
            }
        ],
    )

    assert compute_drift.load_ph_state("error_rate") is None


def test_fetch_prediction_data_projects_monitored_fields(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        compute_drift,
        "query_axiom",
        lambda apl: captured.setdefault("apl", apl) or [],
    )

    compute_drift.fetch_prediction_data("where _time > ago(1h)")

    assert "event_type == 'prediction'" in captured["apl"]
    assert "feature_hour_of_day" in captured["apl"]
    assert "confidence" in captured["apl"]


def test_fetch_signal_since_uses_raw_feedback_events(monkeypatch):
    captured = {}

    def fake_query(apl: str):
        captured["apl"] = apl
        return [{"_time": "t1", "val": 1.0}]

    monkeypatch.setattr(compute_drift, "query_axiom", fake_query)

    rows = compute_drift.fetch_signal_since("error_rate", "2026-01-01T00:00:00Z")

    assert rows == [{"_time": "t1", "val": 1.0}]
    assert "project _time, val = iff(correct, 0.0, 1.0)" in captured["apl"]
    assert "summarize" not in captured["apl"]


def test_fetch_signal_since_returns_empty_for_unknown_signal():
    assert compute_drift.fetch_signal_since("unknown", "") == []


def test_ingest_events_sends_json_payload(monkeypatch):
    calls = {}

    class Client:
        def ingest(self, dataset, payload, content_type, content_encoding):
            calls.update(
                {
                    "dataset": dataset,
                    "payload": payload,
                    "content_type": content_type,
                    "content_encoding": content_encoding,
                }
            )

    monkeypatch.setattr(compute_drift, "AxiomClient", Client)

    compute_drift.ingest_events([{"event_type": "test"}])

    assert calls["dataset"] == compute_drift.AXIOM_DATASET
    assert b'"event_type": "test"' in calls["payload"]


def test_ingest_events_exits_on_client_error(monkeypatch):
    class Client:
        def ingest(self, *_, **__):
            raise RuntimeError("boom")

    monkeypatch.setattr(compute_drift, "AxiomClient", Client)

    with pytest.raises(SystemExit):
        compute_drift.ingest_events([{"event_type": "test"}])


def test_main_emits_training_baseline_psi_and_page_hinkley_events(monkeypatch):
    ingested = {}
    current_rows = [
        {
            "feature_hour_of_day": 22,
            "feature_ad_position": 5,
            "feature_user_age": 60,
            "feature_session_duration_sec": 1500.0,
            "feature_page_views": 2,
            "confidence": 0.5,
        }
    ]

    monkeypatch.setattr(compute_drift, "AXIOM_TOKEN", "token")
    monkeypatch.setattr(compute_drift, "AXIOM_ORG_ID", "org")
    monkeypatch.setattr(
        compute_drift,
        "load_training_baseline",
        lambda: {
            "feature_hour_of_day": np.array([10, 11, 12]),
            "feature_ad_position": np.array([1, 2, 2]),
            "feature_user_age": np.array([25, 30, 35]),
            "feature_session_duration_sec": np.array([100.0, 200.0, 300.0]),
            "feature_page_views": np.array([10, 11, 12]),
        },
    )
    monkeypatch.setattr(compute_drift, "fetch_prediction_data", lambda _: current_rows)
    monkeypatch.setattr(compute_drift, "load_ph_state", lambda _: None)
    monkeypatch.setattr(
        compute_drift,
        "fetch_signal_since",
        lambda *_: [{"_time": "2026-01-01T00:00:00Z", "val": 1.0}],
    )
    monkeypatch.setattr(
        compute_drift,
        "ingest_events",
        lambda events: ingested.setdefault("events", events),
    )

    compute_drift.main()

    events = ingested["events"]
    assert any(e["event_type"] == "drift_psi" for e in events)
    assert any(e.get("baseline") == "training" for e in events)
    assert any(e["event_type"] == "drift_page_hinkley" for e in events)
    assert any(e["event_type"] == "drift_ph_state" for e in events)
    assert events[-1]["event_type"] == "drift_summary"


def test_main_exits_without_axiom_credentials(monkeypatch):
    monkeypatch.setattr(compute_drift, "AXIOM_TOKEN", None)
    monkeypatch.setattr(compute_drift, "AXIOM_ORG_ID", None)

    with pytest.raises(SystemExit):
        compute_drift.main()


def test_main_keeps_previous_page_hinkley_state_when_no_new_rows(monkeypatch):
    ingested = {}
    previous_state = {
        "cumsum": 55.0,
        "min_cumsum": 0.0,
        "running_mean": 0.2,
        "n": 25,
        "last_timestamp": "2026-01-01T00:00:00Z",
    }

    monkeypatch.setattr(compute_drift, "AXIOM_TOKEN", "token")
    monkeypatch.setattr(compute_drift, "AXIOM_ORG_ID", "org")
    monkeypatch.setattr(
        compute_drift,
        "load_training_baseline",
        lambda: {
            "feature_hour_of_day": np.array([10]),
            "feature_ad_position": np.array([1]),
            "feature_user_age": np.array([25]),
            "feature_session_duration_sec": np.array([100.0]),
            "feature_page_views": np.array([10]),
        },
    )
    monkeypatch.setattr(compute_drift, "fetch_prediction_data", lambda _: [])
    monkeypatch.setattr(compute_drift, "load_ph_state", lambda _: previous_state)
    monkeypatch.setattr(compute_drift, "fetch_signal_since", lambda *_: [])
    monkeypatch.setattr(
        compute_drift,
        "ingest_events",
        lambda events: ingested.setdefault("events", events),
    )

    compute_drift.main()

    events = ingested["events"]
    assert any(e["event_type"] == "drift_page_hinkley" for e in events)
    assert events[-1]["max_psi_feature"] == "none"
