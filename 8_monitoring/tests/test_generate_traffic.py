from scripts import generate_traffic


def test_random_sample_uses_feature_ranges(monkeypatch):
    monkeypatch.setattr(generate_traffic.random, "randint", lambda lo, hi: hi)
    monkeypatch.setattr(generate_traffic.random, "uniform", lambda lo, hi: hi)

    sample = generate_traffic.random_sample(generate_traffic.DATA_DRIFT_RANGES)

    assert sample == {
        "hour_of_day": 5,
        "device_type": 2,
        "ad_position": 5,
        "user_age": 65,
        "session_duration_sec": 1800,
        "page_views": 50,
    }


def test_data_drift_ranges_shift_only_hour_of_day():
    stable = generate_traffic.STABLE_FALLBACK_RANGES
    drifted = generate_traffic.DATA_DRIFT_RANGES
    changed = [name for name in stable if stable[name] != drifted[name]]

    assert changed == ["hour_of_day"]
    assert drifted["hour_of_day"] == (0, 5)


def test_load_training_baseline_rows_reads_csv(tmp_path, monkeypatch):
    baseline = tmp_path / "training_baseline.csv"
    baseline.write_text(
        "hour_of_day,device_type,ad_position,user_age,session_duration_sec,page_views\n"
        "10,1,2,28,450.0,12\n"
        "21,0,5,60,1200.0,3\n"
    )
    monkeypatch.setattr(generate_traffic, "TRAINING_BASELINE_PATH", baseline)

    rows = generate_traffic.load_training_baseline_rows()

    assert rows == [
        {
            "hour_of_day": 10,
            "device_type": 1,
            "ad_position": 2,
            "user_age": 28,
            "session_duration_sec": 450.0,
            "page_views": 12,
        },
        {
            "hour_of_day": 21,
            "device_type": 0,
            "ad_position": 5,
            "user_age": 60,
            "session_duration_sec": 1200.0,
            "page_views": 3,
        },
    ]


def test_sample_stable_row_uses_training_baseline(monkeypatch):
    monkeypatch.setattr(
        generate_traffic,
        "load_training_baseline_rows",
        lambda: [
            {
                "hour_of_day": 9,
                "device_type": 1,
                "ad_position": 2,
                "user_age": 28,
                "session_duration_sec": 450.0,
                "page_views": 12,
            }
        ],
    )

    assert generate_traffic.sample_stable_row()["hour_of_day"] == 9


def test_sample_stable_row_falls_back_when_baseline_missing(monkeypatch):
    monkeypatch.setattr(generate_traffic, "load_training_baseline_rows", lambda: [])
    monkeypatch.setattr(generate_traffic.random, "randint", lambda lo, hi: lo)
    monkeypatch.setattr(generate_traffic.random, "uniform", lambda lo, hi: hi)

    sample = generate_traffic.sample_stable_row()

    assert sample["hour_of_day"] == 0
    assert sample["page_views"] == 1


def test_sample_data_drift_row_changes_only_hour(monkeypatch):
    stable_sample = {
        "hour_of_day": 14,
        "device_type": 1,
        "ad_position": 2,
        "user_age": 28,
        "session_duration_sec": 450.0,
        "page_views": 12,
    }
    monkeypatch.setattr(generate_traffic, "sample_stable_row", lambda: stable_sample)
    monkeypatch.setattr(generate_traffic.random, "randint", lambda lo, hi: hi)

    sample = generate_traffic.sample_data_drift_row()

    assert sample == {**stable_sample, "hour_of_day": 5}


def test_concept_drift_changes_click_probability_by_hour_only():
    sample = {
        "hour_of_day": 5,
        "device_type": 0,
        "ad_position": 5,
        "user_age": 60,
        "session_duration_sec": 1200.0,
        "page_views": 3,
    }

    stable_prob = generate_traffic.stable_click_probability(sample)
    concept_prob = generate_traffic.concept_drift_click_probability(sample)

    assert concept_prob > stable_prob
    assert 0.0 <= stable_prob <= 1.0
    assert 0.0 <= concept_prob <= 1.0

    same_hour_different_features = {
        "hour_of_day": 5,
        "device_type": 2,
        "ad_position": 1,
        "user_age": 22,
        "session_duration_sec": 60.0,
        "page_views": 50,
    }
    later_hour = {**sample, "hour_of_day": 18}

    assert (
        generate_traffic.concept_drift_click_probability(same_hour_different_features)
        == concept_prob
    )
    assert generate_traffic.concept_drift_click_probability(later_hour) < concept_prob


def test_simulate_click_uses_probability_function(monkeypatch):
    monkeypatch.setattr(generate_traffic.random, "random", lambda: 0.4)

    assert generate_traffic.simulate_click({}, lambda _: 0.5) is True
    assert generate_traffic.simulate_click({}, lambda _: 0.3) is False


def test_send_predict_and_feedback_use_api(monkeypatch):
    calls = []

    class Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = "body"

        def json(self):
            return self._payload

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        if url.endswith("/predict"):
            return Response(
                201,
                {"prediction_id": "pred-1", "predicted_click": True},
            )
        return Response(201, {"ok": True})

    monkeypatch.setattr(generate_traffic.requests, "post", fake_post)

    prediction = generate_traffic.send_predict({"hour_of_day": 12})
    assert prediction is not None
    feedback_ok = generate_traffic.send_feedback("pred-1", True)

    assert prediction["prediction_id"] == "pred-1"
    assert feedback_ok is True
    assert calls[0][0].endswith("/predict")
    assert calls[1][0].endswith("/feedback")


def test_send_predict_and_feedback_report_failures(monkeypatch):
    class Response:
        status_code = 500
        text = "error"

        @staticmethod
        def json():
            return {}

    monkeypatch.setattr(generate_traffic.requests, "post", lambda *_, **__: Response())

    assert generate_traffic.send_predict({"hour_of_day": 12}) is None
    assert generate_traffic.send_feedback("pred-1", True) is False


def test_generate_counts_successful_feedback(monkeypatch, capsys):
    predictions = iter(
        [
            {"prediction_id": "pred-1", "predicted_click": True},
            {"prediction_id": "pred-2", "predicted_click": False},
        ]
    )

    monkeypatch.setattr(generate_traffic, "send_predict", lambda _: next(predictions))
    monkeypatch.setattr(generate_traffic, "send_feedback", lambda *_: True)
    monkeypatch.setattr(generate_traffic, "simulate_click", lambda *_: True)
    monkeypatch.setattr(generate_traffic.time, "sleep", lambda _: None)

    generate_traffic.generate(
        label="test",
        ranges=generate_traffic.STABLE_RANGES,
        probability_fn=generate_traffic.stable_click_probability,
        count=2,
        delay=0,
    )

    assert "2/2 successful" in capsys.readouterr().out


def test_generate_skips_failed_predictions(monkeypatch, capsys):
    monkeypatch.setattr(generate_traffic, "send_predict", lambda _: None)
    monkeypatch.setattr(generate_traffic.time, "sleep", lambda _: None)

    generate_traffic.generate(
        label="test",
        ranges=generate_traffic.STABLE_RANGES,
        probability_fn=generate_traffic.stable_click_probability,
        count=1,
        delay=0,
    )

    assert "0/1 successful" in capsys.readouterr().out


def test_generate_wrapper_functions_call_generate(monkeypatch):
    calls = []

    def fake_generate(label, ranges, probability_fn, count, delay, sample_fn=None):
        calls.append((label, ranges, probability_fn, count, delay, sample_fn))

    monkeypatch.setattr(generate_traffic, "generate", fake_generate)

    generate_traffic.generate_stable(1, 0.1)
    generate_traffic.generate_data_drift(2, 0.2)
    generate_traffic.generate_concept_drift(3, 0.3)

    assert calls[0][0] == "stable"
    assert calls[1][0] == "data drift"
    assert calls[2][0] == "concept drift"
    assert calls[0][5] == generate_traffic.sample_stable_row
    assert calls[1][5] == generate_traffic.sample_data_drift_row


def test_main_dispatches_selected_scenario(monkeypatch):
    called = []
    monkeypatch.setattr(
        generate_traffic,
        "parse_args",
        lambda: generate_traffic.argparse.Namespace(
            scenario="data-drift", count=7, delay=0.0
        ),
    )
    monkeypatch.setattr(
        generate_traffic,
        "generate_data_drift",
        lambda count, delay: called.append((count, delay)),
    )

    generate_traffic.main()

    assert called == [(7, 0.0)]


def test_main_dispatches_all_scenarios(monkeypatch):
    called = []
    monkeypatch.setattr(
        generate_traffic,
        "parse_args",
        lambda: generate_traffic.argparse.Namespace(
            scenario="all", count=None, delay=0.0
        ),
    )
    monkeypatch.setattr(
        generate_traffic,
        "generate_stable",
        lambda count, delay: called.append(("stable", count, delay)),
    )
    monkeypatch.setattr(
        generate_traffic,
        "generate_data_drift",
        lambda count, delay: called.append(("data", count, delay)),
    )
    monkeypatch.setattr(
        generate_traffic,
        "generate_concept_drift",
        lambda count, delay: called.append(("concept", count, delay)),
    )

    generate_traffic.main()

    assert called == [
        ("stable", 50, 0.0),
        ("data", 50, 0.0),
        ("concept", 100, 0.0),
    ]
