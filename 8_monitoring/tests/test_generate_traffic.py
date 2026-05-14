from scripts import generate_traffic


def test_random_sample_uses_feature_ranges(monkeypatch):
    monkeypatch.setattr(generate_traffic.random, "randint", lambda lo, hi: hi)
    monkeypatch.setattr(generate_traffic.random, "uniform", lambda lo, hi: hi)

    sample = generate_traffic.random_sample(generate_traffic.DATA_DRIFT_RANGES)

    assert sample == {
        "hour_of_day": 5,
        "device_type": 0,
        "ad_position": 5,
        "user_age": 65,
        "session_duration_sec": 1800,
        "page_views": 3,
    }


def test_data_drift_ranges_shift_features_without_changing_click_function():
    stable = generate_traffic.random_sample(generate_traffic.STABLE_RANGES)
    drifted = generate_traffic.random_sample(generate_traffic.DATA_DRIFT_RANGES)

    assert set(stable) == set(drifted)
    assert generate_traffic.DATA_DRIFT_RANGES["hour_of_day"] == (0, 5)
    assert generate_traffic.DATA_DRIFT_RANGES["user_age"] == (55, 65)


def test_concept_drift_changes_click_probability_on_stable_features():
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

    def fake_generate(label, ranges, probability_fn, count, delay):
        calls.append((label, ranges, probability_fn, count, delay))

    monkeypatch.setattr(generate_traffic, "generate", fake_generate)

    generate_traffic.generate_stable(1, 0.1)
    generate_traffic.generate_data_drift(2, 0.2)
    generate_traffic.generate_concept_drift(3, 0.3)

    assert calls[0][0] == "stable"
    assert calls[1][0] == "data drift"
    assert calls[2][0] == "concept drift"


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
        ("stable", 100, 0.0),
        ("data", 100, 0.0),
        ("concept", 400, 0.0),
    ]
