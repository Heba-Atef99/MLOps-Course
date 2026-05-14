import pytest

from scripts import create_dashboard, create_monitors


def test_build_dashboard_contains_monitoring_charts():
    dashboard = create_dashboard.build_dashboard()

    chart_names = {chart["name"] for chart in dashboard["charts"]}
    assert dashboard["name"] == "CTR Model Monitoring"
    assert "Error Rate Over Time (from feedback)" in chart_names
    assert "Confidence Over Time" in chart_names
    assert len(dashboard["layout"]) == len(dashboard["charts"])
    assert dashboard["timeWindowStart"] == "qr-now-2h"


def test_create_dashboard_main_posts_idempotent_payload(monkeypatch):
    posted = {}

    class Response:
        ok = True
        status_code = 200
        text = "ok"

    def fake_post(url, headers, json, timeout):
        posted.update(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return Response()

    monkeypatch.setattr(create_dashboard, "AXIOM_TOKEN", "token")
    monkeypatch.setattr(create_dashboard, "AXIOM_ORG_ID", "org")
    monkeypatch.setattr(create_dashboard.requests, "post", fake_post)

    create_dashboard.main()

    assert posted["url"].endswith("/v2/dashboards")
    assert posted["json"]["overwrite"] is True
    assert posted["json"]["uid"] == "409eed9e-18e5-443e-a685-760acf18ecfc"


def test_create_dashboard_main_exits_without_credentials(monkeypatch):
    monkeypatch.setattr(create_dashboard, "AXIOM_TOKEN", None)
    monkeypatch.setattr(create_dashboard, "AXIOM_ORG_ID", None)

    with pytest.raises(SystemExit):
        create_dashboard.main()


def test_create_dashboard_main_exits_on_api_error(monkeypatch):
    class Response:
        ok = False
        status_code = 500
        text = "server error"

    monkeypatch.setattr(create_dashboard, "AXIOM_TOKEN", "token")
    monkeypatch.setattr(create_dashboard, "AXIOM_ORG_ID", "org")
    monkeypatch.setattr(create_dashboard.requests, "post", lambda *_, **__: Response())

    with pytest.raises(SystemExit):
        create_dashboard.main()


def test_build_monitors_contains_expected_alerts():
    monitors = create_monitors.build_monitors()
    names = {monitor["name"] for monitor in monitors}

    assert "CTR: High Error Rate" in names
    assert "CTR: Low Confidence" in names
    assert "CTR: PSI feature_hour_of_day" in names
    assert "CTR: PSI feature_page_views" in names
    assert "CTR: PH error_rate" in names
    assert "CTR: PH click_through_rate" in names
    assert len(monitors) == 10
    assert all(monitor["intervalMinutes"] == 1 for monitor in monitors)
    assert all(monitor["rangeMinutes"] == 2 for monitor in monitors)


def test_get_existing_monitors_returns_names(monkeypatch):
    class Response:
        ok = True

        @staticmethod
        def json():
            return [{"name": "A"}, {"name": "B"}]

    monkeypatch.setattr(create_monitors.requests, "get", lambda *_, **__: Response())

    assert create_monitors.get_existing_monitors() == {"A", "B"}


def test_get_existing_monitors_returns_empty_on_api_error(monkeypatch):
    class Response:
        ok = False

    monkeypatch.setattr(create_monitors.requests, "get", lambda *_, **__: Response())

    assert create_monitors.get_existing_monitors() == set()


def test_get_or_create_notifier_returns_none_without_email(monkeypatch):
    monkeypatch.setattr(create_monitors, "ALERT_EMAIL", None)

    assert create_monitors.get_or_create_notifier() is None


def test_get_or_create_notifier_reuses_existing(monkeypatch):
    class Response:
        ok = True

        @staticmethod
        def json():
            return [{"name": "CTR Monitoring Alerts", "id": "notifier-1"}]

    monkeypatch.setattr(create_monitors, "ALERT_EMAIL", "user@example.com")
    monkeypatch.setattr(create_monitors.requests, "get", lambda *_, **__: Response())

    assert create_monitors.get_or_create_notifier() == "notifier-1"


def test_get_or_create_notifier_creates_when_missing(monkeypatch):
    post_payload = {}

    class GetResponse:
        ok = True

        @staticmethod
        def json():
            return []

    class PostResponse:
        ok = True

        @staticmethod
        def json():
            return {"id": "notifier-2"}

    def fake_post(url, headers, json, timeout):
        post_payload.update({"url": url, "json": json})
        return PostResponse()

    monkeypatch.setattr(create_monitors, "ALERT_EMAIL", "user@example.com")
    monkeypatch.setattr(create_monitors.requests, "get", lambda *_, **__: GetResponse())
    monkeypatch.setattr(create_monitors.requests, "post", fake_post)

    assert create_monitors.get_or_create_notifier() == "notifier-2"
    assert post_payload["json"]["properties"]["email"]["emails"] == ["user@example.com"]


def test_get_or_create_notifier_returns_none_on_create_failure(monkeypatch):
    class GetResponse:
        ok = True

        @staticmethod
        def json():
            return []

    class PostResponse:
        ok = False
        status_code = 400
        text = "bad request"

    monkeypatch.setattr(create_monitors, "ALERT_EMAIL", "user@example.com")
    monkeypatch.setattr(create_monitors.requests, "get", lambda *_, **__: GetResponse())
    monkeypatch.setattr(
        create_monitors.requests, "post", lambda *_, **__: PostResponse()
    )

    assert create_monitors.get_or_create_notifier() is None


def test_create_monitors_main_creates_missing_monitors(monkeypatch):
    posted = []

    class Response:
        ok = True

        @staticmethod
        def json():
            return {"id": "monitor-id"}

    def fake_post(url, headers, json, timeout):
        posted.append(json)
        return Response()

    monkeypatch.setattr(create_monitors, "AXIOM_TOKEN", "token")
    monkeypatch.setattr(create_monitors, "AXIOM_ORG_ID", "org")
    monkeypatch.setattr(create_monitors, "get_or_create_notifier", lambda: "notifier")
    monkeypatch.setattr(create_monitors, "get_existing_monitors", set)
    monkeypatch.setattr(create_monitors.requests, "post", fake_post)

    create_monitors.main()

    assert len(posted) == 10
    assert all(monitor["notifierIds"] == ["notifier"] for monitor in posted)


def test_create_monitors_main_exits_without_credentials(monkeypatch):
    monkeypatch.setattr(create_monitors, "AXIOM_TOKEN", None)
    monkeypatch.setattr(create_monitors, "AXIOM_ORG_ID", None)

    with pytest.raises(SystemExit):
        create_monitors.main()


def test_create_monitors_main_skips_existing_and_counts_failures(monkeypatch, capsys):
    class Response:
        ok = False
        status_code = 500
        text = "server error"

    monkeypatch.setattr(create_monitors, "AXIOM_TOKEN", "token")
    monkeypatch.setattr(create_monitors, "AXIOM_ORG_ID", "org")
    monkeypatch.setattr(create_monitors, "get_or_create_notifier", lambda: None)
    monkeypatch.setattr(
        create_monitors,
        "get_existing_monitors",
        lambda: {"CTR: High Error Rate"},
    )
    monkeypatch.setattr(create_monitors.requests, "post", lambda *_, **__: Response())

    create_monitors.main()

    output = capsys.readouterr().out
    assert "Skipped: CTR: High Error Rate" in output
    assert "1 skipped" in output
    assert "9 failed" in output
