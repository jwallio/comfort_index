from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError

import pandas as pd

import comfortwx.data.openmeteo_reliability as reliability
from comfortwx.data.openmeteo_reliability import (
    fetch_with_retries,
    openmeteo_request_context,
    reset_openmeteo_request_records,
    write_openmeteo_request_report,
)


def test_fetch_with_retries_records_retry_and_success(tmp_path: Path) -> None:
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    def _request(_base_url: str, _query: dict[str, object]):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise HTTPError("https://example.com", 429, "Too Many Requests", hdrs=None, fp=None)
        return {"ok": True}

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    reset_openmeteo_request_records()
    reliability._LAST_REQUEST_TS.set(0.0)
    original_sleep = reliability.time.sleep
    original_uniform = reliability.random.uniform
    reliability.time.sleep = _fake_sleep
    reliability.random.uniform = lambda _a, _b: 0.0
    with openmeteo_request_context(workflow="verification_benchmark", label="test-case", run_slug="retry_test"):
        try:
            payload = fetch_with_retries(
                base_url="https://api.open-meteo.com/v1/forecast",
                query={"latitude": 35.0, "longitude": -78.0},
                request_func=_request,
            )
        finally:
            reliability.time.sleep = original_sleep
            reliability.random.uniform = original_uniform

    assert payload == {"ok": True}
    summary_path, detail_path = write_openmeteo_request_report(output_dir=tmp_path, run_slug="retry_test")
    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)

    assert int(summary.loc[0, "total_requests"]) == 2
    assert int(summary.loc[0, "successful_requests"]) == 1
    assert int(summary.loc[0, "retry_events"]) == 1
    assert int(summary.loc[0, "http_429s"]) == 1
    assert set(detail["outcome"]) == {"retrying", "success"}
    assert sleep_calls
    assert sleep_calls[-1] >= reliability.OPENMETEO_REQUEST_429_BACKOFF_SECONDS


def test_fetch_with_retries_sets_global_cooldown_on_429() -> None:
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    def _request(_base_url: str, _query: dict[str, object]):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise HTTPError("https://example.com", 429, "Too Many Requests", hdrs=None, fp=None)
        return {"ok": True}

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    reset_openmeteo_request_records()
    original_sleep = reliability.time.sleep
    original_uniform = reliability.random.uniform
    reliability.time.sleep = _fake_sleep
    reliability.random.uniform = lambda _a, _b: 0.0
    try:
        with openmeteo_request_context(workflow="verification_benchmark", label="cooldown-test", run_slug="cooldown_test"):
            payload = fetch_with_retries(
                base_url="https://api.open-meteo.com/v1/forecast",
                query={"latitude": 35.0, "longitude": -78.0},
                request_func=_request,
            )
    finally:
        reliability.time.sleep = original_sleep
        reliability.random.uniform = original_uniform

    assert payload == {"ok": True}
    assert sleep_calls
    assert reliability._RATE_LIMIT_UNTIL_TS.get() > 0.0
