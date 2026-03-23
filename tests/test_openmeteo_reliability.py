from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError

import pandas as pd

from nicewx.data.openmeteo_reliability import (
    fetch_with_retries,
    openmeteo_request_context,
    reset_openmeteo_request_records,
    write_openmeteo_request_report,
)


def test_fetch_with_retries_records_retry_and_success(tmp_path: Path) -> None:
    attempts = {"count": 0}

    def _request(_base_url: str, _query: dict[str, object]):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise HTTPError("https://example.com", 429, "Too Many Requests", hdrs=None, fp=None)
        return {"ok": True}

    reset_openmeteo_request_records()
    with openmeteo_request_context(workflow="verification_benchmark", label="test-case", run_slug="retry_test"):
        payload = fetch_with_retries(
            base_url="https://api.open-meteo.com/v1/forecast",
            query={"latitude": 35.0, "longitude": -78.0},
            request_func=_request,
        )

    assert payload == {"ok": True}
    summary_path, detail_path = write_openmeteo_request_report(output_dir=tmp_path, run_slug="retry_test")
    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)

    assert int(summary.loc[0, "total_requests"]) == 2
    assert int(summary.loc[0, "successful_requests"]) == 1
    assert int(summary.loc[0, "retry_events"]) == 1
    assert int(summary.loc[0, "http_429s"]) == 1
    assert set(detail["outcome"]) == {"retrying", "success"}
