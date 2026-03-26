"""Reliability instrumentation and controls for Open-Meteo requests."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
import random
import socket
import time
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

import pandas as pd

from comfortwx.config import (
    OPENMETEO_REQUEST_MAX_RETRIES,
    OPENMETEO_REQUEST_429_BACKOFF_SECONDS,
    OPENMETEO_REQUEST_BACKOFF_JITTER_SECONDS,
    OPENMETEO_REQUEST_RETRYABLE_STATUS_CODES,
    OPENMETEO_REQUEST_RETRY_BACKOFF_MAX_SECONDS,
    OPENMETEO_REQUEST_RETRY_BACKOFF_INITIAL_SECONDS,
    OPENMETEO_REQUEST_RETRY_BACKOFF_MULTIPLIER,
    OPENMETEO_REQUEST_THROTTLE_SECONDS,
    OPENMETEO_TUNING_REQUEST_THROTTLE_SECONDS,
    OPENMETEO_VERIFICATION_REQUEST_THROTTLE_SECONDS,
)


_WORKFLOW_CONTEXT: ContextVar[str] = ContextVar("openmeteo_workflow", default="unspecified")
_LABEL_CONTEXT: ContextVar[str] = ContextVar("openmeteo_label", default="")
_RUN_SLUG_CONTEXT: ContextVar[str] = ContextVar("openmeteo_run_slug", default="")
_LAST_REQUEST_TS: ContextVar[float] = ContextVar("openmeteo_last_request_ts", default=0.0)


@dataclass
class OpenMeteoRequestRecord:
    request_id: int
    workflow: str
    label: str
    run_slug: str
    endpoint_name: str
    endpoint_type: str
    started_at: str
    ended_at: str
    elapsed_seconds: float
    outcome: str
    status_code: int | None
    error_type: str
    retry_count: int
    query_summary: str


_REQUEST_RECORDS: list[OpenMeteoRequestRecord] = []


def reset_openmeteo_request_records() -> None:
    _REQUEST_RECORDS.clear()
    _LAST_REQUEST_TS.set(0.0)


def current_openmeteo_workflow() -> str:
    return _WORKFLOW_CONTEXT.get()


def current_openmeteo_label() -> str:
    return _LABEL_CONTEXT.get()


@contextmanager
def openmeteo_request_context(*, workflow: str, label: str = "", run_slug: str | None = None) -> Iterator[None]:
    workflow_token = _WORKFLOW_CONTEXT.set(workflow)
    label_token = _LABEL_CONTEXT.set(label)
    run_slug_token = _RUN_SLUG_CONTEXT.set(run_slug or _RUN_SLUG_CONTEXT.get())
    try:
        yield
    finally:
        _WORKFLOW_CONTEXT.reset(workflow_token)
        _LABEL_CONTEXT.reset(label_token)
        _RUN_SLUG_CONTEXT.reset(run_slug_token)


def _endpoint_metadata(base_url: str) -> tuple[str, str]:
    parsed = urlparse(base_url)
    host = parsed.netloc.lower()
    if "single-runs-api" in host:
        return "openmeteo_single_run", "forecast_archive"
    if "archive-api" in host:
        return "openmeteo_archive", "analysis_archive"
    if "air-quality-api" in host:
        return "openmeteo_air_quality", "air_quality"
    return "openmeteo_forecast", "forecast"


def _query_summary(query: dict[str, object]) -> str:
    keys = ["start_date", "end_date", "run", "models", "latitude", "longitude", "forecast_days", "forecast_hours"]
    parts: list[str] = []
    for key in keys:
        if key not in query:
            continue
        value = query[key]
        if isinstance(value, str) and len(value) > 80:
            value = value[:77] + "..."
        parts.append(f"{key}={value}")
    return "; ".join(parts)


def _classify_exception(exc: Exception) -> tuple[int | None, str]:
    if isinstance(exc, HTTPError):
        return exc.code, f"http_{exc.code}"
    if isinstance(exc, socket.timeout | TimeoutError):
        return None, "timeout"
    if isinstance(exc, URLError):
        reason = exc.reason
        if isinstance(reason, socket.timeout):
            return None, "timeout"
        return None, "url_error"
    return None, exc.__class__.__name__


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in OPENMETEO_REQUEST_RETRYABLE_STATUS_CODES
    if isinstance(exc, socket.timeout | TimeoutError):
        return True
    if isinstance(exc, URLError):
        return True
    return False


def _current_throttle_seconds() -> float:
    workflow = current_openmeteo_workflow()
    if workflow == "verification_tuning":
        return max(float(OPENMETEO_REQUEST_THROTTLE_SECONDS), float(OPENMETEO_TUNING_REQUEST_THROTTLE_SECONDS))
    if workflow.startswith("verification"):
        return max(float(OPENMETEO_REQUEST_THROTTLE_SECONDS), float(OPENMETEO_VERIFICATION_REQUEST_THROTTLE_SECONDS))
    return float(OPENMETEO_REQUEST_THROTTLE_SECONDS)


def _maybe_throttle() -> None:
    delay = max(0.0, _current_throttle_seconds())
    if delay <= 0.0:
        return
    last_request_ts = _LAST_REQUEST_TS.get()
    if last_request_ts <= 0.0:
        return
    elapsed = time.monotonic() - last_request_ts
    if elapsed < delay:
        time.sleep(delay - elapsed)


def _retry_after_seconds(exc: Exception) -> float | None:
    if not isinstance(exc, HTTPError) or exc.headers is None:
        return None
    retry_after = exc.headers.get("Retry-After")
    if not retry_after:
        return None
    retry_after = retry_after.strip()
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        return max(0.0, (retry_at - datetime.now(retry_at.tzinfo)).total_seconds())


def _record_request(
    *,
    base_url: str,
    query: dict[str, object],
    started_at: datetime,
    ended_at: datetime,
    outcome: str,
    status_code: int | None,
    error_type: str,
    retry_count: int,
) -> None:
    endpoint_name, endpoint_type = _endpoint_metadata(base_url)
    _REQUEST_RECORDS.append(
        OpenMeteoRequestRecord(
            request_id=len(_REQUEST_RECORDS) + 1,
            workflow=current_openmeteo_workflow(),
            label=current_openmeteo_label(),
            run_slug=_RUN_SLUG_CONTEXT.get(),
            endpoint_name=endpoint_name,
            endpoint_type=endpoint_type,
            started_at=started_at.isoformat(timespec="seconds"),
            ended_at=ended_at.isoformat(timespec="seconds"),
            elapsed_seconds=round((ended_at - started_at).total_seconds(), 3),
            outcome=outcome,
            status_code=status_code,
            error_type=error_type,
            retry_count=retry_count,
            query_summary=_query_summary(query),
        )
    )


def _retry_sleep_seconds(exc: Exception, backoff_seconds: float) -> float:
    sleep_seconds = min(backoff_seconds, float(OPENMETEO_REQUEST_RETRY_BACKOFF_MAX_SECONDS))
    if isinstance(exc, HTTPError) and exc.code == 429:
        retry_after_seconds = _retry_after_seconds(exc)
        sleep_seconds = max(sleep_seconds, retry_after_seconds or float(OPENMETEO_REQUEST_429_BACKOFF_SECONDS))
    jitter = float(OPENMETEO_REQUEST_BACKOFF_JITTER_SECONDS)
    if jitter > 0.0:
        sleep_seconds += random.uniform(0.0, jitter)
    return sleep_seconds


def fetch_with_retries(
    *,
    base_url: str,
    query: dict[str, object],
    request_func,
):
    max_retries = max(0, int(OPENMETEO_REQUEST_MAX_RETRIES))
    backoff_seconds = max(0.0, float(OPENMETEO_REQUEST_RETRY_BACKOFF_INITIAL_SECONDS))

    for attempt in range(max_retries + 1):
        _maybe_throttle()
        started_at = datetime.now()
        try:
            payload = request_func(base_url, query)
            ended_at = datetime.now()
            _LAST_REQUEST_TS.set(time.monotonic())
            _record_request(
                base_url=base_url,
                query=query,
                started_at=started_at,
                ended_at=ended_at,
                outcome="success",
                status_code=200,
                error_type="",
                retry_count=attempt,
            )
            return payload
        except Exception as exc:
            ended_at = datetime.now()
            status_code, error_type = _classify_exception(exc)
            retryable = attempt < max_retries and _is_retryable(exc)
            _record_request(
                base_url=base_url,
                query=query,
                started_at=started_at,
                ended_at=ended_at,
                outcome="retrying" if retryable else "error",
                status_code=status_code,
                error_type=error_type,
                retry_count=attempt,
            )
            if not retryable:
                raise
            time.sleep(_retry_sleep_seconds(exc, backoff_seconds))
            backoff_seconds = min(
                backoff_seconds * float(OPENMETEO_REQUEST_RETRY_BACKOFF_MULTIPLIER),
                float(OPENMETEO_REQUEST_RETRY_BACKOFF_MAX_SECONDS),
            )


def write_openmeteo_request_report(*, output_dir: Path, run_slug: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / f"{run_slug}_openmeteo_request_detail.csv"
    summary_path = output_dir / f"{run_slug}_openmeteo_request_summary.csv"

    frame = pd.DataFrame(asdict(record) for record in _REQUEST_RECORDS)
    if frame.empty:
        pd.DataFrame(
            [
                {
                    "run_slug": run_slug,
                    "total_requests": 0,
                    "successful_requests": 0,
                    "retry_events": 0,
                    "timeouts": 0,
                    "http_429s": 0,
                    "errors": 0,
                    "average_elapsed_seconds": 0.0,
                    "slowest_endpoint_name": "",
                    "slowest_workflow": "",
                    "slowest_label": "",
                }
            ]
        ).to_csv(summary_path, index=False)
        frame.to_csv(detail_path, index=False)
        return summary_path, detail_path

    frame.to_csv(detail_path, index=False)
    success_mask = frame["outcome"] == "success"
    timeout_mask = frame["error_type"] == "timeout"
    rate_limit_mask = frame["status_code"] == 429
    error_mask = frame["outcome"] == "error"
    retry_event_mask = frame["retry_count"] > 0
    slowest = frame.sort_values("elapsed_seconds", ascending=False).iloc[0]

    pd.DataFrame(
        [
            {
                "run_slug": run_slug,
                "total_requests": int(len(frame)),
                "successful_requests": int(success_mask.sum()),
                "retry_events": int(retry_event_mask.sum()),
                "timeouts": int(timeout_mask.sum()),
                "http_429s": int(rate_limit_mask.sum()),
                "errors": int(error_mask.sum()),
                "average_elapsed_seconds": round(float(frame["elapsed_seconds"].mean()), 3),
                "slowest_endpoint_name": slowest["endpoint_name"],
                "slowest_workflow": slowest["workflow"],
                "slowest_label": slowest["label"],
                "slowest_elapsed_seconds": slowest["elapsed_seconds"],
            }
        ]
    ).to_csv(summary_path, index=False)
    return summary_path, detail_path
