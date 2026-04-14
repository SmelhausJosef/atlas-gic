from __future__ import annotations

import json
import logging
import math
import io
import time
import urllib.parse
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pandas as pd


logger = logging.getLogger(__name__)

BINANCE_FAPI_BASE = "https://fapi.binance.com"
BINANCE_ARCHIVE_BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
AGG_TRADES_WINDOW_MS = 15 * 60 * 1000
PAGE_THROTTLE_SECONDS = 0.12
REQUEST_WEIGHT_LIMIT_PER_MINUTE = 2400
REQUEST_WEIGHT_SAFETY_BUDGET = 1800
AGG_TRADES_REQUEST_WEIGHT = 20
FUNDING_SHARED_LIMIT = 500
FUNDING_SHARED_INTERVAL_SECONDS = 300
RATE_LIMIT_COOLDOWN_SECONDS = 120.0


@dataclass(slots=True)
class RequestWindow:
    start_ms: int
    end_ms: int
    estimated_pages: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": _iso_from_ms(self.start_ms),
            "end": _iso_from_ms(self.end_ms),
            "estimated_pages": self.estimated_pages,
        }


@dataclass(slots=True)
class RequestStats:
    rate_limit_hits: int = 0
    waited_seconds: float = 0.0
    requests_sent: int = 0
    windows_processed: int = 0
    throttled_sleeps: int = 0
    recent_weight_events: deque[tuple[float, int]] = field(default_factory=deque)
    recent_funding_events: deque[float] = field(default_factory=deque)


@dataclass(slots=True)
class BinanceDatasetManifest:
    symbol: str
    hours: int
    output_path: str
    row_count: int
    start: str
    end: str
    notes: str
    request_plan_windows: int
    requests_sent: int
    rate_limit_hits: int
    waited_seconds: float
    throttled_sleeps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "hours": self.hours,
            "output_path": self.output_path,
            "row_count": self.row_count,
            "start": self.start,
            "end": self.end,
            "notes": self.notes,
            "request_plan_windows": self.request_plan_windows,
            "requests_sent": self.requests_sent,
            "rate_limit_hits": self.rate_limit_hits,
            "waited_seconds": round(self.waited_seconds, 2),
            "throttled_sleeps": self.throttled_sleeps,
        }


def fetch_and_save_binance_5s_dataset(symbol: str, hours: int, output_path: Path) -> dict[str, Any]:
    end_time = datetime.now(UTC).replace(microsecond=0)
    start_time = end_time - timedelta(hours=hours)
    request_windows = _build_request_windows(start_time, end_time, AGG_TRADES_WINDOW_MS)
    stats = RequestStats()
    logger.info(
        "Starting Binance aggTrades fetch for %s over %s hours using %s windows.",
        symbol,
        hours,
        len(request_windows),
    )
    agg_trades = _fetch_agg_trades(symbol, request_windows, stats)
    if not agg_trades:
        raise RuntimeError(f"No aggregate trades returned for {symbol}.")
    funding = _fetch_funding_history(symbol, start_time - timedelta(hours=8), end_time, stats)
    bars = _resample_agg_trades_to_5s(agg_trades)
    bars = _attach_funding_rates(bars, funding)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(output_path, index=False)
    manifest = BinanceDatasetManifest(
        symbol=symbol,
        hours=hours,
        output_path=str(output_path),
        row_count=len(bars),
        start=bars["timestamp"].iloc[0],
        end=bars["timestamp"].iloc[-1],
        notes=(
            "Derived from Binance USDⓈ-M Futures aggTrades because official klines only support 1m+. "
            "Funding history is distributed across 5s bars over each funding interval."
        ),
        request_plan_windows=len(request_windows),
        requests_sent=stats.requests_sent,
        rate_limit_hits=stats.rate_limit_hits,
        waited_seconds=stats.waited_seconds,
        throttled_sleeps=stats.throttled_sleeps,
    )
    logger.info(
        "Finished Binance fetch for %s: %s bars, %s requests, %s rate-limit hits, %.2fs waited, %s throttle sleeps.",
        symbol,
        len(bars),
        stats.requests_sent,
        stats.rate_limit_hits,
        stats.waited_seconds,
        stats.throttled_sleeps,
    )
    return manifest.to_dict()


def fetch_and_save_binance_archive_5s_dataset(symbol: str, days: int, output_path: Path) -> dict[str, Any]:
    end_day = datetime.now(UTC).date() - timedelta(days=1)
    start_day = end_day - timedelta(days=days - 1)
    stats = RequestStats()
    bars_per_day: list[pd.DataFrame] = []
    logger.info(
        "Starting Binance archive fetch for %s over %s days (%s -> %s).",
        symbol,
        days,
        start_day.isoformat(),
        end_day.isoformat(),
    )
    for offset in range(days):
        current_day = start_day + timedelta(days=offset)
        archive_url = _build_archive_url(symbol, current_day)
        logger.info("Downloading archive day %s/%s: %s", offset + 1, days, archive_url)
        trades = _download_archive_trades(archive_url, stats)
        bars = _resample_agg_trades_to_5s(trades)
        if bars.empty:
            logger.warning("Archive day %s/%s produced no bars.", offset + 1, days)
            continue
        bars_per_day.append(bars)
        logger.info("Processed archive day %s/%s: %s bars", offset + 1, days, len(bars))

    if not bars_per_day:
        raise RuntimeError(f"No archive bars returned for {symbol}.")

    combined_bars = (
        pd.concat(bars_per_day, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    start_time = pd.to_datetime(combined_bars["timestamp"].iloc[0], utc=True).to_pydatetime()
    end_time = pd.to_datetime(combined_bars["timestamp"].iloc[-1], utc=True).to_pydatetime()
    funding = _fetch_funding_history(symbol, start_time - timedelta(hours=8), end_time, stats)
    combined_bars = _attach_funding_rates(combined_bars, funding)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_bars.to_csv(output_path, index=False)
    manifest = BinanceDatasetManifest(
        symbol=symbol,
        hours=days * 24,
        output_path=str(output_path),
        row_count=len(combined_bars),
        start=combined_bars["timestamp"].iloc[0],
        end=combined_bars["timestamp"].iloc[-1],
        notes=(
            "Derived from Binance USDⓈ-M Futures daily aggTrades archives on data.binance.vision. "
            "Funding history is distributed across 5s bars over each funding interval."
        ),
        request_plan_windows=days,
        requests_sent=stats.requests_sent,
        rate_limit_hits=stats.rate_limit_hits,
        waited_seconds=stats.waited_seconds,
        throttled_sleeps=stats.throttled_sleeps,
    )
    logger.info(
        "Finished Binance archive fetch for %s: %s bars, %s requests, %s rate-limit hits, %.2fs waited, %s throttle sleeps.",
        symbol,
        len(combined_bars),
        stats.requests_sent,
        stats.rate_limit_hits,
        stats.waited_seconds,
        stats.throttled_sleeps,
    )
    return manifest.to_dict()


def _build_request_windows(start_time: datetime, end_time: datetime, window_ms: int) -> list[RequestWindow]:
    windows: list[RequestWindow] = []
    start_ms = _ms(start_time)
    end_ms = _ms(end_time)
    cursor = start_ms
    while cursor < end_ms:
        window_end = min(cursor + window_ms - 1, end_ms)
        windows.append(RequestWindow(start_ms=cursor, end_ms=window_end, estimated_pages=1))
        cursor = window_end + 1
    return windows


def _fetch_agg_trades(symbol: str, request_windows: list[RequestWindow], stats: RequestStats) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total_windows = len(request_windows)
    for window_index, window in enumerate(request_windows, start=1):
        stats.windows_processed += 1
        page_start_ms = window.start_ms
        pages = 0
        before_requests = stats.requests_sent
        before_waited_seconds = stats.waited_seconds
        before_rate_limits = stats.rate_limit_hits
        while page_start_ms <= window.end_ms:
            payload = _request_json(
                "/fapi/v1/aggTrades",
                {
                    "symbol": symbol,
                    "startTime": page_start_ms,
                    "endTime": window.end_ms,
                    "limit": 1000,
                },
                stats,
                request_weight=AGG_TRADES_REQUEST_WEIGHT,
            )
            if not payload:
                break
            records.extend(payload)
            pages += 1
            last_trade_time = int(payload[-1]["T"])
            if len(payload) < 1000 or last_trade_time >= window.end_ms:
                break
            page_start_ms = last_trade_time + 1
            time.sleep(PAGE_THROTTLE_SECONDS)
        window.estimated_pages = max(window.estimated_pages, pages)
        logger.info(
            "Window %s/%s %s -> %s: pages=%s, requests=%s, waited=%.2fs, rate_limits=%s",
            window_index,
            total_windows,
            _iso_from_ms(window.start_ms),
            _iso_from_ms(window.end_ms),
            pages,
            stats.requests_sent - before_requests,
            stats.waited_seconds - before_waited_seconds,
            stats.rate_limit_hits - before_rate_limits,
        )
    return records


def _fetch_funding_history(symbol: str, start_time: datetime, end_time: datetime, stats: RequestStats) -> list[dict[str, Any]]:
    payload = _request_json(
        "/fapi/v1/fundingRate",
        {
            "symbol": symbol,
            "startTime": _ms(start_time),
            "endTime": _ms(end_time),
            "limit": 1000,
        },
        stats,
        request_weight=1,
        enforce_funding_bucket=True,
    )
    return payload if isinstance(payload, list) else []


def _download_archive_trades(archive_url: str, stats: RequestStats) -> list[dict[str, Any]]:
    archive_bytes = _request_bytes(archive_url, stats, request_weight=1)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as handle:
            frame = pd.read_csv(handle)
    renamed = frame.rename(columns={"price": "p", "quantity": "q", "transact_time": "T"})
    return renamed[["T", "p", "q"]].to_dict("records")


def _resample_agg_trades_to_5s(agg_trades: list[dict[str, Any]]) -> pd.DataFrame:
    trades = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([trade["T"] for trade in agg_trades], unit="ms", utc=True),
            "price": pd.Series([float(trade["p"]) for trade in agg_trades], dtype="float64"),
            "quantity": pd.Series([float(trade["q"]) for trade in agg_trades], dtype="float64"),
        }
    ).sort_values("timestamp")
    bars = (
        trades.set_index("timestamp")
        .resample("5s")
        .agg(open=("price", "first"), high=("price", "max"), low=("price", "min"), close=("price", "last"), volume=("quantity", "sum"))
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    bars["timestamp"] = bars["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    bars["funding_rate"] = 0.0
    return bars[["timestamp", "open", "high", "low", "close", "volume", "funding_rate"]]


def _attach_funding_rates(bars: pd.DataFrame, funding_events: list[dict[str, Any]]) -> pd.DataFrame:
    if not funding_events or bars.empty:
        return bars

    enriched = bars.copy()
    enriched["_ts"] = pd.to_datetime(enriched["timestamp"], utc=True)
    funding_frame = pd.DataFrame(funding_events)
    funding_frame["fundingTime"] = pd.to_datetime(funding_frame["fundingTime"], unit="ms", utc=True)
    funding_frame["fundingRate"] = funding_frame["fundingRate"].astype(float)
    funding_frame = funding_frame.sort_values("fundingTime").reset_index(drop=True)

    previous_time = enriched["_ts"].iloc[0]
    for _, event in funding_frame.iterrows():
        event_time = event["fundingTime"]
        interval_start = max(previous_time, enriched["_ts"].iloc[0])
        interval_end = min(event_time, enriched["_ts"].iloc[-1] + pd.Timedelta(seconds=5))
        mask = (enriched["_ts"] >= interval_start) & (enriched["_ts"] < interval_end)
        bar_count = int(mask.sum())
        if bar_count > 0:
            enriched.loc[mask, "funding_rate"] = float(event["fundingRate"]) / bar_count
        previous_time = event_time

    enriched.drop(columns=["_ts"], inplace=True)
    return enriched


def _request_json(
    path: str,
    params: dict[str, Any],
    stats: RequestStats,
    request_weight: int,
    enforce_funding_bucket: bool = False,
) -> Any:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{BINANCE_FAPI_BASE}{path}?{query}",
        headers={"User-Agent": "atlas-gic/0.1"},
    )
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            _wait_for_budget(stats, request_weight, enforce_funding_bucket)
            stats.requests_sent += 1
            with urllib.request.urlopen(request, timeout=30) as response:
                if attempt:
                    time.sleep(0.15)
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429:
                raise
            stats.rate_limit_hits += 1
            retry_after = exc.headers.get("Retry-After")
            sleep_seconds = float(retry_after) if retry_after else RATE_LIMIT_COOLDOWN_SECONDS
            logger.warning(
                "Binance returned 429 for %s. Sleeping %.2fs before retry %s/5.",
                path,
                sleep_seconds,
                attempt + 1,
            )
            stats.waited_seconds += sleep_seconds
            stats.throttled_sleeps += 1
            time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Binance request failed without an explicit HTTP error.")


def _request_bytes(url: str, stats: RequestStats, request_weight: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "atlas-gic/0.1"})
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            _wait_for_budget(stats, request_weight, False)
            stats.requests_sent += 1
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429:
                raise
            stats.rate_limit_hits += 1
            retry_after = exc.headers.get("Retry-After")
            sleep_seconds = float(retry_after) if retry_after else RATE_LIMIT_COOLDOWN_SECONDS
            logger.warning(
                "Binance archive returned 429. Sleeping %.2fs before retry %s/5.",
                sleep_seconds,
                attempt + 1,
            )
            stats.waited_seconds += sleep_seconds
            stats.throttled_sleeps += 1
            time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Binance archive request failed without an explicit HTTP error.")


def _ms(value: datetime) -> int:
    return math.floor(value.timestamp() * 1000)


def _iso_from_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()


def _build_archive_url(symbol: str, day: date) -> str:
    return f"{BINANCE_ARCHIVE_BASE}/{symbol}/{symbol}-aggTrades-{day.isoformat()}.zip"


def _wait_for_budget(stats: RequestStats, request_weight: int, enforce_funding_bucket: bool) -> None:
    now = time.monotonic()
    _trim_weight_events(stats.recent_weight_events, now, 60.0)
    current_weight = sum(weight for _, weight in stats.recent_weight_events)
    if current_weight + request_weight > REQUEST_WEIGHT_SAFETY_BUDGET:
        oldest_ts, _ = stats.recent_weight_events[0]
        sleep_seconds = max(60.0 - (now - oldest_ts), 0.5)
        logger.info(
            "Pausing %.2fs to stay under the Binance request weight safety budget.",
            sleep_seconds,
        )
        time.sleep(sleep_seconds)
        stats.waited_seconds += sleep_seconds
        stats.throttled_sleeps += 1
        now = time.monotonic()
        _trim_weight_events(stats.recent_weight_events, now, 60.0)

    if enforce_funding_bucket:
        _trim_timestamp_events(stats.recent_funding_events, now, FUNDING_SHARED_INTERVAL_SECONDS)
        if len(stats.recent_funding_events) >= FUNDING_SHARED_LIMIT:
            oldest_ts = stats.recent_funding_events[0]
            sleep_seconds = max(FUNDING_SHARED_INTERVAL_SECONDS - (now - oldest_ts), 1.0)
            logger.info(
                "Pausing %.2fs to stay under the shared funding history bucket.",
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
            stats.waited_seconds += sleep_seconds
            stats.throttled_sleeps += 1
            now = time.monotonic()
            _trim_timestamp_events(stats.recent_funding_events, now, FUNDING_SHARED_INTERVAL_SECONDS)
        stats.recent_funding_events.append(now)

    stats.recent_weight_events.append((time.monotonic(), request_weight))


def _trim_weight_events(events: deque[tuple[float, int]], now: float, interval_seconds: float) -> None:
    while events and now - events[0][0] >= interval_seconds:
        events.popleft()


def _trim_timestamp_events(events: deque[float], now: float, interval_seconds: float) -> None:
    while events and now - events[0] >= interval_seconds:
        events.popleft()
