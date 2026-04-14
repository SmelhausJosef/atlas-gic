from __future__ import annotations

from datetime import date

from atlas.data.binance import _build_archive_url


def test_build_archive_url():
    url = _build_archive_url("BTCUSDT", date(2026, 4, 13))
    assert url.endswith("/BTCUSDT/BTCUSDT-aggTrades-2026-04-13.zip")
