"""Tests for the IBKR historical-data adapter.

ib_insync is an optional dep and a live Gateway isn't available in CI, so these
mock the ib client (injected) and stub the `ib_insync` module for the parts that
import it. The live reqHistoricalData round-trip is exercised manually against a
running IB Gateway.
"""

from __future__ import annotations

import sys
import types
from datetime import date

import pytest

from hrp.data.sources.factory import DataSourceFactory
from hrp.data.sources.ibkr_source import (
    IBKRDataSource,
    _bars_to_df,
    _duration_str,
)


class FakeBar:
    def __init__(self, d, o, h, low, c, v):
        self.date = d
        self.open = o
        self.high = h
        self.low = low
        self.close = c
        self.volume = v


def test_bars_to_df_maps_schema():
    bars = [
        FakeBar(date(2026, 6, 25), 10.0, 11.0, 9.5, 10.5, 1000),
        FakeBar(date(2026, 6, 26), 10.5, 12.0, 10.0, 11.8, 2000),
    ]
    df = _bars_to_df("NVDA", bars, "ibkr")
    assert list(df.columns) == [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "source",
    ]
    assert df.iloc[0]["symbol"] == "NVDA"
    assert df.iloc[0]["source"] == "ibkr"
    # IBKR TRADES bars are unadjusted -> adj_close mirrors close
    assert df.iloc[1]["adj_close"] == 11.8 == df.iloc[1]["close"]
    assert df.iloc[1]["volume"] == 2000


def test_bars_to_df_empty():
    assert _bars_to_df("X", [], "ibkr").empty


def test_duration_str():
    assert _duration_str(date(2026, 6, 1), date(2026, 6, 20)) == "19 D"
    assert _duration_str(date(2024, 6, 1), date(2026, 6, 1)) == "2 Y"


def test_factory_registers_ibkr():
    primary, fallback = DataSourceFactory.create("ibkr")
    assert isinstance(primary, IBKRDataSource)
    assert primary.source_name == "ibkr"
    # falls back to yfinance when IBKR is unavailable
    assert fallback is not None and fallback.source_name == "yfinance"


class FakeIB:
    def __init__(self, bars):
        self._bars = bars
        self.connected = True
        self.disconnected = False

    def isConnected(self):
        return self.connected

    def connect(self, *a, **k):
        self.connected = True

    def disconnect(self):
        self.disconnected = True
        self.connected = False

    def reqHistoricalData(self, *a, **k):
        return self._bars


@pytest.fixture
def fake_ib_insync(monkeypatch):
    mod = types.ModuleType("ib_insync")
    mod.Stock = lambda *a, **k: ("Stock", a)
    mod.IB = object
    monkeypatch.setitem(sys.modules, "ib_insync", mod)
    return mod


def test_get_daily_bars_with_injected_client(fake_ib_insync):
    bars = [
        FakeBar(date(2026, 6, 24), 1, 2, 1, 1.5, 10),
        FakeBar(date(2026, 6, 25), 1, 2, 1, 1.6, 20),
        FakeBar(date(2026, 6, 26), 1, 2, 1, 1.7, 30),
    ]
    src = IBKRDataSource(ib=FakeIB(bars), pace_seconds=0)
    df = src.get_daily_bars("AAPL", date(2026, 6, 25), date(2026, 6, 26))
    # clipped to [start, end]
    assert list(df["date"]) == [date(2026, 6, 25), date(2026, 6, 26)]
    assert df.iloc[0]["close"] == 1.6


def test_get_multiple_symbols_concats_and_disconnects(fake_ib_insync):
    bars = [FakeBar(date(2026, 6, 26), 1, 2, 1, 1.7, 30)]
    fake = FakeIB(bars)
    src = IBKRDataSource(ib=fake, pace_seconds=0)
    df = src.get_multiple_symbols(["AAPL", "MSFT"], date(2026, 6, 26), date(2026, 6, 26))
    assert set(df["symbol"]) == {"AAPL", "MSFT"}
    assert fake.disconnected is True  # connection cleaned up
