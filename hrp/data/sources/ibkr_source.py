"""Interactive Brokers historical-data adapter.

Pulls daily OHLCV bars from IBKR via ``ib_insync`` (``reqHistoricalData``). Best
suited to ongoing/daily updates: IBKR paces historical requests (~60 / 10 min),
so a full-universe backfill is slow — one request per symbol with a configurable
pause. Requires IB Gateway / TWS running and logged in, and the optional
``ib-insync`` dependency (``pip install -e ".[trading]"``).

``ib_insync`` is imported lazily inside the methods, so importing this module
(and registering it in the factory) does not require the package to be installed.
"""

from __future__ import annotations

import math
import os
import time
from datetime import date, datetime
from typing import Any

import pandas as pd
from loguru import logger

from hrp.data.sources.base import DataSourceBase

_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]


def _bars_to_df(symbol: str, bars: list[Any], source_name: str) -> pd.DataFrame:
    """Map ib_insync historical bars to the standard schema (pure, testable).

    IBKR ``TRADES`` bars are not back-adjusted, so ``adj_close`` mirrors ``close``.
    """
    rows = []
    for b in bars:
        bar_date = getattr(b, "date", None)
        if isinstance(bar_date, datetime):
            bar_date = bar_date.date()
        close = float(getattr(b, "close"))
        rows.append(
            {
                "symbol": symbol,
                "date": bar_date,
                "open": float(getattr(b, "open")),
                "high": float(getattr(b, "high")),
                "low": float(getattr(b, "low")),
                "close": close,
                "adj_close": close,  # IBKR TRADES bars are unadjusted
                "volume": int(getattr(b, "volume", 0) or 0),
                "source": source_name,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)[_COLUMNS]


def _duration_str(start: date, end: date) -> str:
    """IBKR durationStr covering [start, end] for daily bars."""
    days = max(1, (end - start).days)
    if days <= 365:
        return f"{days} D"
    return f"{math.ceil(days / 365)} Y"


class IBKRDataSource(DataSourceBase):
    """Daily bars from Interactive Brokers (requires IB Gateway/TWS + ib-insync)."""

    source_name = "ibkr"

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        pace_seconds: float | None = None,
        ib: Any | None = None,
    ):
        super().__init__()
        self.host = host or os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("IBKR_PORT", "7497"))
        # Distinct client id from the trading connection so both can run.
        self.client_id = int(client_id or os.getenv("IBKR_DATA_CLIENT_ID", "11"))
        self.pace_seconds = float(
            pace_seconds if pace_seconds is not None else os.getenv("HRP_IBKR_PACE_SECONDS", "11")
        )
        self._ib = ib  # injectable for tests
        logger.info("IBKR data source initialized")

    # -- connection ---------------------------------------------------------
    def _connect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            return
        from ib_insync import IB

        if self._ib is None:
            self._ib = IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id, timeout=15)

    def _disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    # -- data ---------------------------------------------------------------
    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        from ib_insync import Stock

        self._connect()
        contract = Stock(symbol, "SMART", "USD")
        end_dt = datetime(end.year, end.month, end.day, 23, 59, 59)
        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr=_duration_str(start, end),
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            logger.warning(f"No IBKR data for {symbol} {start}..{end}")
            return pd.DataFrame()
        df = _bars_to_df(symbol, bars, self.source_name)
        # IBKR returns the whole duration ending at end_dt; clip to [start, end].
        if not df.empty:
            df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
        return df

    def get_multiple_symbols(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        self._connect()
        frames = []
        try:
            for i, symbol in enumerate(symbols):
                try:
                    df = self.get_daily_bars(symbol, start, end)
                    if not df.empty:
                        frames.append(df)
                except Exception as exc:
                    logger.warning(f"Skipping {symbol}: {exc}")
                if i < len(symbols) - 1 and self.pace_seconds > 0:
                    time.sleep(self.pace_seconds)  # respect IBKR pacing limits
        finally:
            self._disconnect()
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def validate_symbol(self, symbol: str) -> bool:
        try:
            from datetime import timedelta

            today = date.today()
            return not self.get_daily_bars(symbol, today - timedelta(days=7), today).empty
        except Exception:
            return False
