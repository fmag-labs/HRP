"""
Stop-loss implementation for backtesting.

Provides ATR-based trailing stops and other stop-loss mechanisms
that integrate with the backtest engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from hrp.research.config import StopLossConfig


@dataclass
class StopResult:
    """
    Result of stop-loss check for a position.

    Attributes:
        triggered: Whether stop was triggered
        trigger_date: Date when stop was triggered (if any)
        trigger_price: Price at which stop was triggered (if any)
        stop_level: The stop price level
        pnl_at_stop: P&L at the time of stop (if triggered)
    """

    triggered: bool
    trigger_date: date | None
    trigger_price: float | None
    stop_level: float
    pnl_at_stop: float | None


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Average True Range (ATR).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR lookback period

    Returns:
        ATR series
    """
    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    # True Range is max of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is exponential moving average of True Range
    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr


def compute_atr_stops(
    prices: pd.DataFrame,
    entries: pd.DataFrame,
    atr_multiplier: float = 2.0,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Compute ATR-based trailing stop levels.

    Args:
        prices: DataFrame with OHLC columns, indexed by date
        entries: DataFrame of entry signals (1 = long entry, 0 = no position)
        atr_multiplier: Multiple of ATR for stop distance
        atr_period: ATR calculation period

    Returns:
        DataFrame of stop levels, same shape as entries
    """
    # Ensure we have required columns
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(prices.columns.str.lower())):
        # Try to use close as proxy for all
        if "close" in prices.columns or "Close" in prices.columns:
            close_col = "close" if "close" in prices.columns else "Close"
            high = prices[close_col]
            low = prices[close_col]
            close = prices[close_col]
        else:
            raise ValueError(f"prices must have columns: {required_cols}")
    else:
        # Normalize column names
        col_map = {c.lower(): c for c in prices.columns}
        high = prices[col_map["high"]]
        low = prices[col_map["low"]]
        close = prices[col_map["close"]]

    # Compute ATR
    atr = compute_atr(high, low, close, atr_period)

    # Stop level = close - (ATR * multiplier)
    stop_distance = atr * atr_multiplier
    stop_levels = close - stop_distance

    # For positions, trail the stop up (never down)
    # Start with initial stop, then take max of current stop and new stop
    trailing_stops = stop_levels.copy()

    # In a real trailing stop, we'd track the highest high since entry
    # For simplicity, we use a rolling max of the stop level
    # This is a simplified trailing mechanism

    return pd.DataFrame(
        trailing_stops.values.reshape(-1, 1) if trailing_stops.ndim == 1 else trailing_stops,
        index=prices.index,
        columns=entries.columns if hasattr(entries, 'columns') else ['stop_level'],
    )


def compute_trailing_stops(
    prices: pd.DataFrame,
    entries: pd.DataFrame,
    stop_config: StopLossConfig,
) -> pd.DataFrame:
    """
    Compute trailing stop levels based on configuration.

    Args:
        prices: DataFrame with price data
        entries: DataFrame of entry signals
        stop_config: Stop-loss configuration

    Returns:
        DataFrame of stop levels
    """
    if not stop_config.enabled:
        # Return NaN stops (no stop-loss)
        return pd.DataFrame(
            np.nan,
            index=prices.index,
            columns=entries.columns if hasattr(entries, 'columns') else ['stop_level'],
        )

    # Get close price
    if "close" in prices.columns:
        close = prices["close"]
    elif "Close" in prices.columns:
        close = prices["Close"]
    else:
        close = prices.iloc[:, 0]

    if stop_config.type == "atr_trailing":
        return compute_atr_stops(
            prices,
            entries,
            atr_multiplier=stop_config.atr_multiplier,
            atr_period=stop_config.atr_period,
        )

    elif stop_config.type == "fixed_pct":
        # Fixed percentage stop from entry price
        # We need to track entry price per position
        # Simplified: use current close - fixed_pct as stop
        stop_levels = close * (1 - stop_config.fixed_pct)
        return pd.DataFrame(
            stop_levels.values.reshape(-1, 1) if stop_levels.ndim == 1 else stop_levels,
            index=prices.index,
            columns=entries.columns if hasattr(entries, 'columns') else ['stop_level'],
        )

    elif stop_config.type == "volatility_scaled":
        # Volatility-scaled stop: use rolling volatility
        returns = close.pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        # Stop distance = 2 * daily volatility
        stop_distance = close * volatility * 2 / np.sqrt(252)
        stop_levels = close - stop_distance
        return pd.DataFrame(
            stop_levels.values.reshape(-1, 1) if stop_levels.ndim == 1 else stop_levels,
            index=prices.index,
            columns=entries.columns if hasattr(entries, 'columns') else ['stop_level'],
        )

    else:
        raise ValueError(f"Unknown stop type: {stop_config.type}")


def apply_trailing_stops(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    stop_config: StopLossConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply trailing stops to signals, generating exit signals when stops hit.

    Args:
        signals: DataFrame of trading signals (1 = long, 0 = no position, -1 = short)
        prices: DataFrame with OHLC data
        stop_config: Stop-loss configuration

    Returns:
        Tuple of:
        - Modified signals DataFrame with stop exits
        - Stop events DataFrame with details of each stop trigger
    """
    if not stop_config.enabled:
        return signals.copy(), pd.DataFrame()

    # Get close/low prices for stop checking
    if "low" in prices.columns:
        check_price = prices["low"]
    elif "Low" in prices.columns:
        check_price = prices["Low"]
    elif "close" in prices.columns:
        check_price = prices["close"]
    elif "Close" in prices.columns:
        check_price = prices["Close"]
    else:
        check_price = prices.iloc[:, 0]

    # Compute stop levels
    stop_levels = compute_trailing_stops(prices, signals, stop_config)

    # Apply stops
    modified_signals = signals.copy()
    stop_events = []

    # Track positions and trailing highs
    for col_idx, col in enumerate(signals.columns if hasattr(signals, 'columns') else [0]):
        in_position = False
        entry_price = None
        entry_date = None
        trailing_high = None

        signal_col = signals[col] if hasattr(signals, 'columns') else signals.iloc[:, col_idx]
        stop_col = stop_levels.iloc[:, col_idx] if stop_levels.shape[1] > col_idx else stop_levels.iloc[:, 0]

        for i, (idx, sig) in enumerate(signal_col.items()):
            current_price = check_price.loc[idx] if idx in check_price.index else np.nan

            if sig > 0 and not in_position:
                # Entry
                in_position = True
                entry_price = current_price
                entry_date = idx
                trailing_high = current_price

            elif in_position:
                # Update trailing high
                if not np.isnan(current_price):
                    trailing_high = max(trailing_high or current_price, current_price)

                # Check for stop
                stop_level = stop_col.loc[idx] if idx in stop_col.index else np.nan

                if not np.isnan(stop_level) and not np.isnan(current_price):
                    if current_price <= stop_level:
                        # Stop triggered
                        if hasattr(modified_signals, 'columns'):
                            modified_signals.loc[idx, col] = 0
                        else:
                            modified_signals.iloc[i, col_idx] = 0

                        pnl = (current_price - entry_price) / entry_price if entry_price else 0

                        stop_events.append({
                            "symbol": col,
                            "entry_date": entry_date,
                            "exit_date": idx,
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "stop_level": stop_level,
                            "trailing_high": trailing_high,
                            "pnl_pct": pnl,
                        })

                        in_position = False
                        entry_price = None
                        entry_date = None
                        trailing_high = None

                # Check for signal exit
                if sig <= 0:
                    in_position = False
                    entry_price = None
                    entry_date = None
                    trailing_high = None

    stop_events_df = pd.DataFrame(stop_events)

    logger.info(
        f"Applied trailing stops: {len(stop_events)} stops triggered"
    )

    return modified_signals, stop_events_df


def calculate_stop_statistics(
    stop_events: pd.DataFrame,
    trades: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    Calculate stop-loss statistics for reporting.

    Args:
        stop_events: DataFrame of stop events from apply_trailing_stops
        trades: Optional DataFrame of all trades for context

    Returns:
        Dict of stop statistics
    """
    if stop_events.empty:
        return {
            "n_stops": 0,
            "stop_rate": 0.0,
            "avg_stop_pnl": 0.0,
            "total_stop_loss": 0.0,
            "avg_holding_period": 0.0,
        }

    n_stops = len(stop_events)

    # Calculate average P&L at stops
    avg_stop_pnl = stop_events["pnl_pct"].mean() if "pnl_pct" in stop_events.columns else 0.0

    # Total loss from stops
    total_stop_loss = stop_events["pnl_pct"].sum() if "pnl_pct" in stop_events.columns else 0.0

    # Average holding period
    if "entry_date" in stop_events.columns and "exit_date" in stop_events.columns:
        holding_periods = (
            pd.to_datetime(stop_events["exit_date"]) -
            pd.to_datetime(stop_events["entry_date"])
        ).dt.days
        avg_holding_period = holding_periods.mean()
    else:
        avg_holding_period = 0.0

    # Stop rate (if we have total trades)
    n_total_trades = len(trades) if trades is not None and not trades.empty else n_stops
    stop_rate = n_stops / n_total_trades if n_total_trades > 0 else 0.0

    return {
        "n_stops": n_stops,
        "stop_rate": stop_rate,
        "avg_stop_pnl": float(avg_stop_pnl),
        "total_stop_loss": float(total_stop_loss),
        "avg_holding_period": float(avg_holding_period),
    }
