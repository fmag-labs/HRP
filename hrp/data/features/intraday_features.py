"""
Intraday feature computation engine for real-time data.

Computes features at minute granularity from intraday_bars table.
Maintains rolling windows in memory for efficient computation.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

import numpy as np
import pandas as pd
from loguru import logger

from hrp.data.db import get_db


@dataclass
class IntradayBar:
    """Represents a single minute bar."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None


@dataclass
class FeatureSpec:
    """Specification for an intraday feature."""

    name: str
    description: str
    window_size: int  # Number of bars needed for computation
    compute_fn: callable


class IntradayFeatureEngine:
    """
    Computes real-time features from intraday bars using rolling windows.

    Architecture:
    - Maintains per-symbol rolling windows of last 60 bars in memory
    - Computes 7 real-time features on each flush
    - Batch writes results to intraday_features table
    - Thread-safe for concurrent ingestion

    Features computed:
    1. intraday_vwap - Running VWAP for the day
    2. intraday_rsi_14 - 14-bar RSI on minute data
    3. intraday_momentum_20 - 20-minute momentum
    4. intraday_volatility_20 - 20-minute realized volatility
    5. intraday_volume_ratio - Current volume vs 20-bar average
    6. intraday_price_to_open - Current price / day's open
    7. intraday_range_position - Position within day's range
    """

    def __init__(self, window_size: int = 60, db_path: str | None = None):
        """
        Initialize the feature engine.

        Args:
            window_size: Number of bars to keep in rolling window (default 60)
            db_path: Path to DuckDB database (optional)
        """
        self.window_size = window_size
        self.db_path = db_path
        self._windows: dict[str, deque[IntradayBar]] = {}
        self._day_opens: dict[str, float] = {}  # Track day's opening price per symbol
        self._day_highs: dict[str, float] = {}  # Track day's high per symbol
        self._day_lows: dict[str, float] = {}  # Track day's low per symbol
        self._lock = Lock()

        # Feature registry
        self.features = [
            FeatureSpec(
                name="intraday_vwap",
                description="Running VWAP for the trading day",
                window_size=1,
                compute_fn=self._compute_vwap,
            ),
            FeatureSpec(
                name="intraday_rsi_14",
                description="14-bar RSI on minute data",
                window_size=15,
                compute_fn=self._compute_rsi_14,
            ),
            FeatureSpec(
                name="intraday_momentum_20",
                description="20-minute momentum (trailing return)",
                window_size=21,
                compute_fn=self._compute_momentum_20,
            ),
            FeatureSpec(
                name="intraday_volatility_20",
                description="20-minute realized volatility (annualized)",
                window_size=21,
                compute_fn=self._compute_volatility_20,
            ),
            FeatureSpec(
                name="intraday_volume_ratio",
                description="Current bar volume vs 20-bar average",
                window_size=21,
                compute_fn=self._compute_volume_ratio,
            ),
            FeatureSpec(
                name="intraday_price_to_open",
                description="Current price relative to day's open",
                window_size=1,
                compute_fn=self._compute_price_to_open,
            ),
            FeatureSpec(
                name="intraday_range_position",
                description="Position within day's high-low range",
                window_size=1,
                compute_fn=self._compute_range_position,
            ),
        ]

    def add_bars(self, bars: list[IntradayBar]) -> None:
        """
        Add new bars to the rolling windows.

        Args:
            bars: List of intraday bars to add
        """
        with self._lock:
            for bar in bars:
                if bar.symbol not in self._windows:
                    self._windows[bar.symbol] = deque(maxlen=self.window_size)
                    self._day_opens[bar.symbol] = bar.open
                    self._day_highs[bar.symbol] = bar.high
                    self._day_lows[bar.symbol] = bar.low

                # Update day highs/lows
                self._day_highs[bar.symbol] = max(self._day_highs[bar.symbol], bar.high)
                self._day_lows[bar.symbol] = min(self._day_lows[bar.symbol], bar.low)

                # Add to window
                self._windows[bar.symbol].append(bar)

    def compute_features(self, bars: list[IntradayBar]) -> pd.DataFrame:
        """
        Compute all features for the given bars.

        This method:
        1. Adds bars to rolling windows
        2. Computes features for each bar
        3. Returns DataFrame ready for batch insert

        Args:
            bars: List of new intraday bars

        Returns:
            DataFrame with columns: symbol, timestamp, feature_name, value, version
        """
        # Add bars to windows first
        self.add_bars(bars)

        features_list = []

        with self._lock:
            for bar in bars:
                if bar.symbol not in self._windows:
                    continue

                window = list(self._windows[bar.symbol])
                current_bar_idx = len(window) - 1

                # Compute each feature
                for feature_spec in self.features:
                    # Check if we have enough bars
                    if len(window) < feature_spec.window_size:
                        continue

                    try:
                        value = feature_spec.compute_fn(window, current_bar_idx, bar)
                        if value is not None and not np.isnan(value):
                            features_list.append(
                                {
                                    "symbol": bar.symbol,
                                    "timestamp": bar.timestamp,
                                    "feature_name": feature_spec.name,
                                    "value": float(value),
                                    "version": "v1",
                                }
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to compute {feature_spec.name} for {bar.symbol} "
                            f"at {bar.timestamp}: {e}"
                        )

        return pd.DataFrame(features_list)

    def persist_features(self, features_df: pd.DataFrame) -> int:
        """
        Persist computed features to the database using batch upsert.

        Args:
            features_df: DataFrame with computed features

        Returns:
            Number of feature rows written
        """
        if features_df.empty:
            return 0

        db = get_db(self.db_path)

        # Use temp table upsert pattern (same as intraday_bars ingestion)
        with db.connection() as conn:
            # Create temporary table
            conn.execute("""
                CREATE TEMPORARY TABLE temp_intraday_features (
                    symbol VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    feature_name VARCHAR NOT NULL,
                    value DECIMAL(24,6),
                    version VARCHAR DEFAULT 'v1'
                )
            """)

            # Insert into temp table
            conn.execute(
                "INSERT INTO temp_intraday_features SELECT * FROM features_df"
            )

            # Upsert into main table
            conn.execute("""
                INSERT INTO intraday_features
                SELECT * FROM temp_intraday_features
                ON CONFLICT (symbol, timestamp, feature_name, version)
                DO UPDATE SET value = EXCLUDED.value, computed_at = CURRENT_TIMESTAMP
            """)

            # Drop temp table
            conn.execute("DROP TABLE temp_intraday_features")

        return len(features_df)

    def reset_day_state(self, symbol: str) -> None:
        """
        Reset per-symbol day state (opens, highs, lows).
        Call this at market open or when starting a new trading day.

        Args:
            symbol: Symbol to reset
        """
        with self._lock:
            if symbol in self._day_opens:
                del self._day_opens[symbol]
            if symbol in self._day_highs:
                del self._day_highs[symbol]
            if symbol in self._day_lows:
                del self._day_lows[symbol]

    def clear_windows(self) -> None:
        """Clear all rolling windows. Useful for testing or market close."""
        with self._lock:
            self._windows.clear()
            self._day_opens.clear()
            self._day_highs.clear()
            self._day_lows.clear()

    # =========================================================================
    # Feature Computation Functions
    # =========================================================================

    def _compute_vwap(
        self, window: list[IntradayBar], current_idx: int, bar: IntradayBar
    ) -> float | None:
        """
        Running VWAP for the day.

        VWAP = Sum(Price * Volume) / Sum(Volume) from market open to current bar
        """
        if bar.vwap is not None:
            # Use pre-computed VWAP from Polygon if available
            return bar.vwap

        # Compute from window (assumes window starts at market open for this symbol)
        total_pv = sum(b.close * b.volume for b in window if b.volume > 0)
        total_volume = sum(b.volume for b in window if b.volume > 0)

        if total_volume == 0:
            return None

        return total_pv / total_volume

    def _compute_rsi_14(
        self, window: list[IntradayBar], current_idx: int, bar: IntradayBar
    ) -> float | None:
        """
        14-bar RSI on minute data.

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over 14 bars
        """
        if len(window) < 15:
            return None

        # Get last 15 bars (14 changes + 1 initial)
        recent = window[-15:]
        prices = [b.close for b in recent]

        # Calculate changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [c if c > 0 else 0 for c in changes]
        losses = [-c if c < 0 else 0 for c in changes]

        # Average gain and loss
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14

        if avg_loss == 0:
            return 100.0  # No losses = max RSI

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _compute_momentum_20(
        self, window: list[IntradayBar], current_idx: int, bar: IntradayBar
    ) -> float | None:
        """
        20-minute momentum (trailing return).

        Momentum = (Current Price / Price 20 bars ago) - 1
        """
        if len(window) < 21:
            return None

        current_price = bar.close
        past_price = window[-21].close

        if past_price == 0:
            return None

        return (current_price / past_price) - 1.0

    def _compute_volatility_20(
        self, window: list[IntradayBar], current_idx: int, bar: IntradayBar
    ) -> float | None:
        """
        20-minute realized volatility (annualized).

        Vol = StdDev(Returns) * sqrt(252 * 390) where 390 = minutes per trading day
        """
        if len(window) < 21:
            return None

        # Get last 21 bars
        recent = window[-21:]
        prices = [b.close for b in recent]

        # Calculate returns
        returns = [(prices[i] / prices[i - 1]) - 1 for i in range(1, len(prices))]

        # Calculate standard deviation
        if len(returns) < 2:
            return None

        vol = np.std(returns, ddof=1) * np.sqrt(252 * 390)

        return vol

    def _compute_volume_ratio(
        self, window: list[IntradayBar], current_idx: int, bar: IntradayBar
    ) -> float | None:
        """
        Current bar volume vs 20-bar average volume.

        VolumeRatio = Current Volume / Average(Last 20 Volumes)
        """
        if len(window) < 21:
            return None

        current_volume = bar.volume
        avg_volume = sum(b.volume for b in window[-21:-1]) / 20

        if avg_volume == 0:
            return None

        return current_volume / avg_volume

    def _compute_price_to_open(
        self, window: list[IntradayBar], current_idx: int, bar: IntradayBar
    ) -> float | None:
        """
        Current price relative to day's open.

        PriceToOpen = (Current Price / Day Open) - 1
        """
        if bar.symbol not in self._day_opens:
            return None

        day_open = self._day_opens[bar.symbol]
        if day_open == 0:
            return None

        return (bar.close / day_open) - 1.0

    def _compute_range_position(
        self, window: list[IntradayBar], current_idx: int, bar: IntradayBar
    ) -> float | None:
        """
        Position within day's high-low range.

        RangePosition = (Close - Day Low) / (Day High - Day Low)
        Returns 0-1 where 0 = at day low, 1 = at day high, 0.5 = midpoint
        """
        if bar.symbol not in self._day_highs or bar.symbol not in self._day_lows:
            return None

        day_high = self._day_highs[bar.symbol]
        day_low = self._day_lows[bar.symbol]

        if day_high == day_low:
            return 0.5  # No range = midpoint

        return (bar.close - day_low) / (day_high - day_low)


def register_intraday_features(db_path: str | None = None) -> None:
    """
    Register intraday features in the feature_definitions table.

    This should be called once during setup or migration.

    Args:
        db_path: Path to DuckDB database (optional)
    """
    db = get_db(db_path)

    engine = IntradayFeatureEngine()
    feature_records = []

    for feature_spec in engine.features:
        feature_records.append(
            {
                "feature_name": feature_spec.name,
                "version": "v1",
                "computation_code": feature_spec.compute_fn.__name__,
                "description": feature_spec.description,
                "is_active": True,
            }
        )

    # Insert into feature_definitions (use INSERT OR IGNORE pattern)
    with db.connection() as conn:
        # DuckDB can query pandas DataFrames by name - features_df is used in SQL
        features_df = pd.DataFrame(feature_records)  # noqa: F841
        conn.execute("""
            INSERT INTO feature_definitions (feature_name, version, computation_code, description, is_active)
            SELECT feature_name, version, computation_code, description, is_active
            FROM features_df
            ON CONFLICT (feature_name, version) DO NOTHING
        """)

    logger.info(f"Registered {len(feature_records)} intraday features")
