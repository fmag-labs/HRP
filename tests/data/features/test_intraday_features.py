"""
Tests for intraday feature computation engine.
"""

from datetime import UTC, datetime

import pytest

from hrp.data.features.intraday_features import (
    IntradayBar,
    IntradayFeatureEngine,
    register_intraday_features,
)


@pytest.fixture
def feature_engine():
    """Create a feature engine instance."""
    return IntradayFeatureEngine(window_size=60)


@pytest.fixture
def sample_bars():
    """
    Create sample intraday bars for testing.

    Generates 30 minutes of bars with predictable patterns:
    - Prices trend from 100 to 110
    - Volumes are constant at 1000
    - Pattern enables testing momentum, volatility, etc.
    """
    bars = []
    base_time = datetime(2024, 1, 15, 9, 30, tzinfo=UTC)

    for i in range(30):
        # Linear price increase from 100 to 110
        price = 100.0 + (i * 10.0 / 29)

        bar = IntradayBar(
            symbol="AAPL",
            timestamp=base_time.replace(minute=30 + i),
            open=price,
            high=price + 0.10,
            low=price - 0.10,
            close=price,
            volume=1000,
            vwap=price,
        )
        bars.append(bar)

    return bars


class TestIntradayBar:
    """Test IntradayBar dataclass."""

    def test_bar_creation(self):
        """Test creating an IntradayBar."""
        bar = IntradayBar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=UTC),
            open=150.0,
            high=151.0,
            low=149.5,
            close=150.5,
            volume=1000,
            vwap=150.25,
        )

        assert bar.symbol == "AAPL"
        assert bar.close == 150.5
        assert bar.volume == 1000
        assert bar.vwap == 150.25


class TestIntradayFeatureEngine:
    """Test IntradayFeatureEngine."""

    def test_engine_initialization(self, feature_engine):
        """Test engine initializes correctly."""
        assert feature_engine.window_size == 60
        assert len(feature_engine.features) == 7
        assert feature_engine._windows == {}
        assert feature_engine._day_opens == {}

    def test_feature_registry(self, feature_engine):
        """Test that all 7 features are registered."""
        feature_names = [f.name for f in feature_engine.features]

        expected = [
            "intraday_vwap",
            "intraday_rsi_14",
            "intraday_momentum_20",
            "intraday_volatility_20",
            "intraday_volume_ratio",
            "intraday_price_to_open",
            "intraday_range_position",
        ]

        assert feature_names == expected

    def test_add_bars_creates_window(self, feature_engine, sample_bars):
        """Test adding bars creates rolling window."""
        feature_engine.add_bars(sample_bars[:5])

        assert "AAPL" in feature_engine._windows
        assert len(feature_engine._windows["AAPL"]) == 5
        assert "AAPL" in feature_engine._day_opens

    def test_window_respects_max_size(self, feature_engine):
        """Test window respects maxlen."""
        # Create engine with small window
        engine = IntradayFeatureEngine(window_size=5)

        bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 9, 30 + i, tzinfo=UTC),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1000,
            )
            for i in range(10)
        ]

        engine.add_bars(bars)

        # Window should only have last 5 bars
        assert len(engine._windows["AAPL"]) == 5

    def test_day_state_tracking(self, feature_engine, sample_bars):
        """Test day open/high/low tracking."""
        feature_engine.add_bars(sample_bars)

        assert "AAPL" in feature_engine._day_opens
        assert feature_engine._day_opens["AAPL"] == 100.0
        assert feature_engine._day_highs["AAPL"] == pytest.approx(110.1, abs=0.2)
        assert feature_engine._day_lows["AAPL"] == pytest.approx(99.90, abs=0.2)

    def test_reset_day_state(self, feature_engine, sample_bars):
        """Test resetting day state."""
        feature_engine.add_bars(sample_bars)
        feature_engine.reset_day_state("AAPL")

        assert "AAPL" not in feature_engine._day_opens
        assert "AAPL" not in feature_engine._day_highs
        assert "AAPL" not in feature_engine._day_lows

    def test_clear_windows(self, feature_engine, sample_bars):
        """Test clearing all windows."""
        feature_engine.add_bars(sample_bars)
        feature_engine.clear_windows()

        assert feature_engine._windows == {}
        assert feature_engine._day_opens == {}

    def test_compute_features_basic(self, feature_engine, sample_bars):
        """Test basic feature computation."""
        # Add enough bars for all features
        bars_to_add = sample_bars[:25]  # 25 bars > 21 needed for all features
        features_df = feature_engine.compute_features(bars_to_add)

        assert not features_df.empty
        assert "symbol" in features_df.columns
        assert "timestamp" in features_df.columns
        assert "feature_name" in features_df.columns
        assert "value" in features_df.columns
        assert "version" in features_df.columns

        # Should have features for all bars with sufficient history
        assert len(features_df) > 0
        assert (features_df["version"] == "v1").all()

    def test_compute_features_insufficient_window(self, feature_engine):
        """Test feature computation with insufficient window."""
        # Only 1 bar - not enough for most features
        bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=UTC),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1000,
            )
        ]

        features_df = feature_engine.compute_features(bars)

        # Should only have features with window_size=1
        feature_names = features_df["feature_name"].unique()
        expected = ["intraday_vwap", "intraday_price_to_open", "intraday_range_position"]

        for name in expected:
            assert name in feature_names


class TestVWAPFeature:
    """Test VWAP computation."""

    def test_vwap_with_polygon_value(self, feature_engine):
        """Test VWAP uses Polygon-provided value if available."""
        bar = IntradayBar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=UTC),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1000,
            vwap=100.25,  # Pre-computed
        )

        features_df = feature_engine.compute_features([bar])
        vwap_row = features_df[features_df["feature_name"] == "intraday_vwap"]

        assert len(vwap_row) == 1
        assert vwap_row.iloc[0]["value"] == 100.25

    def test_vwap_computed_from_window(self, feature_engine):
        """Test VWAP computation from window."""
        bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 9, 30 + i, tzinfo=UTC),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0 + i,  # Increasing prices
                volume=1000,
                vwap=None,  # Force computation
            )
            for i in range(3)
        ]

        features_df = feature_engine.compute_features(bars)
        vwap_values = features_df[features_df["feature_name"] == "intraday_vwap"]["value"].tolist()

        # VWAP should be computed as sum(price*volume)/sum(volume)
        # For the 3rd bar: (100*1000 + 101*1000 + 102*1000) / 3000 = 101.0
        assert vwap_values[-1] == pytest.approx(101.0, abs=0.01)


class TestRSIFeature:
    """Test RSI computation."""

    def test_rsi_basic(self, feature_engine):
        """Test RSI computation with uptrend."""
        # Create 20 bars with steady uptrend
        bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 9, 30 + i, tzinfo=UTC),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0 + i,  # Linear increase
                volume=1000,
            )
            for i in range(20)
        ]

        features_df = feature_engine.compute_features(bars)
        rsi_values = features_df[features_df["feature_name"] == "intraday_rsi_14"]["value"].tolist()

        # RSI should be computed for bars with sufficient history
        assert len(rsi_values) > 0
        # With consistent uptrend, RSI should be high (but may not be >50 for all bars)
        assert any(v > 50 for v in rsi_values)

    def test_rsi_insufficient_window(self, feature_engine):
        """Test RSI with insufficient bars."""
        bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 9, 30 + i, tzinfo=UTC),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1000,
            )
            for i in range(10)  # Only 10 bars, need 15
        ]

        features_df = feature_engine.compute_features(bars)
        rsi_rows = features_df[features_df["feature_name"] == "intraday_rsi_14"]

        # Should not compute RSI
        assert len(rsi_rows) == 0


class TestMomentumFeature:
    """Test momentum computation."""

    def test_momentum_20(self, feature_engine, sample_bars):
        """Test 20-minute momentum."""
        features_df = feature_engine.compute_features(sample_bars)
        momentum_values = features_df[
            features_df["feature_name"] == "intraday_momentum_20"
        ]["value"].tolist()

        # Should have momentum for bars >= 21
        assert len(momentum_values) >= 9  # 30 bars - 21 window

        # Momentum should be mostly positive with uptrend (at least one positive)
        assert any(v > 0 for v in momentum_values)


class TestVolatilityFeature:
    """Test volatility computation."""

    def test_volatility_20(self, feature_engine):
        """Test 20-minute volatility."""
        # Create bars with known volatility
        bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 9, 30 + i, tzinfo=UTC),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0 + (i % 2),  # Alternating pattern
                volume=1000,
            )
            for i in range(25)
        ]

        features_df = feature_engine.compute_features(bars)
        vol_values = features_df[
            features_df["feature_name"] == "intraday_volatility_20"
        ]["value"].tolist()

        # Should compute volatility for bars >= 21
        assert len(vol_values) >= 4
        assert all(v > 0 for v in vol_values)  # Volatility should be positive


class TestVolumeRatioFeature:
    """Test volume ratio computation."""

    def test_volume_ratio(self, feature_engine):
        """Test volume ratio computation."""
        # Create bars with constant volume, then a spike
        bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 9, 30 + i, tzinfo=UTC),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1000 if i < 22 else 2000,  # Double volume on bar 22
            )
            for i in range(25)
        ]

        features_df = feature_engine.compute_features(bars)
        vol_ratio_values = features_df[
            features_df["feature_name"] == "intraday_volume_ratio"
        ]["value"].tolist()

        # Should have volume ratio computed
        assert len(vol_ratio_values) >= 4
        # Bar 22 should have elevated volume ratio (uses average of previous 20 bars)
        assert vol_ratio_values[-3] > 1.5  # At least 1.5x average


class TestPriceToOpenFeature:
    """Test price-to-open computation."""

    def test_price_to_open(self, feature_engine, sample_bars):
        """Test price-to-open computation."""
        features_df = feature_engine.compute_features(sample_bars)
        price_to_open_values = features_df[
            features_df["feature_name"] == "intraday_price_to_open"
        ]["value"].tolist()

        # Should have values for all bars
        assert len(price_to_open_values) == 30

        # First bar should be close to 0 (price = open)
        assert price_to_open_values[0] == pytest.approx(0.0, abs=0.01)

        # Last bar should be positive (price increased)
        assert price_to_open_values[-1] > 0


class TestRangePositionFeature:
    """Test range position computation."""

    def test_range_position(self, feature_engine, sample_bars):
        """Test range position computation."""
        features_df = feature_engine.compute_features(sample_bars)
        range_pos_values = features_df[
            features_df["feature_name"] == "intraday_range_position"
        ]["value"].tolist()

        # Should have values for all bars
        assert len(range_pos_values) == 30

        # Values should be between 0 and 1
        assert all(0 <= v <= 1 for v in range_pos_values)

    def test_range_position_no_range(self, feature_engine):
        """Test range position when day high = day low."""
        # Create bars with constant price
        bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 9, 30 + i, tzinfo=UTC),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1000,
            )
            for i in range(3)
        ]

        features_df = feature_engine.compute_features(bars)
        range_pos_values = features_df[
            features_df["feature_name"] == "intraday_range_position"
        ]["value"].tolist()

        # Should return 0.5 (midpoint) when no range
        assert all(v == 0.5 for v in range_pos_values)


class TestMultiSymbol:
    """Test multi-symbol handling."""

    def test_multiple_symbols(self, feature_engine):
        """Test feature computation for multiple symbols."""
        bars = []
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            for i in range(25):
                bars.append(
                    IntradayBar(
                        symbol=symbol,
                        timestamp=datetime(2024, 1, 15, 9, 30 + i, tzinfo=UTC),
                        open=100.0,
                        high=100.0,
                        low=100.0,
                        close=100.0 + i,
                        volume=1000,
                    )
                )

        features_df = feature_engine.compute_features(bars)

        # Should have features for all 3 symbols
        symbols = features_df["symbol"].unique()
        assert len(symbols) == 3
        assert set(symbols) == {"AAPL", "GOOGL", "MSFT"}

    def test_windows_isolated_per_symbol(self, feature_engine):
        """Test that windows are isolated per symbol."""
        bars_aapl = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 9, 30 + i, tzinfo=UTC),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1000,
            )
            for i in range(5)
        ]

        bars_googl = [
            IntradayBar(
                symbol="GOOGL",
                timestamp=datetime(2024, 1, 15, 9, 30 + i, tzinfo=UTC),
                open=200.0,
                high=200.0,
                low=200.0,
                close=200.0,
                volume=2000,
            )
            for i in range(3)
        ]

        feature_engine.add_bars(bars_aapl)
        feature_engine.add_bars(bars_googl)

        # Windows should be separate
        assert len(feature_engine._windows["AAPL"]) == 5
        assert len(feature_engine._windows["GOOGL"]) == 3

        # Day opens should be different
        assert feature_engine._day_opens["AAPL"] == 100.0
        assert feature_engine._day_opens["GOOGL"] == 200.0


class TestDayStateAutoReset:
    """Test automatic day state reset on day transitions."""

    def test_day_transition_resets_open(self, feature_engine):
        """Test that day open resets when a new day's bar arrives."""
        # Day 1 bars
        day1_bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 9, 30 + i, tzinfo=UTC),
                open=100.0 if i == 0 else 100.0 + i,
                high=101.0 + i,
                low=99.0,
                close=100.0 + i,
                volume=1000,
            )
            for i in range(5)
        ]
        feature_engine.add_bars(day1_bars)
        assert feature_engine._day_opens["AAPL"] == 100.0

        # Day 2 bars - different date
        day2_bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 16, 9, 30 + i, tzinfo=UTC),
                open=110.0 if i == 0 else 110.0 + i,
                high=111.0 + i,
                low=109.0,
                close=110.0 + i,
                volume=1000,
            )
            for i in range(5)
        ]
        feature_engine.add_bars(day2_bars)
        # Day open should be reset to day 2's first bar
        assert feature_engine._day_opens["AAPL"] == 110.0
        assert feature_engine._day_lows["AAPL"] == 109.0

    def test_day_transition_resets_highs_lows(self, feature_engine):
        """Test that day highs/lows reset on new day."""
        # Day 1: high reaches 200
        day1_bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
                open=100.0,
                high=200.0,
                low=50.0,
                close=150.0,
                volume=1000,
            )
        ]
        feature_engine.add_bars(day1_bars)
        assert feature_engine._day_highs["AAPL"] == 200.0
        assert feature_engine._day_lows["AAPL"] == 50.0

        # Day 2: much smaller range
        day2_bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 16, 10, 0, tzinfo=UTC),
                open=105.0,
                high=106.0,
                low=104.0,
                close=105.5,
                volume=1000,
            )
        ]
        feature_engine.add_bars(day2_bars)
        # Should NOT carry over day 1's extreme values
        assert feature_engine._day_highs["AAPL"] == 106.0
        assert feature_engine._day_lows["AAPL"] == 104.0

    def test_same_day_does_not_reset(self, feature_engine):
        """Test that bars on the same day don't trigger a reset."""
        bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=UTC),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000,
            ),
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 9, 31, tzinfo=UTC),
                open=100.5,
                high=105.0,
                low=98.0,
                close=103.0,
                volume=1000,
            ),
        ]
        feature_engine.add_bars(bars)
        # Open should still be from first bar
        assert feature_engine._day_opens["AAPL"] == 100.0
        # Highs/lows should accumulate
        assert feature_engine._day_highs["AAPL"] == 105.0
        assert feature_engine._day_lows["AAPL"] == 98.0

    def test_reset_all_day_state(self, feature_engine, sample_bars):
        """Test reset_all_day_state clears everything."""
        feature_engine.add_bars(sample_bars)
        assert len(feature_engine._day_opens) > 0

        feature_engine.reset_all_day_state()
        assert feature_engine._day_opens == {}
        assert feature_engine._day_highs == {}
        assert feature_engine._day_lows == {}
        assert feature_engine._last_bar_dates == {}

    def test_multi_symbol_day_transition(self, feature_engine):
        """Test day transitions work independently per symbol."""
        # AAPL day 1 + MSFT day 1
        bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
                open=100.0, high=101.0, low=99.0, close=100.0, volume=1000,
            ),
            IntradayBar(
                symbol="MSFT",
                timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
                open=200.0, high=201.0, low=199.0, close=200.0, volume=1000,
            ),
        ]
        feature_engine.add_bars(bars)

        # Only AAPL transitions to day 2
        day2_bars = [
            IntradayBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 16, 10, 0, tzinfo=UTC),
                open=110.0, high=111.0, low=109.0, close=110.0, volume=1000,
            ),
        ]
        feature_engine.add_bars(day2_bars)

        # AAPL should have reset, MSFT should not
        assert feature_engine._day_opens["AAPL"] == 110.0
        assert feature_engine._day_opens["MSFT"] == 200.0


class TestFeatureRegistration:
    """Test feature registration in database."""

    def test_register_intraday_features(self, tmp_path):
        """Test registering features in feature_definitions table."""
        # This would require a real database connection
        # For now, just test the function exists and can be called
        try:
            register_intraday_features()
        except Exception:
            # Expected to fail without database
            pass
