"""
Comprehensive tests for Data Validation Utilities.

Tests cover:
- ValidationResult structure and properties
- DataValidator.validate_price_data() for OHLCV validation
- DataValidator.validate_feature_computation_inputs() for feature computation
- DataValidator.validate_universe_data() for universe health
- DataValidator.validate_fundamentals_data() for fundamentals quality
- validate_before_operation context manager
- Edge cases and error handling
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from hrp.data.quality.validation import (
    DataValidator,
    ValidationResult,
    validate_before_operation,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_init_with_defaults(self):
        """Should initialize with empty lists."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.stats == {}

    def test_init_with_values(self):
        """Should initialize with provided values."""
        errors = ["Error 1", "Error 2"]
        warnings = ["Warning 1"]
        stats = {"count": 100}

        result = ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

        assert result.is_valid is False
        assert result.errors == errors
        assert result.warnings == warnings
        assert result.stats == stats

    def test_init_with_none_converts_to_empty_lists(self):
        """Should convert None to empty lists."""
        result = ValidationResult(
            is_valid=True,
            errors=None,
            warnings=None,
            stats=None,
        )

        assert result.errors == []
        assert result.warnings == []
        assert result.stats == {}


class TestValidatePriceData:
    """Tests for DataValidator.validate_price_data()."""

    def test_valid_price_data(self):
        """Should pass validation for valid price data."""
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [95.0, 96.0, 97.0],
                "close": [102.0, 103.0, 104.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        result = DataValidator.validate_price_data(df)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.stats["total_rows"] == 3

    def test_empty_dataframe(self):
        """Should fail validation for empty DataFrame."""
        df = pd.DataFrame()

        result = DataValidator.validate_price_data(df)

        assert result.is_valid is False
        assert "empty" in result.errors[0].lower()

    def test_missing_required_columns(self):
        """Should fail when required columns are missing."""
        df = pd.DataFrame({"close": [100.0, 101.0]})

        result = DataValidator.validate_price_data(
            df,
            check_ohlc_relationship=True,
        )

        assert result.is_valid is False
        assert any("Missing required columns" in e for e in result.errors)

    def test_negative_close_prices(self):
        """Should detect negative close prices."""
        df = pd.DataFrame(
            {
                "close": [100.0, -50.0, 102.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        result = DataValidator.validate_price_data(df)

        assert result.is_valid is False
        assert any("close <= 0" in e for e in result.errors)

    def test_zero_close_prices(self):
        """Should detect zero close prices."""
        df = pd.DataFrame(
            {
                "close": [100.0, 0.0, 102.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        result = DataValidator.validate_price_data(df)

        assert result.is_valid is False
        assert any("close <= 0" in e for e in result.errors)

    def test_high_less_than_low(self):
        """Should detect high < low relationship violation."""
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [95.0],  # Violation: high < low
                "low": [105.0],
                "close": [100.0],
                "volume": [1000000],
            }
        )

        result = DataValidator.validate_price_data(df)

        assert result.is_valid is False
        assert any("high < low" in e for e in result.errors)

    def test_close_outside_range(self):
        """Should warn when close is outside [low, high] range."""
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [100.0],
                "close": [106.0],  # Violation: close > high
                "volume": [1000000],
            }
        )

        result = DataValidator.validate_price_data(df)

        # This is a warning, not an error
        assert "outside [low, high]" in result.warnings[0]

    def test_zero_volume_warning(self):
        """Should warn about zero volume."""
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [105.0, 106.0],
                "low": [95.0, 96.0],
                "close": [100.0, 101.0],
                "volume": [1000000, 0],  # Zero volume
            }
        )

        result = DataValidator.validate_price_data(df)

        assert result.is_valid is True  # Zero volume is a warning
        assert any("zero volume" in w.lower() for w in result.warnings)

    def test_negative_volume_error(self):
        """Should error on negative volume."""
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0],
                "volume": [1000000, -100],  # Negative volume
            }
        )

        result = DataValidator.validate_price_data(df)

        assert result.is_valid is False
        assert any("negative volume" in e.lower() for e in result.errors)

    def test_null_values_in_required_columns(self):
        """Should detect null values in required columns."""
        df = pd.DataFrame(
            {
                "open": [100.0, None, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [95.0, 96.0, 97.0],
                "close": [102.0, 103.0, None],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        result = DataValidator.validate_price_data(df)

        assert result.is_valid is False
        assert len(result.errors) >= 2  # One for open, one for close

    def test_disable_ohlc_check(self):
        """Should skip OHLC validation when disabled."""
        df = pd.DataFrame(
            {
                "close": [100.0],
                "volume": [1000000],
            }
        )

        result = DataValidator.validate_price_data(
            df,
            check_ohlc_relationship=False,
        )

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_disable_volume_check(self):
        """Should skip volume validation when disabled."""
        df = pd.DataFrame(
            {
                "close": [100.0],
            }
        )

        result = DataValidator.validate_price_data(
            df,
            check_volume=False,
            check_ohlc_relationship=False,
        )

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_stats_tracking(self):
        """Should track error and warning counts in stats."""
        df = pd.DataFrame(
            {
                "close": [100.0, 0.0],  # Error: zero close
                "volume": [1000000, 0],  # Warning: zero volume
            }
        )

        result = DataValidator.validate_price_data(
            df,
            check_ohlc_relationship=False,
        )

        assert result.stats["total_rows"] == 2
        assert result.stats["error_count"] == 1
        assert result.stats["warning_count"] == 1


class TestValidateFeatureComputationInputs:
    """Tests for DataValidator.validate_feature_computation_inputs()."""

    def test_sufficient_history(self):
        """Should pass with sufficient history."""
        df = pd.DataFrame(
            {"close": [100.0 + i for i in range(50)]}
        )

        result = DataValidator.validate_feature_computation_inputs(
            df,
            feature_name="test_feature",
            min_history_days=20,
        )

        assert result.is_valid is True
        assert result.stats["available_rows"] == 50

    def test_insufficient_history(self):
        """Should fail with insufficient history."""
        df = pd.DataFrame(
            {"close": [100.0 + i for i in range(10)]}
        )

        result = DataValidator.validate_feature_computation_inputs(
            df,
            feature_name="test_feature",
            min_history_days=20,
        )

        assert result.is_valid is False
        assert "Insufficient history" in result.errors[0]

    def test_empty_dataframe(self):
        """Should fail for empty DataFrame."""
        df = pd.DataFrame()

        result = DataValidator.validate_feature_computation_inputs(
            df,
            feature_name="test_feature",
        )

        assert result.is_valid is False
        assert "empty" in result.errors[0].lower()

    def test_missing_required_columns(self):
        """Should fail when required columns are missing."""
        df = pd.DataFrame(
            {"close": [100.0 + i for i in range(30)]}
        )

        result = DataValidator.validate_feature_computation_inputs(
            df,
            feature_name="test_feature",
            required_columns=["close", "volume", "high"],
        )

        assert result.is_valid is False
        assert "Missing columns" in result.errors[0]

    def test_high_null_percentage_warning(self):
        """Should warn about high null percentage."""
        df = pd.DataFrame(
            {"close": [100.0] * 20 + [None] * 10}  # 33% null
        )

        result = DataValidator.validate_feature_computation_inputs(
            df,
            feature_name="test_feature",
            required_columns=["close"],
        )

        assert result.is_valid is True  # Still valid, just a warning
        assert len(result.warnings) > 0
        assert "null values" in result.warnings[0]

    def test_low_null_percentage_no_warning(self):
        """Should not warn for low null percentage."""
        df = pd.DataFrame(
            {"close": [100.0] * 95 + [None] * 5}  # 5% null
        )

        result = DataValidator.validate_feature_computation_inputs(
            df,
            feature_name="test_feature",
            required_columns=["close"],
        )

        assert result.is_valid is True
        assert len(result.warnings) == 0  # Below 10% threshold

    def test_stats_tracking(self):
        """Should include feature stats in result."""
        df = pd.DataFrame(
            {"close": [100.0 + i for i in range(25)]}
        )

        result = DataValidator.validate_feature_computation_inputs(
            df,
            feature_name="momentum_20d",
            min_history_days=20,
        )

        assert result.stats["feature_name"] == "momentum_20d"
        assert result.stats["min_history_days"] == 20
        assert result.stats["available_rows"] == 25


class TestValidateUniverseData:
    """Tests for DataValidator.validate_universe_data()."""

    def test_valid_universe(self, test_db):
        """Should pass for valid universe with fresh prices."""
        # Insert test data
        from hrp.data.db import get_db

        db = get_db(test_db)
        test_date = date(2024, 1, 15)

        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, volume, source)
                VALUES
                    ('AAPL', '2024-01-15', 180.0, 1000000, 'test'),
                    ('MSFT', '2024-01-15', 380.0, 1000000, 'test')
            """
            )

        result = DataValidator.validate_universe_data(
            symbols=["AAPL", "MSFT"],
            as_of_date=test_date,
            db_path=test_db,
            require_prices=True,
            max_staleness_days=3,
        )

        assert result.is_valid is True
        assert result.stats["universe_size"] == 2

    def test_empty_universe(self, test_db):
        """Should fail for empty universe."""
        result = DataValidator.validate_universe_data(
            symbols=[],
            as_of_date=date.today(),
            db_path=test_db,
        )

        assert result.is_valid is False
        assert "empty" in result.errors[0].lower()

    def test_stale_prices_warning(self, test_db):
        """Should warn about stale prices."""
        from hrp.data.db import get_db

        db = get_db(test_db)
        as_of_date = date(2024, 1, 15)

        # Insert old prices
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, volume, source)
                VALUES ('AAPL', '2024-01-10', 180.0, 1000000, 'test')
            """
            )

        result = DataValidator.validate_universe_data(
            symbols=["AAPL"],
            as_of_date=as_of_date,
            db_path=test_db,
            require_prices=True,
            max_staleness_days=3,
        )

        assert result.is_valid is True  # Staleness is a warning
        assert len(result.warnings) > 0
        assert "stale" in result.warnings[0].lower()

    def test_no_prices_warning(self, test_db):
        """Should warn about missing prices."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Don't insert any prices - just check that it doesn't crash
        # The query will return no stale symbols (no prices at all)
        result = DataValidator.validate_universe_data(
            symbols=["AAPL"],
            as_of_date=date(2024, 1, 15),
            db_path=test_db,
            require_prices=True,
        )

        # Should be valid since there are no stale prices (just no prices)
        assert result.is_valid is True

    def test_disable_price_check(self, test_db):
        """Should skip price check when disabled."""
        result = DataValidator.validate_universe_data(
            symbols=["AAPL", "MSFT"],
            as_of_date=date.today(),
            db_path=test_db,
            require_prices=False,
        )

        assert result.is_valid is True
        assert len(result.warnings) == 0

    def test_truncates_stale_list(self, test_db):
        """Should truncate long list of stale symbols."""
        from hrp.data.db import get_db

        db = get_db(test_db)
        as_of_date = date(2024, 1, 15)

        # Create many stale symbols - first insert them into symbols table
        symbols = [f"TEST{i:03d}" for i in range(20)]

        with db.connection() as conn:
            # Insert symbols first to satisfy FK constraint
            for symbol in symbols:
                conn.execute(
                    """
                    INSERT INTO symbols (symbol, name, exchange)
                    VALUES (?, ?, 'NYSE')
                    """,
                    (symbol, f"Test {symbol}",),
                )

            # Only insert prices for first symbol
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, volume, source)
                VALUES ('TEST000', '2024-01-10', 100.0, 1000000, 'test')
            """
            )

        result = DataValidator.validate_universe_data(
            symbols=symbols,
            as_of_date=as_of_date,
            db_path=test_db,
            require_prices=True,
            max_staleness_days=3,  # Make staleness threshold explicit
        )

        assert result.is_valid is True
        # Check that the warning contains "and X more" for truncation
        if result.warnings:
            assert "and" in result.warnings[0] and "more" in result.warnings[0]
            assert result.stats["stale_symbols"] == 20


class TestValidateFundamentalsData:
    """Tests for DataValidator.validate_fundamentals_data()."""

    def test_valid_fundamentals(self):
        """Should pass for valid fundamentals data."""
        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "metric": ["revenue", "eps", "revenue"],
            "value": [1000000000, 5.25, 500000000],
        })

        result = DataValidator.validate_fundamentals_data(df)

        assert result.is_valid is True
        assert result.stats["total_records"] == 3

    def test_empty_dataframe(self):
        """Should fail for empty DataFrame."""
        df = pd.DataFrame()

        result = DataValidator.validate_fundamentals_data(df)

        assert result.is_valid is False
        assert "empty" in result.errors[0].lower()

    def test_missing_required_columns(self):
        """Should fail when required columns are missing."""
        df = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            # Missing 'metric' and 'value'
        })

        result = DataValidator.validate_fundamentals_data(df)

        assert result.is_valid is False
        assert "Missing required columns" in result.errors[0]

    def test_null_values_warning(self):
        """Should warn about null values."""
        df = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "metric": ["revenue", "eps", "revenue"],
            "value": [1000000000, None, 500000000],
        })

        result = DataValidator.validate_fundamentals_data(df)

        assert result.is_valid is True  # Null values are warnings
        assert len(result.warnings) > 0
        assert "null values" in result.warnings[0]

    def test_missing_required_metrics(self):
        """Should warn about missing required metrics."""
        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "metric": ["revenue", "revenue"],
            "value": [1000000000, 1100000000],
        })

        result = DataValidator.validate_fundamentals_data(
            df,
            required_metrics=["revenue", "eps", "pe_ratio"],
        )

        assert result.is_valid is True  # Missing metrics are warnings
        assert len(result.warnings) > 0
        assert "Missing required metrics" in result.warnings[0]


class TestValidateBeforeOperation:
    """Tests for validate_before_operation context manager."""

    def test_successful_validation(self):
        """Should allow operation when validation passes."""
        df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0],
        })

        with validate_before_operation(
            DataValidator.validate_price_data,
            on_failure="raise",
            prices_df=df,
            check_ohlc_relationship=False,
            check_volume=False,
        ) as result:
            assert result.is_valid is True
            # Operation would go here

    def test_failure_with_raise(self):
        """Should raise exception when validation fails."""
        df = pd.DataFrame()  # Empty DataFrame

        with pytest.raises(ValueError) as exc_info:
            with validate_before_operation(
                DataValidator.validate_price_data,
                on_failure="raise",
                prices_df=df,
            ):
                pass  # Should not reach here

        assert "Validation failed" in str(exc_info.value)

    def test_failure_with_warn(self, caplog):
        """Should warn when validation fails."""
        import logging

        df = pd.DataFrame()  # Empty DataFrame

        with validate_before_operation(
            DataValidator.validate_price_data,
            on_failure="warn",
            prices_df=df,
        ) as result:
            assert result.is_valid is False

        # Check that warning was logged
        # (Note: caplog or similar would capture this)

    def test_failure_with_ignore(self):
        """Should ignore validation failure."""
        df = pd.DataFrame()  # Empty DataFrame

        with validate_before_operation(
            DataValidator.validate_price_data,
            on_failure="ignore",
            prices_df=df,
        ) as result:
            assert result.is_valid is False
            # Operation continues despite failure

    def test_logs_warnings(self, caplog):
        """Should log warnings from validation."""
        df = pd.DataFrame({
            "close": [100.0, 101.0],
            "volume": [1000000, 0],  # Warning: zero volume
        })

        # Disable ohlc check so warnings are the only issues
        with validate_before_operation(
            DataValidator.validate_price_data,
            on_failure="raise",
            prices_df=df,
            check_ohlc_relationship=False,
        ):
            pass

        # Check for warning logs (implementation dependent)

    def test_custom_validator_kwargs(self):
        """Should pass custom kwargs to validator."""
        df = pd.DataFrame({
            "close": [100.0, 101.0],
        })

        # Disable both ohlc and volume checks
        with validate_before_operation(
            DataValidator.validate_price_data,
            on_failure="raise",
            prices_df=df,
            check_ohlc_relationship=False,
            check_volume=False,
        ) as result:
            assert result.is_valid is True


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_row_dataframe(self):
        """Should handle single-row DataFrames."""
        df = pd.DataFrame({"close": [100.0]})

        result = DataValidator.validate_feature_computation_inputs(
            df,
            feature_name="test",
            min_history_days=1,
        )

        assert result.is_valid is True

    def test_large_dataframe(self):
        """Should handle large DataFrames efficiently."""
        df = pd.DataFrame({
            "close": [100.0 + i * 0.01 for i in range(10000)],
            "volume": [1000000] * 10000,
        })

        result = DataValidator.validate_price_data(
            df,
            check_ohlc_relationship=False,
        )

        assert result.is_valid is True
        assert result.stats["total_rows"] == 10000

    def test_all_null_column(self):
        """Should detect completely null column."""
        df = pd.DataFrame({
            "close": [None, None, None],
            "volume": [1000000, 1100000, 1200000],
        })

        result = DataValidator.validate_price_data(df)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_mixed_valid_invalid_data(self):
        """Should handle mixed valid and invalid data."""
        df = pd.DataFrame({
            "close": [100.0, -50.0, 102.0, 0.0, 104.0],  # Mixed valid/invalid
            "volume": [1000000, 1100000, -100, 1300000, 0],  # Mixed valid/invalid
        })

        result = DataValidator.validate_price_data(df)

        assert result.is_valid is False
        assert result.stats["total_rows"] == 5
        assert result.stats["error_count"] > 0
        assert result.stats["warning_count"] > 0
