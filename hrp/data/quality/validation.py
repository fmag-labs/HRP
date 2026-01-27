"""
Data validation utilities for HRP.

Provides helper classes and functions for validating data integrity
before and after operations. These utilities complement the quality
check framework by providing pre-operation validation and context
management for safe data operations.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd
from loguru import logger


@dataclass
class ValidationResult:
    """
    Result of a data validation operation.

    Attributes:
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages
        stats: Additional validation statistics
    """

    is_valid: bool
    errors: list[str] | None = None
    warnings: list[str] | None = None
    stats: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize lists if None."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.stats is None:
            self.stats = {}


class DataValidator:
    """
    Static utility class for common data validation operations.

    Provides validation methods for different data types and operations.
    All methods are static and return ValidationResult objects.
    """

    @staticmethod
    def validate_price_data(
        prices_df: pd.DataFrame,
        check_ohlc_relationship: bool = True,
        check_positive_values: bool = True,
        check_volume: bool = True,
    ) -> ValidationResult:
        """
        Validate price data for common data quality issues.

        Args:
            prices_df: DataFrame with price data (OHLCV columns)
            check_ohlc_relationship: Verify high >= low, close within [low, high]
            check_positive_values: Verify close > 0, volume >= 0
            check_volume: Check for zero or null volume

        Returns:
            ValidationResult with any issues found
        """
        errors = []
        warnings = []
        stats = {
            "total_rows": len(prices_df),
        }

        if prices_df.empty:
            return ValidationResult(
                is_valid=False,
                errors=["Price DataFrame is empty"],
                stats=stats,
            )

        # Check for required columns
        required_cols = ["close"]
        if check_ohlc_relationship:
            required_cols.extend(["open", "high", "low"])
        if check_volume:
            required_cols.append("volume")

        missing_cols = [c for c in required_cols if c not in prices_df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Check for null values in required columns
        for col in required_cols:
            if col in prices_df.columns:
                null_count = prices_df[col].isna().sum()
                if null_count > 0:
                    errors.append(f"Column '{col}' has {null_count} null values")

        # Check positive values
        if check_positive_values and "close" in prices_df.columns:
            invalid_close = (prices_df["close"] <= 0).sum()
            if invalid_close > 0:
                errors.append(f"Found {invalid_close} rows with close <= 0")

        # Check OHLC relationships
        if check_ohlc_relationship:
            if all(col in prices_df.columns for col in ["high", "low"]):
                invalid_high_low = (prices_df["high"] < prices_df["low"]).sum()
                if invalid_high_low > 0:
                    errors.append(
                        f"Found {invalid_high_low} rows where high < low"
                    )

            if all(col in prices_df.columns for col in ["open", "high", "low", "close"]):
                # Close should be between low and high (inclusive)
                invalid_close_range = (
                    (prices_df["close"] < prices_df["low"]) |
                    (prices_df["close"] > prices_df["high"])
                ).sum()
                if invalid_close_range > 0:
                    warnings.append(
                        f"Found {invalid_close_range} rows where close is outside [low, high]"
                    )

        # Check volume
        if check_volume and "volume" in prices_df.columns:
            zero_volume = (prices_df["volume"] == 0).sum()
            if zero_volume > 0:
                warnings.append(f"Found {zero_volume} rows with zero volume")

            negative_volume = (prices_df["volume"] < 0).sum()
            if negative_volume > 0:
                errors.append(f"Found {negative_volume} rows with negative volume")

        stats["error_count"] = len(errors)
        stats["warning_count"] = len(warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    @staticmethod
    def validate_feature_computation_inputs(
        prices_df: pd.DataFrame,
        feature_name: str,
        min_history_days: int = 20,
        required_columns: list[str] | None = None,
    ) -> ValidationResult:
        """
        Validate that sufficient data exists for feature computation.

        Args:
            prices_df: Price data for feature computation
            feature_name: Name of feature being computed
            min_history_days: Minimum days of history required
            required_columns: Columns required for computation

        Returns:
            ValidationResult indicating if computation can proceed
        """
        errors = []
        warnings = []
        stats = {
            "feature_name": feature_name,
            "available_rows": len(prices_df),
            "min_history_days": min_history_days,
        }

        if prices_df.empty:
            return ValidationResult(
                is_valid=False,
                errors=[f"Cannot compute {feature_name}: empty price data"],
                stats=stats,
            )

        # Check minimum history
        if len(prices_df) < min_history_days:
            errors.append(
                f"Insufficient history for {feature_name}: "
                f"have {len(prices_df)} rows, need {min_history_days}"
            )

        # Check required columns
        if required_columns:
            missing = [c for c in required_columns if c not in prices_df.columns]
            if missing:
                errors.append(
                    f"Missing columns for {feature_name}: {missing}"
                )

        # Check for null values in key columns
        key_cols = required_columns or ["close"]
        for col in key_cols:
            if col in prices_df.columns:
                null_pct = prices_df[col].isna().mean() * 100
                if null_pct > 10:  # More than 10% null
                    warnings.append(
                        f"Column '{col}' has {null_pct:.1f}% null values "
                        f"for {feature_name}"
                    )

        stats["error_count"] = len(errors)
        stats["warning_count"] = len(warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    @staticmethod
    def validate_universe_data(
        symbols: list[str],
        as_of_date: date,
        db_path: str | None = None,
        require_prices: bool = True,
        max_staleness_days: int = 3,
    ) -> ValidationResult:
        """
        Validate universe data completeness and freshness.

        Args:
            symbols: List of universe symbols to validate
            as_of_date: Date to check freshness against
            db_path: Database path
            require_prices: Check if prices exist for all symbols
            max_staleness_days: Maximum allowed staleness in days

        Returns:
            ValidationResult with universe health status
        """
        from hrp.data.db import get_db

        errors = []
        warnings = []
        stats = {
            "universe_size": len(symbols),
            "as_of_date": str(as_of_date),
        }

        if not symbols:
            return ValidationResult(
                is_valid=False,
                errors=["Universe is empty"],
                stats=stats,
            )

        db = get_db(db_path)

        # Check for price data
        if require_prices:
            staleness_cutoff = as_of_date - timedelta(days=max_staleness_days)

            # Find symbols with no recent prices
            # Use VALUES list to ensure all symbols are checked
            placeholders = ",".join(["(?)"] * len(symbols))
            query = f"""
                WITH symbol_list(symbol) AS (
                    VALUES {placeholders}
                ),
                latest_prices AS (
                    SELECT
                        s.symbol,
                        MAX(p.date) as last_date
                    FROM symbol_list s
                    LEFT JOIN prices p ON s.symbol = p.symbol
                    GROUP BY s.symbol
                )
                SELECT symbol, last_date
                FROM latest_prices
                WHERE last_date IS NULL OR last_date < ?
            """

            results = db.fetchall(query, symbols + [staleness_cutoff])

            stale_symbols = []
            for symbol, last_date in results:
                days_stale = (
                    (as_of_date - last_date).days if last_date else None
                )
                stale_symbols.append(
                    f"{symbol} ({days_stale} days stale)" if days_stale else f"{symbol} (no data)"
                )

            if stale_symbols:
                warnings.append(
                    f"Stale or missing prices for {len(stale_symbols)} symbols: "
                    f"{', '.join(stale_symbols[:10])}"
                    + (f" ... and {len(stale_symbols) - 10} more" if len(stale_symbols) > 10 else "")
                )

                stats["stale_symbols"] = len(stale_symbols)

        stats["error_count"] = len(errors)
        stats["warning_count"] = len(warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    @staticmethod
    def validate_fundamentals_data(
        fundamentals_df: pd.DataFrame,
        required_metrics: list[str] | None = None,
    ) -> ValidationResult:
        """
        Validate fundamentals data for quality issues.

        Args:
            fundamentals_df: DataFrame with fundamentals data
            required_metrics: List of required metric columns

        Returns:
            ValidationResult with fundamentals quality status
        """
        errors = []
        warnings = []
        stats = {
            "total_records": len(fundamentals_df),
        }

        if fundamentals_df.empty:
            return ValidationResult(
                is_valid=False,
                errors=["Fundamentals DataFrame is empty"],
                stats=stats,
            )

        # Check for required columns
        expected_cols = ["symbol", "metric", "value"]
        missing = [c for c in expected_cols if c not in fundamentals_df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")

        # Check for null values
        if "value" in fundamentals_df.columns:
            null_count = fundamentals_df["value"].isna().sum()
            if null_count > 0:
                warnings.append(f"Found {null_count} null values in 'value' column")

        # Check for required metrics
        if required_metrics and "metric" in fundamentals_df.columns:
            available = set(fundamentals_df["metric"].unique())
            missing_metrics = set(required_metrics) - available
            if missing_metrics:
                warnings.append(
                    f"Missing required metrics: {missing_metrics}"
                )

        stats["error_count"] = len(errors)
        stats["warning_count"] = len(warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )


@contextmanager
def validate_before_operation(
    validator: callable,
    on_failure: str = "raise",
    **validator_kwargs,
):
    """
    Context manager for validation before an operation.

    Args:
        validator: Validation function that returns ValidationResult
        on_failure: Action on validation failure ('raise', 'warn', 'ignore')
        **validator_kwargs: Arguments to pass to validator

    Raises:
        ValueError: If validation fails and on_failure='raise'

    Example:
        with validate_before_operation(
            DataValidator.validate_price_data,
            on_failure="raise",
            prices_df=price_data
        ):
            # Operation only executes if validation passes
            ingest_prices(...)
    """
    result = validator(**validator_kwargs)

    if not result.is_valid:
        error_msg = f"Validation failed: {result.errors}"
        if result.warnings:
            error_msg += f" | Warnings: {result.warnings}"

        if on_failure == "raise":
            raise ValueError(error_msg)
        elif on_failure == "warn":
            logger.warning(error_msg)

    yield result

    # Post-operation logging
    if result.warnings:
        logger.warning(f"Validation warnings: {result.warnings}")
    if result.stats:
        logger.debug(f"Validation stats: {result.stats}")
