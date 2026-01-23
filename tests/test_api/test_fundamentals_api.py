"""
Tests for the fundamentals API methods.

Tests cover:
- Input validation (empty inputs, future dates)
- Point-in-time correctness (only returns data available by as_of_date)
- Latest value selection (when multiple reports available)
- Missing data handling (some symbols have data, others don't)
- Multiple symbols and metrics
- Integration with backtest
"""

import os
import tempfile
from datetime import date, timedelta

import pandas as pd
import pytest

from hrp.api.platform import PlatformAPI
from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_db():
    """Create a temporary DuckDB database with schema for testing."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    # Delete the empty file so DuckDB can create a fresh database
    os.remove(db_path)

    # Reset the singleton to ensure fresh state
    DatabaseManager.reset()

    # Initialize schema
    create_tables(db_path)

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if os.path.exists(db_path):
        os.remove(db_path)
    # Also remove any wal/tmp files
    for ext in [".wal", ".tmp", "-journal", "-shm"]:
        tmp_file = db_path + ext
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


@pytest.fixture
def test_api(test_db):
    """Create a PlatformAPI instance with a test database."""
    return PlatformAPI(db_path=test_db)


@pytest.fixture
def populated_db(test_api):
    """
    Populate the test database with sample fundamentals data.

    Data setup:
    - AAPL: Revenue reported on 2023-01-10 (period_end 2022-12-31)
    - AAPL: Revenue reported on 2023-04-10 (period_end 2023-03-31)
    - AAPL: EPS reported on 2023-01-10 (period_end 2022-12-31)
    - MSFT: Revenue reported on 2023-01-15 (period_end 2022-12-31)
    - GOOGL: No fundamentals data

    Returns the API instance for convenience.
    """
    # Insert symbols first (needed for foreign key constraints)
    test_api._db.execute(
        """
        INSERT INTO symbols (symbol, name, exchange)
        VALUES
            ('AAPL', 'Apple Inc.', 'NASDAQ'),
            ('MSFT', 'Microsoft Corporation', 'NASDAQ'),
            ('GOOGL', 'Alphabet Inc.', 'NASDAQ')
        """
    )

    # Insert sample universe data
    test_api._db.execute(
        """
        INSERT INTO universe (symbol, date, in_universe, sector, market_cap)
        VALUES
            ('AAPL', '2023-01-01', TRUE, 'Technology', 2500000000000),
            ('MSFT', '2023-01-01', TRUE, 'Technology', 2400000000000),
            ('GOOGL', '2023-01-01', TRUE, 'Technology', 1500000000000),
            ('AAPL', '2023-06-01', TRUE, 'Technology', 2800000000000),
            ('MSFT', '2023-06-01', TRUE, 'Technology', 2600000000000),
            ('GOOGL', '2023-06-01', TRUE, 'Technology', 1600000000000)
        """
    )

    # Insert sample fundamentals data
    # Note: report_date is when the data becomes public, period_end is the fiscal period
    test_api._db.execute(
        """
        INSERT INTO fundamentals (symbol, report_date, period_end, metric, value)
        VALUES
            -- AAPL Q4 2022 (reported Jan 10, 2023)
            ('AAPL', '2023-01-10', '2022-12-31', 'revenue', 117154000000),
            ('AAPL', '2023-01-10', '2022-12-31', 'eps', 1.88),
            ('AAPL', '2023-01-10', '2022-12-31', 'book_value', 50672000000),
            -- AAPL Q1 2023 (reported Apr 10, 2023)
            ('AAPL', '2023-04-10', '2023-03-31', 'revenue', 94836000000),
            ('AAPL', '2023-04-10', '2023-03-31', 'eps', 1.52),
            -- MSFT Q4 2022 (reported Jan 15, 2023)
            ('MSFT', '2023-01-15', '2022-12-31', 'revenue', 52747000000),
            ('MSFT', '2023-01-15', '2022-12-31', 'eps', 2.32)
        """
    )

    return test_api


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestGetFundamentalsAsOfValidation:
    """Tests for input validation in get_fundamentals_as_of."""

    def test_empty_symbols_list_raises(self, test_api):
        """Empty symbols list should raise ValueError."""
        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            test_api.get_fundamentals_as_of(
                symbols=[],
                metrics=["revenue"],
                as_of_date=date(2023, 1, 15),
            )

    def test_empty_metrics_list_raises(self, test_api):
        """Empty metrics list should raise ValueError."""
        with pytest.raises(ValueError, match="metrics list cannot be empty"):
            test_api.get_fundamentals_as_of(
                symbols=["AAPL"],
                metrics=[],
                as_of_date=date(2023, 1, 15),
            )

    def test_future_date_raises(self, test_api):
        """Future as_of_date should raise ValueError."""
        future_date = date.today() + timedelta(days=1)
        with pytest.raises(ValueError, match="as_of_date cannot be in the future"):
            test_api.get_fundamentals_as_of(
                symbols=["AAPL"],
                metrics=["revenue"],
                as_of_date=future_date,
            )

    def test_today_date_is_valid(self, populated_db):
        """Today's date should be valid (not in future)."""
        # Should not raise
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL"],
            metrics=["revenue"],
            as_of_date=date.today(),
        )
        assert isinstance(result, pd.DataFrame)


# =============================================================================
# No Data Available Tests
# =============================================================================


class TestGetFundamentalsAsOfNoData:
    """Tests for handling cases where no data is available."""

    def test_no_fundamentals_returns_empty_dataframe(self, test_api):
        """When no fundamentals exist, return empty DataFrame."""
        # Insert symbol to universe so validation passes
        test_api._db.execute(
            "INSERT INTO symbols (symbol, name, exchange) VALUES ('AAPL', 'Apple Inc.', 'NASDAQ')"
        )
        test_api._db.execute(
            """
            INSERT INTO universe (symbol, date, in_universe, sector, market_cap)
            VALUES ('AAPL', '2023-01-01', TRUE, 'Technology', 2500000000000)
            """
        )

        result = test_api.get_fundamentals_as_of(
            symbols=["AAPL"],
            metrics=["revenue"],
            as_of_date=date(2023, 1, 15),
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_no_data_before_as_of_date(self, populated_db):
        """When fundamentals exist but not before as_of_date, return empty."""
        # Query for date before any data was reported
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL"],
            metrics=["revenue"],
            as_of_date=date(2023, 1, 5),  # Before Jan 10 report
        )

        assert result.empty

    def test_symbol_with_no_data_returns_empty(self, populated_db):
        """Symbol with no fundamentals returns empty DataFrame."""
        # GOOGL has no fundamentals data in the fixture
        result = populated_db.get_fundamentals_as_of(
            symbols=["GOOGL"],
            metrics=["revenue"],
            as_of_date=date(2023, 6, 1),
        )

        assert result.empty


# =============================================================================
# Point-in-Time Correctness Tests
# =============================================================================


class TestGetFundamentalsAsOfPointInTime:
    """Tests for point-in-time correctness to prevent look-ahead bias."""

    def test_only_returns_data_available_by_as_of_date(self, populated_db):
        """Only returns fundamentals where report_date <= as_of_date."""
        # Query on Jan 12 - should only see AAPL data (reported Jan 10)
        # Not MSFT (reported Jan 15)
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL", "MSFT"],
            metrics=["revenue"],
            as_of_date=date(2023, 1, 12),
        )

        assert not result.empty
        assert len(result) == 1  # Only AAPL
        assert result.iloc[0]["symbol"] == "AAPL"

    def test_returns_data_on_exact_report_date(self, populated_db):
        """Data is available on the exact report_date."""
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL"],
            metrics=["revenue"],
            as_of_date=date(2023, 1, 10),  # Exact report date
        )

        assert len(result) == 1
        assert result.iloc[0]["value"] == 117154000000

    def test_latest_report_returned_when_multiple_exist(self, populated_db):
        """When multiple reports exist, return the most recent available."""
        # Query on May 1 - AAPL has both Q4 2022 (Jan 10) and Q1 2023 (Apr 10)
        # Should return Q1 2023 data
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL"],
            metrics=["revenue"],
            as_of_date=date(2023, 5, 1),
        )

        assert len(result) == 1
        assert result.iloc[0]["value"] == 94836000000  # Q1 2023 revenue

    def test_earlier_report_returned_when_querying_before_later_report(self, populated_db):
        """Query before later report returns earlier report."""
        # Query on March 1 - only Q4 2022 data should be available
        # Q1 2023 wasn't reported until Apr 10
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL"],
            metrics=["revenue"],
            as_of_date=date(2023, 3, 1),
        )

        assert len(result) == 1
        assert result.iloc[0]["value"] == 117154000000  # Q4 2022 revenue


# =============================================================================
# Multiple Symbols/Metrics Tests
# =============================================================================


class TestGetFundamentalsAsOfMultiple:
    """Tests for handling multiple symbols and metrics."""

    def test_single_symbol_single_metric(self, populated_db):
        """Basic case: single symbol, single metric."""
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL"],
            metrics=["revenue"],
            as_of_date=date(2023, 1, 15),
        )

        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"
        assert result.iloc[0]["metric"] == "revenue"
        assert result.iloc[0]["value"] == 117154000000

    def test_single_symbol_multiple_metrics(self, populated_db):
        """Single symbol with multiple metrics."""
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL"],
            metrics=["revenue", "eps", "book_value"],
            as_of_date=date(2023, 1, 15),
        )

        assert len(result) == 3
        metrics = set(result["metric"].tolist())
        assert metrics == {"revenue", "eps", "book_value"}

    def test_multiple_symbols_single_metric(self, populated_db):
        """Multiple symbols with single metric."""
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL", "MSFT"],
            metrics=["revenue"],
            as_of_date=date(2023, 1, 20),  # Both have data by this date
        )

        assert len(result) == 2
        symbols = set(result["symbol"].tolist())
        assert symbols == {"AAPL", "MSFT"}

    def test_multiple_symbols_multiple_metrics(self, populated_db):
        """Multiple symbols with multiple metrics."""
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL", "MSFT"],
            metrics=["revenue", "eps"],
            as_of_date=date(2023, 1, 20),
        )

        # AAPL has both metrics, MSFT has both metrics
        assert len(result) == 4

    def test_mixed_data_availability(self, populated_db):
        """Some symbols have data, others don't."""
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL", "GOOGL"],  # GOOGL has no fundamentals
            metrics=["revenue"],
            as_of_date=date(2023, 1, 15),
        )

        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"


# =============================================================================
# DataFrame Output Tests
# =============================================================================


class TestGetFundamentalsAsOfOutput:
    """Tests for DataFrame output format and content."""

    def test_returns_dataframe(self, populated_db):
        """Method returns a pandas DataFrame."""
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL"],
            metrics=["revenue"],
            as_of_date=date(2023, 1, 15),
        )

        assert isinstance(result, pd.DataFrame)

    def test_dataframe_has_expected_columns(self, populated_db):
        """DataFrame has required columns."""
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL"],
            metrics=["revenue"],
            as_of_date=date(2023, 1, 15),
        )

        required_columns = ["symbol", "metric", "value", "report_date", "period_end"]
        for col in required_columns:
            assert col in result.columns

    def test_report_date_is_lte_as_of_date(self, populated_db):
        """All returned report_dates are <= as_of_date."""
        as_of = date(2023, 3, 15)
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL", "MSFT"],
            metrics=["revenue", "eps"],
            as_of_date=as_of,
        )

        for _, row in result.iterrows():
            report_date = row["report_date"]
            # Convert if needed
            if hasattr(report_date, "date"):
                report_date = report_date.date()
            assert report_date <= as_of


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestGetFundamentalsAsOfEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_metric_not_available_for_symbol(self, populated_db):
        """Metric exists for one symbol but not another."""
        # book_value only exists for AAPL, not MSFT
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL", "MSFT"],
            metrics=["book_value"],
            as_of_date=date(2023, 1, 20),
        )

        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"
        assert result.iloc[0]["metric"] == "book_value"

    def test_nonexistent_metric_returns_empty(self, populated_db):
        """Querying for a metric that doesn't exist returns empty."""
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL"],
            metrics=["nonexistent_metric"],
            as_of_date=date(2023, 6, 1),
        )

        assert result.empty

    def test_same_symbol_different_metrics_same_report_date(self, populated_db):
        """Multiple metrics from the same report are all returned."""
        result = populated_db.get_fundamentals_as_of(
            symbols=["AAPL"],
            metrics=["revenue", "eps"],
            as_of_date=date(2023, 1, 15),
        )

        assert len(result) == 2
        # Both should have same report_date
        report_dates = result["report_date"].unique()
        assert len(report_dates) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestGetFundamentalsAsOfIntegration:
    """Integration tests for fundamentals with backtest workflow."""

    def test_point_in_time_correctness_across_date_range(self, populated_db):
        """Verify point-in-time correctness across multiple dates."""
        # Simulate querying fundamentals on different backtest dates

        # Day 1: Before any data
        result_day1 = populated_db.get_fundamentals_as_of(
            symbols=["AAPL"],
            metrics=["revenue"],
            as_of_date=date(2023, 1, 5),
        )
        assert result_day1.empty

        # Day 2: After AAPL Q4 2022 but before MSFT
        result_day2 = populated_db.get_fundamentals_as_of(
            symbols=["AAPL", "MSFT"],
            metrics=["revenue"],
            as_of_date=date(2023, 1, 12),
        )
        assert len(result_day2) == 1
        assert result_day2.iloc[0]["symbol"] == "AAPL"

        # Day 3: After MSFT Q4 2022
        result_day3 = populated_db.get_fundamentals_as_of(
            symbols=["AAPL", "MSFT"],
            metrics=["revenue"],
            as_of_date=date(2023, 1, 20),
        )
        assert len(result_day3) == 2

        # Day 4: After AAPL Q1 2023
        result_day4 = populated_db.get_fundamentals_as_of(
            symbols=["AAPL"],
            metrics=["revenue"],
            as_of_date=date(2023, 4, 15),
        )
        assert len(result_day4) == 1
        # Should show Q1 2023 data, not Q4 2022
        assert result_day4.iloc[0]["value"] == 94836000000

    def test_fundamentals_usable_in_backtest_context(self, populated_db):
        """Verify fundamentals can be used in typical backtest pattern."""
        # Typical usage: for each date in backtest, get fundamentals
        backtest_dates = [
            date(2023, 1, 15),
            date(2023, 2, 15),
            date(2023, 3, 15),
            date(2023, 4, 15),
        ]

        all_fundamentals = []
        for trade_date in backtest_dates:
            df = populated_db.get_fundamentals_as_of(
                symbols=["AAPL"],
                metrics=["revenue", "eps"],
                as_of_date=trade_date,
            )
            if not df.empty:
                df["trade_date"] = trade_date
                all_fundamentals.append(df)

        combined = pd.concat(all_fundamentals, ignore_index=True)

        # Should have data for all dates
        assert len(combined["trade_date"].unique()) == 4

        # Revenue on Jan 15 vs Apr 15 should be different (Q4 2022 vs Q1 2023)
        jan_rev = combined[
            (combined["trade_date"] == date(2023, 1, 15)) & (combined["metric"] == "revenue")
        ]["value"].iloc[0]
        apr_rev = combined[
            (combined["trade_date"] == date(2023, 4, 15)) & (combined["metric"] == "revenue")
        ]["value"].iloc[0]

        assert jan_rev == 117154000000  # Q4 2022
        assert apr_rev == 94836000000  # Q1 2023


