"""
Integration tests for data pipeline validation.

These tests validate the end-to-end data pipeline with quality checks:
- Price ingestion creates valid data
- Feature computation validates inputs
- Quality checks detect real issues
- Validation utilities work with real data
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hrp.data.db import DatabaseManager
from hrp.data.quality.checks import (
    CheckResult,
    CompletenessCheck,
    GapDetectionCheck,
    IssueSeverity,
    PriceAnomalyCheck,
    StaleDataCheck,
    VolumeAnomalyCheck,
)
from hrp.data.quality.validation import (
    DataValidator,
    ValidationResult,
    validate_before_operation,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pipeline_test_db():
    """
    Create a database with sample data for pipeline testing.
    """
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    os.remove(db_path)
    DatabaseManager.reset()
    from hrp.data.schema import create_tables

    create_tables(db_path)

    from hrp.data.db import get_db

    db = get_db(db_path)

    # Insert sample universe
    with db.connection() as conn:
        # Insert symbols first to satisfy FK constraints
        conn.execute(
            """
            INSERT INTO symbols (symbol, name, exchange)
            VALUES
                ('AAPL', 'Apple Inc.', 'NASDAQ'),
                ('MSFT', 'Microsoft Corporation', 'NASDAQ'),
                ('GOOGL', 'Alphabet Inc.', 'NASDAQ'),
                ('TSLA', 'Tesla Inc.', 'NASDAQ')
        """
        )

        conn.execute(
            """
            INSERT INTO universe (symbol, date, in_universe, sector)
            VALUES
                ('AAPL', '2024-01-01', TRUE, 'Technology'),
                ('MSFT', '2024-01-01', TRUE, 'Technology'),
                ('GOOGL', '2024-01-01', TRUE, 'Technology'),
                ('TSLA', '2024-01-01', TRUE, 'Consumer Discretionary')
        """
        )

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = [
        date(2024, 1, 2) + timedelta(days=i)
        for i in range(10)
        if (date(2024, 1, 2) + timedelta(days=i)).weekday() < 5
    ]

    data = []
    for dt in dates:
        data.extend([
            {
                "symbol": "AAPL",
                "date": dt,
                "open": 180.0 + dt.day * 0.1,
                "high": 182.0 + dt.day * 0.1,
                "low": 178.0 + dt.day * 0.1,
                "close": 180.5 + dt.day * 0.1,
                "volume": 10000000,
            },
            {
                "symbol": "MSFT",
                "date": dt,
                "open": 380.0 + dt.day * 0.2,
                "high": 382.0 + dt.day * 0.2,
                "low": 378.0 + dt.day * 0.2,
                "close": 380.5 + dt.day * 0.2,
                "volume": 8000000,
            },
        ])

    return data


# =============================================================================
# Pipeline Integration Tests
# =============================================================================


class TestPriceIngestionValidation:
    """Integration tests for price ingestion with validation."""

    def test_ingest_valid_prices(self, pipeline_test_db, sample_price_data):
        """Should successfully ingest valid price data."""
        from hrp.data.db import get_db

        db = get_db(pipeline_test_db)

        # Insert prices
        with db.connection() as conn:
            for row in sample_price_data:
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, open, high, low, close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'test')
                    """,
                    (
                        row["symbol"],
                        row["date"],
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["volume"],
                    ),
                )

        # Verify insertion
        result = db.fetchall("SELECT COUNT(*) FROM prices")
        assert result[0][0] == len(sample_price_data)

        # Validate with DataValidator
        prices_df = pd.DataFrame([
            {
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": r["volume"],
            }
            for r in sample_price_data
        ])

        validation = DataValidator.validate_price_data(prices_df)
        assert validation.is_valid is True
        assert len(validation.errors) == 0

    def test_detect_invalid_prices_on_ingest(self, pipeline_test_db):
        """Should detect invalid prices during ingestion."""
        from hrp.data.db import get_db

        db = get_db(pipeline_test_db)

        # Try to insert invalid price (negative close)
        with pytest.raises(Exception):
            with db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, close, source)
                    VALUES ('INVALID', '2024-01-15', -100.0, 'test')
                """
                )

        # Verify it was not inserted
        result = db.fetchone("SELECT COUNT(*) FROM prices WHERE symbol = 'INVALID'")
        assert result[0] == 0

    def test_validate_before_ingest(self, pipeline_test_db):
        """Should use validation context manager before ingestion."""
        invalid_df = pd.DataFrame({
            "close": [-100.0],  # Invalid
            "volume": [1000000],
        })

        # Should raise exception (need to disable ohlc check to focus on close validation)
        with pytest.raises(ValueError):
            with validate_before_operation(
                DataValidator.validate_price_data,
                on_failure="raise",
                prices_df=invalid_df,
                check_ohlc_relationship=False,
            ):
                # Ingestion would happen here
                pass


class TestFeatureComputationValidation:
    """Integration tests for feature computation with validation."""

    def test_sufficient_history_for_computation(self, pipeline_test_db):
        """Should compute features with sufficient history."""
        from hrp.data.db import get_db

        db = get_db(pipeline_test_db)

        # Insert 30 days of price history
        base_date = date(2024, 1, 15)
        with db.connection() as conn:
            for i in range(30):
                dt = base_date - timedelta(days=i)
                if dt.weekday() < 5:  # Weekdays only
                    conn.execute(
                        """
                        INSERT INTO prices (symbol, date, close, volume, source)
                        VALUES (?, ?, ?, ?, 'test')
                        """,
                        ("AAPL", dt, 180.0 + i * 0.1, 10000000),
                    )

        # Get price data
        prices_df = pd.read_sql(
            "SELECT date, close FROM prices WHERE symbol = 'AAPL' ORDER BY date",
            db.connection().__enter__(),
        )

        # Validate before computation
        result = DataValidator.validate_feature_computation_inputs(
            prices_df,
            feature_name="momentum_20d",
            min_history_days=20,
        )

        assert result.is_valid is True
        assert result.stats["available_rows"] >= 20

    def test_insufficient_history_for_computation(self, pipeline_test_db):
        """Should reject computation with insufficient history."""
        from hrp.data.db import get_db

        db = get_db(pipeline_test_db)

        # Insert only 5 days of history
        base_date = date(2024, 1, 15)
        with db.connection() as conn:
            for i in range(5):
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, close, volume, source)
                    VALUES (?, ?, ?, ?, 'test')
                    """,
                    ("AAPL", base_date - timedelta(days=i), 180.0, 10000000),
                )

        # Get price data
        prices_df = pd.read_sql(
            "SELECT date, close FROM prices WHERE symbol = 'AAPL' ORDER BY date",
            db.connection().__enter__(),
        )

        # Validate before computation
        result = DataValidator.validate_feature_computation_inputs(
            prices_df,
            feature_name="momentum_20d",
            min_history_days=20,
        )

        assert result.is_valid is False
        assert "Insufficient history" in result.errors[0]


class TestQualityChecksIntegration:
    """Integration tests for quality checks with real data."""

    def test_price_anomaly_check_detects_issues(self, pipeline_test_db):
        """Should detect price anomalies in real data."""
        from hrp.data.db import get_db

        db = get_db(pipeline_test_db)

        # Insert normal prices followed by anomalous spike
        # Use a date range that won't conflict with existing data
        base_date = date(2024, 3, 10)
        with db.connection() as conn:
            # Insert symbol first (use a symbol not in universe)
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('SPIKE', 'Spike Test', 'NASDAQ')"
            )
            # Normal prices
            for i in range(5):
                dt = base_date - timedelta(days=(i + 1) * 2)  # Skip days to avoid weekends
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, close, volume, source)
                    VALUES (?, ?, ?, ?, 'test')
                    """,
                    ("SPIKE", dt, 100.0, 1000000),
                )
            # Anomalous spike (100% increase)
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, volume, source)
                VALUES ('SPIKE', ?, 200.0, 5000000, 'test')
                """,
                (base_date,),
            )

        # Run price anomaly check
        check = PriceAnomalyCheck(pipeline_test_db, threshold=0.5)
        result = check.run(base_date)

        assert not result.passed
        assert len(result.issues) > 0
        assert result.issues[0].symbol == "SPIKE"

    def test_completeness_check_detects_missing(self, pipeline_test_db):
        """Should detect missing prices for universe symbols."""
        from hrp.data.db import get_db

        db = get_db(pipeline_test_db)

        # Only insert prices for 2 of 4 universe symbols
        base_date = date(2024, 1, 15)
        with db.connection() as conn:
            for symbol in ["AAPL", "MSFT"]:
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, close, volume, source)
                    VALUES (?, ?, ?, ?, 'test')
                    """,
                    (symbol, base_date, 150.0, 1000000),
                )

        # Run completeness check
        check = CompletenessCheck(pipeline_test_db)
        result = check.run(base_date)

        # Should have 2 missing symbols (GOOGL, TSLA)
        assert len(result.issues) == 2
        assert result.warning_count == 2

    def test_gap_detection_finds_gaps(self, pipeline_test_db):
        """Should detect gaps in price history."""
        from hrp.data.db import get_db

        db = get_db(pipeline_test_db)

        # Insert prices with gaps - use a different date range
        base_date = date(2024, 3, 15)
        with db.connection() as conn:
            # Insert symbol first
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('GAPPY', 'Gap Test', 'NASDAQ')"
            )
            # Only insert prices for 3 days over 10 day period
            # This will create gaps in the price history
            for i in [0, 4, 8]:  # Only 3 prices over 9 days
                dt = base_date - timedelta(days=i)
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, close, volume, source)
                    VALUES (?, ?, ?, ?, 'test')
                    """,
                    ("GAPPY", dt, 100.0, 1000000),
                )

        # Also insert a reference symbol with complete history
        with db.connection() as conn:
            # Insert REF symbol first
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('REF', 'Reference', 'NASDAQ')"
            )
            for i in range(10):
                dt = base_date - timedelta(days=i)
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, close, volume, source)
                    VALUES (?, ?, ?, ?, 'test')
                    """,
                    ("REF", dt, 100.0, 1000000),
                )

        # Run gap detection
        check = GapDetectionCheck(pipeline_test_db, lookback_days=10)
        result = check.run(base_date)

        # Should detect gaps
        gappy_issues = [i for i in result.issues if i.symbol == "GAPPY"]
        assert len(gappy_issues) > 0

    def test_stale_data_check_detects_staleness(self, pipeline_test_db):
        """Should detect stale data."""
        from hrp.data.db import get_db

        db = get_db(pipeline_test_db)

        # Insert old prices and add STALE to universe
        old_date = date(2024, 1, 1)
        with db.connection() as conn:
            # Insert symbol first
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('STALE', 'Stale Test', 'NASDAQ')"
            )
            # Add to universe so stale check will look for it
            conn.execute(
                "INSERT INTO universe (symbol, date, in_universe, sector) VALUES ('STALE', '2024-01-15', TRUE, 'Technology')"
            )
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, volume, source)
                VALUES ('STALE', ?, 100.0, 1000000, 'test')
                """,
                (old_date,),
            )

        # Run stale data check
        check = StaleDataCheck(pipeline_test_db, stale_threshold_days=3)
        result = check.run(date(2024, 1, 15))

        # Should detect stale data
        stale_issues = [i for i in result.issues if i.symbol == "STALE"]
        assert len(stale_issues) == 1

    def test_volume_anomaly_check_detects_zeros(self, pipeline_test_db):
        """Should detect zero volume days."""
        from hrp.data.db import get_db

        db = get_db(pipeline_test_db)

        # Insert prices with zero volume
        test_date = date(2024, 1, 15)
        with db.connection() as conn:
            # Insert symbol first
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('ZEROVOL', 'Zero Vol Test', 'NASDAQ')"
            )
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, volume, source)
                VALUES ('ZEROVOL', ?, 100.0, 0, 'test')
                """,
                (test_date,),
            )

        # Run volume anomaly check
        check = VolumeAnomalyCheck(pipeline_test_db)
        result = check.run(test_date)

        # Should detect zero volume
        zero_issues = [i for i in result.issues if i.symbol == "ZEROVOL"]
        assert len(zero_issues) == 1


class TestValidationWorkflow:
    """Integration tests for validation workflows."""

    def test_complete_ingestion_validation_workflow(self, pipeline_test_db):
        """Should validate at each step of ingestion workflow."""
        from hrp.data.db import get_db

        db = get_db(pipeline_test_db)

        # Step 1: Validate input data
        input_df = pd.DataFrame({
            "open": [180.0],
            "high": [182.0],
            "low": [178.0],
            "close": [180.5],
            "volume": [10000000],
        })

        validation = DataValidator.validate_price_data(input_df)
        assert validation.is_valid is True

        # Step 2: Insert into database - use a date not conflicting with existing data
        test_date = date(2024, 2, 15)
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO prices (symbol, date, open, high, low, close, volume, source)
                VALUES ('AAPL', ?, ?, ?, ?, ?, ?, 'test')
                """,
                (test_date, float(input_df["open"][0]), float(input_df["high"][0]),
                 float(input_df["low"][0]), float(input_df["close"][0]), int(input_df["volume"][0])),
            )

        # Step 3: Verify with quality checks
        check = CompletenessCheck(pipeline_test_db)
        result = check.run(test_date)

        # AAPL should not be in missing list
        aapl_missing = [i for i in result.issues if i.symbol == "AAPL"]
        assert len(aapl_missing) == 0

    def test_validate_universe_before_feature_computation(self, pipeline_test_db):
        """Should validate universe health before feature computation."""
        from hrp.data.db import get_db

        db = get_db(pipeline_test_db)

        # Insert prices for all universe symbols
        test_date = date(2024, 1, 15)
        with db.connection() as conn:
            for symbol in ["AAPL", "MSFT", "GOOGL", "TSLA"]:
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, close, volume, source)
                    VALUES (?, ?, ?, ?, 'test')
                    """,
                    (symbol, test_date, 150.0, 1000000),
                )

        # Validate universe
        result = DataValidator.validate_universe_data(
            symbols=["AAPL", "MSFT", "GOOGL", "TSLA"],
            as_of_date=test_date,
            db_path=pipeline_test_db,
            require_prices=True,
        )

        assert result.is_valid is True
        assert len(result.warnings) == 0

    def test_feature_computation_with_validation_context(self, pipeline_test_db):
        """Should use validation context manager for feature computation."""
        from hrp.data.db import get_db

        db = get_db(pipeline_test_db)

        # Insert price history
        base_date = date(2024, 1, 15)
        with db.connection() as conn:
            for i in range(25):
                dt = base_date - timedelta(days=i)
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, close, volume, source)
                    VALUES (?, ?, ?, ?, 'test')
                    """,
                    ("AAPL", dt, 180.0 + i * 0.1, 10000000),
                )

        # Get price data
        prices_df = pd.read_sql(
            "SELECT date, close FROM prices WHERE symbol = 'AAPL' ORDER BY date",
            db.connection().__enter__(),
        )

        # Use validation context manager
        with validate_before_operation(
            DataValidator.validate_feature_computation_inputs,
            on_failure="raise",
            prices_df=prices_df,
            feature_name="momentum_20d",
            min_history_days=20,
        ) as result:
            assert result.is_valid is True
            # Feature computation would go here


class TestQualityReportIntegration:
    """Integration tests for quality reports with pipeline data."""

    def test_generate_report_after_ingestion(self, pipeline_test_db, sample_price_data):
        """Should generate quality report after price ingestion."""
        from hrp.data.db import get_db
        from hrp.data.quality.report import QualityReportGenerator

        db = get_db(pipeline_test_db)

        # Ingest sample prices
        with db.connection() as conn:
            for row in sample_price_data:
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, open, high, low, close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'test')
                    """,
                    (
                        row["symbol"],
                        row["date"],
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["volume"],
                    ),
                )

        # Generate quality report
        generator = QualityReportGenerator(pipeline_test_db)
        report = generator.generate_report(date(2024, 1, 15))

        assert report is not None
        assert report.checks_run > 0
        assert 0 <= report.health_score <= 100
        assert report.total_issues >= 0  # May have warnings

    def test_report_detects_missing_universe_data(self, pipeline_test_db):
        """Quality report should detect missing universe data."""
        from hrp.data.quality.report import QualityReportGenerator

        # Don't insert any prices - universe is empty

        # Generate quality report
        generator = QualityReportGenerator(pipeline_test_db)
        report = generator.generate_report(date(2024, 1, 15))

        # Should have completeness issues
        completeness_result = next(
            (r for r in report.results if r.check_name == "completeness"),
            None,
        )
        assert completeness_result is not None
        assert len(completeness_result.issues) > 0


class TestErrorHandling:
    """Integration tests for error handling in validation."""

    def test_validation_failure_prevents_operation(self, pipeline_test_db):
        """Should prevent operation on validation failure."""
        from hrp.data.db import get_db

        db = get_db(pipeline_test_db)

        invalid_df = pd.DataFrame({
            "close": [-100.0],  # Invalid
        })

        initial_count = db.fetchone("SELECT COUNT(*) FROM prices")[0]

        # Should raise and prevent insertion
        try:
            with validate_before_operation(
                DataValidator.validate_price_data,
                on_failure="raise",
                prices_df=invalid_df,
                check_ohlc_relationship=False,
            ):
                # This code should not execute
                with db.connection() as conn:
                    conn.execute(
                        "INSERT INTO prices (symbol, date, close, source) VALUES ('BAD', '2024-01-15', -100.0, 'test')"
                    )
        except ValueError:
            pass  # Expected

        # Verify no data was inserted
        final_count = db.fetchone("SELECT COUNT(*) FROM prices")[0]
        assert final_count == initial_count

    def test_validation_with_warnings_continues(self, pipeline_test_db):
        """Should continue operation with validation warnings."""
        from hrp.data.db import get_db

        db = get_db(pipeline_test_db)

        warning_df = pd.DataFrame({
            "close": [100.0, 101.0],
            "volume": [1000000, 0],  # Warning: zero volume
        })

        # Should not raise, but produce warnings
        with validate_before_operation(
            DataValidator.validate_price_data,
            on_failure="raise",
            prices_df=warning_df,
            check_ohlc_relationship=False,
        ) as result:
            assert result.is_valid is True  # No errors
            assert len(result.warnings) > 0
            # Operation would continue here
