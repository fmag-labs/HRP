"""Integration test fixtures with seed data.

Provides isolated database instances with realistic test data
for end-to-end integration testing.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pytest

from hrp.data.db import DatabaseManager, get_db
from hrp.data.schema import create_tables


def create_seed_data(db_path: str) -> dict:
    """
    Seed database with test data.

    Creates test data for:
    - symbols (required for FK constraints)
    - hypotheses (one in testing status)
    - prices (30 days of data for 3 symbols)
    - features (sample momentum feature)
    - lineage events (creation event)
    - data_sources (required for ingestion tests)

    Args:
        db_path: Path to the test database

    Returns:
        Dict with created entity IDs for test reference
    """
    db = get_db(db_path)
    seed_ids = {}

    with db.connection() as conn:
        # Create test symbols (required for FK constraints on prices/features)
        conn.execute("""
            INSERT INTO symbols (symbol, name, exchange, asset_type)
            VALUES
                ('AAPL', 'Apple Inc.', 'NASDAQ', 'equity'),
                ('MSFT', 'Microsoft Corporation', 'NASDAQ', 'equity'),
                ('GOOGL', 'Alphabet Inc.', 'NASDAQ', 'equity')
            ON CONFLICT DO NOTHING
        """)

        # Create test hypothesis
        conn.execute("""
            INSERT INTO hypotheses (
                hypothesis_id, title, thesis, testable_prediction,
                falsification_criteria, status, pipeline_stage,
                created_at, updated_at, created_by
            )
            VALUES (
                'HYP-TEST-001',
                'Test Momentum Hypothesis',
                'Momentum predicts short-term returns',
                '5-day returns > 0 for top momentum quintile',
                'Sharpe ratio < 0.5',
                'testing',
                'ml_training',
                CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP,
                'integration_test'
            )
        """)
        seed_ids["hypothesis_id"] = "HYP-TEST-001"

        # Create test prices (30 days of data)
        base_date = date.today() - timedelta(days=30)
        for i in range(30):
            trade_date = base_date + timedelta(days=i)
            # Skip weekends
            if trade_date.weekday() >= 5:
                continue
            for symbol in ["AAPL", "MSFT", "GOOGL"]:
                base_price = {"AAPL": 150.0, "MSFT": 350.0, "GOOGL": 140.0}[symbol]
                price = base_price + i * 0.5  # Gradual uptrend
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test')
                    ON CONFLICT DO NOTHING
                    """,
                    [
                        symbol,
                        trade_date,
                        price * 0.99,
                        price * 1.02,
                        price * 0.98,
                        price,
                        price,
                        1000000 + i * 10000,
                    ],
                )

        # Create test features
        yesterday = date.today() - timedelta(days=1)
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            conn.execute(
                """
                INSERT INTO features (symbol, date, feature_name, value, version)
                VALUES (?, ?, 'momentum_20d', ?, 'v1')
                ON CONFLICT DO NOTHING
                """,
                [symbol, yesterday, 0.05 if symbol == "AAPL" else 0.03],
            )
        seed_ids["feature_date"] = yesterday

        # Create test lineage event (lineage_id is required)
        conn.execute("""
            INSERT INTO lineage (lineage_id, event_type, actor, hypothesis_id, details)
            VALUES (
                (SELECT COALESCE(MAX(lineage_id), 0) + 1 FROM lineage),
                'hypothesis_created',
                'integration_test',
                'HYP-TEST-001',
                '{"source": "integration_test_fixture"}'
            )
        """)
        seed_ids["lineage_count"] = 1

        # Create data sources (required for ingestion tests)
        conn.execute("""
            INSERT INTO data_sources (source_id, source_type, status)
            VALUES
                ('test', 'test', 'active'),
                ('yfinance', 'api', 'active'),
                ('polygon', 'api', 'active')
            ON CONFLICT DO NOTHING
        """)

    return seed_ids


@pytest.fixture
def integration_db(tmp_path):
    """
    Provide an isolated database with seed data for integration tests.

    Creates a fresh DuckDB database with:
    - Full HRP schema
    - Seed data for testing (symbols, hypothesis, prices, features, lineage)

    The database is completely isolated from the production database.

    Yields:
        Tuple of (db_path, seed_ids) where:
        - db_path: String path to the test database
        - seed_ids: Dict with created entity IDs for test reference

    Example:
        def test_hypothesis_workflow(integration_db):
            db_path, seed_ids = integration_db
            hypothesis_id = seed_ids["hypothesis_id"]
            # Test against isolated database with seed data
    """
    db_path = tmp_path / "integration_test.duckdb"

    # Reset singleton to ensure fresh state
    DatabaseManager.reset()

    # Set environment variable for this test
    original_db_path = os.environ.get("HRP_DB_PATH")
    os.environ["HRP_DB_PATH"] = str(db_path)

    try:
        # Initialize schema (creates tables via get_db)
        create_tables(str(db_path))

        # Seed data
        seed_ids = create_seed_data(str(db_path))

        yield str(db_path), seed_ids

    finally:
        # Cleanup
        DatabaseManager.reset()

        # Restore original env var
        if original_db_path:
            os.environ["HRP_DB_PATH"] = original_db_path
        elif "HRP_DB_PATH" in os.environ:
            del os.environ["HRP_DB_PATH"]

        # Remove database files
        if db_path.exists():
            db_path.unlink()
        # Also clean up WAL and other DuckDB files
        for ext in [".wal", "-journal", "-shm"]:
            wal_path = Path(str(db_path) + ext)
            if wal_path.exists():
                wal_path.unlink()


@pytest.fixture
def integration_api(integration_db, monkeypatch):
    """
    Provide a PlatformAPI configured with the integration database.

    Uses the integration_db fixture to provide a fully-configured
    PlatformAPI instance pointing to the isolated test database.

    Yields:
        PlatformAPI instance connected to test database

    Example:
        def test_api_operations(integration_api):
            # integration_api is already connected to isolated DB with seed data
            hypotheses = integration_api.query_readonly(
                "SELECT * FROM hypotheses WHERE status = 'testing'"
            )
            assert len(hypotheses) > 0
    """
    db_path, seed_ids = integration_db

    # Environment is already set by integration_db fixture
    # Import here to ensure env var is set first
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI()

    yield api

    api.close()


@pytest.fixture
def integration_db_with_experiments(integration_db):
    """
    Provide integration database with linked experiments.

    Extends integration_db with:
    - Experiment entries in hypothesis_experiments table
    - Additional lineage events for experiments

    Yields:
        Tuple of (db_path, seed_ids) with additional experiment IDs

    Example:
        def test_experiment_analysis(integration_db_with_experiments):
            db_path, seed_ids = integration_db_with_experiments
            experiment_id = seed_ids["experiment_id"]
    """
    db_path, seed_ids = integration_db
    db = get_db(db_path)

    with db.connection() as conn:
        # Create experiment link
        experiment_id = "test-run-001"
        conn.execute(
            """
            INSERT INTO hypothesis_experiments (hypothesis_id, experiment_id, relationship)
            VALUES (?, ?, 'primary')
            """,
            [seed_ids["hypothesis_id"], experiment_id],
        )
        seed_ids["experiment_id"] = experiment_id

        # Create experiment lineage event (lineage_id is required)
        conn.execute(
            """
            INSERT INTO lineage (lineage_id, event_type, actor, hypothesis_id, experiment_id, details)
            VALUES (
                (SELECT COALESCE(MAX(lineage_id), 0) + 1 FROM lineage),
                'experiment_run',
                'integration_test',
                ?,
                ?,
                '{"model_type": "ridge", "sharpe": 1.2}'
            )
            """,
            [seed_ids["hypothesis_id"], experiment_id],
        )
        seed_ids["lineage_count"] = 2

    yield db_path, seed_ids
