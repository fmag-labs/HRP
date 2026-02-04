"""Tests to verify integration fixtures work correctly."""

import pytest


def test_integration_db_fixture(integration_db):
    """Test that integration_db fixture creates database with seed data."""
    db_path, seed_ids = integration_db

    # Verify path is a string
    assert isinstance(db_path, str)
    assert db_path.endswith(".duckdb")

    # Verify seed_ids contains expected keys
    assert "hypothesis_id" in seed_ids
    assert seed_ids["hypothesis_id"] == "HYP-TEST-001"
    assert "lineage_count" in seed_ids
    assert "feature_date" in seed_ids

    # Verify database has data
    from hrp.data.db import get_db

    db = get_db(db_path)

    # Check hypothesis exists
    result = db.fetchone(
        "SELECT status FROM hypotheses WHERE hypothesis_id = ?",
        (seed_ids["hypothesis_id"],),
    )
    assert result is not None
    assert result[0] == "testing"

    # Check prices exist
    price_count = db.fetchone("SELECT COUNT(*) FROM prices")[0]
    assert price_count > 0

    # Check symbols exist
    symbol_count = db.fetchone("SELECT COUNT(*) FROM symbols")[0]
    assert symbol_count >= 3


def test_integration_api_fixture(integration_api):
    """Test that integration_api fixture provides working PlatformAPI."""
    # Verify API is connected and can query
    result = integration_api.query_readonly("SELECT COUNT(*) as cnt FROM hypotheses")
    assert len(result) == 1
    assert result["cnt"].iloc[0] >= 1

    # Verify can query prices
    prices = integration_api.query_readonly("SELECT COUNT(*) as cnt FROM prices")
    assert prices["cnt"].iloc[0] > 0


def test_integration_db_with_experiments_fixture(integration_db_with_experiments):
    """Test that integration_db_with_experiments adds experiment data."""
    db_path, seed_ids = integration_db_with_experiments

    # Verify experiment_id is in seed_ids
    assert "experiment_id" in seed_ids
    assert seed_ids["experiment_id"] == "test-run-001"

    # Verify experiment is linked in database
    from hrp.data.db import get_db

    db = get_db(db_path)

    result = db.fetchone(
        """
        SELECT experiment_id FROM hypothesis_experiments
        WHERE hypothesis_id = ?
        """,
        (seed_ids["hypothesis_id"],),
    )
    assert result is not None
    assert result[0] == seed_ids["experiment_id"]

    # Verify lineage count increased
    assert seed_ids["lineage_count"] == 2


def test_fixtures_isolation(integration_db):
    """Test that fixture provides isolated database."""
    db_path, seed_ids = integration_db

    from hrp.data.db import get_db

    db = get_db(db_path)

    # Insert test data
    db.execute(
        """
        INSERT INTO hypotheses (
            hypothesis_id, title, thesis, testable_prediction, status
        )
        VALUES ('HYP-ISOLATION-TEST', 'Isolation Test', 'Testing isolation', 'Should be isolated', 'draft')
        """
    )

    # Verify it was inserted
    result = db.fetchone(
        "SELECT COUNT(*) FROM hypotheses WHERE hypothesis_id = 'HYP-ISOLATION-TEST'"
    )
    assert result[0] == 1


def test_seed_data_completeness(integration_db):
    """Test that seed data includes all expected entities."""
    db_path, seed_ids = integration_db

    from hrp.data.db import get_db

    db = get_db(db_path)

    # Check symbols
    symbols = db.fetchall("SELECT symbol FROM symbols ORDER BY symbol")
    symbol_list = [s[0] for s in symbols]
    assert "AAPL" in symbol_list
    assert "MSFT" in symbol_list
    assert "GOOGL" in symbol_list

    # Check features
    features = db.fetchall("SELECT DISTINCT feature_name FROM features")
    assert len(features) > 0
    assert features[0][0] == "momentum_20d"

    # Check lineage
    lineage = db.fetchall("SELECT event_type FROM lineage WHERE hypothesis_id = ?", (seed_ids["hypothesis_id"],))
    assert len(lineage) >= 1

    # Check data sources
    sources = db.fetchall("SELECT source_id FROM data_sources")
    source_list = [s[0] for s in sources]
    assert "test" in source_list
