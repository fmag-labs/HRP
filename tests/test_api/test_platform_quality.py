"""Tests for PlatformAPI quality check methods."""

import os
import tempfile
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


@pytest.fixture
def quality_db():
    """Create a temporary DuckDB database with schema."""
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


class TestPlatformQualityMethods:
    """Tests for PlatformAPI quality check methods."""

    def test_run_quality_checks_basic(self, quality_db):
        """Test running quality checks via PlatformAPI."""
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI(db_path=quality_db)
        result = api.run_quality_checks(as_of_date=date.today())

        assert "health_score" in result
        assert "critical_issues" in result
        assert "warning_issues" in result
        assert "passed" in result
        assert "results" in result
        assert isinstance(result["results"], list)

    def test_get_quality_trend(self, quality_db):
        """Test getting quality score trend."""
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI(db_path=quality_db)
        trend = api.get_quality_trend(days=30)

        assert "dates" in trend
        assert "health_scores" in trend
        assert "critical_issues" in trend
        assert "warning_issues" in trend
        assert len(trend["dates"]) == len(trend["health_scores"])

    def test_get_data_health_summary(self, quality_db):
        """Test getting data health summary."""
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI(db_path=quality_db)
        summary = api.get_data_health_summary()

        assert "symbol_count" in summary
        assert "date_range" in summary
        assert "total_records" in summary
        assert "data_freshness" in summary
        assert "ingestion_summary" in summary
        assert summary["symbol_count"] >= 0
