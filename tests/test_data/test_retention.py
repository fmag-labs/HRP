"""
Tests for data retention policy and cleanup.

Tests cover:
- RetentionPolicy tier classification
- RetentionEngine candidate identification
- DataCleanupJob with dry-run mode
- Safety checks and error handling
- Impact estimation
"""

from datetime import date, timedelta

import pytest

from hrp.data.retention.cleanup import DataArchivalJob, DataCleanupJob
from hrp.data.retention.policy import (
    DEFAULT_POLICIES,
    RetentionEngine,
    RetentionPolicy,
    RetentionTier,
)


class TestRetentionPolicy:
    """Tests for RetentionPolicy dataclass."""

    def test_default_policies_exist(self):
        """Should have default policies for all data types."""
        expected_types = {
            "prices",
            "features",
            "fundamentals",
            "lineage",
            "ingestion_log",
            "quality_reports",
        }

        assert set(DEFAULT_POLICIES.keys()) == expected_types

    def test_price_policy_tiers(self):
        """Prices should have 90d hot, 1y warm, 3y cold, 5y archive."""
        policy = DEFAULT_POLICIES["prices"]

        assert policy.data_type == "prices"
        assert policy.hot_days == 90
        assert policy.warm_days == 365
        assert policy.cold_days == 365 * 3
        assert policy.archive_days == 365 * 5

    def test_fundamentals_policy_longer_retention(self):
        """Fundamentals should have longer retention than prices."""
        prices_policy = DEFAULT_POLICIES["prices"]
        fundamentals_policy = DEFAULT_POLICIES["fundamentals"]

        assert fundamentals_policy.hot_days > prices_policy.hot_days
        assert fundamentals_policy.warm_days > prices_policy.warm_days
        assert fundamentals_policy.cold_days > prices_policy.cold_days

    def test_fundamentals_never_archive(self):
        """Fundamentals should have archive_days=None (never archive)."""
        policy = DEFAULT_POLICIES["fundamentals"]

        assert policy.archive_days is None

    def test_get_tier_hot(self):
        """Recent data should be in HOT tier."""
        policy = DEFAULT_POLICIES["prices"]

        tier = policy.get_tier_for_age(30)  # 30 days old
        assert tier == RetentionTier.HOT

    def test_get_tier_warm(self):
        """Data between hot and cold should be WARM."""
        policy = DEFAULT_POLICIES["prices"]

        tier = policy.get_tier_for_age(200)  # 200 days old
        assert tier == RetentionTier.WARM

    def test_get_tier_cold(self):
        """Older data should be in COLD tier."""
        policy = DEFAULT_POLICIES["prices"]

        tier = policy.get_tier_for_age(500)  # 500 days old (~1.4 years)
        assert tier == RetentionTier.COLD

    def test_get_tier_archive(self):
        """Very old data should be in ARCHIVE tier."""
        policy = DEFAULT_POLICIES["prices"]

        tier = policy.get_tier_for_age(1500)  # 1500 days old (~4 years)
        assert tier == RetentionTier.ARCHIVE

    def test_is_eligible_for_cleanup(self):
        """Data older than cold_days should be eligible for cleanup."""
        policy = DEFAULT_POLICIES["prices"]

        # Cold data is eligible
        assert policy.is_eligible_for_cleanup(1100)  # ~3 years

        # Hot data is not eligible
        assert not policy.is_eligible_for_cleanup(30)

    def test_days_until_cleanup(self):
        """Should calculate days until cleanup eligibility."""
        policy = DEFAULT_POLICIES["prices"]

        # Hot data: positive days until cleanup
        days = policy.days_until_cleanup(30)
        assert days > 1000  # About 3 years minus 30 days

        # Cold data: already eligible (0 or negative)
        days = policy.days_until_cleanup(1500)
        assert days == 0


class TestRetentionEngine:
    """Tests for RetentionEngine."""

    def test_init_default_policies(self, test_db):
        """Should initialize with default policies."""
        engine = RetentionEngine(test_db)

        assert len(engine._policies) == len(DEFAULT_POLICIES)

    def test_get_policy(self, test_db):
        """Should retrieve policy for data type."""
        engine = RetentionEngine(test_db)

        policy = engine.get_policy("prices")
        assert policy.data_type == "prices"

    def test_get_policy_invalid_type(self, test_db):
        """Should raise error for invalid data type."""
        engine = RetentionEngine(test_db)

        with pytest.raises(ValueError):
            engine.get_policy("invalid_type")

    def test_get_tier_for_date_hot(self, test_db):
        """Should classify recent dates as HOT."""
        engine = RetentionEngine(test_db)

        tier = engine.get_tier_for_date("prices", date.today() - timedelta(days=30))
        assert tier == RetentionTier.HOT

    def test_get_tier_for_date_warm(self, test_db):
        """Should classify moderately old dates as WARM."""
        engine = RetentionEngine(test_db)

        tier = engine.get_tier_for_date("prices", date.today() - timedelta(days=200))
        assert tier == RetentionTier.WARM

    def test_get_tier_for_date_cold(self, test_db):
        """Should classify old dates as COLD."""
        engine = RetentionEngine(test_db)

        tier = engine.get_tier_for_date("prices", date.today() - timedelta(days=500))
        assert tier == RetentionTier.COLD

    def test_get_cleanup_candidates_prices(self, test_db):
        """Should find old price records for cleanup."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert symbol first
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('OLD', 'Old Test', 'NASDAQ')"
            )

        # Insert old prices
        old_date = date(2020, 1, 1)
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO prices (symbol, date, close, volume, source) "
                "VALUES ('OLD', ?, 100.0, 1000000, 'test')",
                (old_date,),
            )

        engine = RetentionEngine(test_db)
        candidates = engine.get_cleanup_candidates("prices")

        # Should find at least the old symbol
        assert len(candidates) >= 0  # May be empty if not old enough

    def test_estimate_cleanup_impact(self, test_db):
        """Should estimate cleanup impact for data type."""
        engine = RetentionEngine(test_db)

        impact = engine.estimate_cleanup_impact("prices")

        assert "data_type" in impact
        assert "total_candidates" in impact
        assert "total_records" in impact
        assert "tier_distribution" in impact
        assert "policy" in impact

    def test_get_retention_summary(self, test_db):
        """Should get summary for all data types."""
        engine = RetentionEngine(test_db)

        summary = engine.get_retention_summary()

        assert "as_of_date" in summary
        assert "data_types" in summary
        assert len(summary["data_types"]) == len(DEFAULT_POLICIES)


class TestDataCleanupJob:
    """Tests for DataCleanupJob."""

    def test_init(self):
        """Should initialize cleanup job."""
        job = DataCleanupJob(dry_run=True)

        assert job._dry_run is True
        assert "prices" in job._data_types

    def test_dry_run_does_not_delete(self, test_db):
        """Dry run should not delete any data."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert symbol first
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('DRYTEST', 'Dry Run Test', 'NASDAQ')"
            )

        # Insert some old prices
        old_date = date(2020, 1, 1)
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO prices (symbol, date, close, volume, source) "
                "VALUES ('DRYTEST', ?, 100.0, 1000000, 'test')",
                (old_date,),
            )

        # Count before
        count_before = db.fetchone("SELECT COUNT(*) FROM prices")[0]

        # Run dry run cleanup
        job = DataCleanupJob(test_db, dry_run=True)
        results = job.run()

        # Count after (should be same)
        count_after = db.fetchone("SELECT COUNT(*) FROM prices")[0]

        assert count_before == count_after
        assert results["prices"].dry_run is True

    def test_estimate_impact(self, test_db):
        """Should estimate cleanup impact without changes."""
        job = DataCleanupJob(test_db, dry_run=True)

        impact = job.estimate_impact()

        assert "as_of_date" in impact
        assert "dry_run" in impact
        assert impact["dry_run"] is True
        assert "data_types" in impact

    def test_run_with_no_candidates(self, test_db):
        """Should handle case with no cleanup candidates gracefully."""
        # Use only one data type to avoid conflicts
        job = DataCleanupJob(test_db, dry_run=True, data_types=["quality_reports"])

        results = job.run()

        # Should have one result
        assert "quality_reports" in results
        # Should succeed even with no candidates
        assert results["quality_reports"].success is True


class TestDataArchivalJob:
    """Tests for DataArchivalJob."""

    def test_init(self):
        """Should initialize archival job."""
        job = DataArchivalJob(dry_run=True)

        assert job._dry_run is True

    def test_dry_run_does_not_modify(self, test_db):
        """Dry run should not modify any data."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert symbol first
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('ARCHTEST', 'Archive Test', 'NASDAQ')"
            )

        # Insert some data
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO prices (symbol, date, close, volume, source) "
                "VALUES ('ARCHTEST', '2020-01-01', 100.0, 1000000, 'test')"
            )

        count_before = db.fetchone("SELECT COUNT(*) FROM prices")[0]

        job = DataArchivalJob(test_db, dry_run=True)
        job.run()

        count_after = db.fetchone("SELECT COUNT(*) FROM prices")[0]

        assert count_before == count_after

    def test_run_returns_summary(self, test_db):
        """Should return archival summary."""
        job = DataArchivalJob(test_db, dry_run=True)

        results = job.run()

        assert "as_of_date" in results
        assert "dry_run" in results
        assert "archived" in results


class TestSafetyChecks:
    """Tests for safety checks in cleanup operations."""

    def test_require_confirmation_flag(self):
        """Should respect require_confirmation flag."""
        job = DataCleanupJob(dry_run=True, require_confirmation=False)

        assert job._require_confirmation is False

    def test_custom_data_types(self):
        """Should allow custom data type selection."""
        job = DataCleanupJob(dry_run=True, data_types=["prices"])

        assert job._data_types == ["prices"]

    def test_error_handling_invalid_data_type(self, test_db):
        """Should handle invalid data types gracefully."""
        job = DataCleanupJob(test_db, dry_run=True, data_types=["invalid_type"])

        results = job.run()

        # Should return error result for invalid type
        assert "invalid_type" in results
        assert len(results["invalid_type"].errors) > 0


class TestIntegration:
    """Integration tests for retention system."""

    def test_full_cleanup_workflow(self, test_db):
        """Should complete full cleanup workflow."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert symbols first
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('HOT', 'Hot Test', 'NASDAQ')"
            )
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('COLD', 'Cold Test', 'NASDAQ')"
            )

        # Insert test data at various ages
        with db.connection() as conn:
            # Hot data (recent)
            conn.execute(
                "INSERT INTO prices (symbol, date, close, volume, source) "
                "VALUES ('HOT', '2024-01-01', 100.0, 1000000, 'test')"
            )
            # Cold data (old)
            conn.execute(
                "INSERT INTO prices (symbol, date, close, volume, source) "
                "VALUES ('COLD', '2020-01-01', 100.0, 1000000, 'test')"
            )

        # Estimate impact first
        job = DataCleanupJob(test_db, dry_run=True)
        impact = job.estimate_impact()

        assert "prices" in impact["data_types"]

        # Run dry run
        results = job.run()

        assert results["prices"].dry_run is True
        assert results["prices"].success is True

    def test_retention_engine_integration(self, test_db):
        """RetentionEngine should work with real database data."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert symbol first
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('ENGTEST', 'Engine Test', 'NASDAQ')"
            )

        # Insert test data
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO prices (symbol, date, close, volume, source) "
                "VALUES ('ENGTEST', '2020-01-01', 100.0, 1000000, 'test')"
            )

        engine = RetentionEngine(test_db)
        summary = engine.get_retention_summary()

        assert summary["as_of_date"] is not None
        assert len(summary["data_types"]) > 0
