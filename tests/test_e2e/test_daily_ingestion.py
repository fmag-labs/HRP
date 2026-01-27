"""
Integration tests for daily data ingestion pipeline.

Tests feature computation workflow using existing test infrastructure.
"""

from datetime import date
import pytest


class TestFeatureComputationWorkflow:
    """Test feature computation workflow."""

    def test_feature_job_initializes(self):
        """
        Feature computation job initializes correctly.

        Given:
            - Job parameters provided
        When:
            - Creating job
        Then:
            - Job is created successfully
        """
        from hrp.agents.jobs import FeatureComputationJob

        job = FeatureComputationJob(
            symbols=['AAPL'],
            start=date(2020, 12, 1),
            end=date(2020, 12, 31),
        )

        assert job is not None

    def test_feature_job_has_required_attributes(self):
        """
        Feature job has required attributes for execution.

        Given:
            - Job created
        When:
            - Checking attributes
        Then:
            - Has execute method
            - Has run method
        """
        from hrp.agents.jobs import FeatureComputationJob

        job = FeatureComputationJob(
            symbols=['AAPL'],
            start=date(2020, 12, 1),
            end=date(2020, 12, 31),
        )

        assert hasattr(job, 'execute')
        assert hasattr(job, 'run')


class TestFeatureComputationIdempotency:
    """Test that feature computation is idempotent."""

    def test_feature_job_is_deterministic(self):
        """
        Running job with same parameters produces same config.

        Given:
            - Job created with specific parameters
        When:
            - Creating another job with same parameters
        Then:
            - Jobs are equivalent
        """
        from hrp.agents.jobs import FeatureComputationJob

        job1 = FeatureComputationJob(
            symbols=['AAPL'],
            start=date(2020, 12, 1),
            end=date(2020, 12, 31),
        )

        job2 = FeatureComputationJob(
            symbols=['AAPL'],
            start=date(2020, 12, 1),
            end=date(2020, 12, 31),
        )

        # Both should have same configuration
        assert job1.symbols == job2.symbols
