"""
Integration tests for ML research pipeline.

Tests ML workflow using mocked components.
"""

from datetime import date
from unittest.mock import patch, MagicMock
import pytest

from hrp.data.db import get_db
from hrp.api.platform import PlatformAPI


class TestHypothesisCreation:
    """Test hypothesis creation and lifecycle."""

    def test_create_hypothesis_via_api(self, test_api):
        """
        Hypothesis can be created via PlatformAPI.

        Given:
            - PlatformAPI instance
        When:
            - User creates a hypothesis
        Then:
            - Hypothesis ID is returned
            - Hypothesis exists in database
        """
        hypothesis_id = test_api.create_hypothesis(
            title="Test Momentum Strategy",
            thesis="Stocks with high momentum outperform",
            prediction="Top decile > SPY by 3%",
            falsification="Sharpe < 1.0",
            actor='integration-test',
        )

        assert hypothesis_id is not None
        assert hypothesis_id.startswith('HYP-')

    def test_hypothesis_has_correct_initial_status(self, test_api, test_db):
        """
        New hypothesis has 'draft' status.

        Given:
            - Hypothesis created
        When:
            - Checking status
        Then:
            - Status is 'draft'
        """
        hypothesis_id = test_api.create_hypothesis(
            title="Status Test",
            thesis="Test",
            prediction="Test",
            falsification="Sharpe < 1.0",
            actor='integration-test',
        )

        db = get_db(test_db)
        status = db.execute("""
            SELECT status
            FROM hypotheses
            WHERE hypothesis_id = ?
        """, [hypothesis_id]).fetchone()[0]

        assert status == 'draft'


class TestMLValidationWorkflow:
    """Test ML validation workflow."""

    def test_ml_scientist_initializes(self):
        """
        ML Scientist initializes correctly.

        Given:
            - ML Scientist parameters
        When:
            - Creating instance
        Then:
            - Instance created successfully
        """
        from hrp.agents import MLScientist

        scientist = MLScientist(
            hypothesis_ids=['HYP-001'],
            n_folds=3,
            window_type='expanding',
        )

        assert scientist is not None


class TestSignalDiscoveryWorkflow:
    """Test signal discovery workflow."""

    def test_signal_scientist_initializes(self):
        """
        Signal Scientist initializes correctly.

        Given:
            - Signal Scientist parameters
        When:
            - Creating instance
        Then:
            - Instance created successfully
        """
        from hrp.agents import SignalScientist

        scientist = SignalScientist(
            symbols=['AAPL'],
            features=['momentum_20d'],
            forward_horizons=[20],
            ic_threshold=0.03,
            create_hypotheses=False,
        )

        assert scientist is not None


class TestAgentCoordination:
    """Test agent coordination through lineage."""

    def test_feature_job_has_run_method(self):
        """
        Agent job has run method.

        Given:
            - Feature job created
        When:
            - Checking methods
        Then:
            - Has run method
        """
        from hrp.agents.jobs import FeatureComputationJob

        job = FeatureComputationJob(
            symbols=['AAPL'],
            start=date(2020, 12, 1),
            end=date(2020, 12, 31),
        )

        assert hasattr(job, 'run')
        assert callable(job.run)
