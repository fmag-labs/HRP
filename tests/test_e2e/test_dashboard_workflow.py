"""
Integration tests for dashboard workflows.

Tests dashboard component functionality using mocked Streamlit.
"""

from datetime import date
from unittest.mock import patch, MagicMock
import pytest

from hrp.data.db import get_db
from hrp.api.platform import PlatformAPI


class TestDashboardSchedulerControl:
    """Test scheduler control component."""

    def test_detects_duckdb_lock_error(self):
        """
        Scheduler control detects DuckDB lock errors.

        Given:
            - Database lock exception occurs
        When:
            - Checking if error is lock error
        Then:
            - Returns True for lock errors
            - Returns False for other errors
        """
        from hrp.utils.scheduler import is_duckdb_lock_error

        # Lock error
        lock_error = Exception("Conflicting lock is held by process")
        assert is_duckdb_lock_error(lock_error) is True

        # Non-lock error
        other_error = Exception("Connection failed")
        assert is_duckdb_lock_error(other_error) is False

    def test_extracts_pid_from_lock_error(self):
        """
        Extracts PID from DuckDB lock error message.

        Given:
            - Lock error with PID
        When:
            - Parsing error message
        Then:
            - Correct PID is returned
        """
        from hrp.utils.scheduler import get_lock_holder_pid

        error = Exception("Conflicting lock is held by /path/to/db (PID 12345)")
        pid = get_lock_holder_pid(error)

        assert pid == 12345

    def test_returns_none_for_error_without_pid(self):
        """
        Returns None when error has no PID.

        Given:
            - Error without PID information
        When:
            - Parsing error message
        Then:
            - None is returned
        """
        from hrp.utils.scheduler import get_lock_holder_pid

        error = Exception("Conflicting lock")
        pid = get_lock_holder_pid(error)

        assert pid is None


class TestDashboardHypothesisManagement:
    """Test hypothesis management through dashboard."""

    def test_create_hypothesis_via_dashboard(self, test_api):
        """
        Dashboard can create hypothesis via PlatformAPI.

        Given:
            - User enters hypothesis details
        When:
            - Form is submitted
        Then:
            - Hypothesis is created
            - ID is returned
        """
        hypothesis_id = test_api.create_hypothesis(
            title="Dashboard Test Strategy",
            thesis="Test thesis for dashboard",
            prediction="Test prediction",
            falsification="Sharpe < 1.0",
            actor='dashboard',
        )

        assert hypothesis_id is not None
        assert hypothesis_id.startswith('HYP-')

    def test_retrieve_hypothesis_details(self, test_api, test_db):
        """
        Dashboard can retrieve hypothesis details.

        Given:
            - Hypothesis exists
        When:
            - Fetching details
        Then:
            - All fields are returned
        """
        # Create hypothesis
        hypothesis_id = test_api.create_hypothesis(
            title="Details Test",
            thesis="Test thesis",
            prediction="Test prediction",
            falsification="Sharpe < 1.0",
            actor='dashboard',
        )

        # Fetch details
        db = get_db(test_db)
        details = db.execute("""
            SELECT hypothesis_id, title, thesis, testable_prediction, falsification_criteria, status
            FROM hypotheses
            WHERE hypothesis_id = ?
        """, [hypothesis_id]).fetchone()

        assert details is not None
        assert details[0] == hypothesis_id
        assert details[1] == "Details Test"
        assert details[5] == 'draft'


class TestDashboardBacktestWorkflow:
    """Test backtest workflow components."""

    def test_get_prices_for_backtest(self, test_api):
        """
        Dashboard API has get_prices method.

        Given:
            - PlatformAPI instance
        When:
            - Calling get_prices
        Then:
            - Method exists and can be called
        """
        # Method should exist
        assert hasattr(test_api, 'get_prices')
        assert callable(test_api.get_prices)

    def test_get_features_for_backtest(self, test_api):
        """
        Dashboard API has get_features method.

        Given:
            - PlatformAPI instance
        When:
            - Calling get_features
        Then:
            - Method exists and can be called
        """
        # Method should exist
        assert hasattr(test_api, 'get_features')
        assert callable(test_api.get_features)


class TestDashboardErrorHandling:
    """Test dashboard error handling."""

    def test_handles_invalid_hypothesis_id(self, test_api, test_db):
        """
        Dashboard handles invalid hypothesis ID gracefully.

        Given:
            - Invalid hypothesis ID
        When:
            - Attempting to fetch details
        Then:
            - Error is handled gracefully
            - No crash occurs
        """
        db = get_db(test_db)

        # Query non-existent hypothesis
        result = db.execute("""
            SELECT * FROM hypotheses WHERE hypothesis_id = ?
        """, ['HYP-999999']).fetchone()

        # Should return None, not crash
        assert result is None
