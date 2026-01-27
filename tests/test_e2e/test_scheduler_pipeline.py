"""
Integration tests for scheduler coordination.

Tests scheduler job configuration and execution.
"""

from datetime import date
import pytest


class TestSchedulerConfiguration:
    """Test scheduler job configuration."""

    def test_scheduler_initializes_empty(self):
        """
        Scheduler initializes with no jobs.

        Given:
            - New scheduler instance
        When:
            - Checking jobs
        Then:
            - No jobs are configured
        """
        from hrp.agents.scheduler import IngestionScheduler

        scheduler = IngestionScheduler()
        assert len(scheduler._jobs) == 0

    def test_setup_daily_ingestion_configures_jobs(self):
        """
        Setting up daily ingestion configures jobs.

        Given:
            - Scheduler instance
        When:
            - setup_daily_ingestion is called
        Then:
            - Jobs are configured
        """
        from hrp.agents.scheduler import IngestionScheduler

        scheduler = IngestionScheduler()

        scheduler.setup_daily_ingestion(
            symbols=['AAPL'],
            price_job_time='18:00',
            universe_job_time='18:05',
            feature_job_time='18:10',
        )

        # Should have 3 jobs configured
        assert len(scheduler._jobs) == 3

        # Cleanup
        scheduler.shutdown(wait=False)

    def test_setup_weekly_signal_scan_configures_job(self):
        """
        Setting up weekly signal scan configures job.

        Given:
            - Scheduler instance
        When:
            - setup_weekly_signal_scan is called
        Then:
            - Signal scan job is configured
        """
        from hrp.agents.scheduler import IngestionScheduler

        scheduler = IngestionScheduler()

        scheduler.setup_weekly_signal_scan(
            scan_time='19:00',
            day_of_week='mon',
            ic_threshold=0.03,
            create_hypotheses=True,
        )

        # Should have 1 job configured
        assert len(scheduler._jobs) == 1

        # Cleanup
        scheduler.shutdown(wait=False)

    def test_setup_daily_report_configures_job(self):
        """
        Setting up daily report configures job.

        Given:
            - Scheduler instance
        When:
            - setup_daily_report is called
        Then:
            - Report job is configured
        """
        from hrp.agents.scheduler import IngestionScheduler

        scheduler = IngestionScheduler()

        scheduler.setup_daily_report(report_time='07:00')

        # Should have 1 job configured
        assert len(scheduler._jobs) == 1

        # Cleanup
        scheduler.shutdown(wait=False)


class TestSchedulerJobExecution:
    """Test scheduler job execution."""

    def test_feature_job_initializes(self):
        """
        Feature computation job initializes successfully.

        Given:
            - Job parameters
        When:
            - Creating job
        Then:
            - Job is created
        """
        from hrp.agents.jobs import FeatureComputationJob

        job = FeatureComputationJob(
            symbols=['AAPL'],
            start=date(2020, 12, 1),
            end=date(2020, 12, 31),
        )

        assert job is not None
        assert hasattr(job, 'run')


class TestSchedulerStatus:
    """Test scheduler status checks."""

    def test_scheduler_status_returned(self):
        """
        Scheduler status can be retrieved.

        Given:
            - Scheduler may or may not be running
        When:
            - Getting status
        Then:
            - Status object is returned
            - Contains is_installed flag
        """
        from hrp.utils.scheduler import get_scheduler_status

        status = get_scheduler_status()

        # Should return status object
        assert hasattr(status, 'is_installed')
        assert hasattr(status, 'is_running')
        assert isinstance(status.is_installed, bool)
        assert isinstance(status.is_running, bool)


class TestSchedulerLockDetection:
    """Test scheduler lock detection for dashboard."""

    def test_is_duckdb_lock_error_identifies_locks(self):
        """
        Correctly identifies DuckDB lock errors.

        Given:
            - Various error messages
        When:
            - Checking if lock error
        Then:
            - Lock errors return True
            - Other errors return False
        """
        from hrp.utils.scheduler import is_duckdb_lock_error

        # Test lock error patterns
        lock_errors = [
            "Conflicting lock is held by process",
            "Could not set lock",
            "Database is locked",
            "Lock file error",
        ]

        for error_msg in lock_errors:
            error = Exception(error_msg)
            assert is_duckdb_lock_error(error) is True

        # Test non-lock error
        other_error = Exception("Connection failed")
        assert is_duckdb_lock_error(other_error) is False

    def test_get_lock_holder_pid_extracts_pid(self):
        """
        Extracts PID from lock error message.

        Given:
            - Lock error with PID
        When:
            - Extracting PID
        Then:
            - Correct PID returned
        """
        from hrp.utils.scheduler import get_lock_holder_pid

        error = Exception("Conflicting lock is held by /path/to/db.duckdb (PID 12345) by user fer")
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
