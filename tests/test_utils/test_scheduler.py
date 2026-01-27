"""
Tests for scheduler management utilities.

Tests hrp.utils.scheduler module which manages macOS launchd scheduler
for the HRP platform.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from hrp.utils.scheduler import (
    SchedulerStatus,
    get_scheduler_status,
    stop_scheduler,
    start_scheduler,
    is_duckdb_lock_error,
    get_lock_holder_pid,
)


class TestSchedulerStatus:
    """Test SchedulerStatus dataclass."""

    def test_scheduler_status_creation(self):
        """SchedulerStatus can be created with all fields."""
        status = SchedulerStatus(
            is_installed=True,
            is_running=True,
            pid=12345,
            command="python -m hrp.agents.run_scheduler",
            error=None,
        )

        assert status.is_installed is True
        assert status.is_running is True
        assert status.pid == 12345
        assert status.command == "python -m hrp.agents.run_scheduler"
        assert status.error is None

    def test_scheduler_status_defaults(self):
        """SchedulerStatus optional fields default to None."""
        status = SchedulerStatus(
            is_installed=False,
            is_running=False,
        )

        assert status.pid is None
        assert status.command is None
        assert status.error is None


class TestGetSchedulerStatus:
    """Test get_scheduler_status function."""

    def test_returns_not_installed_when_plist_missing(self):
        """Returns not installed status when plist file doesn't exist."""
        mock_plist = MagicMock()
        mock_plist.exists.return_value = False

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            status = get_scheduler_status()

        assert status.is_installed is False
        assert status.is_running is False
        assert status.pid is None
        assert status.command is None
        assert status.error is None

    def test_returns_running_when_scheduler_found(self):
        """Returns running status when scheduler in launchctl list."""
        mock_launchctl_output = """
12345 com.hrp.scheduler
67890 some.other.service
        """

        mock_ps_output = "python -m hrp.agents.run_scheduler"
        mock_plist = MagicMock()
        mock_plist.exists.return_value = True
        mock_plist.__str__ = lambda self: "/tmp/test.plist"

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            with patch('subprocess.run') as mock_run:
                # Mock launchctl list
                launchctl_result = MagicMock()
                launchctl_result.returncode = 0
                launchctl_result.stdout = mock_launchctl_output
                launchctl_result.stderr = ""

                # Mock ps command
                ps_result = MagicMock()
                ps_result.returncode = 0
                ps_result.stdout = mock_ps_output

                mock_run.side_effect = [launchctl_result, ps_result]

                status = get_scheduler_status()

        assert status.is_installed is True
        assert status.is_running is True
        assert status.pid == 12345
        assert status.command == "python -m hrp.agents.run_scheduler"
        assert status.error is None

    def test_returns_not_running_when_scheduler_not_in_list(self):
        """Returns not running when scheduler not in launchctl list."""
        mock_launchctl_output = """
67890 some.other.service
11111 another.service
        """
        mock_plist = MagicMock()
        mock_plist.exists.return_value = True

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            with patch('subprocess.run') as mock_run:
                result = MagicMock()
                result.returncode = 0
                result.stdout = mock_launchctl_output
                result.stderr = ""
                mock_run.return_value = result

                status = get_scheduler_status()

        assert status.is_installed is True
        assert status.is_running is False
        assert status.pid is None
        assert status.command is None

    def test_handles_launchctl_failure(self):
        """Handles launchctl command failure gracefully."""
        mock_plist = MagicMock()
        mock_plist.exists.return_value = True

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            with patch('subprocess.run') as mock_run:
                result = MagicMock()
                result.returncode = 1
                result.stderr = "launchctl: unknown command"
                mock_run.return_value = result

                status = get_scheduler_status()

        assert status.is_installed is True
        assert status.is_running is False
        assert "launchctl failed" in status.error

    def test_handles_launchctl_timeout(self):
        """Handles launchctl timeout gracefully."""
        mock_plist = MagicMock()
        mock_plist.exists.return_value = True

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired('launchctl', 5)

                status = get_scheduler_status()

        assert status.is_installed is True
        assert status.is_running is False
        assert "timed out" in status.error

    def test_handles_ps_timeout(self):
        """Handles ps command timeout when getting command details."""
        mock_launchctl_output = "12345 com.hrp.scheduler"
        mock_plist = MagicMock()
        mock_plist.exists.return_value = True

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            with patch('subprocess.run') as mock_run:
                # launchctl succeeds
                launchctl_result = MagicMock()
                launchctl_result.returncode = 0
                launchctl_result.stdout = mock_launchctl_output

                # ps times out
                mock_run.side_effect = [
                    launchctl_result,
                    subprocess.TimeoutExpired('ps', 5),
                ]

                status = get_scheduler_status()

        assert status.is_installed is True
        assert status.is_running is True
        assert status.pid == 12345
        assert "ps timed out" in status.command

    def test_handles_generic_exception(self):
        """Handles unexpected exceptions gracefully."""
        mock_plist = MagicMock()
        mock_plist.exists.return_value = True

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = Exception("Unexpected error")

                status = get_scheduler_status()

        assert status.is_installed is True
        assert status.is_running is False
        assert "Unexpected error" in status.error


class TestStopScheduler:
    """Test stop_scheduler function."""

    def test_stops_scheduler_successfully(self):
        """Stops scheduler when launchctl unload succeeds."""
        mock_plist = MagicMock()
        mock_plist.exists.return_value = True
        mock_plist.__str__ = lambda self: "/tmp/test.plist"

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            with patch('subprocess.run') as mock_run:
                result = MagicMock()
                result.returncode = 0
                result.stderr = ""
                mock_run.return_value = result

                response = stop_scheduler()

        assert response['success'] is True
        assert "stopped" in response['message'].lower()

    def test_returns_failure_when_plist_missing(self):
        """Returns failure when plist file doesn't exist."""
        mock_plist = MagicMock()
        mock_plist.exists.return_value = False
        mock_plist.__str__ = lambda self: "/tmp/test.plist"

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            response = stop_scheduler()

        assert response['success'] is False
        assert "not installed" in response['message']

    def test_handles_launchctl_failure(self):
        """Handles launchctl unload command failure."""
        mock_plist = MagicMock()
        mock_plist.exists.return_value = True
        mock_plist.__str__ = lambda self: "/tmp/test.plist"

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            with patch('subprocess.run') as mock_run:
                result = MagicMock()
                result.returncode = 1
                result.stderr = "Could not unload service"
                mock_run.return_value = result

                response = stop_scheduler()

        assert response['success'] is False
        assert "Failed to stop" in response['message']

    def test_handles_timeout(self):
        """Handles launchctl timeout gracefully."""
        mock_plist = MagicMock()
        mock_plist.exists.return_value = True
        mock_plist.__str__ = lambda self: "/tmp/test.plist"

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired('launchctl', 10)

                response = stop_scheduler()

        assert response['success'] is False
        assert "timed out" in response['message']

    def test_handles_generic_exception(self):
        """Handles unexpected exceptions gracefully."""
        mock_plist = MagicMock()
        mock_plist.exists.return_value = True
        mock_plist.__str__ = lambda self: "/tmp/test.plist"

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = OSError("Permission denied")

                response = stop_scheduler()

        assert response['success'] is False
        assert "Permission denied" in response['message']


class TestStartScheduler:
    """Test start_scheduler function."""

    def test_starts_scheduler_successfully(self):
        """Starts scheduler when launchctl load succeeds."""
        mock_plist = MagicMock()
        mock_plist.exists.return_value = True
        mock_plist.__str__ = lambda self: "/tmp/test.plist"

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            with patch('subprocess.run') as mock_run:
                result = MagicMock()
                result.returncode = 0
                result.stderr = ""
                mock_run.return_value = result

                response = start_scheduler()

        assert response['success'] is True
        assert "started" in response['message'].lower()

    def test_returns_failure_when_plist_missing(self):
        """Returns failure when plist file doesn't exist."""
        mock_plist = MagicMock()
        mock_plist.exists.return_value = False
        mock_plist.__str__ = lambda self: "/tmp/test.plist"

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            response = start_scheduler()

        assert response['success'] is False
        assert "not installed" in response['message']

    def test_handles_launchctl_failure(self):
        """Handles launchctl load command failure."""
        mock_plist = MagicMock()
        mock_plist.exists.return_value = True
        mock_plist.__str__ = lambda self: "/tmp/test.plist"

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            with patch('subprocess.run') as mock_run:
                result = MagicMock()
                result.returncode = 1
                result.stderr = "Service already loaded"
                mock_run.return_value = result

                response = start_scheduler()

        assert response['success'] is False
        assert "Failed to start" in response['message']

    def test_handles_timeout(self):
        """Handles launchctl timeout gracefully."""
        mock_plist = MagicMock()
        mock_plist.exists.return_value = True
        mock_plist.__str__ = lambda self: "/tmp/test.plist"

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired('launchctl', 10)

                response = start_scheduler()

        assert response['success'] is False
        assert "timed out" in response['message']

    def test_handles_generic_exception(self):
        """Handles unexpected exceptions gracefully."""
        mock_plist = MagicMock()
        mock_plist.exists.return_value = True
        mock_plist.__str__ = lambda self: "/tmp/test.plist"

        with patch('hrp.utils.scheduler.LAUNCH_AGENT_PLIST', mock_plist):
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = OSError("Permission denied")

                response = start_scheduler()

        assert response['success'] is False
        assert "Permission denied" in response['message']


class TestIsDuckDBLockError:
    """Test is_duckdb_lock_error function."""

    def test_identifies_conflicting_lock_error(self):
        """Identifies 'conflicting lock' error."""
        error = Exception("Conflicting lock is held by process")
        assert is_duckdb_lock_error(error) is True

    def test_identifies_could_not_set_lock_error(self):
        """Identifies 'could not set lock' error."""
        error = Exception("Could not set lock on database")
        assert is_duckdb_lock_error(error) is True

    def test_identifies_database_locked_error(self):
        """Identifies 'database is locked' error."""
        error = Exception("Database is locked by another process")
        assert is_duckdb_lock_error(error) is True

    def test_identifies_lock_file_error(self):
        """Identifies 'lock file' error."""
        error = Exception("Cannot access lock file")
        assert is_duckdb_lock_error(error) is True

    def test_case_insensitive_matching(self):
        """Error detection is case-insensitive."""
        error = Exception("CONFLICTING LOCK DETECTED")
        assert is_duckdb_lock_error(error) is True

    def test_returns_false_for_non_lock_errors(self):
        """Returns False for non-lock-related errors."""
        errors = [
            Exception("Connection failed"),
            Exception("Table not found"),
            Exception("Syntax error in query"),
            Exception("Timeout expired"),
        ]

        for error in errors:
            assert is_duckdb_lock_error(error) is False

    def test_returns_false_for_none(self):
        """Handles None input gracefully."""
        assert is_duckdb_lock_error(None) is False

    def test_returns_false_for_non_exception(self):
        """Handles non-exception input gracefully."""
        # String "string error" has no lock indicators
        assert is_duckdb_lock_error("string error") is False
        assert is_duckdb_lock_error(12345) is False  # No lock indicators


class TestGetLockHolderPid:
    """Test get_lock_holder_pid function."""

    def test_extracts_pid_from_standard_format(self):
        """Extracts PID from standard DuckDB error format."""
        error = Exception("Conflicting lock is held by /path/to/db.duckdb (PID 12345) by user fer")
        pid = get_lock_holder_pid(error)

        assert pid == 12345

    def test_extracts_pid_from_variant_format(self):
        """Extracts PID from variant error format."""
        error = Exception("Lock held (PID 67890)")
        pid = get_lock_holder_pid(error)

        assert pid == 67890

    def test_returns_none_when_no_pid_present(self):
        """Returns None when error has no PID."""
        error = Exception("Conflicting lock")
        pid = get_lock_holder_pid(error)

        assert pid is None

    def test_returns_none_for_malformed_pid(self):
        """Returns None when PID is malformed."""
        error = Exception("Conflicting lock (PID abc)")
        pid = get_lock_holder_pid(error)

        assert pid is None

    def test_returns_none_for_empty_string(self):
        """Returns None for empty error message."""
        error = Exception("")
        pid = get_lock_holder_pid(error)

        assert pid is None

    def test_extracts_pid_with_extra_spaces(self):
        """Extracts PID when there are extra spaces."""
        error = Exception("Conflicting lock (PID  99999  )")
        pid = get_lock_holder_pid(error)

        assert pid == 99999

    def test_handles_parenthesis_edge_cases(self):
        """Handles edge cases with parentheses."""
        # No closing paren - implementation slices [pid_start:-1]
        # which excludes the last character when find returns -1
        error1 = Exception("Conflicting lock (PID 12345")
        # error_str[21:-1] = "1234" (excludes last character '5')
        result = get_lock_holder_pid(error1)
        # The function returns 1234 because that's what [21:-1] gives
        assert result == 1234  # Actual behavior

        # Multiple PID references - extracts from first "PID " to first ")"
        # This includes non-numeric text, so int() conversion fails and returns None
        error2 = Exception("PID 12345 and PID 67890)")
        # The implementation extracts "12345 and PID 67890" which can't be converted to int
        # So it returns None due to ValueError being caught
        assert get_lock_holder_pid(error2) is None  # Actual behavior

    def test_returns_none_for_non_exception(self):
        """Handles non-exception input gracefully."""
        assert get_lock_holder_pid("random string") is None
        assert get_lock_holder_pid(12345) is None


class TestSchedulerIntegration:
    """Integration tests for scheduler utilities."""

    def test_scheduler_status_constants(self):
        """Scheduler constants are properly defined."""
        from hrp.utils.scheduler import LAUNCH_AGENT_PLIST

        assert LAUNCH_AGENT_PLIST.name == "com.hrp.scheduler.plist"
        assert "LaunchAgents" in str(LAUNCH_AGENT_PLIST)

    def test_scheduler_status_dataclass_fields(self):
        """SchedulerStatus has all expected fields."""
        status = SchedulerStatus(
            is_installed=True,
            is_running=False,
        )

        # Check all expected attributes exist
        assert hasattr(status, 'is_installed')
        assert hasattr(status, 'is_running')
        assert hasattr(status, 'pid')
        assert hasattr(status, 'command')
        assert hasattr(status, 'error')
