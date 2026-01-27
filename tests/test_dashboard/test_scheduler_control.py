"""
Tests for dashboard scheduler control component.

Tests hrp.dashboard.components.scheduler_control module which provides
UI for managing database lock conflicts with the scheduler.
"""

from unittest.mock import patch, MagicMock, call
import pytest

from hrp.dashboard.components.scheduler_control import (
    render_scheduler_conflict,
    render_scheduler_status,
)
from hrp.utils.scheduler import SchedulerStatus


class TestRenderSchedulerConflict:
    """Test render_scheduler_conflict function."""

    def test_returns_false_for_non_lock_errors(self):
        """Returns False when error is not a DuckDB lock error."""
        error = Exception("Table not found")

        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            result = render_scheduler_conflict(error)

        assert result is False
        # No UI should be rendered for non-lock errors
        mock_st.error.assert_not_called()

    def test_detects_duckdb_lock_error(self):
        """Detects DuckDB lock errors and renders UI."""
        error = Exception("Conflicting lock is held by process")

        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            # Two columns calls - first for 2-column, second for 3-column layout
            col1_1, col2_1 = MagicMock(), MagicMock()
            col1_2, col2_2, col3_2 = MagicMock(), MagicMock(), MagicMock()
            mock_st.columns.side_effect = [[col1_1, col2_1], [col1_2, col2_2, col3_2]]
            mock_st.button.return_value = False  # No button clicked

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                with patch('hrp.dashboard.components.scheduler_control.get_lock_holder_pid') as mock_pid:
                    mock_status.return_value = SchedulerStatus(
                        is_installed=True,
                        is_running=True,
                        pid=12345,
                        command="python -m hrp.agents.run_scheduler",
                    )
                    mock_pid.return_value = 12345

                    result = render_scheduler_conflict(error)

        assert result is False  # Conflict not resolved yet
        # UI should be rendered
        mock_st.error.assert_called_once()
        mock_st.markdown.assert_called()

    def test_displays_running_scheduler_status(self):
        """Displays correct status when scheduler is running."""
        error = Exception("Conflicting lock is held by process")

        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            # Two columns calls - first for 2-column, second for 3-column layout
            col1_1, col2_1 = MagicMock(), MagicMock()
            col1_2, col2_2, col3_2 = MagicMock(), MagicMock(), MagicMock()
            mock_st.columns.side_effect = [[col1_1, col2_1], [col1_2, col2_2, col3_2]]
            mock_st.button.return_value = False  # No button clicked

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                with patch('hrp.dashboard.components.scheduler_control.get_lock_holder_pid'):
                    mock_status.return_value = SchedulerStatus(
                        is_installed=True,
                        is_running=True,
                        pid=12345,
                        command="python -m hrp.agents.run_scheduler",
                    )

                    render_scheduler_conflict(error)

        # Verify success message was displayed with running status
        mock_st.success.assert_called()
        success_call = str(mock_st.success.call_args)
        assert "Running" in success_call
        assert "12345" in success_call

    def test_displays_stopped_scheduler_status(self):
        """Displays correct status when scheduler is stopped."""
        error = Exception("Conflicting lock is held by process")

        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            # Two columns calls
            col1_1, col2_1 = MagicMock(), MagicMock()
            col1_2, col2_2, col3_2 = MagicMock(), MagicMock(), MagicMock()
            mock_st.columns.side_effect = [[col1_1, col2_1], [col1_2, col2_2, col3_2]]
            mock_st.button.return_value = False  # No button clicked

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                with patch('hrp.dashboard.components.scheduler_control.get_lock_holder_pid'):
                    mock_status.return_value = SchedulerStatus(
                        is_installed=True,
                        is_running=False,
                    )

                    render_scheduler_conflict(error)

        # Verify info message was displayed
        mock_st.info.assert_called_once()
        info_call = str(mock_st.info.call_args)
        assert "Stopped" in info_call

    def test_displays_warning_when_not_installed(self):
        """Displays warning when scheduler not installed."""
        error = Exception("Conflicting lock is held by process")

        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            # Two columns calls
            col1_1, col2_1 = MagicMock(), MagicMock()
            col1_2, col2_2, col3_2 = MagicMock(), MagicMock(), MagicMock()
            mock_st.columns.side_effect = [[col1_1, col2_1], [col1_2, col2_2, col3_2]]

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                with patch('hrp.dashboard.components.scheduler_control.get_lock_holder_pid'):
                    mock_status.return_value = SchedulerStatus(
                        is_installed=False,
                        is_running=False,
                    )

                    render_scheduler_conflict(error)

        # Verify warning was displayed
        mock_st.warning.assert_called()

    def test_displays_lock_holder_pid_mismatch(self):
        """Displays warning when lock holder PID differs from scheduler PID."""
        error = Exception("Conflicting lock is held by /path (PID 99999)")

        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            # Two columns calls
            col1_1, col2_1 = MagicMock(), MagicMock()
            col1_2, col2_2, col3_2 = MagicMock(), MagicMock(), MagicMock()
            mock_st.columns.side_effect = [[col1_1, col2_1], [col1_2, col2_2, col3_2]]

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                with patch('hrp.dashboard.components.scheduler_control.get_lock_holder_pid') as mock_pid:
                    mock_status.return_value = SchedulerStatus(
                        is_installed=True,
                        is_running=True,
                        pid=12345,
                    )
                    mock_pid.return_value = 99999

                    render_scheduler_conflict(error)

        # Verify warning about mismatched PID
        mock_st.warning.assert_called()
        warning_call = str(mock_st.warning.call_args)
        assert "99999" in warning_call

    def test_stop_button_calls_stop_scheduler(self):
        """Stop button triggers stop_scheduler function."""
        error = Exception("Conflicting lock is held by process")

        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            # Mock columns and button
            col1 = MagicMock()
            col2 = MagicMock()
            col3 = MagicMock()
            mock_st.columns.side_effect = [[col1, col2], [col1, col2, col3]]
            mock_st.button.return_value = True  # Button clicked
            mock_st.spinner.return_value.__enter__ = MagicMock()
            mock_st.spinner.return_value.__exit__ = MagicMock()

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                with patch('hrp.dashboard.components.scheduler_control.get_lock_holder_pid'):
                    with patch('hrp.dashboard.components.scheduler_control.stop_scheduler') as mock_stop:
                        mock_status.return_value = SchedulerStatus(
                            is_installed=True,
                            is_running=True,
                            pid=12345,
                        )
                        mock_stop.return_value = {
                            "success": True,
                            "message": "Scheduler stopped",
                        }

                        render_scheduler_conflict(error)

        # Verify stop_scheduler was called
        mock_stop.assert_called_once()

    def test_start_button_calls_start_scheduler(self):
        """Start button triggers start_scheduler function."""
        error = Exception("Conflicting lock is held by process")

        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            # Mock columns and button
            col1 = MagicMock()
            col2 = MagicMock()
            col3 = MagicMock()
            mock_st.columns.side_effect = [[col1, col2], [col1, col2, col3]]
            mock_st.button.return_value = True  # Button clicked
            mock_st.spinner.return_value.__enter__ = MagicMock()
            mock_st.spinner.return_value.__exit__ = MagicMock()

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                with patch('hrp.dashboard.components.scheduler_control.get_lock_holder_pid'):
                    with patch('hrp.dashboard.components.scheduler_control.start_scheduler') as mock_start:
                        mock_status.return_value = SchedulerStatus(
                            is_installed=True,
                            is_running=False,
                        )
                        mock_start.return_value = {
                            "success": True,
                            "message": "Scheduler started",
                        }

                        render_scheduler_conflict(error)

        # Verify start_scheduler was called
        mock_start.assert_called_once()

    def test_displays_information_expander(self):
        """Displays information section with explanation."""
        error = Exception("Conflicting lock is held by process")

        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            # Two columns calls
            col1_1, col2_1 = MagicMock(), MagicMock()
            col1_2, col2_2, col3_2 = MagicMock(), MagicMock(), MagicMock()
            mock_st.columns.side_effect = [[col1_1, col2_1], [col1_2, col2_2, col3_2]]
            mock_st.button.return_value = False

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                with patch('hrp.dashboard.components.scheduler_control.get_lock_holder_pid'):
                    mock_status.return_value = SchedulerStatus(
                        is_installed=True,
                        is_running=True,
                        pid=12345,
                    )

                    render_scheduler_conflict(error)

        # Verify expander was created
        mock_st.expander.assert_called_once()
        expander_call = str(mock_st.expander.call_args)
        assert "information" in expander_call.lower()

    def test_handles_stop_failure(self):
        """Handles stop_scheduler failure gracefully."""
        error = Exception("Conflicting lock is held by process")

        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            col1 = MagicMock()
            col2 = MagicMock()
            col3 = MagicMock()
            mock_st.columns.side_effect = [[col1, col2], [col1, col2, col3]]
            mock_st.button.return_value = True
            mock_st.spinner.return_value.__enter__ = MagicMock()
            mock_st.spinner.return_value.__exit__ = MagicMock()

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                with patch('hrp.dashboard.components.scheduler_control.get_lock_holder_pid'):
                    with patch('hrp.dashboard.components.scheduler_control.stop_scheduler') as mock_stop:
                        mock_status.return_value = SchedulerStatus(
                            is_installed=True,
                            is_running=True,
                            pid=12345,
                        )
                        mock_stop.return_value = {
                            "success": False,
                            "message": "Failed to stop scheduler",
                        }

                        render_scheduler_conflict(error)

        # Verify error message displayed
        mock_st.error.assert_called()


class TestRenderSchedulerStatus:
    """Test render_scheduler_status function."""

    def test_displays_running_status_with_html(self):
        """Displays running status with custom HTML styling."""
        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                mock_status.return_value = SchedulerStatus(
                    is_installed=True,
                    is_running=True,
                    pid=12345,
                )

                render_scheduler_status()

        # Verify markdown was called with HTML
        mock_st.markdown.assert_called()
        markdown_call = str(mock_st.markdown.call_args)
        assert "Running" in markdown_call
        assert "12345" in markdown_call

    def test_displays_stopped_status_with_html(self):
        """Displays stopped status with custom HTML styling."""
        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            mock_st.button.return_value = False

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                mock_status.return_value = SchedulerStatus(
                    is_installed=True,
                    is_running=False,
                )

                render_scheduler_status()

        # Verify markdown was called with HTML
        mock_st.markdown.assert_called()
        markdown_call = str(mock_st.markdown.call_args)
        assert "Stopped" in markdown_call

    def test_displays_not_installed_message(self):
        """Displays message when scheduler not installed."""
        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                mock_status.return_value = SchedulerStatus(
                    is_installed=False,
                    is_running=False,
                )

                render_scheduler_status()

        # Verify caption was shown
        mock_st.caption.assert_called_once()

    def test_stop_button_in_sidebar(self):
        """Stop button appears in sidebar when scheduler running."""
        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            mock_st.button.return_value = False

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                mock_status.return_value = SchedulerStatus(
                    is_installed=True,
                    is_running=True,
                    pid=12345,
                )

                render_scheduler_status()

        # Verify stop button was created
        mock_st.button.assert_called()
        button_call = str(mock_st.button.call_args)
        assert "Stop" in button_call

    def test_start_button_in_sidebar(self):
        """Start button appears in sidebar when scheduler stopped."""
        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            mock_st.button.return_value = False

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                mock_status.return_value = SchedulerStatus(
                    is_installed=True,
                    is_running=False,
                )

                render_scheduler_status()

        # Verify start button was created
        mock_st.button.assert_called()
        button_call = str(mock_st.button.call_args)
        assert "Start" in button_call

    def test_sidebar_stop_calls_stop_scheduler(self):
        """Sidebar stop button calls stop_scheduler."""
        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            mock_st.button.return_value = True  # Button clicked

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                with patch('hrp.dashboard.components.scheduler_control.stop_scheduler') as mock_stop:
                    mock_status.return_value = SchedulerStatus(
                        is_installed=True,
                        is_running=True,
                        pid=12345,
                    )
                    mock_stop.return_value = {
                        "success": True,
                        "message": "Scheduler stopped",
                    }

                    render_scheduler_status()

        # Verify stop_scheduler was called
        mock_stop.assert_called_once()

    def test_sidebar_start_calls_start_scheduler(self):
        """Sidebar start button calls start_scheduler."""
        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            mock_st.button.return_value = True  # Button clicked

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                with patch('hrp.dashboard.components.scheduler_control.start_scheduler') as mock_start:
                    mock_status.return_value = SchedulerStatus(
                        is_installed=True,
                        is_running=False,
                    )
                    mock_start.return_value = {
                        "success": True,
                        "message": "Scheduler started",
                    }

                    render_scheduler_status()

        # Verify start_scheduler was called
        mock_start.assert_called_once()

    def test_handles_operation_failure(self):
        """Handles scheduler operation failure gracefully."""
        with patch('hrp.dashboard.components.scheduler_control.st') as mock_st:
            mock_st.button.return_value = True

            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status') as mock_status:
                with patch('hrp.dashboard.components.scheduler_control.stop_scheduler') as mock_stop:
                    mock_status.return_value = SchedulerStatus(
                        is_installed=True,
                        is_running=True,
                        pid=12345,
                    )
                    mock_stop.return_value = {
                        "success": False,
                        "message": "Failed to stop",
                    }

                    render_scheduler_status()

        # Verify error message displayed
        mock_st.error.assert_called()


class TestSchedulerControlIntegration:
    """Integration tests for scheduler control component."""

    def test_imports_scheduler_utilities(self):
        """Component correctly imports scheduler utilities."""
        from hrp.dashboard.components import scheduler_control

        assert hasattr(scheduler_control, 'get_scheduler_status')
        assert hasattr(scheduler_control, 'is_duckdb_lock_error')
        assert hasattr(scheduler_control, 'stop_scheduler')
        assert hasattr(scheduler_control, 'start_scheduler')
        assert hasattr(scheduler_control, 'get_lock_holder_pid')

    def test_functions_are_callable(self):
        """Exported functions are callable."""
        assert callable(render_scheduler_conflict)
        assert callable(render_scheduler_status)

    def test_render_scheduler_conflict_signature(self):
        """render_scheduler_conflict has correct signature."""
        import inspect

        sig = inspect.signature(render_scheduler_conflict)
        params = list(sig.parameters.keys())

        assert 'error' in params
        assert len(params) == 1

    def test_render_scheduler_status_signature(self):
        """render_scheduler_status has correct signature."""
        import inspect

        sig = inspect.signature(render_scheduler_status)
        params = list(sig.parameters.keys())

        assert len(params) == 0  # No parameters

    def test_scheduler_conflict_returns_bool(self):
        """render_scheduler_conflict always returns boolean."""
        error = Exception("Some error")

        with patch('hrp.dashboard.components.scheduler_control.st'):
            with patch('hrp.dashboard.components.scheduler_control.is_duckdb_lock_error', return_value=False):
                result = render_scheduler_conflict(error)

        assert isinstance(result, bool)

    def test_scheduler_status_returns_none(self):
        """render_scheduler_status always returns None."""
        with patch('hrp.dashboard.components.scheduler_control.st'):
            with patch('hrp.dashboard.components.scheduler_control.get_scheduler_status'):
                result = render_scheduler_status()

        assert result is None
