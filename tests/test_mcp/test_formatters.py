"""Tests for MCP formatters module."""

from datetime import date, datetime

import pandas as pd
import pytest

from hrp.mcp.formatters import (
    df_to_dict,
    format_experiment,
    format_hypothesis,
    format_lineage_event,
    format_response,
    parse_date,
)


class TestParseDate:
    """Tests for parse_date function."""

    def test_parse_date_iso_string(self):
        """Parse ISO format date string."""
        result = parse_date("2023-01-15")
        assert result == date(2023, 1, 15)

    def test_parse_date_none(self):
        """Return None for None input."""
        assert parse_date(None) is None

    def test_parse_date_already_date(self):
        """Return date object unchanged."""
        d = date(2023, 5, 20)
        assert parse_date(d) == d

    def test_parse_date_datetime(self):
        """Extract date from datetime."""
        dt = datetime(2023, 6, 15, 10, 30, 0)
        assert parse_date(dt) == date(2023, 6, 15)

    def test_parse_date_invalid_format(self):
        """Raise ValueError for invalid format."""
        with pytest.raises(ValueError, match="Invalid date format"):
            parse_date("15/01/2023")

    def test_parse_date_invalid_string(self):
        """Raise ValueError for non-date string."""
        with pytest.raises(ValueError, match="Invalid date format"):
            parse_date("not-a-date")


class TestDfToDict:
    """Tests for df_to_dict function."""

    def test_df_to_dict_empty(self):
        """Convert empty DataFrame."""
        df = pd.DataFrame()
        result = df_to_dict(df)
        assert result["columns"] == []
        assert result["data"] == []
        assert result["shape"] == [0, 0]

    def test_df_to_dict_none(self):
        """Handle None input."""
        result = df_to_dict(None)
        assert result["columns"] == []
        assert result["data"] == []

    def test_df_to_dict_simple(self):
        """Convert simple DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = df_to_dict(df)
        assert result["columns"] == ["index", "a", "b"]
        assert len(result["data"]) == 2
        assert result["shape"] == [2, 2]

    def test_df_to_dict_with_dates(self):
        """Convert DataFrame with date column."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "value": [100, 200],
        })
        result = df_to_dict(df)
        assert result["data"][0]["date"] == "2023-01-01"
        assert result["data"][1]["date"] == "2023-01-02"

    def test_df_to_dict_with_nan(self):
        """Convert DataFrame with NaN values to None."""
        df = pd.DataFrame({"a": [1, None, 3]})
        result = df_to_dict(df)
        assert result["data"][1]["a"] is None


class TestFormatResponse:
    """Tests for format_response function."""

    def test_format_response_success(self):
        """Format successful response."""
        result = format_response(
            success=True,
            data={"key": "value"},
            message="Operation completed",
        )
        assert result["success"] is True
        assert result["data"] == {"key": "value"}
        assert result["message"] == "Operation completed"
        assert "error" not in result

    def test_format_response_error(self):
        """Format error response."""
        result = format_response(
            success=False,
            message="Operation failed",
            error="Something went wrong",
        )
        assert result["success"] is False
        assert result["message"] == "Operation failed"
        assert result["error"] == "Something went wrong"

    def test_format_response_with_dataframe(self):
        """Format response containing DataFrame."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = format_response(success=True, data=df, message="Data retrieved")
        assert isinstance(result["data"], dict)
        assert "columns" in result["data"]
        assert "data" in result["data"]


class TestFormatHypothesis:
    """Tests for format_hypothesis function."""

    def test_format_hypothesis_none(self):
        """Return None for None input."""
        assert format_hypothesis(None) is None

    def test_format_hypothesis_basic(self):
        """Format hypothesis with datetime fields."""
        hypothesis = {
            "hypothesis_id": "HYP-2026-001",
            "title": "Test hypothesis",
            "created_at": datetime(2026, 1, 15, 10, 30, 0),
            "updated_at": None,
        }
        result = format_hypothesis(hypothesis)
        assert result["hypothesis_id"] == "HYP-2026-001"
        assert "2026-01-15" in result["created_at"]
        assert result["updated_at"] is None

    def test_format_hypothesis_string_dates(self):
        """Handle already-string datetime fields."""
        hypothesis = {
            "hypothesis_id": "HYP-2026-002",
            "created_at": "2026-01-15T10:30:00",
            "updated_at": "2026-01-16T12:00:00",
        }
        result = format_hypothesis(hypothesis)
        assert result["created_at"] == "2026-01-15T10:30:00"


class TestFormatExperiment:
    """Tests for format_experiment function."""

    def test_format_experiment_none(self):
        """Return None for None input."""
        assert format_experiment(None) is None

    def test_format_experiment_with_timestamps(self):
        """Convert millisecond timestamps to ISO format."""
        # MLflow uses millisecond timestamps
        experiment = {
            "experiment_id": "abc123",
            "start_time": 1705312200000,  # 2024-01-15 10:30:00 UTC
            "end_time": 1705313100000,  # 2024-01-15 10:45:00 UTC
        }
        result = format_experiment(experiment)
        assert "experiment_id" in result
        assert "T" in result["start_time"]  # ISO format contains T separator


class TestFormatLineageEvent:
    """Tests for format_lineage_event function."""

    def test_format_lineage_event_datetime(self):
        """Format lineage event with datetime timestamp."""
        event = {
            "lineage_id": 1,
            "event_type": "hypothesis_created",
            "timestamp": datetime(2026, 1, 15, 10, 30, 0),
            "actor": "user",
        }
        result = format_lineage_event(event)
        assert "2026-01-15" in result["timestamp"]

    def test_format_lineage_event_string_timestamp(self):
        """Handle already-string timestamp."""
        event = {
            "lineage_id": 2,
            "event_type": "backtest_run",
            "timestamp": "2026-01-15T10:30:00",
            "actor": "agent:claude-interactive",
        }
        result = format_lineage_event(event)
        assert result["timestamp"] == "2026-01-15T10:30:00"
