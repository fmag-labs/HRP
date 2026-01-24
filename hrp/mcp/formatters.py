"""
Response formatting utilities for MCP server.

Handles date parsing, DataFrame conversion, and standardized response formatting.
"""

from datetime import date, datetime
from typing import Any

import pandas as pd
from loguru import logger


def parse_date(date_str: str | date | datetime | None) -> date | None:
    """
    Parse a date string into a date object.

    Accepts ISO format strings (YYYY-MM-DD) or returns None if input is None.

    Args:
        date_str: Date string in ISO format, date, datetime, or None

    Returns:
        date object or None

    Raises:
        ValueError: If date string is invalid format
    """
    if date_str is None:
        return None

    # Handle datetime objects first (datetime is subclass of date)
    if isinstance(date_str, datetime):
        return date_str.date()

    if isinstance(date_str, date):
        return date_str

    try:
        # Parse ISO format string
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as e:
        logger.warning(f"Invalid date format: {date_str}")
        raise ValueError(
            f"Invalid date format: '{date_str}'. Expected ISO format (YYYY-MM-DD)."
        ) from e


def df_to_dict(df: pd.DataFrame) -> dict[str, Any]:
    """
    Convert a pandas DataFrame to a dictionary suitable for JSON serialization.

    Handles MultiIndex, date columns, and nested data structures.

    Args:
        df: DataFrame to convert

    Returns:
        Dictionary with 'columns', 'data', and 'shape' keys
    """
    if df is None or df.empty:
        return {
            "columns": [],
            "data": [],
            "shape": [0, 0],
        }

    # Reset index to convert MultiIndex to columns
    df_reset = df.reset_index()

    # Convert date columns to ISO strings
    for col in df_reset.columns:
        if pd.api.types.is_datetime64_any_dtype(df_reset[col]):
            df_reset[col] = df_reset[col].dt.strftime("%Y-%m-%d")
        elif df_reset[col].dtype == "object":
            # Handle date objects in object columns
            df_reset[col] = df_reset[col].apply(
                lambda x: x.isoformat() if isinstance(x, (date, datetime)) else x
            )

    # Convert to records format
    records = df_reset.to_dict(orient="records")

    # Handle NaN values - convert to None for JSON compatibility
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None

    return {
        "columns": list(df_reset.columns),
        "data": records,
        "shape": list(df.shape),
    }


def format_response(
    success: bool,
    data: Any = None,
    message: str = "",
    error: str | None = None,
) -> dict[str, Any]:
    """
    Format a standardized API response.

    All MCP tool responses use this format for consistency.

    Args:
        success: Whether the operation succeeded
        data: Response data (any JSON-serializable type)
        message: Human-readable message
        error: Error message if success=False

    Returns:
        Standardized response dictionary
    """
    response: dict[str, Any] = {
        "success": success,
        "message": message,
    }

    # Handle DataFrame data
    if isinstance(data, pd.DataFrame):
        response["data"] = df_to_dict(data)
    else:
        response["data"] = data

    if error:
        response["error"] = error

    return response


def format_hypothesis(hypothesis: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Format a hypothesis dictionary for output.

    Converts datetime fields to ISO strings.

    Args:
        hypothesis: Hypothesis dictionary from API

    Returns:
        Formatted hypothesis or None
    """
    if hypothesis is None:
        return None

    result = dict(hypothesis)

    # Convert datetime fields
    for field in ["created_at", "updated_at"]:
        if field in result and result[field]:
            if isinstance(result[field], datetime):
                result[field] = result[field].isoformat()
            elif isinstance(result[field], str):
                pass  # Already a string
            else:
                result[field] = str(result[field])

    return result


def format_experiment(experiment: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Format an experiment dictionary for output.

    Converts timestamps and handles nested data.

    Args:
        experiment: Experiment dictionary from API

    Returns:
        Formatted experiment or None
    """
    if experiment is None:
        return None

    result = dict(experiment)

    # Convert timestamp fields (MLflow uses milliseconds since epoch)
    for field in ["start_time", "end_time"]:
        if field in result and result[field]:
            if isinstance(result[field], (int, float)):
                # MLflow timestamps are in milliseconds
                result[field] = datetime.fromtimestamp(
                    result[field] / 1000
                ).isoformat()

    return result


def format_lineage_event(event: dict[str, Any]) -> dict[str, Any]:
    """
    Format a lineage event dictionary for output.

    Args:
        event: Lineage event dictionary from API

    Returns:
        Formatted lineage event
    """
    result = dict(event)

    # Convert timestamp
    if "timestamp" in result and result["timestamp"]:
        if isinstance(result["timestamp"], datetime):
            result["timestamp"] = result["timestamp"].isoformat()
        elif not isinstance(result["timestamp"], str):
            result["timestamp"] = str(result["timestamp"])

    return result
