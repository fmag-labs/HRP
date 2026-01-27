"""
Lineage query utilities for HRP.

Provides functions for querying data flow, feature dependencies,
and impact analysis for quality issues.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from loguru import logger

from hrp.data.db import get_db


def get_data_flow(
    record_identifier: str,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """
    Get complete data flow from source to consumption.

    Traces the complete history of a data record including:
    - Original source system
    - All transformations applied
    - Quality checks performed
    - Current location/state

    Args:
        record_identifier: Unique identifier for the record
        db_path: Optional database path

    Returns:
        List of dicts describing the data flow

    Example:
        flow = get_data_flow("prices_AAPL_2024-01-15")
        # Returns: [
        #   {'source': 'yfinance', 'timestamp': '...', 'transformations': [...]},
        #   {'source': 'feature_computation', 'timestamp': '...', 'transformations': [...]},
        # ]
    """
    import json

    db = get_db(db_path)
    flow = []

    # Get provenance record
    query = """
        SELECT
            data_type,
            source_system,
            source_timestamp,
            transformation_history,
            quality_checks,
            created_at
        FROM data_provenance
        WHERE record_identifier = ?
        ORDER BY created_at
    """
    results = db.execute(query, (record_identifier,)).fetchall()

    for row in results:
        transformations = json.loads(row[3]) if row[3] else []
        quality_checks = json.loads(row[4]) if row[4] else {}

        flow.append(
            {
                "data_type": row[0],
                "source_system": row[1],
                "source_timestamp": row[2],
                "transformations": transformations,
                "quality_checks": quality_checks,
                "created_at": row[5],
            }
        )

    return flow


def get_feature_dependencies(
    feature_name: str,
    db_path: str | None = None,
) -> dict[str, Any]:
    """
    Get input and derived features for a given feature.

    Analyzes feature computation lineage to determine:
    - What input features are required
    - What derived features depend on this one
    - Computation parameters

    Args:
        feature_name: Name of the feature to analyze
        db_path: Optional database path

    Returns:
        Dictionary with dependencies

    Example:
        deps = get_feature_dependencies("momentum_20d")
        # Returns:
        # {
        #     'inputs': ['close', 'returns_1d'],
        #     'derived': ['momentum_20d_rank', 'momentum_signal'],
        #     'computation_params': {'window': 20},
        # }
    """
    import json

    db = get_db(db_path)

    # Get input features from lineage
    input_features = set()
    input_symbols = set()

    query = """
        SELECT DISTINCT input_features, input_symbols, computation_params
        FROM feature_lineage
        WHERE feature_name = ?
    """
    results = db.execute(query, (feature_name,)).fetchall()

    computation_params = {}
    for row in results:
        if row[0]:  # input_features
            input_features.update(json.loads(row[0]))
        if row[1]:  # input_symbols
            input_symbols.update(json.loads(row[1]))
        if row[2]:  # computation_params
            computation_params.update(json.loads(row[2]))

    # Get derived features (features that use this as input)
    derived_features = []
    query = """
        SELECT DISTINCT feature_name
        FROM feature_lineage
        WHERE input_features LIKE ?
    """
    results = db.execute(query, (f'%"{feature_name}"%',)).fetchall()

    for row in results:
        derived_features.append(row[0])

    return {
        "feature_name": feature_name,
        "inputs": list(input_features),
        "input_symbols": list(input_symbols),
        "derived": derived_features,
        "computation_params": computation_params,
    }


def get_impact_analysis(
    quality_issue: dict[str, Any],
    db_path: str | None = None,
) -> dict[str, Any]:
    """
    Analyze the impact of a data quality issue.

    Determines:
    - Which features are affected
    - Which symbols are affected
    - What downstream computations would be impacted
    - Recommended remediation actions

    Args:
        quality_issue: Quality issue details (symbol, date, feature, etc.)
        db_path: Optional database path

    Returns:
        Dictionary with impact analysis

    Example:
        issue = {'symbol': 'AAPL', 'date': '2024-01-15', 'issue_type': 'missing_data'}
        impact = get_impact_analysis(issue)
        # Returns:
        # {
        #     'affected_features': ['momentum_20d', 'volatility_60d', ...],
        #     'affected_symbols': ['AAPL'],
        #     'downstream_impact': 'High - 15 features depend on this data',
        #     'remediation': 'Re-ingest price data from yfinance',
        # }
    """
    import json

    db = get_db(db_path)

    symbol = quality_issue.get("symbol")
    issue_date = quality_issue.get("date")
    feature_name = quality_issue.get("feature_name")

    affected_features = []
    affected_symbols = []

    # Find features that depend on the affected data
    if symbol:
        query = """
            SELECT DISTINCT feature_name
            FROM feature_lineage
            WHERE (input_symbols LIKE ? OR input_features LIKE ?)
        """
        pattern = f'%"{symbol}"%'
        results = db.execute(query, (pattern, pattern)).fetchall()

        for row in results:
            affected_features.append(row[0])

        affected_symbols.append(symbol)

    # If a specific feature is mentioned, check for derived features
    if feature_name:
        derived = get_feature_dependencies(feature_name, db_path)
        affected_features.extend(derived["derived"])

    # Calculate impact level
    impact_level = "Low"
    if len(affected_features) > 10:
        impact_level = "High"
    elif len(affected_features) > 5:
        impact_level = "Medium"

    # Determine remediation
    remediation = "No action required"
    if quality_issue.get("issue_type") == "missing_data":
        remediation = f"Re-ingest {quality_issue.get('data_type', 'data')} from source"
    elif quality_issue.get("issue_type") == "stale_data":
        remediation = f"Refresh {quality_issue.get('data_type', 'data')} from source"
    elif quality_issue.get("issue_type") == "quality_anomaly":
        remediation = "Investigate data source and apply corrections"

    return {
        "quality_issue": quality_issue,
        "affected_features": list(set(affected_features)),
        "affected_symbols": list(set(affected_symbols)),
        "downstream_impact": f"{impact_level} - {len(affected_features)} features depend on this data",
        "remediation": remediation,
        "impact_level": impact_level,
    }


def get_lineage_summary(
    db_path: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    """
    Get a summary of lineage tracking activity.

    Provides statistics on:
    - Features tracked
    - Computations recorded
    - Data provenance entries
    - Transformation history

    Args:
        db_path: Optional database path
        start_date: Filter by start date
        end_date: Filter by end date

    Returns:
        Dictionary with lineage summary
    """
    db = get_db(db_path)

    summary = {
        "feature_lineage": {},
        "data_provenance": {},
    }

    # Feature lineage summary
    lineage_query = """
        SELECT
            feature_name,
            COUNT(DISTINCT symbol) as unique_symbols,
            COUNT(DISTINCT date) as computation_days,
            COUNT(*) as total_records
        FROM feature_lineage
    """

    if start_date or end_date:
        lineage_query += " WHERE 1=1"
        params = []

        if start_date:
            lineage_query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            lineage_query += " AND date <= ?"
            params.append(end_date)

        lineage_query += " GROUP BY feature_name"

        results = db.execute(lineage_query, tuple(params) if start_date or end_date else ()).fetchall()

        summary["feature_lineage"] = {
            row[0]: {
                "unique_symbols": row[1],
                "computation_days": row[2],
                "total_records": row[3],
            }
            for row in results
        }

    # Provenance summary
    provenance_query = """
        SELECT
            data_type,
            source_system,
            COUNT(*) as total_records,
            COUNT(DISTINCT record_identifier) as unique_records
        FROM data_provenance
        GROUP BY data_type, source_system
    """

    results = db.execute(provenance_query).fetchall()

    summary["data_provenance"] = [
        {
            "data_type": row[0],
            "source_system": row[1],
            "total_records": row[2],
            "unique_records": row[3],
        }
        for row in results
    ]

    return summary
