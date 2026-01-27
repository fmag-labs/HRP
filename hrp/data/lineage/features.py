"""
Feature computation lineage tracking for HRP.

Tracks the complete history of feature computations including:
- Input symbols and features
- Computation parameters
- Performance metrics
- Transformation history
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from loguru import logger

from hrp.data.db import get_db


@dataclass
class FeatureComputationRecord:
    """
    Record of a single feature computation event.

    Attributes:
        lineage_id: Unique identifier for the computation
        feature_name: Name of the computed feature
        symbol: Ticker symbol
        date: Computation date
        version: Feature version (e.g., 'v1')
        computation_source: Source of computation (batch/incremental/manual)
        input_symbols: JSON list of input symbols
        input_features: JSON list of input feature names
        computation_params: JSON of computation parameters
        rows_computed: Number of rows computed
        duration_ms: Computation time in milliseconds
        timestamp: When computation was performed
    """

    lineage_id: int
    feature_name: str
    symbol: str
    date: date
    version: str
    computation_source: str
    input_symbols: str  # JSON
    input_features: str  # JSON
    computation_params: str  # JSON
    rows_computed: int
    duration_ms: float
    timestamp: datetime


class FeatureLineage:
    """
    Manages feature computation lineage tracking.

    Provides methods for tracking feature computations and querying
    computation history.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize the feature lineage tracker.

        Args:
            db_path: Optional database path
        """
        self._db = get_db(db_path)
        self._db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Ensure feature_lineage table exists."""
        with self._db.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_lineage (
                    lineage_id INTEGER PRIMARY KEY,
                    feature_name VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    date DATE NOT NULL,
                    version VARCHAR DEFAULT 'v1',
                    computation_source VARCHAR,
                    input_symbols VARCHAR,  -- JSON
                    input_features VARCHAR,  -- JSON
                    computation_params JSON,
                    rows_computed INTEGER,
                    duration_ms FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol) REFERENCES symbols(symbol)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feature_lineage_feature_symbol
                ON feature_lineage(feature_name, symbol)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feature_lineage_date
                ON feature_lineage(date)
                """
            )

    def track_computation(
        self,
        feature_name: str,
        symbols: list[str],
        computation_date: date,
        computation_source: str = "batch",
        input_symbols: list[str] | None = None,
        input_features: list[str] | None = None,
        computation_params: dict[str, Any] | None = None,
        rows_computed: int = 0,
        duration_ms: float = 0.0,
        version: str = "v1",
    ) -> int:
        """
        Track a feature computation event.

        Args:
            feature_name: Name of the computed feature
            symbols: List of symbols computed
            computation_date: Date of computation
            computation_source: Source (batch/incremental/manual)
            input_symbols: Input symbols for computation
            input_features: Input feature names
            computation_params: Computation parameters
            rows_computed: Number of rows computed
            duration_ms: Computation time
            version: Feature version

        Returns:
            lineage_id of the tracked computation
        """
        import json

        with self._db.connection() as conn:
            # Get next lineage_id
            result = conn.execute(
                "SELECT COALESCE(MAX(lineage_id), 0) + 1 FROM feature_lineage"
            ).fetchone()
            lineage_id = result[0]

            # Insert lineage record for each symbol
            for symbol in symbols:
                conn.execute(
                    """
                    INSERT INTO feature_lineage (
                        lineage_id,
                        feature_name,
                        symbol,
                        date,
                        version,
                        computation_source,
                        input_symbols,
                        input_features,
                        computation_params,
                        rows_computed,
                        duration_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lineage_id,
                        feature_name,
                        symbol,
                        computation_date,
                        version,
                        computation_source,
                        json.dumps(input_symbols or []),
                        json.dumps(input_features or []),
                        json.dumps(computation_params or {}),
                        rows_computed,
                        duration_ms,
                    ),
                )
                lineage_id += 1

        logger.debug(
            f"Tracked feature computation: {feature_name} for {len(symbols)} symbols "
            f"(source={computation_source})"
        )

        return lineage_id - 1  # Return the first lineage_id

    def get_computation_history(
        self,
        feature_name: str | None = None,
        symbol: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 100,
    ) -> list[FeatureComputationRecord]:
        """
        Get computation history for features.

        Args:
            feature_name: Filter by feature name
            symbol: Filter by symbol
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of records to return

        Returns:
            List of FeatureComputationRecord
        """
        query = "SELECT * FROM feature_lineage WHERE 1=1"
        params = []

        if feature_name:
            query += " AND feature_name = ?"
            params.append(feature_name)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._db.connection() as conn:
            results = conn.execute(query, params).fetchall()

        return [
            FeatureComputationRecord(
                lineage_id=row[0],
                feature_name=row[1],
                symbol=row[2],
                date=row[3],
                version=row[4],
                computation_source=row[5],
                input_symbols=row[6],
                input_features=row[7],
                computation_params=row[8],
                rows_computed=row[9],
                duration_ms=row[10],
                timestamp=row[11],
            )
            for row in results
        ]

    def get_feature_statistics(
        self,
        feature_name: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, Any]:
        """
        Get statistics for a feature's computations.

        Args:
            feature_name: Name of the feature
            start_date: Start date for statistics
            end_date: End date for statistics

        Returns:
            Dictionary with feature statistics
        """
        query = """
            SELECT
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(DISTINCT date) as computation_days,
                SUM(rows_computed) as total_rows,
                AVG(duration_ms) as avg_duration_ms,
                MIN(timestamp) as first_computation,
                MAX(timestamp) as last_computation
            FROM feature_lineage
            WHERE feature_name = ?
        """
        params = [feature_name]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        with self._db.connection() as conn:
            result = conn.execute(query, params).fetchone()

            if result and result[0]:
                return {
                    "feature_name": feature_name,
                    "unique_symbols": result[0],
                    "computation_days": result[1],
                    "total_rows": result[2],
                    "avg_duration_ms": float(result[3]) if result[3] else 0.0,
                    "first_computation": result[4],
                    "last_computation": result[5],
                }

        return {
            "feature_name": feature_name,
            "unique_symbols": 0,
            "computation_days": 0,
            "total_rows": 0,
            "avg_duration_ms": 0.0,
        }

    def get_computation_chain(
        self,
        feature_name: str,
        symbol: str,
        computation_date: date,
    ) -> list[dict[str, Any]]:
        """
        Get the complete computation chain for a feature.

        Traces back through input features to show the complete
        dependency graph for a specific feature computation.

        Args:
            feature_name: Name of the feature
            symbol: Ticker symbol
            computation_date: Date of computation

        Returns:
            List of dicts describing the computation chain
        """
        chain = []
        visited = set()

        def trace_back(current_feature: str, current_date: date) -> None:
            """Recursively trace feature dependencies."""
            if (current_feature, current_date) in visited:
                return

            visited.add((current_feature, current_date))

            # Get lineage records for this feature
            records = self.get_computation_history(
                feature_name=current_feature,
                symbol=symbol,
                start_date=current_date,
                end_date=current_date,
                limit=1,
            )

            for record in records:
                import json

                input_features = json.loads(record.input_features) if record.input_features else []
                input_symbols = json.loads(record.input_symbols) if record.input_symbols else []

                chain.append(
                    {
                        "feature": current_feature,
                        "date": current_date,
                        "lineage_id": record.lineage_id,
                        "input_features": input_features,
                        "input_symbols": input_symbols,
                        "params": json.loads(record.computation_params) if record.computation_params else {},
                    }
                )

                # Trace back through input features
                for input_feature in input_features:
                    trace_back(input_feature, current_date)

        trace_back(feature_name, computation_date)
        return chain
