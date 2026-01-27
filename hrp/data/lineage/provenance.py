"""
Data provenance tracking for HRP.

Tracks the origin and transformation history of data throughout the system.

Provides:
- Source system tracking
- Transformation history logging
- Quality check results storage
- Data integrity verification with hashes
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from loguru import logger

from hrp.data.db import get_db


@dataclass
class ProvenanceRecord:
    """
    Record of data provenance.

    Attributes:
        provenance_id: Unique identifier
        data_type: Type of data (prices, features, fundamentals, etc.)
        record_identifier: Unique identifier for the record
        source_system: Original source system
        source_timestamp: Timestamp from source system
        transformation_history: JSON list of transformations applied
        quality_checks: JSON of quality check results
        data_hash: Hash of data for integrity verification
        lineage_id: Reference to lineage table
        created_at: When provenance was recorded
    """

    provenance_id: int
    data_type: str
    record_identifier: str
    source_system: str
    source_timestamp: datetime | None
    transformation_history: str  # JSON
    quality_checks: str  # JSON
    data_hash: str
    lineage_id: int | None
    created_at: datetime


class DataProvenance:
    """
    Manages data provenance tracking.

    Tracks where data came from, how it was transformed, and
    validates data integrity with cryptographic hashes.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize the data provenance tracker.

        Args:
            db_path: Optional database path
        """
        self._db = get_db(db_path)
        self._db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Ensure data_provenance table exists."""
        with self._db.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_provenance (
                    provenance_id INTEGER PRIMARY KEY,
                    data_type VARCHAR NOT NULL,
                    record_identifier VARCHAR NOT NULL,
                    source_system VARCHAR NOT NULL,
                    source_timestamp TIMESTAMP,
                    transformation_history JSON,
                    quality_checks JSON,
                    data_hash VARCHAR,
                    lineage_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (lineage_id) REFERENCES lineage(lineage_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_provenance_data_type
                ON data_provenance(data_type)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_provenance_identifier
                ON data_provenance(record_identifier)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_provenance_source
                ON data_provenance(source_system)
                """
            )

    def track_source(
        self,
        data_type: str,
        record_identifier: str,
        source_system: str,
        source_timestamp: datetime | None = None,
        data_content: Any = None,
        lineage_id: int | None = None,
    ) -> int:
        """
        Track the source of a data record.

        Args:
            data_type: Type of data (prices, features, etc.)
            record_identifier: Unique identifier (e.g., 'AAPL_2024-01-15')
            source_system: Original source system name
            source_timestamp: Timestamp from source system
            data_content: Data content for hash calculation
            lineage_id: Optional reference to lineage table

        Returns:
            provenance_id of the tracked record
        """
        # Calculate data hash if content provided
        data_hash = self._calculate_hash(data_content) if data_content else None

        with self._db.connection() as conn:
            # Get next provenance_id
            result = conn.execute(
                "SELECT COALESCE(MAX(provenance_id), 0) + 1 FROM data_provenance"
            ).fetchone()
            provenance_id = result[0]

            conn.execute(
                """
                INSERT INTO data_provenance (
                    provenance_id,
                    data_type,
                    record_identifier,
                    source_system,
                    source_timestamp,
                    transformation_history,
                    quality_checks,
                    data_hash,
                    lineage_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provenance_id,
                    data_type,
                    record_identifier,
                    source_system,
                    source_timestamp,
                    json.dumps([]),  # Empty transformation history
                    json.dumps({}),  # Empty quality checks
                    data_hash,
                    lineage_id,
                ),
            )

        logger.debug(
            f"Tracked data source: {data_type}/{record_identifier} from {source_system}"
        )

        return provenance_id

    def add_transformation(
        self,
        provenance_id: int,
        transformation_type: str,
        transformation_params: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> None:
        """
        Add a transformation step to provenance history.

        Args:
            provenance_id: ID of provenance record
            transformation_type: Type of transformation
            transformation_params: Parameters of transformation
            timestamp: When transformation was applied
        """
        timestamp = timestamp or datetime.now()

        with self._db.connection() as conn:
            # Get current history
            result = conn.execute(
                "SELECT transformation_history FROM data_provenance WHERE provenance_id = ?",
                (provenance_id,),
            ).fetchone()

            if result:
                history = json.loads(result[0]) if result[0] else []
                history.append(
                    {
                        "type": transformation_type,
                        "params": transformation_params,
                        "timestamp": timestamp.isoformat(),
                    }
                )

                # Update history
                conn.execute(
                    "UPDATE data_provenance SET transformation_history = ? WHERE provenance_id = ?",
                    (json.dumps(history), provenance_id),
                )

    def add_quality_check(
        self,
        provenance_id: int,
        check_name: str,
        check_result: dict[str, Any],
    ) -> None:
        """
        Add quality check results to provenance.

        Args:
            provenance_id: ID of provenance record
            check_name: Name of quality check
            check_result: Result of quality check
        """
        with self._db.connection() as conn:
            # Get current quality checks
            result = conn.execute(
                "SELECT quality_checks FROM data_provenance WHERE provenance_id = ?",
                (provenance_id,),
            ).fetchone()

            if result:
                checks = json.loads(result[0]) if result[0] else {}
                checks[check_name] = {
                    "result": check_result,
                    "timestamp": datetime.now().isoformat(),
                }

                # Update quality checks
                conn.execute(
                    "UPDATE data_provenance SET quality_checks = ? WHERE provenance_id = ?",
                    (json.dumps(checks), provenance_id),
                )

    def verify_integrity(self, provenance_id: int, data_content: Any) -> bool:
        """
        Verify data integrity using stored hash.

        Args:
            provenance_id: ID of provenance record
            data_content: Current data content

        Returns:
            True if hash matches, False otherwise
        """
        current_hash = self._calculate_hash(data_content)

        with self._db.connection() as conn:
            result = conn.execute(
                "SELECT data_hash FROM data_provenance WHERE provenance_id = ?",
                (provenance_id,),
            ).fetchone()

            if result and result[0]:
                return result[0] == current_hash

        return False

    def get_provenance(
        self,
        data_type: str | None = None,
        record_identifier: str | None = None,
        source_system: str | None = None,
        limit: int = 100,
    ) -> list[ProvenanceRecord]:
        """
        Get provenance records matching criteria.

        Args:
            data_type: Filter by data type
            record_identifier: Filter by record identifier
            source_system: Filter by source system
            limit: Maximum records to return

        Returns:
            List of ProvenanceRecord
        """
        query = "SELECT * FROM data_provenance WHERE 1=1"
        params = []

        if data_type:
            query += " AND data_type = ?"
            params.append(data_type)

        if record_identifier:
            query += " AND record_identifier = ?"
            params.append(record_identifier)

        if source_system:
            query += " AND source_system = ?"
            params.append(source_system)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._db.connection() as conn:
            results = conn.execute(query, params).fetchall()

        return [
            ProvenanceRecord(
                provenance_id=row[0],
                data_type=row[1],
                record_identifier=row[2],
                source_system=row[3],
                source_timestamp=row[4],
                transformation_history=row[5],
                quality_checks=row[6],
                data_hash=row[7],
                lineage_id=row[8],
                created_at=row[9],
            )
            for row in results
        ]

    def get_transformation_history(
        self,
        provenance_id: int,
    ) -> list[dict[str, Any]]:
        """
        Get the complete transformation history for a provenance record.

        Args:
            provenance_id: ID of provenance record

        Returns:
            List of transformation steps
        """
        with self._db.connection() as conn:
            result = conn.execute(
                "SELECT transformation_history FROM data_provenance WHERE provenance_id = ?",
                (provenance_id,),
            ).fetchone()

            if result and result[0]:
                return json.loads(result[0])

        return []

    @staticmethod
    def _calculate_hash(data: Any) -> str:
        """
        Calculate SHA-256 hash of data.

        Args:
            data: Data to hash (will be converted to JSON string)

        Returns:
            Hexadecimal hash string
        """
        if data is None:
            return ""

        # Convert to JSON string for consistent hashing
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    @staticmethod
    def create_record_identifier(data_type: str, **kwargs) -> str:
        """
        Create a unique record identifier from components.

        Args:
            data_type: Type of data
            **kwargs: Components for identifier (e.g., symbol, date)

        Returns:
            Unique identifier string

        Examples:
            create_record_identifier("prices", symbol="AAPL", date="2024-01-15")
            # Returns: "prices_AAPL_2024-01-15"

            create_record_identifier("features", symbol="AAPL", feature="momentum_20d", date="2024-01-15")
            # Returns: "features_AAPL_momentum_20d_2024-01-15"
        """
        parts = [data_type]

        # Add relevant components based on data type
        if "symbol" in kwargs:
            parts.append(kwargs["symbol"])
        if "date" in kwargs:
            parts.append(str(kwargs["date"]))
        if "feature_name" in kwargs:
            parts.append(kwargs["feature_name"])
        if "feature" in kwargs:
            parts.append(kwargs["feature"])

        return "_".join(parts)
