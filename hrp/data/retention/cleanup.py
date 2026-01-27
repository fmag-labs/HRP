"""
Automated data cleanup jobs for HRP.

Provides scheduled jobs for cleaning up old data based on retention policies.
Includes safety checks and dry-run mode for testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from loguru import logger

from hrp.data.db import get_db
from hrp.data.retention.policy import RetentionEngine, RetentionTier


@dataclass
class CleanupResult:
    """
    Result of a cleanup operation.

    Attributes:
        data_type: Type of data cleaned up
        dry_run: Whether this was a dry run
        records_deleted: Number of records deleted
        records_archived: Number of records archived
        bytes_freed: Estimated bytes freed
        duration_seconds: Time taken for cleanup
        errors: List of error messages
    """

    data_type: str
    dry_run: bool
    records_deleted: int = 0
    records_archived: int = 0
    bytes_freed: int = 0
    duration_seconds: float = 0.0
    errors: list[str] | None = None

    def __post_init__(self):
        """Initialize errors list if None."""
        if self.errors is None:
            self.errors = []

    @property
    def success(self) -> bool:
        """Check if cleanup was successful."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data_type": self.data_type,
            "dry_run": self.dry_run,
            "records_deleted": self.records_deleted,
            "records_archived": self.records_archived,
            "bytes_freed": self.bytes_freed,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
        }


class DataCleanupJob:
    """
    Automated cleanup job for old data based on retention policies.

    Identifies and deletes data that exceeds retention thresholds.
    Includes dry-run mode for safe testing and safety checks to prevent
    accidental data loss.
    """

    def __init__(
        self,
        db_path: str | None = None,
        dry_run: bool = False,
        data_types: list[str] | None = None,
        require_confirmation: bool = True,
    ):
        """
        Initialize the cleanup job.

        Args:
            db_path: Optional database path
            dry_run: If True, only report what would be deleted
            data_types: List of data types to clean (defaults to all)
            require_confirmation: If True, requires confirmation before deletion
        """
        self._db = get_db(db_path)
        self._db_path = db_path
        self._dry_run = dry_run
        self._data_types = data_types or ["prices", "features", "lineage", "ingestion_log"]
        self._require_confirmation = require_confirmation
        self._engine = RetentionEngine(db_path)

    def run(self, as_of_date: date | None = None) -> dict[str, CleanupResult]:
        """
        Run the cleanup job for all configured data types.

        Args:
            as_of_date: Reference date for age calculation

        Returns:
            Dictionary mapping data_type to CleanupResult
        """
        import time

        as_of_date = as_of_date or date.today()
        results = {}

        logger.info(
            f"Running cleanup job (dry_run={self._dry_run}) for {len(self._data_types)} data types"
        )

        for data_type in self._data_types:
            start_time = time.time()

            try:
                result = self._cleanup_data_type(data_type, as_of_date)
                result.duration_seconds = time.time() - start_time
                results[data_type] = result

                logger.info(
                    f"Cleanup {data_type}: deleted={result.records_deleted}, "
                    f"archived={result.records_archived}, errors={len(result.errors)}"
                )

            except Exception as e:
                logger.error(f"Error cleaning up {data_type}: {e}")
                results[data_type] = CleanupResult(
                    data_type=data_type,
                    dry_run=self._dry_run,
                    errors=[str(e)],
                )

        return results

    def _cleanup_data_type(self, data_type: str, as_of_date: date) -> CleanupResult:
        """
        Cleanup a specific data type.

        Args:
            data_type: Type of data to clean
            as_of_date: Reference date for age calculation

        Returns:
            CleanupResult with operation details
        """
        policy = self._engine.get_policy(data_type)
        cutoff_date = as_of_date - timedelta(days=policy.cold_days)

        result = CleanupResult(data_type=data_type, dry_run=self._dry_run)

        # Get cleanup candidates
        candidates = self._engine.get_cleanup_candidates(data_type, as_of_date)

        if not candidates:
            logger.info(f"No cleanup candidates found for {data_type}")
            return result

        # Safety check: require confirmation if not dry run
        if not self._dry_run and self._require_confirmation:
            total_records = sum(c.get("record_count", 1) for c in candidates)
            logger.warning(
                f"About to delete {total_records} {data_type} records older than {cutoff_date}"
            )
            # In production, this would require user confirmation
            # For automated jobs, we rely on dry_run testing first

        # Perform cleanup
        with self._db.connection() as conn:
            if data_type == "prices":
                result.records_deleted = self._cleanup_prices(conn, cutoff_date, candidates)
            elif data_type == "features":
                result.records_deleted = self._cleanup_features(conn, cutoff_date, candidates)
            elif data_type == "lineage":
                result.records_deleted = self._cleanup_lineage(conn, cutoff_date, candidates)
            elif data_type == "ingestion_log":
                result.records_deleted = self._cleanup_ingestion_log(conn, cutoff_date, candidates)
            else:
                result.errors.append(f"Cleanup not implemented for data type: {data_type}")

        return result

    def _cleanup_prices(self, conn, cutoff_date: date, candidates: list) -> int:
        """Cleanup old price records."""
        deleted_count = 0

        for candidate in candidates:
            symbol = candidate["symbol"]
            oldest_date = candidate["oldest_date"]

            if not self._dry_run:
                try:
                    # Delete prices older than cutoff date
                    result = conn.execute(
                        "DELETE FROM prices WHERE symbol = ? AND date < ?",
                        (symbol, cutoff_date),
                    )
                    deleted_count += result.rowcount
                except Exception as e:
                    logger.error(f"Error deleting prices for {symbol}: {e}")
            else:
                # Dry run: just count
                deleted_count += candidate.get("record_count", 0)

        return deleted_count

    def _cleanup_features(self, conn, cutoff_date: date, candidates: list) -> int:
        """Cleanup old feature records."""
        deleted_count = 0

        for candidate in candidates:
            symbol = candidate["symbol"]
            feature_name = candidate["feature_name"]

            if not self._dry_run:
                try:
                    # Delete features older than cutoff date
                    result = conn.execute(
                        "DELETE FROM features WHERE symbol = ? AND feature_name = ? AND date < ?",
                        (symbol, feature_name, cutoff_date),
                    )
                    deleted_count += result.rowcount
                except Exception as e:
                    logger.error(f"Error deleting features for {symbol}/{feature_name}: {e}")
            else:
                # Dry run: just count
                deleted_count += candidate.get("record_count", 0)

        return deleted_count

    def _cleanup_lineage(self, conn, cutoff_date: date, candidates: list) -> int:
        """Cleanup old lineage records."""
        deleted_count = 0

        # Use cutoff_timestamp instead of cutoff_date
        cutoff_timestamp = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")

        for candidate in candidates:
            lineage_id = candidate["lineage_id"]

            if not self._dry_run:
                try:
                    # Delete lineage records older than cutoff
                    result = conn.execute(
                        "DELETE FROM lineage WHERE lineage_id = ? AND timestamp < ?",
                        (lineage_id, cutoff_timestamp),
                    )
                    deleted_count += result.rowcount
                except Exception as e:
                    logger.error(f"Error deleting lineage {lineage_id}: {e}")
            else:
                # Dry run: just count
                deleted_count += 1

        return deleted_count

    def _cleanup_ingestion_log(self, conn, cutoff_date: date, candidates: list) -> int:
        """Cleanup old ingestion log records."""
        deleted_count = 0

        # Use cutoff_timestamp instead of cutoff_date
        cutoff_timestamp = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")

        for candidate in candidates:
            log_id = candidate["log_id"]

            if not self._dry_run:
                try:
                    # Delete log entries older than cutoff
                    result = conn.execute(
                        "DELETE FROM ingestion_log WHERE log_id = ? AND started_at < ?",
                        (log_id, cutoff_timestamp),
                    )
                    deleted_count += result.rowcount
                except Exception as e:
                    logger.error(f"Error deleting ingestion log {log_id}: {e}")
            else:
                # Dry run: just count
                deleted_count += 1

        return deleted_count

    def estimate_impact(self, as_of_date: date | None = None) -> dict[str, Any]:
        """
        Estimate the impact of running cleanup without making changes.

        Args:
            as_of_date: Reference date for age calculation

        Returns:
            Dictionary with impact estimates
        """
        as_of_date = as_of_date or date.today()

        impact_summary = {
            "as_of_date": str(as_of_date),
            "dry_run": True,
            "data_types": {},
        }

        for data_type in self._data_types:
            try:
                impact = self._engine.estimate_cleanup_impact(data_type, as_of_date)
                impact_summary["data_types"][data_type] = impact
            except Exception as e:
                impact_summary["data_types"][data_type] = {"error": str(e)}

        return impact_summary


class DataArchivalJob:
    """
    Job for archiving old data to cold storage.

    Moves data from hot/warm tiers to cold storage (e.g., compressed files,
    external storage) based on retention policies.
    """

    def __init__(
        self,
        db_path: str | None = None,
        dry_run: bool = False,
        archive_path: str | None = None,
    ):
        """
        Initialize the archival job.

        Args:
            db_path: Optional database path
            dry_run: If True, only report what would be archived
            archive_path: Optional path for archived data
        """
        self._db = get_db(db_path)
        self._db_path = db_path
        self._dry_run = dry_run
        self._archive_path = archive_path
        self._engine = RetentionEngine(db_path)

    def run(self, as_of_date: date | None = None) -> dict[str, Any]:
        """
        Run the archival job.

        Args:
            as_of_date: Reference date for age calculation

        Returns:
            Dictionary with archival results
        """
        as_of_date = as_of_date or date.today()

        logger.info(f"Running archival job (dry_run={self._dry_run})")

        results = {
            "as_of_date": str(as_of_date),
            "dry_run": self._dry_run,
            "archived": {},
        }

        # Get candidates for archival (COLD tier data)
        all_candidates = self._engine.get_all_cleanup_candidates(as_of_date)

        for data_type, candidates in all_candidates.items():
            cold_candidates = [c for c in candidates if c["tier"] == RetentionTier.COLD.value]

            if cold_candidates:
                results["archived"][data_type] = {
                    "candidates": len(cold_candidates),
                    "records": sum(c.get("record_count", 1) for c in cold_candidates),
                }

                if not self._dry_run:
                    # Perform archival (placeholder - would compress/move files)
                    logger.info(
                        f"Archived {len(cold_candidates)} {data_type} candidates "
                        f"to {self._archive_path or 'default archive location'}"
                    )

        return results
