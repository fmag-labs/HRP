"""
Data retention policy engine for HRP.

Defines tiered retention policies for different data types and provides
utilities for determining cleanup candidates and estimating cleanup impact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Any

from loguru import logger

from hrp.data.db import get_db


class RetentionTier(Enum):
    """Retention tiers for data classification."""

    HOT = "hot"  # Frequently accessed active data (90 days)
    WARM = "warm"  # Recent historical data (1 year)
    COLD = "cold"  # Long-term storage for backtesting (3 years)
    ARCHIVE = "archive"  # Compressed archival (5+ years)


@dataclass(frozen=True)
class RetentionPolicy:
    """
    Retention policy configuration for a data type.

    Defines how long data should be kept in each retention tier.
    """

    data_type: str
    hot_days: int
    warm_days: int
    cold_days: int
    archive_days: int | None  # None means never archive

    def get_tier_for_age(self, age_days: int) -> RetentionTier:
        """
        Determine the retention tier for data of a given age.

        Args:
            age_days: Age of data in days

        Returns:
            RetentionTier for the data
        """
        if age_days < self.hot_days:
            return RetentionTier.HOT
        elif age_days < self.warm_days:
            return RetentionTier.WARM
        elif age_days < self.cold_days:
            return RetentionTier.COLD
        elif self.archive_days is None or age_days < self.archive_days:
            return RetentionTier.ARCHIVE
        else:
            # Exceeds archive retention - eligible for deletion
            return RetentionTier.ARCHIVE  # Keep for manual review

    def is_eligible_for_cleanup(self, age_days: int) -> bool:
        """
        Check if data is eligible for cleanup (archival or deletion).

        Args:
            age_days: Age of data in days

        Returns:
            True if data should be cleaned up
        """
        return age_days >= self.cold_days

    def days_until_cleanup(self, age_days: int) -> int:
        """
        Calculate days until data is eligible for cleanup.

        Args:
            age_days: Current age of data in days

        Returns:
            Days until cleanup (negative if already eligible)
        """
        return max(0, self.cold_days - age_days)


# Default retention policies for each data type
DEFAULT_POLICIES = {
    # Prices: 90d hot, 1y warm, 3y cold, 5y archive
    "prices": RetentionPolicy(
        data_type="prices",
        hot_days=90,
        warm_days=365,
        cold_days=365 * 3,
        archive_days=365 * 5,
    ),
    # Features: 90d hot, 1y warm, 3y cold, 5y archive
    "features": RetentionPolicy(
        data_type="features",
        hot_days=90,
        warm_days=365,
        cold_days=365 * 3,
        archive_days=365 * 5,
    ),
    # Fundamentals: 1y hot, 5y warm, 10y cold, never archive
    "fundamentals": RetentionPolicy(
        data_type="fundamentals",
        hot_days=365,
        warm_days=365 * 5,
        cold_days=365 * 10,
        archive_days=None,
    ),
    # Lineage: 30d hot, 90d warm, 1y cold, 2y archive
    "lineage": RetentionPolicy(
        data_type="lineage",
        hot_days=30,
        warm_days=90,
        cold_days=365,
        archive_days=365 * 2,
    ),
    # Ingestion log: 30d hot, 90d warm, 1y cold, 2y archive
    "ingestion_log": RetentionPolicy(
        data_type="ingestion_log",
        hot_days=30,
        warm_days=90,
        cold_days=365,
        archive_days=365 * 2,
    ),
    # Quality reports: 30d hot, 90d warm, 1y cold, 2y archive
    "quality_reports": RetentionPolicy(
        data_type="quality_reports",
        hot_days=30,
        warm_days=90,
        cold_days=365,
        archive_days=365 * 2,
    ),
}


class RetentionEngine:
    """
    Engine for applying retention policies to HRP data.

    Provides methods for identifying cleanup candidates and estimating
    the impact of cleanup operations.
    """

    def __init__(
        self,
        db_path: str | None = None,
        policies: dict[str, RetentionPolicy] | None = None,
    ):
        """
        Initialize the retention engine.

        Args:
            db_path: Optional database path
            policies: Optional custom retention policies (defaults to DEFAULT_POLICIES)
        """
        self._db = get_db(db_path)
        self._db_path = db_path
        self._policies = policies or DEFAULT_POLICIES
        logger.info(f"Retention engine initialized with {len(self._policies)} policies")

    def get_policy(self, data_type: str) -> RetentionPolicy:
        """
        Get the retention policy for a data type.

        Args:
            data_type: Type of data (e.g., 'prices', 'features')

        Returns:
            RetentionPolicy for the data type

        Raises:
            ValueError: If data_type has no defined policy
        """
        if data_type not in self._policies:
            raise ValueError(f"No retention policy defined for data type: {data_type}")
        return self._policies[data_type]

    def get_tier_for_date(
        self,
        data_type: str,
        data_date: date,
        as_of_date: date | None = None,
    ) -> RetentionTier:
        """
        Determine the retention tier for a specific date.

        Args:
            data_type: Type of data
            data_date: Date of the data
            as_of_date: Reference date (defaults to today)

        Returns:
            RetentionTier for the data
        """
        as_of_date = as_of_date or date.today()
        policy = self.get_policy(data_type)
        age_days = (as_of_date - data_date).days
        return policy.get_tier_for_age(age_days)

    def get_cleanup_candidates(
        self,
        data_type: str,
        as_of_date: date | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get records eligible for cleanup (archival or deletion).

        Args:
            data_type: Type of data to check
            as_of_date: Reference date for age calculation
            limit: Maximum number of candidates to return

        Returns:
            List of candidate records with metadata
        """
        as_of_date = as_of_date or date.today()
        policy = self.get_policy(data_type)

        candidates = []

        with self._db.connection() as conn:
            # Build query based on data type
            if data_type == "prices":
                cutoff_date = as_of_date - timedelta(days=policy.cold_days)
                query = """
                    SELECT
                        symbol,
                        MIN(date) as first_date,
                        MAX(date) as last_date,
                        COUNT(*) as record_count,
                        MIN(date) as oldest_date
                    FROM prices
                    WHERE date < ?
                    GROUP BY symbol
                    ORDER BY oldest_date ASC
                """
                results = conn.execute(query, (cutoff_date,)).fetchall()

                for symbol, first_date, last_date, count, oldest_date in results:
                    age_days = (as_of_date - oldest_date).days
                    tier = policy.get_tier_for_age(age_days)
                    candidates.append(
                        {
                            "data_type": data_type,
                            "symbol": symbol,
                            "first_date": first_date,
                            "last_date": last_date,
                            "record_count": count,
                            "oldest_date": oldest_date,
                            "age_days": age_days,
                            "tier": tier.value,
                            "action": "archive" if tier == RetentionTier.COLD else "keep",
                        }
                    )

            elif data_type == "features":
                cutoff_date = as_of_date - timedelta(days=policy.cold_days)
                query = """
                    SELECT
                        symbol,
                        feature_name,
                        MIN(date) as first_date,
                        MAX(date) as last_date,
                        COUNT(*) as record_count,
                        MIN(date) as oldest_date
                    FROM features
                    WHERE date < ?
                    GROUP BY symbol, feature_name
                    ORDER BY oldest_date ASC
                """
                results = conn.execute(query, (cutoff_date,)).fetchall()

                for symbol, feature_name, first_date, last_date, count, oldest_date in results:
                    age_days = (as_of_date - oldest_date).days
                    tier = policy.get_tier_for_age(age_days)
                    candidates.append(
                        {
                            "data_type": data_type,
                            "symbol": symbol,
                            "feature_name": feature_name,
                            "first_date": first_date,
                            "last_date": last_date,
                            "record_count": count,
                            "oldest_date": oldest_date,
                            "age_days": age_days,
                            "tier": tier.value,
                            "action": "archive" if tier == RetentionTier.COLD else "keep",
                        }
                    )

            elif data_type == "lineage":
                cutoff_date = as_of_date - timedelta(days=policy.cold_days)
                query = """
                    SELECT
                        lineage_id,
                        event_type,
                        timestamp,
                        MIN(timestamp) as oldest_timestamp
                    FROM lineage
                    WHERE timestamp < ?
                    GROUP BY lineage_id, event_type, timestamp
                    ORDER BY oldest_timestamp ASC
                """
                results = conn.execute(query, (cutoff_date,)).fetchall()

                for lineage_id, event_type, ts, oldest_ts in results:
                    age_days = (as_of_date - oldest_ts.date()).days if oldest_ts else 0
                    tier = policy.get_tier_for_age(age_days)
                    candidates.append(
                        {
                            "data_type": data_type,
                            "lineage_id": lineage_id,
                            "event_type": event_type,
                            "timestamp": ts,
                            "oldest_timestamp": oldest_ts,
                            "age_days": age_days,
                            "tier": tier.value,
                            "action": "archive" if tier == RetentionTier.COLD else "keep",
                        }
                    )

            elif data_type == "ingestion_log":
                cutoff_date = as_of_date - timedelta(days=policy.cold_days)
                query = """
                    SELECT
                        log_id,
                        source_id,
                        status,
                        started_at,
                        completed_at,
                        MIN(started_at) as oldest_start
                    FROM ingestion_log
                    WHERE started_at < ?
                    GROUP BY log_id, source_id, status, started_at, completed_at
                    ORDER BY oldest_start ASC
                """
                results = conn.execute(query, (cutoff_date,)).fetchall()

                for log_id, source_id, status, started_at, completed_at, oldest_start in results:
                    age_days = (as_of_date - oldest_start.date()).days if oldest_start else 0
                    tier = policy.get_tier_for_age(age_days)
                    candidates.append(
                        {
                            "data_type": data_type,
                            "log_id": log_id,
                            "source_id": source_id,
                            "status": status,
                            "started_at": started_at,
                            "completed_at": completed_at,
                            "age_days": age_days,
                            "tier": tier.value,
                            "action": "archive" if tier == RetentionTier.COLD else "keep",
                        }
                    )

            else:
                logger.warning(f"Cleanup candidates not implemented for data type: {data_type}")
                return []

        # Apply limit if specified
        if limit is not None:
            candidates = candidates[:limit]

        logger.info(f"Found {len(candidates)} cleanup candidates for {data_type}")
        return candidates

    def estimate_cleanup_impact(
        self,
        data_type: str,
        as_of_date: date | None = None,
    ) -> dict[str, Any]:
        """
        Estimate the impact of cleanup operations.

        Args:
            data_type: Type of data to analyze
            as_of_date: Reference date for age calculation

        Returns:
            Dictionary with impact metrics
        """
        as_of_date = as_of_date or date.today()
        policy = self.get_policy(data_type)
        candidates = self.get_cleanup_candidates(data_type, as_of_date)

        total_records = sum(c.get("record_count", 1) for c in candidates)
        oldest_date = min((c["oldest_date"] for c in candidates), default=None)
        newest_date = max((c["last_date"] for c in candidates), default=None)

        # Count by tier
        tier_counts = {}
        for candidate in candidates:
            tier = candidate["tier"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        return {
            "data_type": data_type,
            "total_candidates": len(candidates),
            "total_records": total_records,
            "oldest_date": str(oldest_date) if oldest_date else None,
            "newest_date": str(newest_date) if newest_date else None,
            "tier_distribution": tier_counts,
            "policy": {
                "hot_days": policy.hot_days,
                "warm_days": policy.warm_days,
                "cold_days": policy.cold_days,
                "archive_days": policy.archive_days,
            },
        }

    def get_all_cleanup_candidates(
        self,
        as_of_date: date | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get cleanup candidates for all data types.

        Args:
            as_of_date: Reference date for age calculation

        Returns:
            Dictionary mapping data_type to list of candidates
        """
        all_candidates = {}
        for data_type in self._policies.keys():
            try:
                candidates = self.get_cleanup_candidates(data_type, as_of_date)
                if candidates:
                    all_candidates[data_type] = candidates
            except Exception as e:
                logger.error(f"Error getting cleanup candidates for {data_type}: {e}")

        return all_candidates

    def get_retention_summary(
        self,
        as_of_date: date | None = None,
    ) -> dict[str, Any]:
        """
        Get a summary of retention status across all data types.

        Args:
            as_of_date: Reference date for age calculation

        Returns:
            Dictionary with retention summary
        """
        as_of_date = as_of_date or date.today()

        summary = {
            "as_of_date": str(as_of_date),
            "data_types": {},
        }

        for data_type, policy in self._policies.items():
            try:
                impact = self.estimate_cleanup_impact(data_type, as_of_date)
                summary["data_types"][data_type] = {
                    "policy": {
                        "hot_days": policy.hot_days,
                        "warm_days": policy.warm_days,
                        "cold_days": policy.cold_days,
                        "archive_days": policy.archive_days,
                    },
                    "cleanup_candidates": impact["total_candidates"],
                    "total_records": impact["total_records"],
                    "tier_distribution": impact["tier_distribution"],
                }
            except Exception as e:
                logger.error(f"Error summarizing {data_type}: {e}")
                summary["data_types"][data_type] = {"error": str(e)}

        return summary
