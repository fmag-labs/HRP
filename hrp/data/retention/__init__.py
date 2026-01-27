"""
Data retention policy management for HRP.

Provides tiered data retention with automated cleanup and archival.

Usage:
    from hrp.data.retention import RetentionPolicy, RetentionEngine

    # Get retention tier for a date
    tier = RetentionEngine.get_tier_for_date(date(2024, 1, 1))

    # Get cleanup candidates
    engine = RetentionEngine()
    candidates = engine.get_cleanup_candidates(
        data_type='prices',
        as_of_date=date.today(),
    )
"""

from hrp.data.retention.policy import RetentionEngine, RetentionPolicy
from hrp.data.retention.cleanup import DataCleanupJob, DataArchivalJob

__all__ = [
    "RetentionPolicy",
    "RetentionPolicy",
    "RetentionEngine",
    "DataCleanupJob",
    "DataArchivalJob",
]
