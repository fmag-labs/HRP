#!/usr/bin/env python3
"""
Backfill universe table for all dates with price data.

The universe table is required for symbol validation. When querying historical
dates without universe entries, symbols are rejected even if price data exists.

This script populates the universe table for all dates where price data exists
by using the most recent available universe snapshot as a proxy.

Usage:
    python scripts/backfill_universe.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import duckdb
from loguru import logger


def backfill_universe(db_path: str = None) -> None:
    """
    Backfill universe table for all dates with price data.

    Args:
        db_path: Path to database (uses default if not specified)
    """
    if db_path is None:
        from hrp.data.db import DEFAULT_DB_PATH
        db_path = str(DEFAULT_DB_PATH)

    logger.info(f"Connecting to database: {db_path}")
    conn = duckdb.connect(db_path)

    # Get all dates that have price data
    dates_with_prices = conn.execute("""
        SELECT DISTINCT date FROM prices
        ORDER BY date
    """).fetchdf()

    logger.info(f"Found {len(dates_with_prices)} dates with price data")

    # Get dates that already have universe
    dates_with_universe = conn.execute("""
        SELECT DISTINCT date FROM universe
        ORDER BY date
    """).fetchdf()

    logger.info(f"Found {len(dates_with_universe)} dates with universe data")

    # Find missing dates
    all_dates = set(dates_with_prices['date'].astype(str))
    existing_dates = set(dates_with_universe['date'].astype(str))
    missing_dates = sorted(all_dates - existing_dates)

    logger.info(f"Dates needing universe backfill: {len(missing_dates)}")

    if not missing_dates:
        logger.info("No backfill needed - universe already complete!")
        conn.close()
        return

    # Get the most recent universe snapshot to use as template
    latest_universe_date = conn.execute("""
        SELECT MAX(date) as max_date FROM universe WHERE in_universe = TRUE
    """).fetchone()[0]

    logger.info(f"Using universe from {latest_universe_date} as template")

    # For each missing date, copy the latest universe
    # Note: This is a simplification. In production, you'd track historical
    # universe changes (additions/removals) over time.
    insert_count = 0
    batch_size = 100

    for i, target_date in enumerate(missing_dates, 1):
        conn.execute(f"""
            INSERT INTO universe (symbol, date, in_universe, exclusion_reason, sector, market_cap, created_at)
            SELECT
                symbol,
                '{target_date}' as date,
                in_universe,
                exclusion_reason,
                sector,
                market_cap,
                CURRENT_TIMESTAMP as created_at
            FROM universe
            WHERE date = '{latest_universe_date}' AND in_universe = TRUE
        """)

        if i % batch_size == 0:
            logger.info(f"Processed {i}/{len(missing_dates)} dates...")

    # Verify results
    total_universe = conn.execute("SELECT COUNT(*) FROM universe").fetchone()[0]
    logger.info(f"Total universe records after backfill: {total_universe}")

    # Check specific date that was failing
    jan_17_count = conn.execute("SELECT COUNT(*) FROM universe WHERE date = '2025-01-17' AND in_universe = TRUE").fetchone()[0]
    logger.info(f"2025-01-17 universe records: {jan_17_count}")

    conn.close()
    logger.info("Universe backfill complete!")


if __name__ == "__main__":
    backfill_universe()
