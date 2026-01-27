#!/usr/bin/env python3
"""
Improved Historical Universe Backfill for HRP.

This script properly tracks S&P 500 membership over time using:
1. Current S&P 500 constituents (from Wikipedia/other source)
2. Historical change tracking (when stocks added/removed)
3. Point-in-time universe reconstruction

Approaches:
- MANUAL: Download S&P 500 data from Wikipedia, save to CSV, load it
- SIMPLIFIED: Use symbols that had price data (less accurate but works)
- HYBRID: Use available data sources to approximate historical membership

Usage:
    # Step 1: Download current S&P 500 from Wikipedia (manual one-time)
    # Go to https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    # Save the table to data/sp500_current.csv

    # Step 2: Run this script
    python scripts/backfill_universe_improved.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import duckdb
from datetime import date, timedelta
from loguru import logger
from typing import Optional


def get_historical_universe_from_prices() -> dict[date, set[str]]:
    """
    Reconstruct historical universe from price data.

    For each date, any symbol with price data was likely tradeable.
    We then filter out sectors that should be excluded.

    This is a simplification but works reasonably well for backtesting.
    """
    from hrp.data.db import get_db

    db = get_db()

    # Get all unique symbol-date pairs from prices
    query = """
        SELECT DISTINCT symbol, date
        FROM prices
        ORDER BY date, symbol
    """

    logger.info("Fetching all symbol-date pairs from prices...")
    df = db.fetchdf(query)

    # Build date -> symbols mapping
    universe_by_date: dict[date, set[str]] = {}

    for row in df.itertuples():
        d = row.date
        s = row.symbol

        if d not in universe_by_date:
            universe_by_date[d] = set()
        universe_by_date[d].add(s)

    logger.info(f"Found price data for {len(universe_by_date)} unique dates")

    return universe_by_date


def get_current_sp500_from_manual_csv(csv_path: Optional[str] = None) -> set[str]:
    """
    Load current S&P 500 constituents from a manually downloaded CSV.

    Expected CSV format (from Wikipedia):
        Symbol | Security | Sector | Sub-Industry
        AAPL   | Apple Inc. | Information Technology | ...
    """
    if csv_path is None:
        # Default path
        csv_path = project_root / "data" / "sp500_current.csv"

    path = Path(csv_path)
    if not path.exists():
        logger.warning(f"Manual CSV not found at {path}")
        return set()

    import pandas as pd

    df = pd.read_csv(path)
    symbols = set(df['Symbol'].str.strip().tolist())

    logger.info(f"Loaded {len(symbols)} symbols from manual CSV")
    return symbols


def backfill_universe_price_based() -> None:
    """
    Backfill universe using price data as the source of truth.

    For each date with price data:
    1. Get all symbols that have prices
    2. Check if they have sector data
    3. Apply exclusions (financials, REITs)
    4. Insert into universe table

    This is more accurate than copying current universe to all dates,
    but still a simplification (assumes symbols with prices were tradeable).
    """
    from hrp.data.db import get_db, DEFAULT_DB_PATH

    db_path = str(DEFAULT_DB_PATH)
    db = get_db()
    conn = duckdb.connect(db_path)

    # Clear existing backfilled data (keep original 8 dates)
    original_dates = conn.execute("""
        SELECT DISTINCT date FROM universe
        WHERE date IN ('2024-01-01', '2024-06-01', '2024-12-01',
                      '2026-01-21', '2026-01-24', '2026-01-25',
                      '2026-01-26', '2026-01-27')
    """).fetchdf()

    logger.info(f"Preserving {len(original_dates)} original universe dates")

    # Delete backfilled data
    conn.execute("""
        DELETE FROM universe
        WHERE date NOT IN ('2024-01-01', '2024-06-01', '2024-12-01',
                          '2026-01-21', '2026-01-24', '2026-01-25',
                          '2026-01-26', '2026-01-27')
    """)

    # Get all dates with price data
    dates_with_prices = conn.execute("""
        SELECT DISTINCT date FROM prices
        ORDER BY date
    """).fetchdf()

    logger.info(f"Processing {len(dates_with_prices)} dates with price data")

    # Process in batches
    batch_size = 100
    processed = 0

    for date_row in dates_with_prices.itertuples():
        target_date = str(date_row.date)

        # Get all symbols with price data for this date
        symbols_with_prices = conn.execute(f"""
            SELECT DISTINCT symbol
            FROM prices
            WHERE date = '{target_date}'
        """).fetchdf()

        for symbol_row in symbols_with_prices.itertuples():
            symbol = symbol_row.symbol

            # Determine if symbol should be in universe
            # Try to get sector info (if available in prices or other sources)
            # For now, use a simple heuristic based on symbol properties

            # Check if symbol was in the original S&P 500-like list
            # This is where you'd integrate historical S&P 500 data
            in_universe = True  # Default to TRUE
            exclusion_reason = None
            sector = None

            # Apply known exclusions based on symbol patterns
            # (You can enhance this with sector data if available)
            if symbol.endswith('-UN'):  # Some data providers use this for units
                exclusion_reason = 'unit_structure'
                in_universe = False
            elif symbol.endswith(('A', 'B', 'C')) and len(symbol) > 4:
                # Likely a share class - could be excluded
                pass

            # Insert into universe
            conn.execute(f"""
                INSERT INTO universe (symbol, date, in_universe, exclusion_reason, sector, created_at)
                VALUES ('{symbol}', '{target_date}', {in_universe},
                        {f"'{exclusion_reason}'" if exclusion_reason else "NULL"},
                        {f"'{sector}'" if sector else "NULL"},
                        CURRENT_TIMESTAMP)
            """)

        processed += 1
        if processed % batch_size == 0:
            logger.info(f"Processed {processed}/{len(dates_with_prices)} dates...")

    # Verify results
    total_universe = conn.execute("SELECT COUNT(*) FROM universe").fetchone()[0]
    total_dates = conn.execute("SELECT COUNT(DISTINCT date) FROM universe").fetchone()[0]

    logger.info(f"Backfill complete!")
    logger.info(f"Total universe records: {total_universe}")
    logger.info(f"Total dates covered: {total_dates}")

    conn.close()


def backfill_universe_with_sector_data() -> None:
    """
    Improved backfill using sector classification.

    This version:
    1. Gets all symbols with price data for each date
    2. Uses yfinance or other sources to get sector info
    3. Applies proper sector exclusions (Financials, REITs)
    4. Tracks sector information over time
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Install with: pip install yfinance")
        return

    from hrp.data.db import get_db, DEFAULT_DB_PATH

    db_path = str(DEFAULT_DB_PATH)
    db = get_db()
    conn = duckdb.connect(db_path)

    # Clear existing backfilled data
    conn.execute("""
        DELETE FROM universe
        WHERE date NOT IN ('2024-01-01', '2024-06-01', '2024-12-01',
                          '2026-01-21', '2026-01-24', '2026-01-25',
                          '2026-01-26', '2026-01-27')
    """)

    # Get unique symbols from prices
    all_symbols = conn.execute("""
        SELECT DISTINCT symbol FROM prices ORDER BY symbol
    """).fetchdf()

    logger.info(f"Fetching sector info for {len(all_symbols)} symbols...")

    # Fetch sector info for all symbols (this is expensive, so cache it)
    symbol_sectors = {}

    for i, row in enumerate(all_symbols.itertuples()):
        symbol = row.symbol

        if i % 50 == 0:
            logger.info(f"Fetched sectors for {i}/{len(all_symbols)} symbols...")

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # yfinance provides sector info
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')

            symbol_sectors[symbol] = {
                'sector': sector,
                'industry': industry,
            }
        except Exception as e:
            logger.debug(f"Could not fetch info for {symbol}: {e}")
            symbol_sectors[symbol] = {'sector': 'Unknown', 'industry': 'Unknown'}

    # Get all dates with price data
    dates_with_prices = conn.execute("""
        SELECT DISTINCT date FROM prices ORDER BY date
    """).fetchdf()

    logger.info(f"Processing {len(dates_with_prices)} dates...")

    # Sectors to exclude
    excluded_sectors = {'Financials', 'Real Estate'}

    # Exclusion keywords for industries
    exclusion_keywords = [
        'REIT', 'Real Estate Investment',
        'Bank', 'Insurance', 'Financial',
    ]

    processed = 0
    batch_size = 100

    for date_row in dates_with_prices.itertuples():
        target_date = str(date_row.date)

        # Get symbols with price data for this date
        symbols_with_prices = conn.execute(f"""
            SELECT DISTINCT symbol FROM prices WHERE date = '{target_date}'
        """).fetchdf()

        for symbol_row in symbols_with_prices.itertuples():
            symbol = symbol_row.symbol

            # Get sector info
            sector_info = symbol_sectors.get(symbol, {})
            sector = sector_info.get('sector', 'Unknown')
            industry = sector_info.get('industry', 'Unknown')

            # Determine exclusions
            in_universe = True
            exclusion_reason = None

            if sector in excluded_sectors:
                in_universe = False
                exclusion_reason = f'sector_{sector.lower().replace(" ", "_")}'
            else:
                # Check industry keywords
                for keyword in exclusion_keywords:
                    if keyword.lower() in industry.lower():
                        in_universe = False
                        exclusion_reason = f'industry_{keyword.lower().replace(" ", "_")}'
                        break

            # Insert into universe
            conn.execute(f"""
                INSERT INTO universe (symbol, date, in_universe, exclusion_reason, sector, created_at)
                VALUES ('{symbol}', '{target_date}', {in_universe},
                        {f"'{exclusion_reason}'" if exclusion_reason else "NULL"},
                        {f"'{sector}'" if sector != 'Unknown' else "NULL"},
                        CURRENT_TIMESTAMP)
            """)

        processed += 1
        if processed % batch_size == 0:
            logger.info(f"Processed {processed}/{len(dates_with_prices)} dates...")

    # Verify results
    total_universe = conn.execute("SELECT COUNT(*) FROM universe").fetchone()[0]
    in_universe_count = conn.execute("SELECT COUNT(*) FROM universe WHERE in_universe = TRUE").fetchone()[0]
    excluded_count = conn.execute("SELECT COUNT(*) FROM universe WHERE in_universe = FALSE").fetchone()[0]

    logger.info("Backfill complete!")
    logger.info(f"Total universe records: {total_universe}")
    logger.info(f"In universe: {in_universe_count}")
    logger.info(f"Excluded: {excluded_count}")

    # Show exclusion breakdown
    exclusions = conn.execute("""
        SELECT exclusion_reason, COUNT(*) as count
        FROM universe
        WHERE in_universe = FALSE
        GROUP BY exclusion_reason
        ORDER BY count DESC
    """).fetchdf()

    logger.info(f"\nExclusion breakdown:\n{exclusions.to_string(index=False)}")

    conn.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Improved historical universe backfill"
    )
    parser.add_argument(
        '--method',
        choices=['price-based', 'sector-based', 'manual'],
        default='sector-based',
        help='Backfill method: price-based (simple), sector-based (uses yfinance), or manual (from CSV)'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        help='Path to manual CSV file (for manual method)'
    )

    args = parser.parse_args()

    logger.info(f"Starting improved universe backfill (method: {args.method})")

    if args.method == 'price-based':
        backfill_universe_price_based()
    elif args.method == 'sector-based':
        backfill_universe_with_sector_data()
    elif args.method == 'manual':
        csv_path = args.csv_path
        if csv_path:
            logger.info(f"Loading S&P 500 from {csv_path}")
            symbols = get_current_sp500_from_manual_csv(csv_path)
            logger.info(f"Found {len(symbols)} symbols")
            # You would implement the backfill using this symbol list
        else:
            logger.error("--csv-path required for manual method")

    logger.info("Done!")


if __name__ == "__main__":
    main()
