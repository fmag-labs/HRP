#!/usr/bin/env python3
"""
Backfill Universe Using Public S&P 500 Historical Data.

This script downloads and processes historical S&P 500 constituents
from public sources to create an accurate point-in-time universe.

Data Sources:
- SlickCharts (https://www.slickcharts.com/sp500/) - Current constituents
- Various GitHub repositories with historical data

Usage:
    python scripts/backfill_universe_from_source.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import duckdb
import pandas as pd
from datetime import date
from loguru import logger
from typing import Optional


def fetch_sp500_from_slickcharts() -> set[str]:
    """
    Fetch current S&P 500 constituents from SlickCharts.

    SlickCharts has a clean table of all S&P 500 stocks.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("Install requests and beautifulsoup4: pip install requests beautifulsoup4")
        return set()

    url = "https://www.slickcharts.com/sp500"

    logger.info(f"Fetching S&P 500 from {url}...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table with S&P 500 stocks
    table = soup.find('table', {'class': 'wp-table'})

    if not table:
        logger.error("Could not find S&P 500 table on SlickCharts")
        return set()

    symbols = set()
    rows = table.find_all('tr')[1:]  # Skip header

    for row in rows:
        cells = row.find_all('td')
        if cells:
            symbol = cells[0].get_text(strip=True)
            # Remove dots (used for share classes on some sites)
            symbol = symbol.replace('.', '')
            if symbol:
                symbols.add(symbol)

    logger.info(f"Fetched {len(symbols)} S&P 500 symbols from SlickCharts")
    return symbols


def process_historical_sp500_changes() -> dict[str, list[tuple[str, str, str]]]:
    """
    Process historical S&P 500 changes.

    Returns:
        Dict mapping date to list of (symbol, action, reason) tuples
        action can be 'added', 'removed', or 'readded'
    """
    # For now, return empty - this would be populated from:
    # 1. Wikipedia historical changes page
    # 2. S&P official documentation
    # 3. Third-party datasets

    # TODO: Implement historical change tracking
    logger.warning("Historical change tracking not yet implemented")
    return {}


def backfill_with_current_sp500(min_date: Optional[str] = None,
                                  max_date: Optional[str] = None) -> None:
    """
    Backfill universe using current S&P 500 as base.

    This approach:
    1. Fetches current S&P 500 constituents
    2. For each date in the database, marks these symbols as "in universe"
    3. Applies sector-based exclusions if sector data is available

    This is more accurate than copying all symbols, but still doesn't
    account for historical S&P 500 composition changes.

    Args:
        min_date: Earliest date to backfill (default: first date in prices)
        max_date: Latest date to backfill (default: last date in prices)
    """
    from hrp.data.db import get_db, DEFAULT_DB_PATH

    db_path = str(DEFAULT_DB_PATH)
    conn = duckdb.connect(db_path)

    # Fetch current S&P 500
    sp500_symbols = fetch_sp500_from_slickcharts()

    if not sp500_symbols:
        logger.error("No S&P 500 symbols fetched")
        conn.close()
        return

    # Get date range from prices if not specified
    if not min_date or not max_date:
        date_range = conn.execute("""
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM prices
        """).fetchdf()
        min_date = str(date_range.iloc[0]['min_date'])
        max_date = str(date_range.iloc[0]['max_date'])

    logger.info(f"Backfilling universe from {min_date} to {max_date}")
    logger.info(f"Using {len(sp500_symbols)} current S&P 500 symbols")

    # Clear existing backfilled data (keep original dates)
    original_dates = conn.execute("""
        SELECT DISTINCT date FROM universe
        WHERE date IN ('2024-01-01', '2024-06-01', '2024-12-01',
                      '2026-01-21', '2026-01-24', '2026-01-25',
                      '2026-01-26', '2026-01-27')
    """).fetchdf()

    logger.info(f"Preserving {len(original_dates)} original universe dates")

    conn.execute("""
        DELETE FROM universe
        WHERE date NOT IN ('2024-01-01', '2024-06-01', '2024-12-01',
                          '2026-01-21', '2026-01-24', '2026-01-25',
                          '2026-01-26', '2026-01-27')
    """)

    # Get all dates to process
    dates = conn.execute(f"""
        SELECT DISTINCT date FROM prices
        WHERE date >= '{min_date}' AND date <= '{max_date}'
        ORDER BY date
    """).fetchdf()

    logger.info(f"Processing {len(dates)} dates...")

    # For each date, insert S&P 500 symbols
    batch_size = 100
    processed = 0

    for date_row in dates.itertuples():
        target_date = str(date_row.date)

        # Insert all S&P 500 symbols for this date
        for symbol in sp500_symbols:
            # Check if this symbol has price data for this date
            # (to avoid inserting symbols that didn't exist yet)
            has_price = conn.execute(f"""
                SELECT 1 FROM prices
                WHERE symbol = '{symbol}' AND date = '{target_date}'
                LIMIT 1
            """).fetchone()

            if has_price:
                conn.execute(f"""
                    INSERT INTO universe (symbol, date, in_universe, created_at)
                    VALUES ('{symbol}', '{target_date}', TRUE, CURRENT_TIMESTAMP)
                """)

        processed += 1
        if processed % batch_size == 0:
            logger.info(f"Processed {processed}/{len(dates)} dates...")

    # Now apply sector-based exclusions
    logger.info("Applying sector-based exclusions...")
    apply_sector_exclusions(conn, min_date, max_date)

    # Verify results
    total_universe = conn.execute("SELECT COUNT(*) FROM universe").fetchone()[0]
    total_dates = conn.execute("SELECT COUNT(DISTINCT date) FROM universe").fetchone()[0]
    in_universe_count = conn.execute("SELECT COUNT(*) FROM universe WHERE in_universe = TRUE").fetchone()[0]

    logger.info(f"Backfill complete!")
    logger.info(f"Total universe records: {total_universe}")
    logger.info(f"Total dates covered: {total_dates}")
    logger.info(f"In universe (tradable): {in_universe_count}")

    # Check specific date
    check_date = '2015-01-05'
    count = conn.execute(f"SELECT COUNT(*) FROM universe WHERE date = '{check_date}' AND in_universe = TRUE").fetchone()[0]
    logger.info(f"{check_date}: {count} symbols in universe")

    conn.close()


def apply_sector_exclusions(conn, min_date: str, max_date: str) -> None:
    """
    Apply sector-based exclusions to universe data.

    Excludes:
    - Financials (banks, insurance, etc.)
    - Real Estate (REITs)
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed, skipping sector exclusions")
        return

    logger.info("Fetching sector data for universe symbols...")

    # Get all unique symbols in universe
    symbols = conn.execute(f"""
        SELECT DISTINCT symbol FROM universe
        WHERE date >= '{min_date}' AND date <= '{max_date}'
        ORDER BY symbol
    """).fetchdf()

    # Fetch sector info
    symbol_sectors = {}
    excluded_sectors = {'Financials', 'Real Estate'}

    for i, row in enumerate(symbols.itertuples()):
        symbol = row.symbol

        if i % 50 == 0:
            logger.info(f"Fetched sectors for {i}/{len(symbols)} symbols...")

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get('sector', 'Unknown')

            symbol_sectors[symbol] = sector
        except Exception as e:
            logger.debug(f"Could not fetch sector for {symbol}: {e}")
            symbol_sectors[symbol] = 'Unknown'

    # Apply exclusions
    logger.info("Applying sector exclusions...")

    for symbol, sector in symbol_sectors.items():
        if sector in excluded_sectors:
            # Update all universe records for this symbol
            exclusion_reason = f'sector_{sector.lower().replace(" ", "_")}'

            conn.execute(f"""
                UPDATE universe
                SET in_universe = FALSE,
                    exclusion_reason = '{exclusion_reason}',
                    sector = '{sector}'
                WHERE symbol = '{symbol}'
                  AND date >= '{min_date}'
                  AND date <= '{max_date}'
            """)

            logger.debug(f"Excluded {symbol} ({sector})")
        else:
            # Update sector info
            conn.execute(f"""
                UPDATE universe
                SET sector = '{sector}'
                WHERE symbol = '{symbol}'
                  AND date >= '{min_date}'
                  AND date <= '{max_date}'
            """)

    # Count exclusions
    excluded_count = conn.execute(f"""
        SELECT COUNT(DISTINCT symbol) FROM universe
        WHERE date >= '{min_date}' AND date <= '{max_date}'
          AND in_universe = FALSE
    """).fetchone()[0]

    logger.info(f"Excluded {excluded_count} symbols based on sector")


def main():
    """Main entry point."""
    logger.info("Starting S&P 500 universe backfill...")

    # First, try to fetch from SlickCharts
    sp500 = fetch_sp500_from_slickcharts()

    if sp500:
        logger.info(f"Successfully fetched {len(sp500)} S&P 500 symbols")
        backfill_with_current_sp500()
    else:
        logger.error("Could not fetch S&P 500 data")

    logger.info("Done!")


if __name__ == "__main__":
    main()
