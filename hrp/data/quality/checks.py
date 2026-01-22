"""
Data quality check functions for HRP.

Performs completeness, anomaly, gap, and freshness checks on price data.
"""

from datetime import datetime, date, timedelta
from typing import Any

import pandas as pd
from loguru import logger

from hrp.data.db import get_db


def check_completeness(symbols: list[str] | None = None) -> dict[str, Any]:
    """
    Check for missing price data for active symbols.

    Identifies symbols with missing data or incomplete coverage.

    Args:
        symbols: Optional list of symbols to check. If None, checks all symbols.

    Returns:
        Dictionary with check results:
        {
            "status": "pass" | "warning" | "fail",
            "total_symbols": int,
            "symbols_checked": int,
            "incomplete_symbols": int,
            "issues": list[dict],  # List of symbols with low coverage
            "details": str
        }
    """
    db = get_db()

    try:
        # Build query to check coverage per symbol
        if symbols:
            symbol_filter = f"WHERE symbol IN ({','.join(['?'] * len(symbols))})"
            params = tuple(symbols)
        else:
            symbol_filter = ""
            params = ()

        query = f"""
            SELECT
                symbol,
                COUNT(*) as record_count,
                MIN(date) as first_date,
                MAX(date) as last_date,
                DATEDIFF('day', MIN(date), MAX(date) + INTERVAL 1 DAY) as total_days,
                COUNT(*) * 1.0 / DATEDIFF('day', MIN(date), MAX(date) + INTERVAL 1 DAY) as coverage_ratio
            FROM prices
            {symbol_filter}
            GROUP BY symbol
            HAVING coverage_ratio < 0.8  -- Flag symbols with <80% coverage
            ORDER BY coverage_ratio ASC
        """

        df = db.fetchdf(query, params)

        # Get total symbol count
        if symbols:
            total_symbols = len(symbols)
        else:
            total_result = db.fetchone("SELECT COUNT(DISTINCT symbol) FROM prices")
            total_symbols = total_result[0] if total_result else 0

        incomplete_count = len(df)

        # Build results
        issues = []
        for _, row in df.iterrows():
            issues.append({
                "symbol": row["symbol"],
                "record_count": int(row["record_count"]),
                "first_date": str(row["first_date"]),
                "last_date": str(row["last_date"]),
                "coverage_ratio": float(row["coverage_ratio"]),
            })

        # Determine status
        if incomplete_count == 0:
            status = "pass"
            details = "All symbols have complete data coverage (>=80%)"
        elif incomplete_count <= total_symbols * 0.1:  # <10% of symbols incomplete
            status = "warning"
            details = f"{incomplete_count} symbols have incomplete coverage (<80%)"
        else:
            status = "fail"
            details = f"{incomplete_count} symbols have incomplete coverage (<80%) - exceeds threshold"

        logger.info(f"Completeness check: {status} - {details}")

        return {
            "status": status,
            "total_symbols": total_symbols,
            "symbols_checked": total_symbols,
            "incomplete_symbols": incomplete_count,
            "issues": issues,
            "details": details,
        }

    except Exception as e:
        logger.error(f"Completeness check failed: {e}")
        return {
            "status": "error",
            "total_symbols": 0,
            "symbols_checked": 0,
            "incomplete_symbols": 0,
            "issues": [],
            "details": f"Check failed: {str(e)}",
        }


def check_anomalies(threshold: float = 0.5, limit: int = 100) -> dict[str, Any]:
    """
    Detect anomalies in price data.

    Checks for:
    - Negative or zero prices
    - High < Low
    - Negative volume
    - Null required fields
    - Extreme daily moves (>threshold%)

    Args:
        threshold: Percentage threshold for extreme moves (default 0.5 = 50%)
        limit: Maximum number of anomalies to return in details

    Returns:
        Dictionary with check results:
        {
            "status": "pass" | "warning" | "fail",
            "total_anomalies": int,
            "anomaly_types": dict[str, int],  # Count per type
            "issues": list[dict],  # Sample of anomalies
            "details": str
        }
    """
    db = get_db()

    try:
        # Check for structural anomalies
        structural_query = """
            SELECT
                symbol,
                date,
                open,
                high,
                low,
                close,
                volume,
                CASE
                    WHEN close <= 0 THEN 'zero_or_negative_close'
                    WHEN high < low THEN 'high_less_than_low'
                    WHEN volume < 0 THEN 'negative_volume'
                    WHEN close IS NULL THEN 'null_close'
                    WHEN open IS NULL THEN 'null_open'
                    ELSE 'unknown'
                END as anomaly_type
            FROM prices
            WHERE close <= 0
               OR high < low
               OR volume < 0
               OR close IS NULL
               OR open IS NULL
            ORDER BY date DESC
            LIMIT ?
        """

        structural_df = db.fetchdf(structural_query, (limit,))

        # Check for extreme price moves (requires lag calculation)
        extreme_moves_query = f"""
            WITH price_changes AS (
                SELECT
                    symbol,
                    date,
                    close,
                    LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_close,
                    (close - LAG(close) OVER (PARTITION BY symbol ORDER BY date))
                        / LAG(close) OVER (PARTITION BY symbol ORDER BY date) as pct_change
                FROM prices
            )
            SELECT
                symbol,
                date,
                close,
                prev_close,
                pct_change,
                'extreme_move' as anomaly_type
            FROM price_changes
            WHERE ABS(pct_change) > ?
                AND prev_close IS NOT NULL
            ORDER BY ABS(pct_change) DESC
            LIMIT ?
        """

        extreme_df = db.fetchdf(extreme_moves_query, (threshold, limit))

        # Combine anomalies
        all_anomalies = pd.concat([structural_df, extreme_df], ignore_index=True)

        # Count anomalies by type
        anomaly_counts = all_anomalies["anomaly_type"].value_counts().to_dict()
        total_anomalies = len(all_anomalies)

        # Build issues list (sample)
        issues = []
        for _, row in all_anomalies.head(limit).iterrows():
            issue = {
                "symbol": row["symbol"],
                "date": str(row["date"]),
                "anomaly_type": row["anomaly_type"],
            }
            # Add relevant fields based on anomaly type
            if row["anomaly_type"] == "extreme_move" and "pct_change" in row:
                issue["pct_change"] = float(row["pct_change"])
                issue["close"] = float(row["close"]) if pd.notna(row["close"]) else None
                issue["prev_close"] = float(row["prev_close"]) if pd.notna(row["prev_close"]) else None
            else:
                issue["close"] = float(row["close"]) if pd.notna(row["close"]) else None
                if "high" in row:
                    issue["high"] = float(row["high"]) if pd.notna(row["high"]) else None
                if "low" in row:
                    issue["low"] = float(row["low"]) if pd.notna(row["low"]) else None
                if "volume" in row:
                    issue["volume"] = float(row["volume"]) if pd.notna(row["volume"]) else None

            issues.append(issue)

        # Determine status
        if total_anomalies == 0:
            status = "pass"
            details = "No anomalies detected in price data"
        elif total_anomalies <= 10:
            status = "warning"
            details = f"{total_anomalies} anomalies detected - review recommended"
        else:
            status = "fail"
            details = f"{total_anomalies} anomalies detected - immediate attention required"

        logger.info(f"Anomaly check: {status} - {details}")

        return {
            "status": status,
            "total_anomalies": total_anomalies,
            "anomaly_types": anomaly_counts,
            "issues": issues,
            "details": details,
        }

    except Exception as e:
        logger.error(f"Anomaly check failed: {e}")
        return {
            "status": "error",
            "total_anomalies": 0,
            "anomaly_types": {},
            "issues": [],
            "details": f"Check failed: {str(e)}",
        }


def check_gaps(min_gap_days: int = 5, limit: int = 50) -> dict[str, Any]:
    """
    Detect gaps in date sequences per symbol.

    Identifies missing trading days in price history.

    Args:
        min_gap_days: Minimum gap size to report (default 5 trading days)
        limit: Maximum number of gaps to return in details

    Returns:
        Dictionary with check results:
        {
            "status": "pass" | "warning" | "fail",
            "symbols_with_gaps": int,
            "total_gaps": int,
            "total_missing_days": int,
            "issues": list[dict],  # List of significant gaps
            "details": str
        }
    """
    db = get_db()

    try:
        # Find gaps in date sequences
        gap_query = """
            WITH date_gaps AS (
                SELECT
                    symbol,
                    date as gap_start,
                    LEAD(date) OVER (PARTITION BY symbol ORDER BY date) as gap_end,
                    DATEDIFF('day', date, LEAD(date) OVER (PARTITION BY symbol ORDER BY date)) as gap_days
                FROM prices
            )
            SELECT
                symbol,
                gap_start,
                gap_end,
                gap_days - 1 as missing_days
            FROM date_gaps
            WHERE gap_days > ? AND gap_days <= 365  -- Exclude weekends but cap at 1 year
            ORDER BY gap_days DESC
            LIMIT ?
        """

        df = db.fetchdf(gap_query, (min_gap_days, limit))

        # Calculate summary stats
        total_gaps = len(df)
        if total_gaps > 0:
            symbols_with_gaps = df["symbol"].nunique()
            total_missing_days = int(df["missing_days"].sum())
        else:
            symbols_with_gaps = 0
            total_missing_days = 0

        # Build issues list
        issues = []
        for _, row in df.iterrows():
            issues.append({
                "symbol": row["symbol"],
                "gap_start": str(row["gap_start"]),
                "gap_end": str(row["gap_end"]),
                "missing_days": int(row["missing_days"]),
            })

        # Determine status
        if total_gaps == 0:
            status = "pass"
            details = "No significant gaps detected in price data"
        elif total_gaps <= 5:
            status = "warning"
            details = f"{total_gaps} gaps detected across {symbols_with_gaps} symbols"
        else:
            status = "fail"
            details = f"{total_gaps} gaps detected across {symbols_with_gaps} symbols - data quality issue"

        logger.info(f"Gap check: {status} - {details}")

        return {
            "status": status,
            "symbols_with_gaps": symbols_with_gaps,
            "total_gaps": total_gaps,
            "total_missing_days": total_missing_days,
            "issues": issues,
            "details": details,
        }

    except Exception as e:
        logger.error(f"Gap check failed: {e}")
        return {
            "status": "error",
            "symbols_with_gaps": 0,
            "total_gaps": 0,
            "total_missing_days": 0,
            "issues": [],
            "details": f"Check failed: {str(e)}",
        }


def check_freshness(max_stale_days: int = 3) -> dict[str, Any]:
    """
    Check how recent the price data is.

    Verifies that data is up-to-date relative to current date.

    Args:
        max_stale_days: Maximum acceptable days since last data (default 3 for weekends)

    Returns:
        Dictionary with check results:
        {
            "status": "pass" | "warning" | "fail",
            "last_date": str | None,
            "days_stale": int | None,
            "is_fresh": bool,
            "details": str
        }
    """
    db = get_db()

    try:
        # Get most recent date in prices table
        result = db.fetchone("SELECT MAX(date) FROM prices")

        if result and result[0]:
            last_date = result[0]

            # Convert to date if string
            if isinstance(last_date, str):
                last_date = datetime.strptime(last_date, "%Y-%m-%d").date()

            today = datetime.now().date()
            days_stale = (today - last_date).days

            # Determine status
            is_fresh = days_stale <= max_stale_days

            if is_fresh:
                status = "pass"
                details = f"Data is current (last updated {last_date}, {days_stale} days ago)"
            elif days_stale <= max_stale_days * 2:
                status = "warning"
                details = f"Data is slightly stale (last updated {last_date}, {days_stale} days ago)"
            else:
                status = "fail"
                details = f"Data is stale (last updated {last_date}, {days_stale} days ago)"

            logger.info(f"Freshness check: {status} - {details}")

            return {
                "status": status,
                "last_date": str(last_date),
                "days_stale": days_stale,
                "is_fresh": is_fresh,
                "details": details,
            }
        else:
            # No data found
            status = "fail"
            details = "No price data found in database"

            logger.warning(f"Freshness check: {status} - {details}")

            return {
                "status": status,
                "last_date": None,
                "days_stale": None,
                "is_fresh": False,
                "details": details,
            }

    except Exception as e:
        logger.error(f"Freshness check failed: {e}")
        return {
            "status": "error",
            "last_date": None,
            "days_stale": None,
            "is_fresh": False,
            "details": f"Check failed: {str(e)}",
        }
