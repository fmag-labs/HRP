"""
Quality report generation for HRP.

Generates comprehensive quality reports combining all checks and stores results.
"""

import argparse
import json
from datetime import date, datetime
from typing import Any

from loguru import logger

from hrp.data.db import get_db
from hrp.data.quality.checks import (
    check_anomalies,
    check_completeness,
    check_freshness,
    check_gaps,
)


def generate_quality_report(
    check_date: date | None = None,
    symbols: list[str] | None = None,
    store_results: bool = True,
) -> dict[str, Any]:
    """
    Generate comprehensive quality report for price data.

    Runs all quality checks and optionally stores results in quality_metrics table.

    Args:
        check_date: Date of the quality check (defaults to today)
        symbols: Optional list of symbols to check (None = all symbols)
        store_results: Whether to store results in database (default True)

    Returns:
        Dictionary with report results:
        {
            "check_date": str,
            "overall_status": "pass" | "warning" | "fail" | "error",
            "checks": {
                "completeness": {...},
                "anomalies": {...},
                "gaps": {...},
                "freshness": {...}
            },
            "summary": {
                "total_checks": int,
                "passed": int,
                "warnings": int,
                "failed": int,
                "errors": int
            },
            "critical_issues": list[str],
            "metrics_stored": int
        }
    """
    if check_date is None:
        check_date = datetime.now().date()

    logger.info(f"Generating quality report for {check_date}")

    # Run all quality checks
    checks = {}
    try:
        logger.info("Running completeness check...")
        checks["completeness"] = check_completeness(symbols)

        logger.info("Running anomaly check...")
        checks["anomalies"] = check_anomalies()

        logger.info("Running gap check...")
        checks["gaps"] = check_gaps()

        logger.info("Running freshness check...")
        checks["freshness"] = check_freshness()

    except Exception as e:
        logger.error(f"Failed to run quality checks: {e}")
        return {
            "check_date": str(check_date),
            "overall_status": "error",
            "checks": checks,
            "summary": {
                "total_checks": 0,
                "passed": 0,
                "warnings": 0,
                "failed": 0,
                "errors": 1,
            },
            "critical_issues": [f"Quality check execution failed: {str(e)}"],
            "metrics_stored": 0,
        }

    # Calculate summary stats
    summary = {
        "total_checks": len(checks),
        "passed": sum(1 for c in checks.values() if c.get("status") == "pass"),
        "warnings": sum(1 for c in checks.values() if c.get("status") == "warning"),
        "failed": sum(1 for c in checks.values() if c.get("status") == "fail"),
        "errors": sum(1 for c in checks.values() if c.get("status") == "error"),
    }

    # Determine overall status
    if summary["errors"] > 0:
        overall_status = "error"
    elif summary["failed"] > 0:
        overall_status = "fail"
    elif summary["warnings"] > 0:
        overall_status = "warning"
    else:
        overall_status = "pass"

    # Collect critical issues
    critical_issues = []
    for check_name, check_result in checks.items():
        if check_result.get("status") in ["fail", "error"]:
            critical_issues.append(f"{check_name}: {check_result.get('details', 'Unknown error')}")

    # Store results in database
    metrics_stored = 0
    if store_results:
        try:
            metrics_stored = _store_quality_metrics(check_date, checks)
            logger.info(f"Stored {metrics_stored} quality metrics in database")
        except Exception as e:
            logger.error(f"Failed to store quality metrics: {e}")
            critical_issues.append(f"Failed to store metrics: {str(e)}")

    report = {
        "check_date": str(check_date),
        "overall_status": overall_status,
        "checks": checks,
        "summary": summary,
        "critical_issues": critical_issues,
        "metrics_stored": metrics_stored,
    }

    logger.info(
        f"Quality report complete: {overall_status} - "
        f"{summary['passed']} passed, {summary['warnings']} warnings, "
        f"{summary['failed']} failed, {summary['errors']} errors"
    )

    return report


def _store_quality_metrics(check_date: date, checks: dict[str, Any]) -> int:
    """
    Store quality check results in the quality_metrics table.

    Args:
        check_date: Date of the check
        checks: Dictionary of check results

    Returns:
        Number of metrics stored
    """
    db = get_db()
    metrics_stored = 0

    with db.connection() as conn:
        # Store completeness metrics
        completeness = checks.get("completeness", {})
        if completeness:
            conn.execute(
                """
                INSERT INTO quality_metrics (
                    check_type, check_date, table_name, metric_name,
                    metric_value, status, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    "completeness",
                    check_date,
                    "prices",
                    "incomplete_symbols",
                    completeness.get("incomplete_symbols", 0),
                    completeness.get("status", "error"),
                    json.dumps(
                        {
                            "total_symbols": completeness.get("total_symbols", 0),
                            "symbols_checked": completeness.get("symbols_checked", 0),
                            "details": completeness.get("details", ""),
                            "issues_count": len(completeness.get("issues", [])),
                        }
                    ),
                ),
            )
            metrics_stored += 1

        # Store anomaly metrics
        anomalies = checks.get("anomalies", {})
        if anomalies:
            conn.execute(
                """
                INSERT INTO quality_metrics (
                    check_type, check_date, table_name, metric_name,
                    metric_value, status, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    "anomalies",
                    check_date,
                    "prices",
                    "total_anomalies",
                    anomalies.get("total_anomalies", 0),
                    anomalies.get("status", "error"),
                    json.dumps(
                        {
                            "anomaly_types": anomalies.get("anomaly_types", {}),
                            "details": anomalies.get("details", ""),
                            "issues_count": len(anomalies.get("issues", [])),
                        }
                    ),
                ),
            )
            metrics_stored += 1

        # Store gap metrics
        gaps = checks.get("gaps", {})
        if gaps:
            conn.execute(
                """
                INSERT INTO quality_metrics (
                    check_type, check_date, table_name, metric_name,
                    metric_value, status, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    "gaps",
                    check_date,
                    "prices",
                    "total_gaps",
                    gaps.get("total_gaps", 0),
                    gaps.get("status", "error"),
                    json.dumps(
                        {
                            "symbols_with_gaps": gaps.get("symbols_with_gaps", 0),
                            "total_missing_days": gaps.get("total_missing_days", 0),
                            "details": gaps.get("details", ""),
                            "issues_count": len(gaps.get("issues", [])),
                        }
                    ),
                ),
            )
            metrics_stored += 1

        # Store freshness metrics
        freshness = checks.get("freshness", {})
        if freshness:
            conn.execute(
                """
                INSERT INTO quality_metrics (
                    check_type, check_date, table_name, metric_name,
                    metric_value, status, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    "freshness",
                    check_date,
                    "prices",
                    "days_stale",
                    freshness.get("days_stale") if freshness.get("days_stale") is not None else -1,
                    freshness.get("status", "error"),
                    json.dumps(
                        {
                            "last_date": freshness.get("last_date"),
                            "is_fresh": freshness.get("is_fresh", False),
                            "details": freshness.get("details", ""),
                        }
                    ),
                ),
            )
            metrics_stored += 1

    return metrics_stored


def get_quality_history(
    start_date: date | None = None,
    end_date: date | None = None,
    check_type: str | None = None,
) -> dict[str, Any]:
    """
    Get historical quality metrics.

    Args:
        start_date: Start date for history (optional)
        end_date: End date for history (optional)
        check_type: Filter by check type (optional)

    Returns:
        Dictionary with historical metrics:
        {
            "metrics": list[dict],
            "summary": {
                "total_records": int,
                "date_range": {...},
                "check_types": list[str]
            }
        }
    """
    db = get_db()

    # Build query with optional filters
    conditions = []
    params = []

    if start_date:
        conditions.append("check_date >= ?")
        params.append(start_date)

    if end_date:
        conditions.append("check_date <= ?")
        params.append(end_date)

    if check_type:
        conditions.append("check_type = ?")
        params.append(check_type)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT
            metric_id,
            check_type,
            check_date,
            table_name,
            symbol,
            metric_name,
            metric_value,
            threshold,
            status,
            details,
            created_at
        FROM quality_metrics
        {where_clause}
        ORDER BY check_date DESC, created_at DESC
    """

    df = db.fetchdf(query, tuple(params))

    # Parse JSON details
    metrics = []
    for _, row in df.iterrows():
        metric = {
            "metric_id": int(row["metric_id"]),
            "check_type": row["check_type"],
            "check_date": str(row["check_date"]),
            "table_name": row["table_name"],
            "symbol": row["symbol"] if row["symbol"] else None,
            "metric_name": row["metric_name"],
            "metric_value": float(row["metric_value"]) if row["metric_value"] is not None else None,
            "threshold": float(row["threshold"]) if row["threshold"] is not None else None,
            "status": row["status"],
            "created_at": str(row["created_at"]),
        }

        # Parse JSON details
        if row["details"]:
            try:
                metric["details"] = json.loads(row["details"])
            except json.JSONDecodeError:
                metric["details"] = {}

        metrics.append(metric)

    # Calculate summary
    summary = {
        "total_records": len(metrics),
        "date_range": {
            "start": str(df["check_date"].min()) if not df.empty else None,
            "end": str(df["check_date"].max()) if not df.empty else None,
        },
        "check_types": sorted(df["check_type"].unique().tolist()) if not df.empty else [],
    }

    return {"metrics": metrics, "summary": summary}


def get_latest_quality_report() -> dict[str, Any] | None:
    """
    Get the most recent quality report.

    Returns:
        Latest quality report dictionary or None if no reports exist
    """
    db = get_db()

    # Get the latest check_date
    result = db.fetchone("SELECT MAX(check_date) FROM quality_metrics")

    if not result or not result[0]:
        logger.info("No quality reports found")
        return None

    latest_date = result[0]

    # Convert to date if string
    if isinstance(latest_date, str):
        latest_date = datetime.strptime(latest_date, "%Y-%m-%d").date()

    logger.info(f"Retrieving quality report for {latest_date}")

    # Get metrics for that date
    history = get_quality_history(start_date=latest_date, end_date=latest_date)

    if not history["metrics"]:
        return None

    # Reconstruct report structure
    checks = {}
    for metric in history["metrics"]:
        check_type = metric["check_type"]
        if check_type not in checks:
            checks[check_type] = {
                "status": metric["status"],
                "details": metric.get("details", {}),
            }

    # Calculate summary
    summary = {
        "total_checks": len(checks),
        "passed": sum(1 for c in checks.values() if c.get("status") == "pass"),
        "warnings": sum(1 for c in checks.values() if c.get("status") == "warning"),
        "failed": sum(1 for c in checks.values() if c.get("status") == "fail"),
        "errors": sum(1 for c in checks.values() if c.get("status") == "error"),
    }

    # Determine overall status
    if summary["errors"] > 0:
        overall_status = "error"
    elif summary["failed"] > 0:
        overall_status = "fail"
    elif summary["warnings"] > 0:
        overall_status = "warning"
    else:
        overall_status = "pass"

    return {
        "check_date": str(latest_date),
        "overall_status": overall_status,
        "checks": checks,
        "summary": summary,
        "metrics_count": len(history["metrics"]),
    }


def main():
    """CLI entry point for quality report generation."""
    parser = argparse.ArgumentParser(description="HRP Quality Report Generation")
    parser.add_argument(
        "--date",
        type=str,
        help="Date for quality check (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to check (default: all symbols)",
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Don't store results in database",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show quality history instead of generating new report",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Show latest quality report",
    )

    args = parser.parse_args()

    if args.history:
        # Show historical metrics
        history = get_quality_history()
        print(f"\nQuality History Summary:")
        print(f"  Total Records: {history['summary']['total_records']}")
        print(f"  Date Range: {history['summary']['date_range']['start']} to {history['summary']['date_range']['end']}")
        print(f"  Check Types: {', '.join(history['summary']['check_types'])}")

        print(f"\nRecent Metrics:")
        for metric in history["metrics"][:10]:  # Show last 10
            print(f"  {metric['check_date']} - {metric['check_type']}: {metric['status']}")

    elif args.latest:
        # Show latest report
        report = get_latest_quality_report()
        if report:
            print(f"\nLatest Quality Report ({report['check_date']}):")
            print(f"  Overall Status: {report['overall_status']}")
            print(f"  Summary: {report['summary']['passed']} passed, {report['summary']['warnings']} warnings, "
                  f"{report['summary']['failed']} failed, {report['summary']['errors']} errors")
            print(f"\nChecks:")
            for check_name, check_data in report["checks"].items():
                print(f"  {check_name}: {check_data['status']}")
        else:
            print("No quality reports found")

    else:
        # Generate new report
        check_date = None
        if args.date:
            check_date = datetime.strptime(args.date, "%Y-%m-%d").date()

        report = generate_quality_report(
            check_date=check_date,
            symbols=args.symbols,
            store_results=not args.no_store,
        )

        print(f"\nQuality Report ({report['check_date']}):")
        print(f"  Overall Status: {report['overall_status']}")
        print(f"  Summary: {report['summary']['passed']} passed, {report['summary']['warnings']} warnings, "
              f"{report['summary']['failed']} failed, {report['summary']['errors']} errors")
        print(f"  Metrics Stored: {report['metrics_stored']}")

        if report["critical_issues"]:
            print(f"\nCritical Issues:")
            for issue in report["critical_issues"]:
                print(f"  - {issue}")

        print(f"\nCheck Details:")
        for check_name, check_data in report["checks"].items():
            print(f"  {check_name}: {check_data['status']} - {check_data.get('details', '')}")


if __name__ == "__main__":
    main()
