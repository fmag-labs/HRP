"""
CLI commands for HRP data ingestion job management.

Allows manual triggering of scheduled jobs for testing and debugging.
"""

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

# Repository root (hrp/agents/cli.py -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_project_script(rel_path: str, script_args: list[str] | None = None) -> int:
    """Run a project shell script, streaming its output. Returns the exit code."""
    script = PROJECT_ROOT / rel_path
    if not script.exists():
        logger.error(f"Script not found: {script}")
        return 1
    cmd = ["bash", str(script), *(script_args or [])]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def _print_data_health() -> None:
    """Print a best-effort data-freshness + advisory summary for `hrp doctor`."""
    print("\n=== Data Health ===")
    try:
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI(read_only=True)
        summary = api.get_data_health_summary()
        freshness = summary.get("data_freshness", {})
        symbols = summary.get("symbol_count", 0)
        last = freshness.get("last_date")
        days = freshness.get("days_stale")
        state = "OK" if freshness.get("is_fresh") else "STALE"
        print(f"  Symbols loaded:  {symbols}")
        print(f"  Last price date: {last} ({days} days ago) [{state}]")
        try:
            recs = api.fetchone_readonly("SELECT COUNT(*) FROM recommendations")[0]
            positions = api.fetchone_readonly("SELECT COUNT(*) FROM live_positions")[0]
            print(f"  Recommendations: {recs}")
            print(f"  Open positions:  {positions}")
        except Exception:
            pass
        if symbols == 0:
            print("  -> No data loaded. Bootstrap: " "python -m hrp.agents.run_job --job prices")
        elif not freshness.get("is_fresh"):
            print("  -> Data is stale. Refresh: hrp start --full (or the daily job)")
    except Exception as exc:
        print(f"  (could not read data health: {exc})")


# NOTE: heavy modules (jobs, scheduler, PlatformAPI, ingestion) are imported
# lazily inside the handlers that need them, so lightweight service commands
# (start/stop/status/doctor) and --help stay fast and quiet.


def run_job_now(job_name: str, symbols: list[str] | None = None) -> dict[str, Any]:
    """
    Manually trigger a job to run immediately.

    Args:
        job_name: Name of the job to run ('prices', 'features', or 'universe')
        symbols: Optional list of symbols to process

    Returns:
        Dictionary with job execution results
    """
    from hrp.agents.jobs import (
        FeatureComputationJob,
        PriceIngestionJob,
        UniverseUpdateJob,
    )
    from hrp.data.ingestion.prices import TEST_SYMBOLS

    logger.info(f"Manually triggering job: {job_name}")

    if job_name == "prices":
        # Run price ingestion job
        job = PriceIngestionJob(
            symbols=symbols or TEST_SYMBOLS,
            start=date.today() - timedelta(days=1),
            end=date.today(),
        )
        result = job.run()
        logger.info(f"Price ingestion result: {result}")
        return result

    elif job_name == "features":
        # Run feature computation job
        job = FeatureComputationJob(
            symbols=symbols,  # None = all symbols in database
            start=date.today() - timedelta(days=30),
            end=date.today(),
        )
        result = job.run()
        logger.info(f"Feature computation result: {result}")
        return result

    elif job_name == "universe":
        # Run universe update job
        job = UniverseUpdateJob(
            as_of_date=date.today(),
            actor="user:manual_cli",
        )
        result = job.run()
        logger.info(f"Universe update result: {result}")
        return result

    else:
        raise ValueError(f"Unknown job: {job_name}. Must be 'prices', 'features', or 'universe'")


def list_scheduled_jobs() -> list[dict[str, Any]]:
    """
    List all scheduled jobs from the scheduler.

    Returns:
        List of job information dictionaries
    """
    from hrp.agents.scheduler import IngestionScheduler

    scheduler = IngestionScheduler()

    # Setup jobs to query them (without starting scheduler)
    try:
        scheduler.setup_daily_ingestion()
    except Exception as e:
        logger.warning(f"Could not setup jobs: {e}")

    jobs = scheduler.list_jobs()

    if not jobs:
        logger.info("No scheduled jobs found")
        return []

    logger.info(f"Found {len(jobs)} scheduled jobs:")
    for job in jobs:
        logger.info(f"  - {job['id']}: next run at {job['next_run']}")

    return jobs


def get_job_status(job_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    """
    Get job execution status from ingestion_log table.

    Args:
        job_id: Optional job ID to filter by (None = all jobs)
        limit: Maximum number of records to return

    Returns:
        List of job execution records
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI()
    logs = api.get_ingestion_logs(job_id=job_id, limit=limit)

    results = [
        (
            log["log_id"],
            log["source_id"],
            log["started_at"],
            log["completed_at"],
            log["status"],
            log["records_fetched"],
            log["records_inserted"],
            log["error_message"],
        )
        for log in logs
    ]

    if not results:
        logger.info(f"No job history found{f' for {job_id}' if job_id else ''}")
        return []

    records = []
    for row in results:
        record = {
            "log_id": row[0],
            "source_id": row[1],
            "started_at": row[2],
            "completed_at": row[3],
            "status": row[4],
            "records_fetched": row[5],
            "records_inserted": row[6],
            "error_message": row[7],
        }
        records.append(record)

        # Log summary
        status_emoji = {
            "success": "✅",
            "failed": "❌",
            "running": "🔄",
        }.get(row[4], "❓")

        logger.info(
            f"{status_emoji} {row[1]} - {row[4]} - {row[2]} - "
            f"fetched: {row[5]}, inserted: {row[6]}"
        )

    return records


def clear_job_history(
    job_id: str | None = None,
    before: datetime | None = None,
    status: str | None = None,
) -> int:
    """
    Clear job history from ingestion_log table.

    Args:
        job_id: Optional job ID to filter by (None = all jobs)
        before: Optional datetime to clear records before
        status: Optional status to filter by ('success', 'failed', etc.)

    Returns:
        Number of records deleted
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI()
    rows_deleted = api.purge_ingestion_logs(
        job_id=job_id,
        before=before.isoformat() if before else None,
        status=status,
    )
    logger.info(f"Deleted {rows_deleted} records from ingestion_log")
    return rows_deleted


def main():
    """CLI entry point for job management."""
    parser = argparse.ArgumentParser(
        description="HRP Data Ingestion Job Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run price ingestion now
  python -m hrp.agents.cli run-now --job prices

  # Run feature computation now
  python -m hrp.agents.cli run-now --job features

  # Run universe update now
  python -m hrp.agents.cli run-now --job universe

  # Run with specific symbols
  python -m hrp.agents.cli run-now --job prices --symbols AAPL MSFT GOOGL

  # List scheduled jobs
  python -m hrp.agents.cli list-jobs

  # View job status history
  python -m hrp.agents.cli job-status

  # View status for specific job
  python -m hrp.agents.cli job-status --job-id price_ingestion

  # Clear old job history
  python -m hrp.agents.cli clear-history --before 2025-01-01
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # run-now command
    run_parser = subparsers.add_parser(
        "run-now",
        help="Manually trigger a job to run immediately",
    )
    run_parser.add_argument(
        "--job",
        type=str,
        required=True,
        choices=["prices", "features", "universe"],
        help="Job to run",
    )
    run_parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to process (default: TEST_SYMBOLS for prices, all for features)",
    )

    # list-jobs command
    subparsers.add_parser(
        "list-jobs",
        help="List all scheduled jobs",
    )

    # job-status command
    status_parser = subparsers.add_parser(
        "job-status",
        help="Get job execution status from history",
    )
    status_parser.add_argument(
        "--job-id",
        type=str,
        help="Filter by specific job ID",
    )
    status_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of records to show (default: 10)",
    )

    # clear-history command
    clear_parser = subparsers.add_parser(
        "clear-history",
        help="Clear job history from ingestion_log",
    )
    clear_parser.add_argument(
        "--job-id",
        type=str,
        help="Clear history for specific job ID",
    )
    clear_parser.add_argument(
        "--before",
        type=str,
        help="Clear records before this date (ISO format: YYYY-MM-DD)",
    )
    clear_parser.add_argument(
        "--status",
        type=str,
        choices=["success", "failed", "running"],
        help="Clear records with specific status",
    )
    clear_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm deletion without prompting",
    )

    # --- Service management (wraps scripts/startup.sh and scripts/setup.sh) ---

    # start command
    start_parser = subparsers.add_parser(
        "start",
        help="Start HRP services (API, MLflow, scheduler)",
    )
    start_scope = start_parser.add_mutually_exclusive_group()
    start_scope.add_argument(
        "--full",
        action="store_true",
        help="Start with all research agents enabled",
    )
    start_scope.add_argument(
        "--api-only",
        action="store_true",
        help="Start only the API server",
    )
    start_scope.add_argument(
        "--mlflow-only",
        action="store_true",
        help="Start only the MLflow UI",
    )

    # stop command
    subparsers.add_parser("stop", help="Stop all HRP services")

    # restart command
    subparsers.add_parser("restart", help="Restart all HRP services")

    # status command (service status; see job-status for ingestion history)
    subparsers.add_parser(
        "status",
        help="Show running HRP services (use job-status for ingestion history)",
    )

    # doctor command
    subparsers.add_parser(
        "doctor",
        help="Run setup verification checks (PASS/FAIL summary)",
    )

    # consult command — ask any configured LLM
    consult_parser = subparsers.add_parser(
        "consult",
        help="Ask any configured LLM a question (claude|gpt|glm)",
    )
    consult_parser.add_argument("question", nargs="?", help="The question to ask")
    consult_parser.add_argument(
        "--model", default=None, help="Model key: claude | gpt | glm (default: claude)"
    )
    consult_parser.add_argument(
        "--list-models", action="store_true", help="List models and configuration"
    )

    args = parser.parse_args()

    # Handle commands
    if args.command == "run-now":
        result = run_job_now(args.job, args.symbols)
        if result.get("status") == "failed":
            logger.error(f"Job failed: {result.get('error')}")
            sys.exit(1)
        else:
            logger.info(f"Job completed successfully: {result}")
            sys.exit(0)

    elif args.command == "list-jobs":
        jobs = list_scheduled_jobs()
        if jobs:
            print("\nScheduled Jobs:")
            print("-" * 80)
            for job in jobs:
                print(f"ID: {job['id']}")
                print(f"  Name: {job['name']}")
                print(f"  Next Run: {job['next_run']}")
                print(f"  Trigger: {job['trigger']}")
                print()
        else:
            print("No scheduled jobs found")

    elif args.command == "job-status":
        records = get_job_status(args.job_id, args.limit)
        if records:
            print(f"\nJob Status History (last {len(records)} records):")
            print("-" * 120)
            print(
                f"{'ID':<6} {'Job':<20} {'Status':<10} {'Started':<20} "
                f"{'Fetched':<10} {'Inserted':<10}"
            )
            print("-" * 120)
            for record in records:
                print(
                    f"{record['log_id']:<6} {record['source_id']:<20} "
                    f"{record['status']:<10} {str(record['started_at']):<20} "
                    f"{record['records_fetched'] or 0:<10} "
                    f"{record['records_inserted'] or 0:<10}"
                )
                if record["error_message"]:
                    print(f"  Error: {record['error_message']}")
            print()
        else:
            print("No job history found")

    elif args.command == "clear-history":
        # Parse before date if provided
        before_dt = None
        if args.before:
            try:
                before_dt = datetime.fromisoformat(args.before)
            except ValueError:
                logger.error(f"Invalid date format: {args.before}. Use YYYY-MM-DD")
                sys.exit(1)

        # Confirm deletion
        if not args.confirm:
            conditions = []
            if args.job_id:
                conditions.append(f"job_id={args.job_id}")
            if args.before:
                conditions.append(f"before {args.before}")
            if args.status:
                conditions.append(f"status={args.status}")

            filter_desc = " AND ".join(conditions) if conditions else "ALL RECORDS"
            response = input(f"Delete {filter_desc} from ingestion_log? [y/N]: ")
            if response.lower() != "y":
                print("Cancelled")
                sys.exit(0)

        count = clear_job_history(args.job_id, before_dt, args.status)
        print(f"Deleted {count} records")

    elif args.command == "start":
        scope_args = []
        if args.full:
            scope_args.append("--full")
        elif args.api_only:
            scope_args.append("--api-only")
        elif args.mlflow_only:
            scope_args.append("--mlflow-only")
        sys.exit(_run_project_script("scripts/startup.sh", ["start", *scope_args]))

    elif args.command == "stop":
        sys.exit(_run_project_script("scripts/startup.sh", ["stop"]))

    elif args.command == "restart":
        sys.exit(_run_project_script("scripts/startup.sh", ["restart"]))

    elif args.command == "status":
        sys.exit(_run_project_script("scripts/startup.sh", ["status"]))

    elif args.command == "doctor":
        code = _run_project_script("scripts/setup.sh", ["--check"])
        _print_data_health()
        sys.exit(code)

    elif args.command == "consult":
        from hrp import llm

        if args.list_models:
            for m in llm.list_models():
                mark = "available" if m["available"] else "no key"
                print(f"  {m['key']:8} {m['label']:20} {m['model']:26} [{mark}]")
            sys.exit(0)
        if not args.question:
            print("error: provide a question, or use --list-models")
            sys.exit(1)
        model_key = args.model or llm.default_model()
        try:
            print(
                llm.complete(
                    model_key,
                    "You are a helpful, concise expert assistant.",
                    args.question,
                )
            )
        except llm.LLMError as exc:
            print(f"error: {exc}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
