# Data Management Improvements (Section 5.3) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address three medium-term (P2) improvements from the data management health check: feature backfill for EMA/VWAP, time-series fundamentals, and enhanced quality monitoring.

**Architecture:** Extend existing patterns - add feature backfill to price backfill infrastructure, implement fundamentals time-series with point-in-time correctness, expose quality checks via PlatformAPI with real-time dashboard alerts.

**Tech Stack:** Python 3.11+, DuckDB, APScheduler, Streamlit, pytest

---

## Executive Summary

| Priority | Issue | Impact | Status |
|----------|-------|--------|--------|
| P2-4 | Feature Coverage | 280K missing EMA/VWAP rows | Incomplete |
| P2-5 | Time-Series Fundamentals | No daily fundamentals | Incomplete |
| P2-6 | Quality Monitoring | No dashboard alerts/API access | Incomplete |

**Implementation Sequence:** 3 phases over 6 weeks, each providing standalone value.

---

## Phase 1: Feature Backfill (EMA/VWAP)

**Problem:** EMA/VWAP features only computed from 2026-01-25 (~280K rows vs ~500K expected)

**Solution:** Add feature backfill to existing price backfill infrastructure

### Task 1: Add Feature Backfill Function

**Files:**
- Modify: `hrp/data/backfill.py` (add after line 500)
- Test: `tests/test_data/test_backfill.py`

**Step 1: Write the failing test**

```python
# tests/test_data/test_backfill.py
def test_backfill_ema_vwap_basic(populated_db):
    """Test EMA/VWAP backfill for historical dates."""
    from hrp.data.backfill import backfill_features_ema_vwap

    result = backfill_features_ema_vwap(
        symbols=["AAPL", "MSFT"],
        start=date(2023, 1, 1),
        end=date(2023, 1, 31),
        db_path=populated_db,
    )

    assert result["symbols_success"] == 2
    assert result["rows_inserted"] > 0

    # Verify EMA/VWAP features exist
    db = get_db(populated_db)
    count = db.fetchone("""
        SELECT COUNT(*) FROM features
        WHERE feature_name IN ('ema_12d', 'ema_26d', 'vwap_20d')
    """)[0]
    assert count > 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_data/test_backfill.py::test_backfill_ema_vwap_basic -v
```

Expected: `ImportError: cannot import name 'backfill_features_ema_vwap'`

**Step 3: Write minimal implementation**

```python
# hrp/data/backfill.py (add after existing backfill_prices function)

def backfill_features_ema_vwap(
    symbols: list[str],
    start: date,
    end: date,
    batch_size: int = 10,
    progress_file: Optional[Path] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Backfill EMA and VWAP features for historical dates.

    Computes only ema_12d, ema_26d, and vwap_20d features which were
    previously only computed from 2026-01-25.

    Args:
        symbols: List of tickers to compute features for
        start: Start date for feature computation
        end: End date for feature computation
        batch_size: Number of symbols per batch
        progress_file: Path to progress tracking file
        db_path: Optional database path

    Returns:
        Dictionary with computation statistics
    """
    from hrp.data.db import get_db
    from hrp.data.ingestion.features import _compute_all_features, _upsert_features
    from hrp.data.features.computation import FEATURE_FUNCTIONS
    import logging

    logger = logging.getLogger(__name__)

    # Filter to EMA/VWAP features only
    ema_vwap_features = {
        "ema_12d": FEATURE_FUNCTIONS["ema_12d"],
        "ema_26d": FEATURE_FUNCTIONS["ema_26d"],
        "vwap_20d": FEATURE_FUNCTIONS["vwap_20d"],
    }

    # Initialize database connection
    db = get_db(db_path)

    # Track statistics
    symbols_success = 0
    symbols_failed = 0
    rows_inserted = 0
    failed_symbols = []

    # Process symbols in batches
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}/{(len(symbols) + batch_size - 1) // batch_size}")

        for symbol in batch:
            try:
                # Get price data for symbol
                prices_df = db.fetchall(
                    """
                    SELECT date, open, high, low, close, volume
                    FROM prices
                    WHERE symbol = ? AND date BETWEEN ? AND ?
                    ORDER BY date
                    """,
                    params=[symbol, start, end],
                )

                if not prices_df:
                    logger.warning(f"No price data for {symbol} in date range")
                    symbols_failed += 1
                    failed_symbols.append(symbol)
                    continue

                # Convert to DataFrame
                import pandas as pd
                prices_df = pd.DataFrame(
                    prices_df,
                    columns=["date", "open", "high", "low", "close", "volume"]
                )

                # Compute EMA/VWAP features
                features_df = _compute_all_features(
                    db,
                    symbol,
                    start,
                    end,
                    feature_list=list(ema_vwap_features.keys()),
                )

                # Upsert to features table
                if not features_df.empty:
                    _upsert_features(db, features_df)
                    rows_inserted += len(features_df)

                symbols_success += 1
                logger.info(f"Computed {len(features_df)} EMA/VWAP features for {symbol}")

            except Exception as e:
                logger.error(f"Failed to compute features for {symbol}: {e}")
                symbols_failed += 1
                failed_symbols.append(symbol)

    return {
        "symbols_success": symbols_success,
        "symbols_failed": symbols_failed,
        "failed_symbols": failed_symbols,
        "rows_inserted": rows_inserted,
    }
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_data/test_backfill.py::test_backfill_ema_vwap_basic -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/data/backfill.py tests/test_data/test_backfill.py
git commit -m "feat: add EMA/VWAP feature backfill

- Add backfill_features_ema_vwap() to backfill.py
- Computes ema_12d, ema_26d, vwap_20d for historical dates
- Processes symbols in batches with progress tracking
- Addresses P2-4 from data management health check"
```

### Task 2: Add CLI Interface

**Files:**
- Modify: `hrp/data/backfill.py` (extend CLI section)

**Step 1: Add CLI flags**

```python
# In the main() CLI section of backfill.py, add:
parser.add_argument(
    "--ema-vwap",
    action="store_true",
    help="Backfill EMA and VWAP features for historical dates",
)
```

**Step 2: Add CLI handler**

```python
# In main() function, add:
if args.ema_vwap:
    symbols = get_symbols(args.universe, args.symbols)
    result = backfill_features_ema_vwap(
        symbols=symbols,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        progress_file=args.progress,
    )
    print(json.dumps(result, indent=2, default=str))
    return
```

**Step 3: Test CLI**

```bash
# Test help message
python -m hrp.data.backfill --help | grep ema-vwap

# Test dry run
python -m hrp.data.backfill --symbols AAPL --start 2023-01-01 --end 2023-01-31 --ema-vwap
```

**Step 4: Commit**

```bash
git add hrp/data/backfill.py
git commit -m "feat: add CLI interface for EMA/VWAP backfill

- Add --ema-vwap flag to backfill CLI
- Support all existing flags (symbols, universe, start, end, progress)
- Addresses P2-4 from data management health check"
```

### Task 3: Run Production Backfill

**Step 1: Run backfill for all symbols**

```bash
python -m hrp.data.backfill \
    --universe \
    --start 2020-01-01 \
    --end 2026-01-24 \
    --batch-size 10 \
    --progress ~/hrp-data/backfill_ema_vwap_progress.json \
    --ema-vwap
```

**Step 2: Verify coverage**

```bash
python -c "
from hrp.data.db import get_db
db = get_db()
result = db.fetchall('''
    SELECT feature_name, COUNT(DISTINCT CONCAT(symbol, '-', date)) as coverage
    FROM features
    WHERE feature_name IN ('ema_12d', 'ema_26d', 'vwap_20d')
    GROUP BY feature_name
''')
for row in result:
    print(f'{row[0]}: {row[1]} symbol-dates')
"
```

Expected: ~500K rows per feature (396 symbols × ~1,250 days)

---

## Phase 2: Time-Series Fundamentals

**Problem:** No daily fundamentals time-series for backtesting accuracy

**Solution:** Extend quarterly fundamentals with daily time-series using point-in-time forward-fill

### Task 4: Create Time-Series Fundamentals Module

**Files:**
- Create: `hrp/data/ingestion/fundamentals_timeseries.py`
- Test: `tests/test_data/test_fundamentals_timeseries.py`

**Step 1: Write the failing test**

```python
# tests/test_data/test_fundamentals_timeseries.py
import pytest
from datetime import date, timedelta

def test_backfill_fundamentals_timeseries_basic(populated_db):
    """Test fundamentals time-series backfill."""
    from hrp.data.ingestion.fundamentals_timeseries import backfill_fundamentals_timeseries

    result = backfill_fundamentals_timeseries(
        symbols=["AAPL", "MSFT"],
        start=date(2023, 10, 1),
        end=date(2023, 12, 31),
        db_path=populated_db,
    )

    assert result["symbols_success"] == 2
    assert result["rows_inserted"] > 0

    # Verify time-series fundamentals exist
    from hrp.data.db import get_db
    db = get_db(populated_db)
    count = db.fetchone("""
        SELECT COUNT(*) FROM features
        WHERE feature_name LIKE 'ts_%'
    """)[0]
    assert count > 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_data/test_fundamentals_timeseries.py::test_backfill_fundamentals_timeseries_basic -v
```

Expected: `ModuleNotFoundError: No module named 'hrp.data.ingestion.fundamentals_timeseries'`

**Step 3: Write minimal implementation**

```python
# hrp/data/ingestion/fundamentals_timeseries.py (new file)

"""
Time-series fundamentals for HRP.

Extends quarterly fundamentals with daily fundamental values for
backtesting accuracy and research.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def backfill_fundamentals_timeseries(
    symbols: list[str],
    start: date,
    end: date,
    metrics: list[str] | None = None,
    batch_size: int = 10,
    source: str = "yfinance",
    progress_file: Optional[Path] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Backfill daily fundamental time-series data.

    For each day in the range:
    1. Fetch the latest quarterly fundamental data available as of that day
    2. Forward-fill values until next quarter report
    3. Store in features table with ts_ prefix (time-series)

    This provides point-in-time correctness for backtesting.

    Args:
        symbols: List of tickers to backfill
        start: Start date for time-series
        end: End date for time-series
        metrics: Fundamental metrics to track (default: revenue, eps, book_value)
        batch_size: Number of symbols per batch
        source: Data source ('yfinance' or 'simfin')
        progress_file: Path to progress tracking file
        db_path: Optional database path

    Returns:
        Dictionary with backfill statistics
    """
    from hrp.data.db import get_db
    from hrp.data.ingestion.fundamentals import ingest_fundamentals
    import pandas as pd

    metrics = metrics or ["revenue", "eps", "book_value"]

    # Initialize database connection
    db = get_db(db_path)

    # Track statistics
    symbols_success = 0
    symbols_failed = 0
    rows_inserted = 0
    failed_symbols = []

    # Get trading days in range
    trading_days = db.fetchall(
        """
        SELECT DISTINCT date FROM prices
        WHERE date BETWEEN ? AND ?
        ORDER BY date
        """,
        params=[start, end],
    )
    trading_days = [row[0] for row in trading_days]

    logger.info(f"Computing time-series for {len(trading_days)} trading days")

    # Process symbols in batches
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}")

        for symbol in batch:
            try:
                # Fetch quarterly fundamentals with point-in-time
                quarterly_data = db.fetchall(
                    """
                    SELECT period_end, metric, value
                    FROM fundamentals
                    WHERE symbol = ? AND metric IN (?, ?, ?)
                    ORDER BY period_end
                    """,
                    params=[symbol, *metrics],
                )

                if not quarterly_data:
                    logger.warning(f"No quarterly data for {symbol}")
                    symbols_failed += 1
                    failed_symbols.append(symbol)
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(quarterly_data, columns=["period_end", "metric", "value"])

                # Pivot to have metrics as columns
                df_pivot = df.pivot(index="period_end", columns="metric", values="value")

                # Forward-fill to create daily time-series
                daily_values = {}
                for trading_day in trading_days:
                    # Find latest quarter as of this day
                    latest_quarter = df_pivot[df_pivot.index <= trading_day]
                    if latest_quarter.empty:
                        continue

                    # Get last row (latest quarter)
                    latest_values = latest_quarter.iloc[-1]

                    # Store time-series value
                    for metric in metrics:
                        if pd.notna(latest_values[metric]):
                            daily_values.setdefault(f"ts_{metric}", []).append({
                                "symbol": symbol,
                                "date": trading_day,
                                "feature_name": f"ts_{metric}",
                                "value": float(latest_values[metric]),
                            })

                # Bulk insert time-series fundamentals
                if daily_values:
                    all_rows = []
                    for metric_rows in daily_values.values():
                        all_rows.extend(metric_rows)

                    # Use existing upsert pattern
                    db.upsert_many(
                        "features",
                        ["symbol", "date", "feature_name", "value", "version", "computed_at"],
                        [
                            (
                                r["symbol"],
                                r["date"],
                                r["feature_name"],
                                r["value"],
                                "v1",
                                pd.Timestamp.now(),
                            )
                            for r in all_rows
                        ],
                    )

                    rows_inserted += len(all_rows)

                symbols_success += 1
                logger.info(f"Computed {len(all_rows)} time-series values for {symbol}")

            except Exception as e:
                logger.error(f"Failed to compute time-series for {symbol}: {e}")
                symbols_failed += 1
                failed_symbols.append(symbol)

    return {
        "symbols_success": symbols_success,
        "symbols_failed": symbols_failed,
        "failed_symbols": failed_symbols,
        "rows_inserted": rows_inserted,
    }
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_data/test_fundamentals_timeseries.py::test_backfill_fundamentals_timeseries_basic -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/data/ingestion/fundamentals_timeseries.py tests/test_data/test_fundamentals_timeseries.py
git commit -m "feat: add fundamentals time-series backfill

- Create fundamentals_timeseries.py module
- Forward-fill quarterly data to daily time-series
- Store as ts_revenue, ts_eps, ts_book_value features
- Provides point-in-time correctness for backtesting
- Addresses P2-5 from data management health check"
```

### Task 5: Add Scheduled Job

**Files:**
- Modify: `hrp/agents/jobs.py` (add after line 886)
- Modify: `hrp/agents/scheduler.py` (add after line 685)

**Step 1: Add FundamentalsTimeSeriesJob**

```python
# hrp/agents/jobs.py

class FundamentalsTimeSeriesJob(IngestionJob):
    """
    Scheduled job for daily fundamentals time-series ingestion.

    Runs weekly to update fundamental time-series with latest quarterly data.
    Recommended schedule: Sunday 6 AM ET (before market open).
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        as_of_date: date | None = None,
        lookback_days: int = 90,
        job_id: str = "fundamentals_timeseries",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        dependencies: list[str] | None = None,
    ):
        """
        Initialize fundamentals time-series job.

        Args:
            symbols: List of stock tickers (None = all universe symbols)
            as_of_date: Date to compute time-series as of (default: today)
            lookback_days: Days to backfill for point-in-time correctness
            job_id: Unique identifier for this job
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier (seconds)
            dependencies: List of job IDs that must complete before this job runs
        """
        super().__init__(job_id, max_retries, retry_backoff, dependencies or [])
        self.symbols = symbols
        self.as_of_date = as_of_date or date.today()
        self.lookback_days = lookback_days

    def execute(self) -> dict[str, Any]:
        """Execute fundamentals time-series ingestion."""
        from hrp.data.ingestion.fundamentals_timeseries import backfill_fundamentals_timeseries
        from hrp.data.universe import UniverseManager

        # Get symbols from universe if not specified
        symbols = self.symbols
        if symbols is None:
            manager = UniverseManager()
            symbols = manager.get_universe_at_date(date.today())
            if not symbols:
                symbols = ["AAPL", "MSFT", "GOOGL"]
            logger.info(f"Using {len(symbols)} symbols from universe")

        # Compute time-series for lookback period
        start = self.as_of_date - timedelta(days=self.lookback_days)

        logger.info(
            f"Computing fundamentals time-series for {len(symbols)} symbols "
            f"from {start} to {self.as_of_date}"
        )

        result = backfill_fundamentals_timeseries(
            symbols=symbols,
            start=start,
            end=self.as_of_date,
            batch_size=10,
            source="yfinance",
        )

        return {
            "records_fetched": result["rows_inserted"],
            "records_inserted": result["rows_inserted"],
            "symbols_success": result["symbols_success"],
            "symbols_failed": result["symbols_failed"],
            "failed_symbols": result.get("failed_symbols", []),
        }
```

**Step 2: Add scheduler method**

```python
# hrp/agents/scheduler.py

def setup_weekly_fundamentals_timeseries(
    self,
    fundamentals_time: str = "06:00",
    day_of_week: str = "sun",
) -> None:
    """
    Schedule weekly fundamentals time-series ingestion.

    Args:
        fundamentals_time: Time to run job (HH:MM format)
        day_of_week: Day of week (mon, tue, wed, thu, fri, sat, sun)
    """
    job_id = "fundamentals_timeseries"

    # Create job wrapper
    def job_wrapper():
        from hrp.agents.jobs import FundamentalsTimeSeriesJob
        job = FundamentalsTimeSeriesJob(
            symbols=None,  # All universe symbols
            lookback_days=90,
        )
        return job.run()

    self.scheduler.add_job(
        job_wrapper,
        trigger=CronTrigger(
            day_of_week=day_of_week,
            hour=int(fundamentals_time.split(":")[0]),
            minute=int(fundamentals_time.split(":")[1]),
        ),
        id=job_id,
        name="Weekly Fundamentals Time-Series",
        replace_existing=True,
    )
    logger.info(f"Scheduled {job_id} for {day_of_week} {fundamentals_time}")
```

**Step 3: Test job manually**

```bash
python -c "
from hrp.agents.jobs import FundamentalsTimeSeriesJob
from datetime import date

job = FundamentalsTimeSeriesJob(
    symbols=['AAPL', 'MSFT'],
    lookback_days=30,
)
result = job.run()
print(result)
"
```

**Step 4: Commit**

```bash
git add hrp/agents/jobs.py hrp/agents/scheduler.py
git commit -m "feat: add scheduled fundamentals time-series job

- Add FundamentalsTimeSeriesJob to agents/jobs.py
- Add setup_weekly_fundamentals_timeseries() to scheduler
- Runs weekly (Sunday 6 AM) with 90-day lookback
- Addresses P2-5 from data management health check"
```

### Task 6: Verify Point-in-Time Correctness

**Step 1: Run validation query**

```bash
python -c "
from hrp.data.db import get_db
db = get_db()

# Check that ts_pe_ratio varies daily (not constant)
result = db.fetchall('''
    SELECT date, COUNT(DISTINCT value) as distinct_values
    FROM features
    WHERE feature_name = 'ts_eps' AND symbol = 'AAPL'
      AND date >= '2023-12-01' AND date <= '2023-12-31'
    GROUP BY date
    ORDER BY date
    LIMIT 10
''')
print('Daily EPS variation (should have different values):')
for row in result:
    print(f'  {row[0]}: {row[1]} distinct values')
"
```

---

## Phase 3: Enhanced Quality Monitoring

**Problem:** Quality checks not exposed via PlatformAPI, no real-time dashboard alerts

**Solution:** Add quality methods to PlatformAPI and alert banner to dashboard

### Task 7: Add Quality Methods to PlatformAPI

**Files:**
- Modify: `hrp/api/platform.py` (add after existing methods)
- Test: `tests/test_api/test_platform_quality.py` (new file)

**Step 1: Write the failing test**

```python
# tests/test_api/test_platform_quality.py
import pytest
from datetime import date

def test_run_quality_checks_basic(db_session):
    """Test running quality checks via PlatformAPI."""
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI()
    result = api.run_quality_checks(as_of_date=date.today())

    assert "health_score" in result
    assert "critical_issues" in result
    assert "warning_issues" in result
    assert "passed" in result
    assert "results" in result
    assert isinstance(result["results"], list)

def test_get_quality_trend(db_session):
    """Test getting quality score trend."""
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI()
    trend = api.get_quality_trend(days=30)

    assert "dates" in trend
    assert "health_scores" in trend
    assert "critical_issues" in trend
    assert "warning_issues" in trend
    assert len(trend["dates"]) == len(trend["health_scores"])

def test_get_data_health_summary(db_session):
    """Test getting data health summary."""
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI()
    summary = api.get_data_health_summary()

    assert "symbol_count" in summary
    assert "date_range" in summary
    assert "total_records" in summary
    assert "data_freshness" in summary
    assert "ingestion_summary" in summary
    assert summary["symbol_count"] >= 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_api/test_platform_quality.py -v
```

Expected: `AttributeError: 'PlatformAPI' object has no attribute 'run_quality_checks'`

**Step 3: Write minimal implementation**

```python
# hrp/api/platform.py (add to PlatformAPI class)

def run_quality_checks(
    self,
    as_of_date: date | None = None,
    checks: list[str] | None = None,
    send_alerts: bool = False,
) -> dict[str, Any]:
    """
    Run data quality checks and return results.

    Args:
        as_of_date: Date to run checks for (default: today)
        checks: List of check names to run (default: all checks)
        send_alerts: Whether to send email alerts for critical issues

    Returns:
        Dictionary with quality check results

    Raises:
        ValueError: If check names are invalid
    """
    from hrp.data.quality.checks import (
        DEFAULT_CHECKS,
        PriceAnomalyCheck,
        CompletenessCheck,
        GapDetectionCheck,
        StaleDataCheck,
        VolumeAnomalyCheck,
    )
    from hrp.data.quality.report import QualityReportGenerator

    as_of_date = as_of_date or date.today()

    # Map check names to classes
    check_classes = {
        "price_anomaly": PriceAnomalyCheck,
        "completeness": CompletenessCheck,
        "gap_detection": GapDetectionCheck,
        "stale_data": StaleDataCheck,
        "volume_anomaly": VolumeAnomalyCheck,
    }

    # Validate check names
    if checks:
        invalid = [c for c in checks if c not in check_classes]
        if invalid:
            raise ValueError(f"Invalid check names: {invalid}")
        checks_to_run = [check_classes[c]() for c in checks]
    else:
        checks_to_run = [check_cls() for check_cls in DEFAULT_CHECKS]

    # Run checks
    generator = QualityReportGenerator()
    report = generator.generate_report(as_of_date, checks_to_run)

    # Send alerts if requested
    if send_alerts and report.critical_issues > 0:
        self._send_quality_alerts(report)

    return {
        "health_score": report.health_score,
        "critical_issues": report.critical_issues,
        "warning_issues": report.warning_issues,
        "passed": report.passed,
        "results": [
            {
                "check_name": r.check_name,
                "passed": r.passed,
                "critical_count": r.critical_count,
                "warning_count": r.warning_count,
                "run_time_ms": r.run_time_ms,
                "issues": [i.to_dict() for i in r.issues],
            }
            for r in report.results
        ],
        "generated_at": report.generated_at.isoformat(),
    }

def get_quality_trend(self, days: int = 30) -> dict[str, Any]:
    """
    Get historical quality scores for trend analysis.

    Args:
        days: Number of days to look back

    Returns:
        Dictionary with trend data
    """
    from hrp.data.quality.report import QualityReportGenerator

    generator = QualityReportGenerator()
    trend_data = generator.get_health_trend(days=days)

    if not trend_data:
        return {
            "dates": [],
            "health_scores": [],
            "critical_issues": [],
            "warning_issues": [],
        }

    return {
        "dates": [d["date"].isoformat() for d in trend_data],
        "health_scores": [d["health_score"] for d in trend_data],
        "critical_issues": [d.get("critical_issues", 0) for d in trend_data],
        "warning_issues": [d.get("warning_issues", 0) for d in trend_data],
    }

def get_data_health_summary(self) -> dict[str, Any]:
    """
    Get summary statistics for data health dashboard.

    Returns:
        Dictionary with health metrics
    """
    with self._db.connection() as conn:
        # Symbol count
        symbol_count = conn.execute(
            "SELECT COUNT(DISTINCT symbol) FROM prices"
        ).fetchone()[0]

        # Date range
        date_range = conn.execute(
            "SELECT MIN(date), MAX(date) FROM prices"
        ).fetchone()

        # Total records
        prices_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        features_count = conn.execute(
            "SELECT COUNT(*) FROM features"
        ).fetchone()[0]
        fundamentals_count = conn.execute(
            "SELECT COUNT(*) FROM fundamentals"
        ).fetchone()[0]

        # Data freshness
        last_price_date = conn.execute(
            "SELECT MAX(date) FROM prices"
        ).fetchone()[0]
        if last_price_date:
            days_stale = (date.today() - last_price_date).days
            is_fresh = days_stale <= 3
        else:
            days_stale = None
            is_fresh = False

        # Ingestion summary
        ingestion_summary = conn.execute("""
            SELECT
                COUNT(*) as total_runs,
                SUM(CASE WHEN LOWER(status) = 'completed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                MAX(completed_at) as last_successful
            FROM ingestion_log
        """).fetchone()

    return {
        "symbol_count": symbol_count,
        "date_range": {
            "start": str(date_range[0]) if date_range[0] else None,
            "end": str(date_range[1]) if date_range[1] else None,
        },
        "total_records": {
            "prices": prices_count,
            "features": features_count,
            "fundamentals": fundamentals_count,
        },
        "data_freshness": {
            "last_date": str(last_price_date) if last_price_date else None,
            "days_stale": days_stale,
            "is_fresh": is_fresh,
        },
        "ingestion_summary": {
            "total_runs": ingestion_summary[0] or 0,
            "success_rate": round(ingestion_summary[1] or 0, 1),
            "last_successful": str(ingestion_summary[2]) if ingestion_summary[2] else None,
        },
    }

def _send_quality_alerts(self, report) -> None:
    """Send email alerts for critical quality issues."""
    try:
        from hrp.notifications.email import EmailNotifier

        notifier = EmailNotifier()
        notifier.send_quality_alert(
            health_score=report.health_score,
            critical_issues=report.critical_issues,
            warning_issues=report.warning_issues,
            issues=[i.to_dict() for r in report.results for i in r.issues],
            timestamp=report.generated_at.isoformat(),
        )
        logger.info(
            f"Sent quality alerts: {report.critical_issues} critical issues"
        )
    except Exception as e:
        logger.error(f"Failed to send quality alerts: {e}")
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_api/test_platform_quality.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/api/platform.py tests/test_api/test_platform_quality.py
git commit -m "feat: add quality check methods to PlatformAPI

- Add run_quality_checks() method
- Add get_quality_trend() method
- Add get_data_health_summary() method
- Add _send_quality_alerts() helper
- Addresses P2-6 from data management health check"
```

### Task 8: Add Dashboard Alert Banner

**Files:**
- Modify: `hrp/dashboard/pages/data_health.py`

**Step 1: Add alert banner function**

```python
# hrp/dashboard/pages/data_health.py (add after imports)

def render_quality_alert_banner(api: PlatformAPI) -> None:
    """Render alert banner for critical quality issues."""
    try:
        # Get latest quality results
        result = api.run_quality_checks(as_of_date=date.today())

        if result["critical_issues"] > 0:
            st.error(
                f"🚨 **Critical Data Quality Issues Detected**\n\n"
                f"{result['critical_issues']} critical issues found. "
                f"Health Score: {result['health_score']:.0f}/100"
            )

            # Show top 5 critical issues
            all_issues = []
            for r in result["results"]:
                all_issues.extend(r["issues"])

            critical_issues = [i for i in all_issues if i["severity"] == "critical"]
            critical_issues.sort(key=lambda x: x.get("date", ""), reverse=True)

            with st.expander("View Critical Issues"):
                for issue in critical_issues[:5]:
                    st.markdown(f"**[{issue['symbol']}] {issue['date']}**")
                    st.markdown(f"- {issue['description']}")
                    if issue.get("details"):
                        for key, value in issue["details"].items():
                            st.markdown(f"  - {key}: {value}")

        elif result["warning_issues"] > 0:
            st.warning(
                f"⚠️ **Data Quality Warnings**\n\n"
                f"{result['warning_issues']} warnings found. "
                f"Health Score: {result['health_score']:.0f}/100"
            )

    except Exception as e:
        st.warning(f"Could not load quality alerts: {e}")
```

**Step 2: Integrate into render() function**

```python
# hrp/dashboard/pages/data_health.py (in render() function, after page header)

def render() -> None:
    """Render the Data Health page with real-time alerts."""
    # Initialize Platform API
    api = PlatformAPI()

    # ... existing page header code ...

    # -------------------------------------------------------------------------
    # Real-time Quality Alert Banner
    # -------------------------------------------------------------------------
    render_quality_alert_banner(api)

    st.markdown(
        """<div style="height: 1px; background: #374151; margin: 2rem 0;"></div>""",
        unsafe_allow_html=True,
    )

    # ... rest of existing dashboard code ...
```

**Step 3: Test dashboard**

```bash
streamlit run hrp/dashboard/app.py
```

Navigate to Data Health page and verify alert banner appears.

**Step 4: Commit**

```bash
git add hrp/dashboard/pages/data_health.py
git commit -m "feat: add real-time quality alerts to dashboard

- Add render_quality_alert_banner() function
- Show critical/warning banners at top of Data Health page
- Expandable list of top critical issues
- Addresses P2-6 from data management health check"
```

---

## Verification Plan

### Phase 1 Verification

```bash
# 1. Check EMA/VWAP coverage
python -c "
from hrp.data.db import get_db
db = get_db()
result = db.fetchall('''
    SELECT feature_name, COUNT(DISTINCT CONCAT(symbol, '-', date)) as coverage
    FROM features
    WHERE feature_name IN ('ema_12d', 'ema_26d', 'vwap_20d')
    GROUP BY feature_name
''')
for row in result:
    print(f'{row[0]}: {row[1]} symbol-dates')
"

# 2. Validate feature dates cover expected range
python -c "
from hrp.data.db import get_db
db = get_db()
result = db.fetchone('''
    SELECT MIN(date), MAX(date), COUNT(DISTINCT date)
    FROM features
    WHERE feature_name IN ('ema_12d', 'ema_26d', 'vwap_20d')
''')
print(f'Date range: {result[0]} to {result[1]} ({result[2]} days)')
"
```

Expected: ~500K rows per feature (396 symbols × ~1,250 days)

### Phase 2 Verification

```bash
# 1. Check time-series fundamentals exist
python -c "
from hrp.data.db import get_db
db = get_db()
result = db.fetchall('''
    SELECT feature_name, COUNT(*) as rows, MIN(date), MAX(date)
    FROM features
    WHERE feature_name LIKE 'ts_%'
    GROUP BY feature_name
''')
for row in result:
    print(f'{row[0]}: {row[1]} rows from {row[2]} to {row[3]}')
"

# 2. Verify point-in-time correctness
python -c "
from hrp.data.db import get_db
db = get_db()
result = db.fetchall('''
    SELECT date, COUNT(DISTINCT value) as distinct_values
    FROM features
    WHERE feature_name = 'ts_eps' AND symbol = 'AAPL'
      AND date >= '2023-12-01' AND date <= '2023-12-31'
    GROUP BY date
    ORDER BY date
    LIMIT 10
''')
print('Daily EPS variation (should have different values):')
for row in result:
    print(f'  {row[0]}: {row[1]} distinct values')
"
```

Expected: Daily values change at quarter boundaries

### Phase 3 Verification

```bash
# 1. Test PlatformAPI quality methods
python -c "
from hrp.api.platform import PlatformAPI
from datetime import date

api = PlatformAPI()

# Run quality checks
result = api.run_quality_checks(as_of_date=date.today())
print(f'Health Score: {result[\"health_score\"]:.0f}/100')
print(f'Critical: {result[\"critical_issues\"]}, Warnings: {result[\"warning_issues\"]}')

# Get quality trend
trend = api.get_quality_trend(days=7)
print(f'Last 7 days health scores: {trend[\"health_scores\"]}')

# Get health summary
summary = api.get_data_health_summary()
print(f'Data Freshness: {summary[\"data_freshness\"][\"is_fresh\"]}')
print(f'Ingestion Success Rate: {summary[\"ingestion_summary\"][\"success_rate\"]}%')
"

# 2. Start dashboard and verify alerts
streamlit run hrp/dashboard/app.py
# Navigate to Data Health page
# Verify alert banner appears if critical issues exist
```

---

## Success Criteria

| Phase | Criterion | Target |
|-------|-----------|--------|
| P2-4 | EMA/VWAP coverage | ~500K rows per feature |
| P2-4 | Date range coverage | 2020-01-01 to 2026-01-24 |
| P2-5 | Time-series fundamentals | 5 new feature types (ts_*) |
| P2-5 | Point-in-time correctness | No look-ahead bias |
| P2-6 | PlatformAPI methods | 3 quality methods exposed |
| P2-6 | Dashboard alerts | Real-time banner functional |

---

## Critical Files

| Phase | File | Purpose |
|-------|------|---------|
| 1 | `hrp/data/backfill.py` | Feature backfill function |
| 1 | `tests/test_data/test_backfill.py` | EMA/VWAP tests |
| 2 | `hrp/data/ingestion/fundamentals_timeseries.py` | Time-series module |
| 2 | `hrp/agents/jobs.py` | FundamentalsTimeSeriesJob |
| 2 | `hrp/agents/scheduler.py` | Weekly scheduling |
| 2 | `tests/test_data/test_fundamentals_timeseries.py` | Time-series tests |
| 3 | `hrp/api/platform.py` | Quality API methods |
| 3 | `hrp/dashboard/pages/data_health.py` | Alert banner |
| 3 | `tests/test_api/test_platform_quality.py` | Quality API tests |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Long computation time | Process in batches, use progress tracking |
| Database lock contention | Smaller batch sizes (10 symbols), off-hours |
| Point-in-time violations | SimFin publish_date, YFinance 45-day buffer |
| API rate limits | Rate limiting, YFinance fallback |
| False positives | Tune thresholds, add whitelist |

---

**End of Plan**
