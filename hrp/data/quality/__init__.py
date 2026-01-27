"""
Data Quality Framework for HRP.

Provides comprehensive data quality monitoring including:
- Automated quality checks (anomaly, completeness, gaps, stale data)
- Daily quality report generation
- Health score tracking over time
- Email alerts for critical issues
- Data validation utilities for pre-operation checks

Usage:
    from hrp.data.quality import generate_daily_report, run_quality_check_with_alerts, DataValidator

    # Generate a report
    report = generate_daily_report(as_of_date=date.today())
    print(f"Health Score: {report.health_score}")

    # Run checks with alerts
    result = run_quality_check_with_alerts()

    # Validate data before operation
    validation = DataValidator.validate_price_data(prices_df)
    if not validation.is_valid:
        print(f"Validation failed: {validation.errors}")
"""

from hrp.data.quality.alerts import (
    QualityAlertManager,
    run_quality_check_with_alerts,
)
from hrp.data.quality.checks import (
    CheckResult,
    CompletenessCheck,
    GapDetectionCheck,
    IssueSeverity,
    PriceAnomalyCheck,
    QualityCheck,
    QualityIssue,
    StaleDataCheck,
    VolumeAnomalyCheck,
)
from hrp.data.quality.report import (
    QualityReport,
    QualityReportGenerator,
    generate_daily_report,
)
from hrp.data.quality.validation import (
    DataValidator,
    ValidationResult,
    validate_before_operation,
)

__all__ = [
    # Checks
    "QualityCheck",
    "CheckResult",
    "QualityIssue",
    "IssueSeverity",
    "PriceAnomalyCheck",
    "CompletenessCheck",
    "GapDetectionCheck",
    "StaleDataCheck",
    "VolumeAnomalyCheck",
    # Reports
    "QualityReport",
    "QualityReportGenerator",
    "generate_daily_report",
    # Alerts
    "QualityAlertManager",
    "run_quality_check_with_alerts",
    # Validation
    "DataValidator",
    "ValidationResult",
    "validate_before_operation",
]
