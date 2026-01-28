# Data Quality Monitoring - Command Reference

## Quick Commands

### Test Monitoring System
```bash
# Quick test (no alerts)
python -c "
from hrp.monitoring.quality_monitor import DataQualityMonitor
monitor = DataQualityMonitor(send_alerts=False)
result = monitor.run_daily_check()
print(f'Health: {result.health_score}/100, Critical: {result.critical_issues}, Warnings: {result.warning_issues}')
"

# Full test with alerts
python -c "
from hrp.monitoring.quality_monitor import run_quality_monitor_with_alerts
result = run_quality_monitor_with_alerts(send_alerts=True)
print(f'Health: {result.health_score}/100, Alerts Sent: {sum(result.alerts_sent.values())}')
"
```

### Start Automated Scheduler
```bash
# Basic monitoring
python -m hrp.agents.run_scheduler --with-quality-monitoring

# Full monitoring
python -m hrp.agents.run_scheduler \
    --with-quality-monitoring \
    --quality-monitor-time="06:00" \
    --health-threshold=90.0 \
    --with-daily-report \
    --daily-report-time="07:00"

# Everything enabled
python -m hrp.agents.run_scheduler \
    --with-quality-monitoring \
    --with-daily-report \
    --with-weekly-report \
    --with-signal-scan \
    --with-research-triggers
```

### Quick Start Script
```bash
bash scripts/start_monitoring.sh
```

### Dashboard
```bash
streamlit run hrp/dashboard/app.py
# Navigate to: http://localhost:8501/Data_Health
```

## Scheduler CLI Options

### Quality Monitoring Options
```bash
--with-quality-monitoring        # Enable daily quality monitoring
--quality-monitor-time="06:00"   # Time to run check (HH:MM format)
--health-threshold=90.0          # Health score threshold (0-100)
```

### Data Ingestion Options
```bash
--price-time="18:00"             # Price ingestion time (6 PM ET)
--universe-time="18:05"          # Universe update time
--feature-time="18:10"           # Feature computation time
--backup-time="02:00"            # Daily backup time (2 AM ET)
--backup-keep-days=30            # Days of backups to retain
--no-backup                      # Disable daily backup
--symbols AAPL MSFT              # Specific symbols to ingest
```

### Research Agent Options
```bash
--with-research-triggers         # Enable event-driven agent pipeline
--trigger-poll-interval=60       # Lineage poll interval (seconds)
--with-signal-scan               # Enable weekly signal scan
--signal-scan-time="19:00"       # Signal scan time (7 PM ET)
--signal-scan-day="mon"          # Signal scan day (mon-sun)
--ic-threshold=0.03              # Minimum IC to create hypothesis
--with-quality-sentinel          # Enable ML Quality Sentinel
--sentinel-time="06:00"          # ML Quality Sentinel time
```

### Report Options
```bash
--with-daily-report              # Enable daily research report
--daily-report-time="07:00"      # Daily report time (7 AM ET)
--with-weekly-report             # Enable weekly research report
--weekly-report-time="20:00"     # Weekly report time (8 PM ET)
```

### Fundamentals Options
```bash
--fundamentals-time="10:00"      # Fundamentals time (10 AM ET)
--fundamentals-day="sat"         # Fundamentals day (mon-sun)
--fundamentals-source="simfin"   # Source: simfin or yfinance
--no-fundamentals                # Disable fundamentals ingestion
```

## Python API

### Basic Usage
```python
from hrp.monitoring.quality_monitor import run_quality_monitor_with_alerts
from datetime import date

# Run with defaults
result = run_quality_monitor_with_alerts(as_of_date=date.today())
```

### Custom Thresholds
```python
from hrp.monitoring.quality_monitor import (
    DataQualityMonitor,
    MonitoringThresholds,
)

# Create custom thresholds
thresholds = MonitoringThresholds(
    health_score_warning=95.0,
    health_score_critical=80.0,
    freshness_warning_days=2,
    freshness_critical_days=4,
)

# Run with custom thresholds
monitor = DataQualityMonitor(
    thresholds=thresholds,
    send_alerts=True,
)
result = monitor.run_daily_check()
```

### Access Results
```python
# Health score and trend
print(f"Health Score: {result.health_score}/100")
print(f"Trend: {result.trend}")  # improving, stable, declining

# Issues
print(f"Critical: {result.critical_issues}")
print(f"Warnings: {result.warning_issues}")
print(f"Total: {result.total_issues}")

# Alerts sent
for alert_type, sent in result.alerts_sent.items():
    print(f"{alert_type}: {sent}")

# Recommendations
for rec in result.recommendations:
    print(f"- {rec}")
```

### Health Summary
```python
from hrp.monitoring.quality_monitor import DataQualityMonitor

monitor = DataQualityMonitor()
summary = monitor.get_health_summary(days=30)

print(f"Current Health: {summary['current_health_score']}")
print(f"Trend: {summary['trend_direction']}")
print(f"Data Points: {summary['trend_data_points']}")
```

## Platform API Integration

### Run Quality Checks via Platform API
```python
from hrp.api.platform import PlatformAPI
from datetime import date

api = PlatformAPI()
result = api.run_quality_checks(
    as_of_date=date.today(),
    send_alerts=True,
)

print(f"Health Score: {result['health_score']}")
print(f"Critical Issues: {result['critical_issues']}")
print(f"Warnings: {result['warning_issues']}")
```

### Get Quality Trend
```python
api = PlatformAPI()
trend = api.get_quality_trend(days=30)

for data_point in trend:
    print(f"{data_point['date']}: {data_point['health_score']}/100")
```

### Get Data Health Summary
```python
api = PlatformAPI()
summary = api.get_data_health_summary()

print(f"Symbols: {summary['symbol_count']}")
print(f"Date Range: {summary['date_range']}")
print(f"Total Records: {summary['total_records']}")
print(f"Data Freshness: {summary['data_freshness']}")
```

## Email Configuration

### Set Environment Variables
```bash
# Required
export RESEND_API_KEY="re_your_api_key_here"
export NOTIFICATION_EMAIL="your_email@example.com"

# Optional
export NOTIFICATION_FROM_EMAIL="noreply@hrp.local"
```

### Test Email Notifications
```python
from hrp.notifications.email import EmailNotifier

notifier = EmailNotifier()
success = notifier.send_email(
    subject="Test Email",
    body="This is a test email from HRP.",
)
print(f"Email sent: {success}")
```

### Test Quality Alerts
```python
from hrp.data.quality.alerts import QualityAlertManager
from hrp.data.quality.report import QualityReportGenerator
from datetime import date

# Generate report
generator = QualityReportGenerator()
report = generator.generate_report(date.today())

# Send alerts
alert_manager = QualityAlertManager()
result = alert_manager.process_report(report, send_summary=True)

print(f"Critical alert sent: {result['critical_alert_sent']}")
print(f"Summary sent: {result['summary_sent']}")
```

## Scheduler Management (launchd)

### Check Status
```bash
launchctl list | grep hrp
```

### Load Scheduler
```bash
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

### Unload Scheduler
```bash
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

### Restart Scheduler
```bash
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist && \
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

### View Logs
```bash
# Error log
tail -f ~/hrp-data/logs/scheduler.error.log

# Output log
tail -f ~/hrp-data/logs/scheduler.out.log

# Both logs
tail -f ~/hrp-data/logs/scheduler.*.log
```

### Filter Logs
```bash
# Email-related logs
tail -f ~/hrp-data/logs/scheduler.error.log | grep -i email

# Quality check logs
tail -f ~/hrp-data/logs/scheduler.error.log | grep -i quality

# Alert logs
tail -f ~/hrp-data/logs/scheduler.error.log | grep -i alert
```

## Database Queries

### Recent Quality Reports
```python
from hrp.data.db import get_db

db = get_db()
query = """
    SELECT report_id, report_date, health_score, critical_issues, warning_issues
    FROM quality_reports
    ORDER BY report_date DESC
    LIMIT 10
"""
df = db.fetchdf(query)
print(df)
```

### Health Score Trend
```python
db = get_db()
query = """
    SELECT report_date, health_score
    FROM quality_reports
    WHERE report_date >= CURRENT_DATE - INTERVAL '30 days'
    ORDER BY report_date ASC
"""
df = db.fetchdf(query)
print(df)
```

### Critical Issues by Check
```python
db = get_db()
query = """
    SELECT check_name, COUNT(*) as issue_count
    FROM quality_reports
    WHERE severity = 'critical'
    GROUP BY check_name
    ORDER BY issue_count DESC
"""
df = db.fetchdf(query)
print(df)
```

### Stale Data Issues
```python
db = get_db()
query = """
    SELECT symbol, date, description
    FROM quality_reports
    WHERE check_name = 'stale_data' AND severity = 'critical'
    ORDER BY date DESC
    LIMIT 20
"""
df = db.fetchdf(query)
print(df)
```

## Troubleshooting Commands

### Check Email Configuration
```bash
echo "RESEND_API_KEY: $RESEND_API_KEY"
echo "NOTIFICATION_EMAIL: $NOTIFICATION_EMAIL"
echo "NOTIFICATION_FROM_EMAIL: $NOTIFICATION_FROM_EMAIL"
```

### Test Database Connection
```python
from hrp.data.db import get_db

db = get_db()
result = db.fetchone("SELECT COUNT(*) FROM prices")
print(f"Total price records: {result[0]}")
```

### Check Quality Tables
```python
from hrp.data.db import get_db

db = get_db()

# Check quality_reports table
result = db.fetchone("SELECT COUNT(*) FROM quality_reports")
print(f"Quality reports: {result[0]}")

# Most recent report
result = db.fetchone("""
    SELECT report_date, health_score, critical_issues
    FROM quality_reports
    ORDER BY report_date DESC
    LIMIT 1
""")
print(f"Latest: {result[0]} - Health: {result[1]}/100 - Critical: {result[2]}")
```

### Verify Scheduler Jobs
```python
from hrp.agents.scheduler import IngestionScheduler

scheduler = IngestionScheduler()
jobs = scheduler.list_jobs()

print("Scheduled Jobs:")
for job in jobs:
    print(f"  - {job['id']}: {job['name']}")
    print(f"    Next run: {job['next_run']}")
```

### Manual Quality Check
```python
from hrp.data.quality.report import QualityReportGenerator
from datetime import date

generator = QualityReportGenerator()
report = generator.generate_report(date.today())

print(f"Health Score: {report.health_score}/100")
print(f"Passed: {report.passed}")
print(f"Critical Issues: {report.critical_issues}")
print(f"Warning Issues: {report.warning_issues}")

print("\nCheck Results:")
for result in report.results:
    status = "✅" if result.passed else "❌"
    print(f"  {status} {result.check_name}: {result.critical_count} critical, {result.warning_count} warnings")
```

## Common Workflows

### Daily Monitoring Check
```bash
# 1. Check health score
python -c "
from hrp.monitoring.quality_monitor import DataQualityMonitor
monitor = DataQualityMonitor()
result = monitor.run_daily_check()
print(f'Health: {result.health_score}/100 ({result.trend})')
print(f'Critical: {result.critical_issues}, Warnings: {result.warning_issues}')
"

# 2. If issues found, review recommendations
python -c "
from hrp.monitoring.quality_monitor import DataQualityMonitor
monitor = DataQualityMonitor()
result = monitor.run_daily_check()
for rec in result.recommendations:
    print(f'- {rec}')
"

# 3. Check dashboard
streamlit run hrp/dashboard/app.py
```

### Investigate Health Score Drop
```bash
# 1. Get historical trend
python -c "
from hrp.data.quality.report import QualityReportGenerator
from datetime import date, timedelta

generator = QualityReportGenerator()
for days_ago in range(7, 0, -1):
    d = date.today() - timedelta(days=days_ago)
    r = generator.generate_report(d)
    print(f'{d}: {r.health_score}/100')
"

# 2. Identify top issues
python -c "
from hrp.data.db import get_db
db = get_db()
query = '''
    SELECT check_name, COUNT(*) as count
    FROM quality_reports
    WHERE report_date >= CURRENT_DATE - INTERVAL '7 days'
    AND severity = \"critical\"
    GROUP BY check_name
    ORDER BY count DESC
'''
df = db.fetchdf(query)
print(df)
"
```

### Setup Production Monitoring
```bash
# 1. Configure email alerts
export RESEND_API_KEY="your_key"
export NOTIFICATION_EMAIL="your_email@example.com"

# 2. Test monitoring
python -c "from hrp.monitoring.quality_monitor import run_quality_monitor_with_alerts; run_quality_monitor_with_alerts()"

# 3. Create launchd plist (see docs)

# 4. Load scheduler
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist

# 5. Verify running
launchctl list | grep hrp
tail -f ~/hrp-data/logs/scheduler.error.log
```

## Quick Reference Card

### Most Common Commands

```bash
# Test monitoring
python -c "from hrp.monitoring.quality_monitor import run_quality_monitor_with_alerts; run_quality_monitor_with_alerts()"

# Start scheduler with monitoring
python -m hrp.agents.run_scheduler --with-quality-monitoring

# Check health trend
python -c "from hrp.monitoring.quality_monitor import DataQualityMonitor; print(DataQualityMonitor().get_health_summary())"

# Start dashboard
streamlit run hrp/dashboard/app.py

# View logs
tail -f ~/hrp-data/logs/scheduler.error.log

# Check scheduler status
launchctl list | grep hrp
```

### Health Score Interpretation

- **90-100**: Healthy ✅ - No action needed
- **70-89**: Warning ⚠️ - Review warnings, schedule maintenance
- **0-69**: Critical ❌ - Immediate investigation required

### Alert Priority

1. **Critical Alerts** - Immediate action
   - Health score < 70
   - Data freshness > 5 days
   - Any critical issues

2. **Warning Alerts** - Review soon
   - Health score 70-89
   - Data freshness 3-5 days
   - High warning count

3. **Daily Summary** - Informational
   - Daily health status
   - All issues summary
   - Recommendations
