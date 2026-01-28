# Automated Data Quality Monitoring System - COMPLETE

## Executive Summary

I've successfully implemented a comprehensive automated monitoring system for data quality issues in the HRP platform. The system is production-ready and includes:

✅ **Daily automated quality checks** (5 quality dimensions)
✅ **Health score tracking** (0-100 metric with trend analysis)
✅ **Threshold-based alerting** (5 different alert types)
✅ **Email notifications** (via Resend)
✅ **Dashboard integration** (Streamlit)
✅ **Scheduler integration** (fully automated)
✅ **Comprehensive documentation**

## Current System Status

### Health Score: 50/100 (CRITICAL)

**Issues Detected:**
- Critical: 1 (stale data)
- Warnings: 387 (completeness)
- Total: 388

**Quality Checks:**
- ✅ Price Anomaly: PASS
- ⚠️ Completeness: 387 issues
- ✅ Gap Detection: PASS
- ❌ Stale Data: 1 critical
- ✅ Volume Anomaly: PASS

**Trend:** Stable

## What Was Created

### 1. Core Monitoring Module
**File:** `/Users/fer/Documents/GitHub/HRP/hrp/monitoring/quality_monitor.py`

**Classes:**
- `DataQualityMonitor` - Main monitoring class
- `MonitoringThresholds` - Configurable alert thresholds
- `MonitoringResult` - Result object with check data
- `run_quality_monitor_with_alerts()` - Convenience function

**Features:**
- Daily quality check execution
- Health score trend calculation (improving/stable/declining)
- Actionable recommendations generation
- Multi-level alerting (warning/critical)
- Integration with existing QualityReport system

### 2. Scheduler Integration
**Modified Files:**
- `/Users/fer/Documents/GitHub/HRP/hrp/agents/scheduler.py` - Added `setup_quality_monitoring()`
- `/Users/fer/Documents/GitHub/HRP/hrp/agents/run_scheduler.py` - Added CLI options

**New CLI Options:**
- `--with-quality-monitoring` - Enable daily quality checks
- `--quality-monitor-time` - Set check time (default: 06:00)
- `--health-threshold` - Set warning threshold (default: 90.0)

### 3. Documentation
**Created:**
- `/Users/fer/Documents/GitHub/HRP/docs/setup/Automated-Monitoring-Setup.md` - Complete setup guide
- `/Users/fer/Documents/GitHub/HRP/docs/setup/Monitoring-Summary.md` - Implementation summary
- `/Users/fer/Documents/GitHub/HRP/docs/setup/Monitoring-Command-Reference.md` - Command reference
- `/Users/fer/Documents/GitHub/HRP/scripts/start_monitoring.sh` - Quick start script

## Quick Start

### 1. Test the System
```bash
bash scripts/start_monitoring.sh
```

### 2. Configure Email Alerts (Optional)
```bash
export RESEND_API_KEY="your_resend_api_key"
export NOTIFICATION_EMAIL="your_email@example.com"
```

### 3. Start Automated Monitoring
```bash
# Basic monitoring
python -m hrp.agents.run_scheduler --with-quality-monitoring

# Full monitoring (recommended)
python -m hrp.agents.run_scheduler \
    --with-quality-monitoring \
    --quality-monitor-time="06:00" \
    --health-threshold=90.0 \
    --with-daily-report \
    --daily-report-time="07:00"
```

### 4. Access Dashboard
```bash
streamlit run hrp/dashboard/app.py
# Navigate to: http://localhost:8501/Data_Health
```

## Alert Thresholds

### Default Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Health Score | < 90 | < 70 |
| Data Freshness | > 3 days stale | > 5 days stale |
| Anomaly Count | - | > 100 anomalies |

### Alert Types

1. **Health Score Warning** - Score < 90
   - Sends daily summary email
   - Includes recommendations

2. **Health Score Critical** - Score < 70
   - Sends immediate critical alert
   - Lists all critical issues

3. **Critical Issues Alert** - Any critical issues
   - Immediate email with details
   - Sent regardless of health score

4. **Data Freshness Alert** - Data too old
   - Warning: > 3 days stale
   - Critical: > 5 days stale

5. **Anomaly Spike Alert** - High anomaly count
   - Triggered when > 100 anomalies
   - Indicates systemic issues

## Usage Examples

### Manual Quality Check
```python
from hrp.monitoring.quality_monitor import run_quality_monitor_with_alerts
from datetime import date

result = run_quality_monitor_with_alerts(as_of_date=date.today())

print(f"Health Score: {result.health_score}/100")
print(f"Trend: {result.trend}")
print(f"Critical: {result.critical_issues}")
print(f"Warnings: {result.warning_issues}")
print(f"Alerts Sent: {sum(result.alerts_sent.values())}")
print(f"Recommendations:")
for rec in result.recommendations:
    print(f"  - {rec}")
```

### Custom Thresholds
```python
from hrp.monitoring.quality_monitor import (
    DataQualityMonitor,
    MonitoringThresholds,
)

# Strict thresholds for production
strict_thresholds = MonitoringThresholds(
    health_score_warning=95.0,
    health_score_critical=80.0,
    freshness_warning_days=1,
    freshness_critical_days=3,
)

monitor = DataQualityMonitor(
    thresholds=strict_thresholds,
    send_alerts=True,
)
result = monitor.run_daily_check()
```

### Health Trend Summary
```python
from hrp.monitoring.quality_monitor import DataQualityMonitor

monitor = DataQualityMonitor()
summary = monitor.get_health_summary(days=30)

print(f"Current Health: {summary['current_health_score']}")
print(f"Trend: {summary['trend_direction']}")
print(f"Data Points: {summary['trend_data_points']}")
```

## Production Deployment

### Using launchd (macOS)

Create `~/Library/LaunchAgents/com.hrp.scheduler.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hrp.scheduler</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/fer/Documents/GitHub/HRP/.venv/bin/python</string>
        <string>-m</string>
        <string>hrp.agents.run_scheduler</string>
        <string>--with-quality-monitoring</string>
        <string>--quality-monitor-time=06:00</string>
        <string>--health-threshold=90.0</string>
        <string>--with-daily-report</string>
        <string>--daily-report-time=07:00</string>
    </array>
    <key>StandardOutPath</key>
    <string>/Users/fer/hrp-data/logs/scheduler.out.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/fer/hrp-data/logs/scheduler.error.log</string>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
```

Load the scheduler:
```bash
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

## Monitoring Dashboard

### Access
```bash
streamlit run hrp/dashboard/app.py
# Navigate to: http://localhost:8501/Data_Health
```

### Features
- Real-time health score display
- Historical trend chart (90 days)
- Quality checks summary table
- Flagged anomalies drill-down
- Ingestion status monitoring
- Symbol coverage analysis

## Troubleshooting

### No Email Alerts
```bash
# Check environment variables
echo $RESEND_API_KEY
echo $NOTIFICATION_EMAIL

# Test email sending
python -c "
from hrp.notifications.email import EmailNotifier
notifier = EmailNotifier()
success = notifier.send_email(
    subject='Test Email',
    body='This is a test.',
)
print(f'Email sent: {success}')
"
```

### Health Score Declining
```python
# Review recommendations
from hrp.monitoring.quality_monitor import DataQualityMonitor

monitor = DataQualityMonitor()
result = monitor.run_daily_check()
for rec in result.recommendations:
    print(f"- {rec}")

# Check historical trend
from hrp.data.quality.report import QualityReportGenerator
from datetime import date, timedelta

generator = QualityReportGenerator()
for days_ago in range(7, 0, -1):
    d = date.today() - timedelta(days=days_ago)
    r = generator.generate_report(d)
    print(f"{d}: {r.health_score}/100")
```

### Scheduler Not Running
```bash
# Check status
launchctl list | grep hrp

# View logs
tail -f ~/hrp-data/logs/scheduler.error.log

# Restart
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

## Testing

All tests pass:
```bash
pytest tests/ -k "quality" -v
# 133 tests passed
```

## Next Steps

### Immediate Actions
1. Configure email alerts (optional but recommended)
2. Address critical issues (1 stale data issue)
3. Start automated monitoring

### Optional Enhancements
1. Customize thresholds based on requirements
2. Add additional quality checks
3. Integrate with external monitoring systems

## Files Created/Modified

### New Files
1. `/Users/fer/Documents/GitHub/HRP/hrp/monitoring/quality_monitor.py` - Core monitoring module
2. `/Users/fer/Documents/GitHub/HRP/docs/setup/Automated-Monitoring-Setup.md` - Setup guide
3. `/Users/fer/Documents/GitHub/HRP/docs/setup/Monitoring-Summary.md` - Implementation summary
4. `/Users/fer/Documents/GitHub/HRP/docs/setup/Monitoring-Command-Reference.md` - Command reference
5. `/Users/fer/Documents/GitHub/HRP/scripts/start_monitoring.sh` - Quick start script

### Modified Files
1. `/Users/fer/Documents/GitHub/HRP/hrp/monitoring/__init__.py` - Added exports
2. `/Users/fer/Documents/GitHub/HRP/hrp/agents/scheduler.py` - Added `setup_quality_monitoring()`
3. `/Users/fer/Documents/GitHub/HRP/hrp/agents/run_scheduler.py` - Added CLI options

## Summary

The automated monitoring system is fully operational and provides:

- ✅ Daily automated quality checks (5 dimensions)
- ✅ Health score tracking with trend analysis
- ✅ Threshold-based alerting (5 alert types)
- ✅ Email notification support (Resend)
- ✅ Dashboard integration for visual monitoring
- ✅ Actionable recommendations
- ✅ Scheduler integration for automation
- ✅ Comprehensive documentation

**Current Status:**
- Health Score: 50/100 (Critical)
- Active Alerts: 1 critical, 387 warnings
- Recommendations: 4 actionable items
- Monitoring: Ready for production deployment

**To start monitoring:**
```bash
bash scripts/start_monitoring.sh
python -m hrp.agents.run_scheduler --with-quality-monitoring
```

**For detailed documentation, see:**
- `docs/setup/Automated-Monitoring-Setup.md` - Complete setup guide
- `docs/setup/Monitoring-Command-Reference.md` - Command reference
- `docs/setup/Monitoring-Summary.md` - Implementation summary
