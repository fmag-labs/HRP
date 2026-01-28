# Scheduler Configuration for Automated Monitoring

## Recommended Configurations

### 1. Minimal Monitoring Setup
**For: Basic quality monitoring with minimal overhead**

```bash
python -m hrp.agents.run_scheduler \
    --with-quality-monitoring \
    --quality-monitor-time="06:00" \
    --health-threshold=90.0
```

**Jobs Scheduled:**
- 06:00 AM ET - Daily quality monitoring
- 18:00 PM ET - Daily price ingestion
- 18:05 PM ET - Daily universe update
- 18:10 PM ET - Daily feature computation
- 02:00 AM ET - Daily backup

### 2. Standard Monitoring Setup (Recommended)
**For: Daily monitoring with reporting**

```bash
python -m hrp.agents.run_scheduler \
    --with-quality-monitoring \
    --quality-monitor-time="06:00" \
    --health-threshold=90.0 \
    --with-daily-report \
    --daily-report-time="07:00"
```

**Jobs Scheduled:**
- 06:00 AM ET - Daily quality monitoring
- 07:00 AM ET - Daily research report
- 18:00 PM ET - Daily price ingestion
- 18:05 PM ET - Daily universe update
- 18:10 PM ET - Daily feature computation
- 02:00 AM ET - Daily backup

### 3. Full Monitoring Setup
**For: Complete monitoring with research agents**

```bash
python -m hrp.agents.run_scheduler \
    --with-quality-monitoring \
    --quality-monitor-time="06:00" \
    --health-threshold=90.0 \
    --with-daily-report \
    --daily-report-time="07:00" \
    --with-weekly-report \
    --weekly-report-time="20:00" \
    --with-signal-scan \
    --signal-scan-time="19:00" \
    --signal-scan-day="mon" \
    --ic-threshold=0.03 \
    --with-quality-sentinel \
    --sentinel-time="06:00"
```

**Jobs Scheduled:**
- 06:00 AM ET - Daily quality monitoring
- 06:00 AM ET - ML Quality Sentinel audit
- 07:00 AM ET - Daily research report
- 19:00 PM ET (Mon) - Weekly signal scan
- 20:00 PM ET (Sun) - Weekly research report
- 18:00 PM ET - Daily price ingestion
- 18:05 PM ET - Daily universe update
- 18:10 PM ET - Daily feature computation
- 02:00 AM ET - Daily backup
- 10:00 AM ET (Sat) - Weekly fundamentals

### 4. Complete Research Pipeline
**For: Full automation with event-driven agents**

```bash
python -m hrp.agents.run_scheduler \
    --with-quality-monitoring \
    --quality-monitor-time="06:00" \
    --health-threshold=90.0 \
    --with-daily-report \
    --daily-report-time="07:00" \
    --with-weekly-report \
    --weekly-report-time="20:00" \
    --with-signal-scan \
    --signal-scan-time="19:00" \
    --signal-scan-day="mon" \
    --ic-threshold=0.03 \
    --with-quality-sentinel \
    --sentinel-time="06:00" \
    --with-research-triggers \
    --trigger-poll-interval=60
```

**Jobs Scheduled:**
- All jobs from "Full Monitoring Setup"
- Event-driven triggers:
  - Signal Scientist → Alpha Researcher
  - Alpha Researcher → ML Scientist
  - ML Scientist → ML Quality Sentinel
  - ML Quality Sentinel → Validation Analyst

## Daily Schedule Timeline

### Early Morning (Quality & Reports)
```
02:00 AM - Daily backup
06:00 AM - Quality monitoring check
06:00 AM - ML Quality Sentinel audit
07:00 AM - Daily research report
```

### Evening (Data Ingestion)
```
18:00 PM - Price ingestion (6 PM ET)
18:05 PM - Universe update
18:10 PM - Feature computation
```

### Weekly Jobs
```
Monday 7:00 PM - Signal scan
Saturday 10:00 AM - Fundamentals ingestion
Sunday 8:00 PM - Weekly research report
Sunday 2:00 AM - Data cleanup (if enabled)
```

## launchd Configuration Examples

### Minimal Monitoring
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

### Standard Monitoring (Recommended)
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

## Timing Considerations

### Quality Check Timing (06:00 AM ET)
- **Before market open** (9:30 AM ET)
- Allows time to address issues before trading
- Email alerts arrive before work day
- Sufficient time for overnight data processing

### Data Ingestion Timing (18:00 PM ET)
- **After market close** (4:00 PM ET)
- Allows 2 hours for data to be available
- Before overnight processing begins
- East Coast data sources most reliable

### Report Timing (07:00 AM ET)
- **After quality check** (06:00 AM ET)
- Before market open
- During morning preparation time
- Includes latest quality status

### Backup Timing (02:00 AM ET)
- **Low activity period**
- Minimal database contention
- Sufficient time before morning jobs
- Allows for lengthy backup operations

## Health Threshold Guidelines

### Development Environment
```bash
--health-threshold=80.0
```
- More lenient threshold
- Focus on critical issues only
- Allows for experimental changes

### Staging Environment
```bash
--health-threshold=90.0
```
- Standard threshold
- Catches most issues
- Balances sensitivity vs noise

### Production Environment
```bash
--health-threshold=95.0
```
- Strict threshold
- Early warning of issues
- Highest data quality standards

## Resource Requirements

### Minimal Monitoring
- **CPU**: Low background usage
- **Memory**: ~50 MB baseline
- **Disk**: ~10 MB/day for logs
- **Email**: 1-2 emails/day (if issues detected)

### Standard Monitoring
- **CPU**: Low background usage
- **Memory**: ~100 MB baseline
- **Disk**: ~20 MB/day for logs and reports
- **Email**: 1-3 emails/day (daily summary + alerts)

### Full Monitoring
- **CPU**: Moderate during scans
- **Memory**: ~200 MB baseline
- **Disk**: ~50 MB/day for logs, reports, MLflow
- **Email**: 2-5 emails/day (daily + weekly + alerts)

## Monitoring Best Practices

### 1. Start Simple
Begin with minimal monitoring, then add features:
```bash
# Week 1: Basic monitoring
python -m hrp.agents.run_scheduler --with-quality-monitoring

# Week 2: Add daily reports
python -m hrp.agents.run_scheduler --with-quality-monitoring --with-daily-report

# Week 3: Add weekly reports
python -m hrp.agents.run_scheduler --with-quality-monitoring --with-daily-report --with-weekly-report
```

### 2. Monitor Logs
Check logs regularly, especially after configuration changes:
```bash
tail -f ~/hrp-data/logs/scheduler.error.log
```

### 3. Verify Jobs
Ensure scheduled jobs are running:
```bash
launchctl list | grep hrp
```

### 4. Test Alerts
Verify email notifications are working:
```bash
python -c "
from hrp.monitoring.quality_monitor import run_quality_monitor_with_alerts
result = run_quality_monitor_with_alerts(send_alerts=True)
print(f'Alerts sent: {sum(result.alerts_sent.values())}')
"
```

### 5. Review Health Trends
Check dashboard regularly:
```bash
streamlit run hrp/dashboard/app.py
# Navigate to: http://localhost:8501/Data_Health
```

## Troubleshooting Scheduler Issues

### Jobs Not Running
```bash
# Check scheduler status
launchctl list | grep hrp

# View error logs
tail -f ~/hrp-data/logs/scheduler.error.log

# Restart scheduler
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

### Jobs Running at Wrong Time
```bash
# Check system timezone
date

# Verify ET timezone in logs
grep "timezone" ~/hrp-data/logs/scheduler.error.log

# Restart with correct timezone
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
# Edit plist with correct times
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

### High Memory Usage
```bash
# Reduce job frequency
# Disable research agents
python -m hrp.agents.run_scheduler \
    --with-quality-monitoring \
    --quality-monitor-time="06:00"

# Disable MLflow logging (if not needed)
export MLFLOW_ENABLED=false
```

## Upgrading Configuration

### From Minimal to Standard
1. Stop scheduler:
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
   ```

2. Update plist with new arguments:
   ```xml
   <string>--with-daily-report</string>
   <string>--daily-report-time=07:00</string>
   ```

3. Reload scheduler:
   ```bash
   launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
   ```

### From Standard to Full
1. Stop scheduler
2. Update plist with research agent options
3. Verify email configuration
4. Reload scheduler
5. Monitor logs for agent activity

## Quick Reference

### Start Scheduler
```bash
# Minimal
python -m hrp.agents.run_scheduler --with-quality-monitoring

# Standard (Recommended)
python -m hrp.agents.run_scheduler --with-quality-monitoring --with-daily-report

# Full
python -m hrp.agents.run_scheduler --with-quality-monitoring --with-daily-report --with-weekly-report --with-signal-scan
```

### Check Status
```bash
# Scheduler running
launchctl list | grep hrp

# View logs
tail -f ~/hrp-data/logs/scheduler.error.log

# Test monitoring
python -c "from hrp.monitoring.quality_monitor import run_quality_monitor_with_alerts; run_quality_monitor_with_alerts()"
```

### Stop Scheduler
```bash
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

### Restart Scheduler
```bash
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

## Support

For issues or questions:
- Logs: `~/hrp-data/logs/scheduler.error.log`
- Dashboard: `http://localhost:8501/Data_Health`
- Documentation: `docs/setup/Automated-Monitoring-Setup.md`
