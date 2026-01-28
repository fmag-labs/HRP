#!/bin/bash
# Quick start script for automated data quality monitoring

set -e

echo "HRP Automated Data Quality Monitoring - Quick Start"
echo "==================================================="
echo

# Check Python environment
if ! python -c "import hrp" 2>/dev/null; then
    echo "Error: HRP module not found. Please activate your virtual environment."
    exit 1
fi

echo "1. Testing quality monitoring system..."
python -c "
from hrp.monitoring.quality_monitor import DataQualityMonitor
from datetime import date

monitor = DataQualityMonitor(send_alerts=False)
result = monitor.run_daily_check()

print(f'   Health Score: {result.health_score}/100')
print(f'   Trend: {result.trend}')
print(f'   Critical Issues: {result.critical_issues}')
print(f'   Warning Issues: {result.warning_issues}')
"
echo

echo "2. Checking email notification configuration..."
if [ -z "$RESEND_API_KEY" ]; then
    echo "   ⚠️  RESEND_API_KEY not set - email alerts will be disabled"
else
    echo "   ✅ RESEND_API_KEY configured"
fi

if [ -z "$NOTIFICATION_EMAIL" ]; then
    echo "   ⚠️  NOTIFICATION_EMAIL not set - email alerts will be disabled"
else
    echo "   ✅ NOTIFICATION_EMAIL configured: $NOTIFICATION_EMAIL"
fi
echo

echo "3. Scheduler configuration:"
echo "   Daily Quality Check: 06:00 AM ET"
echo "   Health Threshold: 90.0"
echo "   Data Ingestion: 18:00 PM ET (6 PM)"
echo "   Daily Backup: 02:00 AM ET (2 AM)"
echo

echo "4. To start the automated scheduler:"
echo
echo "   python -m hrp.agents.run_scheduler --with-quality-monitoring"
echo
echo "   Or with full monitoring:"
echo
echo "   python -m hrp.agents.run_scheduler \\"
echo "     --with-quality-monitoring \\"
echo "     --quality-monitor-time=06:00 \\"
echo "     --health-threshold=90.0 \\"
echo "     --with-daily-report \\"
echo "     --daily-report-time=07:00"
echo
echo "5. Access the dashboard:"
echo "   streamlit run hrp/dashboard/app.py"
echo "   Then navigate to: http://localhost:8501/Data_Health"
echo
echo "Setup complete! See docs/setup/Automated-Monitoring-Setup.md for details."
