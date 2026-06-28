# HRP Consumer Mode

Consumer mode is the local, low-friction way to use HRP as a daily research brief.
It keeps trading in dry-run mode by default and launches the Next.js web app
(backed by the HTTP/JSON API) in your browser.

## First-Time Install

Double-click:

```text
Install HRP.command
```

Or run:

```bash
./scripts/consumer_install.sh
```

This creates `.venv`, installs HRP, creates `~/hrp-data`, creates `.env` if it is
missing, and initializes the DuckDB schema.

## Open HRP

Double-click:

```text
Open HRP.command
```

Or run:

```bash
./scripts/open_hrp.sh
```

This starts the HTTP/JSON API (`http://localhost:8090/api`) and the Next.js web
app, then opens `http://localhost:3000` in your browser.

## Enable Daily Local Runs

Double-click:

```text
Enable Daily HRP.command
```

Or run:

```bash
./scripts/install_consumer_launchd.sh install
```

By default this installs one macOS LaunchAgent, `com.hrp.consumer-daily`, that
runs daily at 18:30 local time. Override the time before installing:

```bash
HRP_CONSUMER_DAILY_HOUR=7 HRP_CONSUMER_DAILY_MINUTE=15 ./scripts/install_consumer_launchd.sh install
```

Check or remove it:

```bash
./scripts/install_consumer_launchd.sh status
./scripts/install_consumer_launchd.sh uninstall
```

## Daily Refresh Contents

The consumer daily run executes:

```text
universe -> prices -> features -> quality-monitoring -> recommendations -> daily-report
```

Recommendations require at least one active production model deployment. If there
are no deployed models yet, the web app still shows data freshness and job
history while the research pipeline builds toward usable recommendations.

## Safety Defaults

The generated `.env` uses:

```text
HRP_BROKER_TYPE=paper
HRP_TRADING_DRY_RUN=true
```

Leave dry-run enabled until recommendations have been reviewed manually and paper
trading has been tested separately.
