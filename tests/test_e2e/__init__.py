"""
End-to-end tests for HRP platform.

These tests verify complete workflows across multiple layers:
- Data ingestion pipelines
- Research workflows (hypothesis → backtest → validation)
- Scheduler coordination
- Dashboard integration

E2E tests use a temporary database and mock external services.
"""
