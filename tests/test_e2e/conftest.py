"""
Shared fixtures for integration tests.

Uses existing test infrastructure from conftest.py.
"""

from datetime import date
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest


@pytest.fixture(scope="function")
def mock_external_sources():
    """
    Mock external data sources for tests.

    Use this to avoid network calls during testing.
    """
    # Mock Polygon.io
    mock_polygon = MagicMock()
    sample_prices = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT'],
        'date': pd.to_datetime(['2024-01-02', '2024-01-02']),
        'open': [185.0, 370.0],
        'high': [188.0, 375.0],
        'low': [184.0, 368.0],
        'close': [187.0, 372.0],
        'volume': [50_000_000, 20_000_000],
        'vwap': [186.0, 371.0],
    })
    mock_polygon.fetch_prices.return_value = sample_prices

    # Mock YFinance
    mock_yfinance = MagicMock()
    mock_yfinance.fetch_prices.return_value = sample_prices

    with patch.multiple(
        'hrp.data.sources',
        polygon_source=mock_polygon,
        yfinance_source=mock_yfinance,
    ):
        yield {
            'polygon': mock_polygon,
            'yfinance': mock_yfinance,
        }


@pytest.fixture(scope="function")
def mock_mlflow():
    """Mock MLflow tracking client."""
    with patch('mlflow') as mock:
        mock.start_run = MagicMock()
        mock.end_run = MagicMock()
        mock.log_params = MagicMock()
        mock.log_metrics = MagicMock()
        mock.log_metric = MagicMock()
        mock.set_tag = MagicMock()
        mock.set_tags = MagicMock()
        mock.active_run = MagicMock()
        mock.search_runs = MagicMock(return_value=[])
        mock.get_run = MagicMock()
        mock.create_experiment = MagicMock()
        mock.get_experiment_by_name = MagicMock()

        yield mock


@pytest.fixture(scope="function")
def frozen_time():
    """
    Freeze time for deterministic testing.

    Usage:
        with frozen_time('2024-01-15 10:00:00'):
            # Time-dependent code uses this timestamp
            pass
    """
    from freezegun import freeze_time
    return freeze_time
