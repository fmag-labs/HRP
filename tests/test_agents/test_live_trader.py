"""Tests for live trading agent."""
import pytest
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch
from hrp.agents.live_trader import LiveTradingAgent, TradingConfig
from hrp.execution.broker import BrokerConfig


@pytest.fixture
def mock_api():
    """Mock PlatformAPI for testing."""
    api = Mock()
    api.get_universe.return_value = ["AAPL", "MSFT"]
    api.log_event = Mock()
    return api


@pytest.fixture
def trading_config():
    """Default trading config for tests."""
    return TradingConfig(
        portfolio_value=Decimal("100000"),
        max_positions=20,
        max_position_pct=0.10,
        dry_run=True,
    )


@pytest.fixture
def broker_config():
    """Mock broker config for tests."""
    return BrokerConfig(
        host="127.0.0.1",
        port=7497,
        client_id=1,
        account="DU123456",
        paper_trading=True,
    )


def test_live_trader_no_deployed_strategies(mock_api, trading_config, broker_config):
    """Test agent handles no deployed strategies."""
    mock_api.get_deployed_strategies.return_value = []

    agent = LiveTradingAgent(
        trading_config=trading_config,
        broker_config=broker_config,
        api=mock_api,
    )
    result = agent.execute()

    assert result["status"] == "no_deployed_strategies"
    assert result["orders_generated"] == 0


def test_live_trader_no_predictions(mock_api, trading_config, broker_config):
    """Test agent handles no predictions."""
    mock_api.get_deployed_strategies.return_value = [
        {"hypothesis_id": "HYP-2026-001", "metadata": {"model_name": "model_v1"}}
    ]
    mock_api.predict_model.return_value = pd.DataFrame()

    agent = LiveTradingAgent(
        trading_config=trading_config,
        broker_config=broker_config,
        api=mock_api,
    )
    result = agent.execute()

    assert result["status"] == "no_predictions"


def test_live_trader_dry_run_mode(mock_api, trading_config, broker_config):
    """Test agent generates orders in dry-run mode."""
    mock_api.get_deployed_strategies.return_value = [
        {"hypothesis_id": "HYP-2026-001", "metadata": {"model_name": "model_v1"}}
    ]

    predictions = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "prediction": [0.05, 0.04, 0.03],
        "signal": [1.0, 1.0, 1.0],
    })
    mock_api.predict_model.return_value = predictions

    agent = LiveTradingAgent(
        trading_config=trading_config,
        broker_config=broker_config,
        api=mock_api,
    )
    result = agent.execute()

    assert result["status"] == "dry_run"
    assert result["orders_generated"] > 0
    assert result["orders_submitted"] == 0


def test_live_trader_skips_strategy_without_model(mock_api, trading_config, broker_config):
    """Test agent skips strategies without model_name."""
    mock_api.get_deployed_strategies.return_value = [
        {"hypothesis_id": "HYP-2026-001", "metadata": {}},  # No model_name
    ]

    agent = LiveTradingAgent(
        trading_config=trading_config,
        broker_config=broker_config,
        api=mock_api,
    )
    result = agent.execute()

    assert result["status"] == "no_predictions"
    mock_api.predict_model.assert_not_called()


def test_live_trader_handles_prediction_error(mock_api, trading_config, broker_config, caplog):
    """Test agent handles prediction errors gracefully."""
    mock_api.get_deployed_strategies.return_value = [
        {"hypothesis_id": "HYP-2026-001", "metadata": {"model_name": "broken"}},
    ]
    mock_api.predict_model.side_effect = ValueError("Model not found")

    agent = LiveTradingAgent(
        trading_config=trading_config,
        broker_config=broker_config,
        api=mock_api,
    )
    result = agent.execute()

    assert result["status"] == "no_predictions"
    assert "Failed to get predictions" in caplog.text


def test_trading_config_from_env():
    """Test TradingConfig loads from environment."""
    with patch.dict("os.environ", {
        "HRP_PORTFOLIO_VALUE": "50000",
        "HRP_MAX_POSITIONS": "10",
        "HRP_TRADING_DRY_RUN": "false",
    }):
        config = TradingConfig.from_env()

        assert config.portfolio_value == Decimal("50000")
        assert config.max_positions == 10
        assert config.dry_run is False


def test_live_trader_combines_multiple_strategies(mock_api, trading_config, broker_config):
    """Test agent combines predictions from multiple strategies."""
    mock_api.get_deployed_strategies.return_value = [
        {"hypothesis_id": "HYP-2026-001", "metadata": {"model_name": "model_a"}},
        {"hypothesis_id": "HYP-2026-002", "metadata": {"model_name": "model_b"}},
    ]

    # Different predictions from each model
    predictions_a = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "prediction": [0.05, 0.03],
        "signal": [1.0, 1.0],
    })
    predictions_b = pd.DataFrame({
        "symbol": ["AAPL", "GOOGL"],
        "prediction": [0.04, 0.06],
        "signal": [1.0, 1.0],
    })

    mock_api.predict_model.side_effect = [predictions_a, predictions_b]

    agent = LiveTradingAgent(
        trading_config=trading_config,
        broker_config=broker_config,
        api=mock_api,
    )
    result = agent.execute()

    assert result["status"] == "dry_run"
    assert result["orders_generated"] > 0
    # Should have combined both prediction sets
    assert mock_api.predict_model.call_count == 2
