"""Live trading agent for executing signals."""
import logging
import os
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from hrp.agents.jobs import DataRequirement, IngestionJob
from hrp.api.platform import PlatformAPI
from hrp.execution.broker import BrokerConfig, IBKRBroker
from hrp.execution.orders import OrderManager
from hrp.execution.positions import PositionTracker
from hrp.execution.signal_converter import ConversionConfig, SignalConverter

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Configuration for live trading."""

    portfolio_value: Decimal
    max_positions: int = 20
    max_position_pct: float = 0.10
    min_order_value: Decimal = Decimal("100.00")
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> "TradingConfig":
        """Load trading config from environment."""
        return cls(
            portfolio_value=Decimal(os.getenv("HRP_PORTFOLIO_VALUE", "100000")),
            max_positions=int(os.getenv("HRP_MAX_POSITIONS", "20")),
            max_position_pct=float(os.getenv("HRP_MAX_POSITION_PCT", "0.10")),
            min_order_value=Decimal(os.getenv("HRP_MIN_ORDER_VALUE", "100")),
            dry_run=os.getenv("HRP_TRADING_DRY_RUN", "true").lower() == "true",
        )


class LiveTradingAgent(IngestionJob):
    """Agent for live trading execution.

    This agent:
    1. Gets latest predictions for deployed strategies
    2. Converts predictions to signals
    3. Compares against current positions
    4. Generates rebalancing orders
    5. Submits orders to broker (if not dry-run)

    IMPORTANT: This agent is disabled by default for safety.
    Set HRP_TRADING_DRY_RUN=false to enable actual order submission.
    """

    def __init__(
        self,
        trading_config: TradingConfig | None = None,
        broker_config: BrokerConfig | None = None,
        api: PlatformAPI | None = None,
        job_id: str = "live_trader",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        """Initialize live trading agent.

        Args:
            trading_config: Trading configuration (from env if None)
            broker_config: Broker configuration (from env if None)
            api: PlatformAPI instance (creates new if None)
            job_id: Job identifier
            max_retries: Maximum retry attempts
            retry_backoff: Exponential backoff multiplier
        """
        data_requirements = [
            DataRequirement(
                table="features",
                min_rows=100,
                max_age_days=3,
                date_column="date",
                description="Recent feature data",
            ),
        ]

        super().__init__(
            job_id,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            data_requirements=data_requirements,
        )

        if api is not None:
            self.api = api

        self.trading_config = trading_config or TradingConfig.from_env()
        self.broker_config = broker_config or self._broker_config_from_env()

    def _broker_config_from_env(self) -> BrokerConfig:
        """Create broker config from environment variables."""
        return BrokerConfig(
            host=os.getenv("IBKR_HOST", "127.0.0.1"),
            port=int(os.getenv("IBKR_PORT", "7497")),
            client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
            account=os.getenv("IBKR_ACCOUNT", ""),
            paper_trading=os.getenv("IBKR_PAPER_TRADING", "true").lower() == "true",
        )

    def execute(self) -> dict[str, Any]:
        """Execute live trading agent.

        Returns:
            Dict with execution stats:
                - status: success, no_predictions, dry_run, or error
                - orders_generated: Count of orders created
                - orders_submitted: Count of orders sent to broker
                - positions_synced: Current position count
        """
        logger.info(f"Starting live trading agent (dry_run={self.trading_config.dry_run})")

        # Get deployed strategies
        deployed = self.api.get_deployed_strategies()
        if not deployed:
            logger.warning("No deployed strategies found")
            return {
                "status": "no_deployed_strategies",
                "orders_generated": 0,
                "orders_submitted": 0,
                "records_fetched": 0,
                "records_inserted": 0,
            }

        # Get latest predictions
        universe = self.api.get_universe(as_of_date=date.today())
        all_predictions = []

        for strategy in deployed:
            hypothesis_id = strategy.get("hypothesis_id") or getattr(
                strategy, "hypothesis_id", None
            )
            metadata = strategy.get("metadata") or getattr(strategy, "metadata", {})
            model_name = metadata.get("model_name") if isinstance(metadata, dict) else None

            if not model_name:
                continue

            try:
                predictions = self.api.predict_model(
                    model_name=model_name,
                    symbols=universe,
                    as_of_date=date.today(),
                )
                if predictions is not None and not predictions.empty:
                    all_predictions.append(predictions)
                    logger.info(
                        f"Got {len(predictions)} predictions from {hypothesis_id}"
                    )
            except Exception as e:
                logger.error(f"Failed to get predictions for {hypothesis_id}: {e}")
                continue

        if not all_predictions:
            logger.warning("No predictions available")
            return {
                "status": "no_predictions",
                "orders_generated": 0,
                "orders_submitted": 0,
                "records_fetched": 0,
                "records_inserted": 0,
            }

        # Combine predictions (simple average for now)
        import pandas as pd

        combined = pd.concat(all_predictions)
        combined = combined.groupby("symbol").agg({
            "prediction": "mean",
            "signal": "max",  # Take the strongest signal
        }).reset_index()

        # Dry-run mode - just generate orders without submitting
        if self.trading_config.dry_run:
            config = ConversionConfig(
                portfolio_value=self.trading_config.portfolio_value,
                max_positions=self.trading_config.max_positions,
                max_position_pct=self.trading_config.max_position_pct,
                min_order_value=self.trading_config.min_order_value,
            )
            converter = SignalConverter(config)
            orders = converter.signals_to_orders(combined, method="rank")

            logger.info(f"[DRY RUN] Would submit {len(orders)} orders")
            for order in orders:
                logger.info(
                    f"[DRY RUN] {order.side.value.upper()} {order.quantity} {order.symbol}"
                )

            return {
                "status": "dry_run",
                "orders_generated": len(orders),
                "orders_submitted": 0,
                "records_fetched": len(combined),
                "records_inserted": 0,
            }

        # Live mode - connect to broker and execute
        with IBKRBroker(self.broker_config) as broker:
            # Sync positions
            tracker = PositionTracker(broker, self.api)
            positions = tracker.sync_positions()
            tracker.persist_positions()

            current_positions = {p.symbol: p.quantity for p in positions}
            current_prices = {
                p.symbol: p.current_price for p in positions
            }

            # Get portfolio value from positions
            portfolio_value = tracker.calculate_portfolio_value()
            if portfolio_value == 0:
                portfolio_value = self.trading_config.portfolio_value

            # Convert signals to orders
            config = ConversionConfig(
                portfolio_value=portfolio_value,
                max_positions=self.trading_config.max_positions,
                max_position_pct=self.trading_config.max_position_pct,
                min_order_value=self.trading_config.min_order_value,
            )
            converter = SignalConverter(config)

            # Generate rebalancing orders
            orders = converter.rebalance_to_orders(
                current_positions, combined, current_prices
            )

            # Submit orders
            order_manager = OrderManager(broker)
            submitted_count = 0

            for order in orders:
                try:
                    order_manager.submit_order(order)
                    submitted_count += 1

                    # Record trade in database
                    self.api.record_trade(
                        order=order,
                        filled_price=current_prices.get(
                            order.symbol, Decimal("0")
                        ),
                    )
                except Exception as e:
                    logger.error(f"Failed to submit order for {order.symbol}: {e}")

            logger.info(
                f"Submitted {submitted_count}/{len(orders)} orders"
            )

            # Log to lineage
            self.api.log_event(
                event_type="agent_run_complete",
                actor="system:live_trader",
                details={
                    "orders_generated": len(orders),
                    "orders_submitted": submitted_count,
                    "positions_synced": len(positions),
                    "dry_run": False,
                },
            )

            return {
                "status": "success",
                "orders_generated": len(orders),
                "orders_submitted": submitted_count,
                "positions_synced": len(positions),
                "records_fetched": len(combined),
                "records_inserted": submitted_count,
            }
