"""
Research agents for HRP automated hypothesis discovery.

Research agents extend the IngestionJob pattern to provide automated
research capabilities with actor tracking and lineage logging.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from hrp.agents.jobs import IngestionJob
from hrp.api.platform import PlatformAPI
from hrp.data.db import get_db
from hrp.notifications.email import EmailNotifier
from hrp.research.lineage import EventType, log_event


@dataclass
class SignalScanResult:
    """Result of a single feature scan."""

    feature_name: str
    forward_horizon: int  # days
    ic: float
    ic_std: float  # Standard deviation across rolling windows
    ic_ir: float  # Information ratio (ic / ic_std)
    sample_size: int
    start_date: date
    end_date: date
    is_combination: bool = False  # True if two-factor combination
    combination_method: str | None = None  # "additive" or "subtractive"


@dataclass
class SignalScanReport:
    """Complete scan report."""

    scan_date: date
    total_features_scanned: int
    promising_signals: list[SignalScanResult]
    hypotheses_created: list[str]  # hypothesis_ids
    mlflow_run_id: str
    duration_seconds: float


class ResearchAgent(IngestionJob, ABC):
    """
    Base class for research agents (extends IngestionJob pattern).

    Research agents perform automated analysis and can create draft
    hypotheses. They track their actor identity for lineage purposes
    and have access to the PlatformAPI.
    """

    def __init__(
        self,
        job_id: str,
        actor: str,
        dependencies: list[str] | None = None,
        max_retries: int = 2,
    ):
        """
        Initialize a research agent.

        Args:
            job_id: Unique identifier for this job
            actor: Actor identity for lineage (e.g., 'agent:signal-scientist')
            dependencies: List of job IDs that must complete before this job runs
            max_retries: Maximum number of retry attempts
        """
        super().__init__(
            job_id=job_id,
            dependencies=dependencies or [],
            max_retries=max_retries,
        )
        self.actor = actor
        self.api = PlatformAPI()

    @abstractmethod
    def execute(self) -> dict[str, Any]:
        """
        Implement research logic.

        Must be implemented by subclasses.

        Returns:
            Dictionary with execution results
        """
        pass

    def _log_agent_event(
        self,
        event_type: str | EventType,
        details: dict,
        hypothesis_id: str | None = None,
        experiment_id: str | None = None,
    ) -> int:
        """
        Log event to lineage with agent actor.

        Args:
            event_type: Type of event (EventType enum or string)
            details: Event-specific details
            hypothesis_id: Optional associated hypothesis
            experiment_id: Optional associated experiment

        Returns:
            lineage_id of the created event
        """
        # Convert EventType to string if needed
        if isinstance(event_type, EventType):
            event_type = event_type.value

        return log_event(
            event_type=event_type,
            actor=self.actor,
            details=details,
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
        )


class SignalScientist(ResearchAgent):
    """
    Scans feature universe for predictive signals.

    The Signal Scientist performs systematic IC (Information Coefficient)
    analysis across all features to identify those with predictive power
    for forward returns. When promising signals are found, it creates
    draft hypotheses for review.

    Features:
    - Rolling IC calculation for robust signal detection
    - Multi-horizon analysis (5, 10, 20 day returns)
    - Two-factor combination scanning
    - Automatic hypothesis creation for strong signals
    - MLflow logging for reproducibility
    - Email notifications with scan results
    """

    DEFAULT_JOB_ID = "signal_scientist_scan"
    ACTOR = "agent:signal-scientist"

    # IC thresholds
    IC_WEAK = 0.02
    IC_MODERATE = 0.03
    IC_STRONG = 0.05

    # Pre-defined two-factor combinations (theoretically motivated)
    FACTOR_PAIRS = [
        ("momentum_20d", "volatility_60d"),  # Momentum + Low Vol
        ("momentum_20d", "rsi_14d"),  # Momentum + Oversold
        ("returns_252d", "volatility_60d"),  # Annual momentum + Vol
        ("price_to_sma_200d", "rsi_14d"),  # Trend + Mean reversion
        ("volume_ratio", "momentum_20d"),  # Volume confirmation
    ]

    # All available features (39 technical + 5 fundamental)
    ALL_FEATURES = [
        # Returns
        "returns_1d",
        "returns_5d",
        "returns_20d",
        "returns_60d",
        "returns_252d",
        # Momentum
        "momentum_20d",
        "momentum_60d",
        "momentum_252d",
        # Volatility
        "volatility_20d",
        "volatility_60d",
        # Volume
        "volume_20d",
        "volume_ratio",
        "obv",
        # Oscillators
        "rsi_14d",
        "cci_20d",
        "roc_10d",
        "stoch_k_14d",
        "stoch_d_14d",
        "williams_r_14d",
        "mfi_14d",
        # Trend
        "atr_14d",
        "adx_14d",
        "macd_line",
        "macd_signal",
        "macd_histogram",
        "trend",
        # Moving Averages
        "sma_20d",
        "sma_50d",
        "sma_200d",
        "ema_12d",
        "ema_26d",
        # EMA Signals
        "ema_crossover",
        # Price Ratios
        "price_to_sma_20d",
        "price_to_sma_50d",
        "price_to_sma_200d",
        # Bollinger Bands
        "bb_upper_20d",
        "bb_lower_20d",
        "bb_width_20d",
        # VWAP
        "vwap_20d",
        # Fundamental
        "market_cap",
        "pe_ratio",
        "pb_ratio",
        "dividend_yield",
        "ev_ebitda",
    ]

    def __init__(
        self,
        symbols: list[str] | None = None,
        features: list[str] | None = None,
        forward_horizons: list[int] | None = None,
        lookback_days: int = 756,  # 3 years
        ic_threshold: float = 0.03,
        create_hypotheses: bool = True,
        as_of_date: date | None = None,
    ):
        """
        Initialize the Signal Scientist agent.

        Args:
            symbols: List of symbols to analyze (None = all universe)
            features: List of features to scan (None = all 44 features)
            forward_horizons: Return horizons in days (default: [5, 10, 20])
            lookback_days: Days of history to use for IC calculation
            ic_threshold: Minimum IC to create hypothesis (default: 0.03)
            create_hypotheses: Whether to create draft hypotheses
            as_of_date: Date to run scan as of (default: today)
        """
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=["feature_computation"],  # Requires fresh features
        )
        self.symbols = symbols  # None = all universe
        self.features = features  # None = all features
        self.forward_horizons = forward_horizons or [5, 10, 20]
        self.lookback_days = lookback_days
        self.ic_threshold = ic_threshold
        self.create_hypotheses = create_hypotheses
        self.as_of_date = as_of_date or date.today()

    def execute(self) -> dict[str, Any]:
        """
        Run signal scan.

        Scans all features for IC against forward returns, creates
        hypotheses for promising signals, and sends notification.

        Returns:
            Dictionary with scan results
        """
        start_time = time.time()

        # 1. Get universe symbols
        symbols = self.symbols or self._get_universe_symbols()

        if not symbols:
            logger.warning("No symbols to scan")
            return {
                "scan_date": self.as_of_date.isoformat(),
                "features_scanned": 0,
                "signals_found": 0,
                "hypotheses_created": [],
                "error": "No symbols in universe",
            }

        # 2. Get features to scan
        features = self.features or self.ALL_FEATURES

        # 3. Load price data for forward returns
        prices = self._load_prices(symbols)

        # 4. Scan each single feature
        results: list[SignalScanResult] = []
        for feature in features:
            for horizon in self.forward_horizons:
                try:
                    result = self._scan_feature(feature, horizon, symbols)
                    if result and abs(result.ic) >= self.IC_WEAK:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to scan {feature}/{horizon}d: {e}")

        # 5. Scan two-factor combinations
        for feature_a, feature_b in self.FACTOR_PAIRS:
            # Check if both features are in our scan list
            if self.features and (
                feature_a not in self.features or feature_b not in self.features
            ):
                continue

            for method in ["additive", "subtractive"]:
                for horizon in self.forward_horizons:
                    try:
                        result = self._scan_combination(
                            feature_a, feature_b, method, horizon, symbols
                        )
                        if result and abs(result.ic) >= self.IC_WEAK:
                            # Only keep if combo beats both individual features
                            ic_a = self._get_single_feature_ic(feature_a, horizon, results)
                            ic_b = self._get_single_feature_ic(feature_b, horizon, results)
                            if abs(result.ic) > max(abs(ic_a), abs(ic_b)) + 0.005:
                                results.append(result)
                    except Exception as e:
                        logger.warning(
                            f"Failed to scan {feature_a}+{feature_b}/{method}: {e}"
                        )

        # 6. Log to MLflow
        mlflow_run_id = self._log_to_mlflow(results)

        # 7. Create hypotheses for promising signals
        hypotheses_created = []
        if self.create_hypotheses:
            promising = [
                r
                for r in results
                if abs(r.ic) >= self.ic_threshold and r.ic_ir >= 0.3
            ]
            for signal in promising:
                try:
                    hyp_id = self._create_hypothesis(signal)
                    if hyp_id:
                        hypotheses_created.append(hyp_id)
                except Exception as e:
                    logger.error(f"Failed to create hypothesis for {signal.feature_name}: {e}")

        # 8. Log agent completion event
        self._log_agent_event(
            event_type=EventType.AGENT_RUN_COMPLETE,
            details={
                "features_scanned": len(features),
                "combinations_scanned": len(self.FACTOR_PAIRS) * 2,
                "signals_found": len(results),
                "hypotheses_created": hypotheses_created,
                "mlflow_run_id": mlflow_run_id,
            },
        )

        duration = time.time() - start_time

        # 9. Send email notification
        self._send_email_notification(results, hypotheses_created, mlflow_run_id, duration)

        return {
            "scan_date": self.as_of_date.isoformat(),
            "features_scanned": len(features),
            "combinations_scanned": len(self.FACTOR_PAIRS) * 2,
            "signals_found": len(results),
            "promising_signals": len(
                [r for r in results if abs(r.ic) >= self.ic_threshold]
            ),
            "hypotheses_created": hypotheses_created,
            "mlflow_run_id": mlflow_run_id,
            "duration_seconds": duration,
        }

    def _get_universe_symbols(self) -> list[str]:
        """Get symbols from the current universe."""
        try:
            return self.api.get_universe(self.as_of_date)
        except Exception as e:
            logger.warning(f"Failed to get universe: {e}")
            # Fallback to symbols with features
            db = get_db()
            result = db.fetchall(
                """
                SELECT DISTINCT symbol
                FROM features
                WHERE date >= ?
                ORDER BY symbol
                """,
                (self.as_of_date - timedelta(days=30),),
            )
            return [row[0] for row in result]

    def _load_prices(self, symbols: list[str]) -> pd.DataFrame:
        """Load price data for forward return calculation."""
        start = self.as_of_date - timedelta(days=self.lookback_days)

        try:
            return self.api.get_prices(symbols, start, self.as_of_date)
        except Exception as e:
            logger.warning(f"Failed to load prices via API: {e}")
            # Direct query fallback
            db = get_db()
            symbols_str = ",".join(f"'{s}'" for s in symbols)
            df = db.fetchdf(
                f"""
                SELECT symbol, date, adj_close as close
                FROM prices
                WHERE symbol IN ({symbols_str})
                  AND date >= ?
                  AND date <= ?
                ORDER BY symbol, date
                """,
                (start, self.as_of_date),
            )
            return df

    def _calculate_rolling_ic(
        self,
        feature_values: np.ndarray,
        forward_returns: np.ndarray,
        window_size: int = 60,
    ) -> dict[str, float]:
        """
        Calculate rolling IC (Information Coefficient) using Spearman correlation.

        Args:
            feature_values: Array of feature values
            forward_returns: Array of forward returns
            window_size: Rolling window size in observations

        Returns:
            Dictionary with mean_ic, ic_std, ic_ir, sample_size
        """
        if len(feature_values) < window_size:
            return {
                "mean_ic": 0.0,
                "ic_std": 1.0,
                "ic_ir": 0.0,
                "sample_size": len(feature_values),
            }

        rolling_ics = []

        for i in range(window_size, len(feature_values)):
            window_features = feature_values[i - window_size : i]
            window_returns = forward_returns[i - window_size : i]

            # Skip if insufficient valid data
            valid_mask = ~(np.isnan(window_features) | np.isnan(window_returns))
            if valid_mask.sum() < window_size * 0.5:  # Require at least 50% valid
                continue

            # Calculate Spearman correlation (IC)
            try:
                ic, _ = stats.spearmanr(
                    window_features[valid_mask], window_returns[valid_mask]
                )
                if not np.isnan(ic):
                    rolling_ics.append(ic)
            except Exception:
                continue

        if not rolling_ics:
            return {
                "mean_ic": 0.0,
                "ic_std": 1.0,
                "ic_ir": 0.0,
                "sample_size": len(feature_values),
            }

        mean_ic = np.mean(rolling_ics)
        ic_std = np.std(rolling_ics) if len(rolling_ics) > 1 else 1.0

        # Avoid division by zero
        if ic_std < 1e-8:
            ic_std = 1e-8

        ic_ir = mean_ic / ic_std

        return {
            "mean_ic": float(mean_ic),
            "ic_std": float(ic_std),
            "ic_ir": float(ic_ir),
            "sample_size": len(feature_values),
        }

    def _scan_feature(
        self,
        feature: str,
        horizon: int,
        symbols: list[str],
    ) -> SignalScanResult | None:
        """
        Calculate IC for a single feature/horizon combination.

        Args:
            feature: Feature name to scan
            horizon: Forward return horizon in days
            symbols: Symbols to include in scan

        Returns:
            SignalScanResult or None if insufficient data
        """
        db = get_db()
        start_date = self.as_of_date - timedelta(days=self.lookback_days)

        # Get feature values
        symbols_str = ",".join(f"'{s}'" for s in symbols)
        feature_df = db.fetchdf(
            f"""
            SELECT symbol, date, value
            FROM features
            WHERE feature_name = ?
              AND symbol IN ({symbols_str})
              AND date >= ?
              AND date <= ?
            ORDER BY symbol, date
            """,
            (feature, start_date, self.as_of_date),
        )

        if feature_df.empty:
            return None

        # Get price data for forward returns
        price_df = db.fetchdf(
            f"""
            SELECT symbol, date, adj_close
            FROM prices
            WHERE symbol IN ({symbols_str})
              AND date >= ?
              AND date <= ?
            ORDER BY symbol, date
            """,
            (start_date, self.as_of_date + timedelta(days=horizon + 10)),
        )

        if price_df.empty:
            return None

        # Calculate forward returns per symbol
        all_features = []
        all_returns = []

        for symbol in symbols:
            sym_features = feature_df[feature_df["symbol"] == symbol].copy()
            sym_prices = price_df[price_df["symbol"] == symbol].copy()

            if sym_features.empty or sym_prices.empty:
                continue

            sym_features = sym_features.sort_values("date")
            sym_prices = sym_prices.sort_values("date")

            # Calculate forward returns
            sym_prices["forward_return"] = (
                sym_prices["adj_close"].shift(-horizon) / sym_prices["adj_close"] - 1
            )

            # Merge features with forward returns
            merged = pd.merge(
                sym_features,
                sym_prices[["date", "forward_return"]],
                on="date",
                how="inner",
            )

            if not merged.empty:
                all_features.extend(merged["value"].tolist())
                all_returns.extend(merged["forward_return"].tolist())

        if len(all_features) < 100:  # Minimum sample size
            return None

        # Calculate rolling IC
        ic_result = self._calculate_rolling_ic(
            np.array(all_features),
            np.array(all_returns),
            window_size=60,
        )

        return SignalScanResult(
            feature_name=feature,
            forward_horizon=horizon,
            ic=ic_result["mean_ic"],
            ic_std=ic_result["ic_std"],
            ic_ir=ic_result["ic_ir"],
            sample_size=int(ic_result["sample_size"]),
            start_date=start_date,
            end_date=self.as_of_date,
        )

    def _scan_combination(
        self,
        feature_a: str,
        feature_b: str,
        method: str,
        horizon: int,
        symbols: list[str],
    ) -> SignalScanResult | None:
        """
        Scan a two-factor combination for IC.

        Args:
            feature_a: First feature name
            feature_b: Second feature name
            method: Combination method ('additive' or 'subtractive')
            horizon: Forward return horizon in days
            symbols: Symbols to include

        Returns:
            SignalScanResult or None if insufficient data
        """
        db = get_db()
        start_date = self.as_of_date - timedelta(days=self.lookback_days)
        symbols_str = ",".join(f"'{s}'" for s in symbols)

        # Get both features
        features_df = db.fetchdf(
            f"""
            SELECT symbol, date, feature_name, value
            FROM features
            WHERE feature_name IN (?, ?)
              AND symbol IN ({symbols_str})
              AND date >= ?
              AND date <= ?
            ORDER BY symbol, date, feature_name
            """,
            (feature_a, feature_b, start_date, self.as_of_date),
        )

        if features_df.empty:
            return None

        # Pivot to wide format
        features_wide = features_df.pivot_table(
            index=["symbol", "date"], columns="feature_name", values="value"
        ).reset_index()

        if feature_a not in features_wide.columns or feature_b not in features_wide.columns:
            return None

        # Rank transform within each date
        all_features = []
        all_returns = []

        for date_val in features_wide["date"].unique():
            day_data = features_wide[features_wide["date"] == date_val].copy()

            if len(day_data) < 5:  # Need enough stocks for ranking
                continue

            # Rank each feature (higher rank = higher value)
            day_data["rank_a"] = day_data[feature_a].rank(pct=True)
            day_data["rank_b"] = day_data[feature_b].rank(pct=True)

            # Combine ranks
            if method == "additive":
                day_data["composite"] = day_data["rank_a"] + day_data["rank_b"]
            else:  # subtractive
                day_data["composite"] = day_data["rank_a"] - day_data["rank_b"]

            # Get forward returns for these symbols on this date
            price_df = db.fetchdf(
                f"""
                SELECT symbol, adj_close
                FROM prices
                WHERE symbol IN ({symbols_str})
                  AND date = ?
                """,
                (date_val,),
            )

            forward_price_df = db.fetchdf(
                f"""
                SELECT symbol, adj_close as forward_close
                FROM prices
                WHERE symbol IN ({symbols_str})
                  AND date >= ?
                ORDER BY date
                LIMIT 1
                """,
                (date_val + timedelta(days=horizon),),
            )

            if price_df.empty or forward_price_df.empty:
                continue

            merged = pd.merge(
                day_data[["symbol", "composite"]],
                price_df,
                on="symbol",
            )
            merged = pd.merge(merged, forward_price_df, on="symbol")
            merged["forward_return"] = merged["forward_close"] / merged["adj_close"] - 1

            all_features.extend(merged["composite"].tolist())
            all_returns.extend(merged["forward_return"].tolist())

        if len(all_features) < 100:
            return None

        ic_result = self._calculate_rolling_ic(
            np.array(all_features),
            np.array(all_returns),
            window_size=60,
        )

        combo_name = f"{feature_a} {'+' if method == 'additive' else '-'} {feature_b}"

        return SignalScanResult(
            feature_name=combo_name,
            forward_horizon=horizon,
            ic=ic_result["mean_ic"],
            ic_std=ic_result["ic_std"],
            ic_ir=ic_result["ic_ir"],
            sample_size=int(ic_result["sample_size"]),
            start_date=start_date,
            end_date=self.as_of_date,
            is_combination=True,
            combination_method=method,
        )

    def _get_single_feature_ic(
        self, feature: str, horizon: int, results: list[SignalScanResult]
    ) -> float:
        """Get IC for a single feature from results list."""
        for r in results:
            if r.feature_name == feature and r.forward_horizon == horizon:
                return r.ic
        return 0.0

    def _create_hypothesis(self, signal: SignalScanResult) -> str | None:
        """
        Create draft hypothesis from promising signal.

        Args:
            signal: SignalScanResult with promising IC

        Returns:
            hypothesis_id or None if creation failed
        """
        direction = "positively" if signal.ic > 0 else "negatively"
        horizon_name = {5: "weekly", 10: "bi-weekly", 20: "monthly"}.get(
            signal.forward_horizon, f"{signal.forward_horizon}d"
        )

        title = f"{signal.feature_name} predicts {horizon_name} returns"

        thesis = (
            f"The {signal.feature_name} feature is {direction} correlated with "
            f"{signal.forward_horizon}-day forward returns (IC={signal.ic:.4f}). "
            f"This signal may capture a persistent market inefficiency."
        )

        prediction = (
            f"A long-short strategy based on {signal.feature_name} will achieve "
            f"IC > {self.IC_MODERATE:.2f} out-of-sample with stability score < 1.0."
        )

        falsification = (
            f"The signal fails if: (1) out-of-sample IC < {self.IC_WEAK:.2f}, "
            f"(2) IC is unstable across time periods (stability > 1.5), or "
            f"(3) the signal decays within 6 months of discovery."
        )

        return self.api.create_hypothesis(
            title=title,
            thesis=thesis,
            prediction=prediction,
            falsification=falsification,
            actor=self.actor,
        )

    def _log_to_mlflow(self, results: list[SignalScanResult]) -> str:
        """
        Log scan results to MLflow.

        Args:
            results: List of SignalScanResult objects

        Returns:
            MLflow run ID
        """
        from hrp.research.mlflow_utils import setup_mlflow

        setup_mlflow()

        with mlflow.start_run(run_name=f"signal_scan_{self.as_of_date}") as run:
            # Log parameters
            mlflow.log_params(
                {
                    "scan_date": self.as_of_date.isoformat(),
                    "features_count": len(self.features or self.ALL_FEATURES),
                    "forward_horizons": str(self.forward_horizons),
                    "lookback_days": self.lookback_days,
                    "ic_threshold": self.ic_threshold,
                    "symbols_count": len(self.symbols or []),
                }
            )

            # Log summary metrics
            mlflow.log_metrics(
                {
                    "signals_above_weak": len(
                        [r for r in results if abs(r.ic) >= self.IC_WEAK]
                    ),
                    "signals_above_moderate": len(
                        [r for r in results if abs(r.ic) >= self.IC_MODERATE]
                    ),
                    "signals_above_strong": len(
                        [r for r in results if abs(r.ic) >= self.IC_STRONG]
                    ),
                    "total_signals": len(results),
                }
            )

            # Log individual signal metrics
            for i, result in enumerate(sorted(results, key=lambda x: abs(x.ic), reverse=True)[:20]):
                mlflow.log_metrics(
                    {
                        f"signal_{i}_ic": result.ic,
                        f"signal_{i}_ir": result.ic_ir,
                    }
                )

            return str(run.info.run_id)

    def _send_email_notification(
        self,
        results: list[SignalScanResult],
        hypotheses_created: list[str],
        mlflow_run_id: str,
        duration: float,
    ) -> None:
        """
        Send email notification with scan results.

        Args:
            results: List of SignalScanResult objects
            hypotheses_created: List of created hypothesis IDs
            mlflow_run_id: MLflow run ID
            duration: Scan duration in seconds
        """
        try:
            notifier = EmailNotifier()

            # Sort by absolute IC
            top_signals = sorted(results, key=lambda x: abs(x.ic), reverse=True)[:10]

            summary_data = {
                "scan_date": self.as_of_date.isoformat(),
                "duration_seconds": f"{duration:.1f}",
                "features_scanned": len(self.features or self.ALL_FEATURES),
                "signals_found": len(results),
                "signals_above_threshold": len(
                    [r for r in results if abs(r.ic) >= self.ic_threshold]
                ),
                "hypotheses_created": len(hypotheses_created),
                "mlflow_run_id": mlflow_run_id,
            }

            # Add top signals
            for i, signal in enumerate(top_signals[:5]):
                summary_data[f"top_{i+1}_signal"] = (
                    f"{signal.feature_name} ({signal.forward_horizon}d): "
                    f"IC={signal.ic:.4f}, IR={signal.ic_ir:.2f}"
                )

            subject = f"[HRP] Signal Scan Complete - {len(top_signals)} signals found"

            notifier.send_summary_email(
                subject=subject,
                summary_data=summary_data,
            )

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")


@dataclass
class ModelExperimentResult:
    """Result of a single model experiment."""

    hypothesis_id: str
    model_type: str
    features: list[str]
    model_params: dict[str, Any]
    mean_ic: float
    ic_std: float
    stability_score: float
    is_stable: bool
    n_folds: int
    fold_results: list[dict]
    mlflow_run_id: str
    training_time_seconds: float


@dataclass
class MLScientistReport:
    """Complete ML Scientist run report."""

    run_date: date
    hypotheses_processed: int
    hypotheses_validated: int
    hypotheses_rejected: int
    total_trials: int
    total_training_time_seconds: float
    best_models: list[ModelExperimentResult]
    mlflow_experiment_id: str


class MLScientist(ResearchAgent):
    """
    Trains and validates ML models for hypotheses in testing status.

    The ML Scientist takes hypotheses created by the Signal Scientist
    (or manually) and systematically trains ML models using walk-forward
    validation. It identifies the best model/feature combinations and
    updates hypothesis status based on statistical rigor.

    Features:
    - Multi-model type testing (ridge, lasso, lightgbm)
    - Walk-forward validation with stability scoring
    - Feature combination search
    - Hyperparameter optimization with trial budget
    - Automatic hypothesis status updates
    - MLflow experiment logging
    - Email notifications with results
    """

    DEFAULT_JOB_ID = "ml_scientist_training"
    ACTOR = "agent:ml-scientist"

    # Default model types to test
    DEFAULT_MODEL_TYPES = ["ridge", "lasso", "lightgbm"]

    # Validation thresholds
    IC_THRESHOLD_VALIDATED = 0.03
    IC_THRESHOLD_PROMISING = 0.02
    STABILITY_THRESHOLD_VALIDATED = 1.0
    STABILITY_THRESHOLD_PROMISING = 1.5

    # Trial limits
    MAX_TRIALS_PER_HYPOTHESIS = 50
    MAX_FEATURE_COMBINATIONS = 10
    MAX_FEATURES_PER_MODEL = 3

    # Hyperparameter grids
    HYPERPARAMETER_GRIDS = {
        "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
        "lasso": {"alpha": [0.001, 0.01, 0.1, 1.0]},
        "elasticnet": {
            "alpha": [0.01, 0.1, 1.0],
            "l1_ratio": [0.2, 0.5, 0.8],
        },
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
        },
        "lightgbm": {
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200],
        },
        "xgboost": {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200],
        },
    }

    # Complementary features for combination search
    COMPLEMENTARY_FEATURES = {
        "momentum_20d": ["volatility_60d", "rsi_14d", "volume_ratio"],
        "momentum_60d": ["volatility_60d", "returns_252d", "adx_14d"],
        "momentum_252d": ["volatility_60d", "price_to_sma_200d"],
        "volatility_60d": ["momentum_20d", "returns_252d", "atr_14d"],
        "volatility_20d": ["momentum_20d", "rsi_14d"],
        "rsi_14d": ["momentum_20d", "price_to_sma_200d", "cci_20d"],
        "returns_252d": ["volatility_60d", "momentum_20d"],
        "price_to_sma_200d": ["rsi_14d", "momentum_20d", "trend"],
        "volume_ratio": ["momentum_20d", "obv"],
    }

    # All features (reuse from SignalScientist)
    ALL_FEATURES = SignalScientist.ALL_FEATURES

    def __init__(
        self,
        hypothesis_ids: list[str] | None = None,
        model_types: list[str] | None = None,
        target: str = "returns_20d",
        n_folds: int = 5,
        window_type: str = "expanding",
        start_date: date | None = None,
        end_date: date | None = None,
        symbols: list[str] | None = None,
        max_trials_per_hypothesis: int | None = None,
        skip_hyperparameter_search: bool = False,
        parallel_folds: bool = True,
    ):
        """
        Initialize the ML Scientist agent.

        Args:
            hypothesis_ids: Specific hypotheses to process (None = all in 'testing')
            model_types: Models to test (default: ridge, lasso, lightgbm)
            target: Target variable name (default: returns_20d)
            n_folds: Number of walk-forward folds (default: 5)
            window_type: 'expanding' or 'rolling' (default: expanding)
            start_date: Start of training date range
            end_date: End of training date range
            symbols: Symbols to use (None = all universe)
            max_trials_per_hypothesis: Max trials per hypothesis (default: 50)
            skip_hyperparameter_search: Use default params only
            parallel_folds: Run folds in parallel (default: True)
        """
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=["signal_scientist_scan"],
        )
        self.hypothesis_ids = hypothesis_ids
        self.model_types = model_types or self.DEFAULT_MODEL_TYPES
        self.target = target
        self.n_folds = n_folds
        self.window_type = window_type
        self.start_date = start_date or date(2015, 1, 1)
        self.end_date = end_date or date.today()
        self.symbols = symbols
        self.max_trials = max_trials_per_hypothesis or self.MAX_TRIALS_PER_HYPOTHESIS
        self.skip_hyperparameter_search = skip_hyperparameter_search
        self.parallel_folds = parallel_folds

    def execute(self) -> dict[str, Any]:
        """
        Run ML experimentation on hypotheses in testing status.

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        # 1. Get hypotheses to process
        hypotheses = self._get_hypotheses_to_process()
        if not hypotheses:
            return {
                "status": "no_hypotheses",
                "message": "No hypotheses in testing status",
            }

        # 2. Get universe symbols
        symbols = self.symbols or self._get_universe_symbols()

        if not symbols:
            return {
                "status": "no_symbols",
                "message": "No symbols in universe",
            }

        # 3. Process each hypothesis
        all_results: list[ModelExperimentResult] = []
        validated_count = 0
        rejected_count = 0
        total_trials = 0

        for hypothesis in hypotheses:
            try:
                hyp_results = self._process_hypothesis(hypothesis, symbols)
                all_results.extend(hyp_results)

                # Update hypothesis status based on best result
                if hyp_results:
                    best = max(hyp_results, key=lambda r: self._calculate_model_score(r))
                    status = self._determine_status(best)
                    self._update_hypothesis(hypothesis, best, status)

                    if status == "validated":
                        validated_count += 1
                    elif status == "rejected":
                        rejected_count += 1

                    total_trials += len(hyp_results)
            except Exception as e:
                logger.error(f"Failed to process hypothesis {hypothesis.get('id')}: {e}")

        # 4. Log completion event
        duration = time.time() - start_time
        self._log_agent_event(
            event_type=EventType.AGENT_RUN_COMPLETE,
            details={
                "hypotheses_processed": len(hypotheses),
                "hypotheses_validated": validated_count,
                "hypotheses_rejected": rejected_count,
                "total_trials": total_trials,
                "duration_seconds": duration,
            },
        )

        # 5. Send email notification
        self._send_ml_email_notification(
            hypotheses, all_results, validated_count, rejected_count, duration
        )

        return {
            "run_date": date.today().isoformat(),
            "hypotheses_processed": len(hypotheses),
            "hypotheses_validated": validated_count,
            "hypotheses_rejected": rejected_count,
            "total_trials": total_trials,
            "duration_seconds": duration,
        }

    def _get_hypotheses_to_process(self) -> list[dict]:
        """Get hypotheses in testing status."""
        if self.hypothesis_ids:
            hypotheses = []
            for hid in self.hypothesis_ids:
                hyp = self.api.get_hypothesis(hid)
                if hyp:
                    hypotheses.append(hyp)
            return hypotheses
        return self.api.list_hypotheses(status="testing")

    def _get_universe_symbols(self) -> list[str]:
        """Get symbols from the current universe."""
        try:
            return self.api.get_universe(date.today())
        except Exception as e:
            logger.warning(f"Failed to get universe: {e}")
            # Fallback to symbols with features
            db = get_db()
            result = db.fetchall(
                """
                SELECT DISTINCT symbol
                FROM features
                WHERE date >= ?
                ORDER BY symbol
                """,
                (date.today() - timedelta(days=30),),
            )
            return [row[0] for row in result]

    def _process_hypothesis(
        self,
        hypothesis: dict,
        symbols: list[str],
    ) -> list[ModelExperimentResult]:
        """Process a single hypothesis through ML pipeline."""
        from hrp.risk.overfitting import HyperparameterTrialCounter

        results = []
        hypothesis_id = hypothesis.get("id", "unknown")

        # Initialize trial counter
        counter = HyperparameterTrialCounter(
            hypothesis_id=hypothesis_id,
            max_trials=self.max_trials,
        )

        # Extract base features from hypothesis
        base_features = self._extract_features_from_hypothesis(hypothesis)

        # Generate feature combinations
        feature_combos = self._generate_feature_combinations(base_features)

        # Test each model type
        for model_type in self.model_types:
            if counter.remaining_trials <= 0:
                logger.info(f"Trial budget exhausted for {hypothesis_id}")
                break

            # Test each feature combination
            for features in feature_combos:
                if counter.remaining_trials <= 0:
                    break

                # Get hyperparameter grid
                if self.skip_hyperparameter_search:
                    param_grid = [{}]  # Default params only
                else:
                    param_grid = self._get_param_grid(model_type)

                # Test each hyperparameter combination
                for params in param_grid:
                    if counter.remaining_trials <= 0:
                        break

                    try:
                        result = self._run_experiment(
                            hypothesis_id=hypothesis_id,
                            model_type=model_type,
                            features=features,
                            model_params=params,
                            symbols=symbols,
                        )

                        if result:
                            results.append(result)
                            try:
                                counter.log_trial(
                                    model_type=model_type,
                                    hyperparameters={"features": features, **params},
                                    metric_name="mean_ic",
                                    metric_value=result.mean_ic,
                                )
                            except Exception as e:
                                logger.warning(f"Failed to log trial: {e}")

                            # Early stopping if we find excellent result
                            if result.mean_ic > 0.05 and result.is_stable:
                                logger.info(
                                    f"Excellent result found for {hypothesis_id}, stopping early"
                                )
                                return results
                    except Exception as e:
                        logger.warning(
                            f"Experiment failed: {model_type}/{features}: {e}"
                        )

        return results

    def _run_experiment(
        self,
        hypothesis_id: str,
        model_type: str,
        features: list[str],
        model_params: dict,
        symbols: list[str],
    ) -> ModelExperimentResult | None:
        """Run a single walk-forward validation experiment."""
        from hrp.ml import WalkForwardConfig, walk_forward_validate

        try:
            start_time = time.time()

            config = WalkForwardConfig(
                model_type=model_type,
                target=self.target,
                features=features,
                start_date=self.start_date,
                end_date=self.end_date,
                n_folds=self.n_folds,
                window_type=self.window_type,
                n_jobs=-1 if self.parallel_folds else 1,
                hyperparameters=model_params,
            )

            result = walk_forward_validate(
                config=config,
                symbols=symbols,
                log_to_mlflow=True,
            )

            training_time = time.time() - start_time

            # Extract fold IC values
            fold_results = []
            for fold in result.fold_results:
                fold_results.append(fold.metrics)

            return ModelExperimentResult(
                hypothesis_id=hypothesis_id,
                model_type=model_type,
                features=features,
                model_params=model_params,
                mean_ic=result.mean_ic,
                ic_std=result.aggregate_metrics.get("std_ic", 0.0),
                stability_score=result.stability_score,
                is_stable=result.is_stable,
                n_folds=len(result.fold_results),
                fold_results=fold_results,
                mlflow_run_id="",  # TODO: capture from walk_forward_validate
                training_time_seconds=training_time,
            )

        except Exception as e:
            logger.error(f"Experiment failed: {model_type}/{features}: {e}")
            return None

    def _extract_features_from_hypothesis(self, hypothesis: dict) -> list[str]:
        """Extract feature names from hypothesis thesis/metadata."""
        # Check metadata first
        metadata = hypothesis.get("metadata", {})
        if isinstance(metadata, dict) and "features" in metadata:
            return metadata["features"]

        # Parse from thesis text
        thesis = hypothesis.get("thesis", "")
        features = []
        for feature in self.ALL_FEATURES:
            if feature in thesis.lower():
                features.append(feature)

        return features if features else ["momentum_20d"]  # Default fallback

    def _generate_feature_combinations(self, base_features: list[str]) -> list[list[str]]:
        """Generate feature combinations to test."""
        combinations = [base_features]  # Start with base

        # Add complementary features
        for base in base_features:
            complements = self.COMPLEMENTARY_FEATURES.get(base, [])
            for comp in complements[:2]:  # Limit to top 2 complements
                combo = base_features + [comp]
                if len(combo) <= self.MAX_FEATURES_PER_MODEL:
                    combinations.append(combo)

        # Deduplicate and limit
        seen = set()
        unique = []
        for combo in combinations:
            key = tuple(sorted(combo))
            if key not in seen:
                seen.add(key)
                unique.append(combo)

        return unique[: self.MAX_FEATURE_COMBINATIONS]

    def _get_param_grid(self, model_type: str) -> list[dict]:
        """Get hyperparameter combinations for model type."""
        from sklearn.model_selection import ParameterGrid

        grid = self.HYPERPARAMETER_GRIDS.get(model_type, {})
        if not grid:
            return [{}]

        return list(ParameterGrid(grid))[:10]  # Limit combinations

    def _calculate_model_score(self, result: ModelExperimentResult) -> float:
        """Calculate composite score for model ranking."""
        ic_score = result.mean_ic
        stability_penalty = 1 / max(result.stability_score, 0.1)

        # Bonus if all folds have positive IC
        all_positive = all(f.get("ic", 0) > 0 for f in result.fold_results)
        consistency_bonus = 1.2 if all_positive else 1.0

        return ic_score * stability_penalty * consistency_bonus

    def _determine_status(self, result: ModelExperimentResult) -> str:
        """Determine hypothesis status based on best model result."""
        if (
            result.mean_ic >= self.IC_THRESHOLD_VALIDATED
            and result.stability_score <= self.STABILITY_THRESHOLD_VALIDATED
            and result.is_stable
        ):
            return "validated"
        elif (
            result.mean_ic >= self.IC_THRESHOLD_PROMISING
            and result.stability_score <= self.STABILITY_THRESHOLD_PROMISING
        ):
            return "testing"  # Keep in testing for further work
        else:
            return "rejected"

    def _update_hypothesis(
        self,
        hypothesis: dict,
        best_result: ModelExperimentResult,
        status: str,
    ) -> None:
        """Update hypothesis with ML results."""
        hypothesis_id = hypothesis.get("id", "unknown")

        # Build outcome string with ML results
        outcome = (
            f"ML Scientist: {best_result.model_type} model with "
            f"features {best_result.features}, IC={best_result.mean_ic:.4f}, "
            f"stability={best_result.stability_score:.2f}"
        )

        try:
            self.api.update_hypothesis(
                hypothesis_id=hypothesis_id,
                status=status,
                outcome=outcome,
                actor=self.ACTOR,
            )
            logger.info(f"Updated hypothesis {hypothesis_id} to status={status}")
        except Exception as e:
            logger.error(f"Failed to update hypothesis {hypothesis_id}: {e}")

    def _send_ml_email_notification(
        self,
        hypotheses: list[dict],
        results: list[ModelExperimentResult],
        validated_count: int,
        rejected_count: int,
        duration: float,
    ) -> None:
        """Send email notification with ML results."""
        try:
            notifier = EmailNotifier()

            # Group best results by hypothesis
            best_by_hypothesis = {}
            for result in results:
                hid = result.hypothesis_id
                if hid not in best_by_hypothesis:
                    best_by_hypothesis[hid] = result
                elif self._calculate_model_score(result) > self._calculate_model_score(
                    best_by_hypothesis[hid]
                ):
                    best_by_hypothesis[hid] = result

            summary_data = {
                "run_date": date.today().isoformat(),
                "duration_seconds": f"{duration:.1f}",
                "hypotheses_processed": len(hypotheses),
                "hypotheses_validated": validated_count,
                "hypotheses_rejected": rejected_count,
                "total_experiments": len(results),
            }

            # Add top results
            sorted_results = sorted(
                best_by_hypothesis.values(),
                key=lambda r: self._calculate_model_score(r),
                reverse=True,
            )
            for i, result in enumerate(sorted_results[:5]):
                summary_data[f"top_{i+1}_model"] = (
                    f"{result.hypothesis_id}: {result.model_type} "
                    f"IC={result.mean_ic:.4f}, stability={result.stability_score:.2f}"
                )

            subject = (
                f"[HRP] ML Scientist Complete - "
                f"{validated_count} validated, {rejected_count} rejected"
            )

            notifier.send_summary_email(
                subject=subject,
                summary_data=summary_data,
            )

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
