# Plan: ML Scientist Agent Implementation

## Overview

Build the second research agent for HRP - the ML Scientist - responsible for automated model training, walk-forward validation, hyperparameter optimization, and strategy generation from discovered signals.

---

## Agent Specification: ML Scientist

### Identity

| Attribute | Value |
|-----------|-------|
| **Name** | ML Scientist |
| **Actor ID** | `agent:ml-scientist` |
| **Role** | Model training, walk-forward validation, hyperparameter optimization |
| **Trigger** | Scheduled (after Signal Scientist) + on-demand via MCP |
| **Upstream** | Signal Scientist (creates hypotheses with promising signals) |
| **Downstream** | ML Quality Sentinel (audits), Quant Developer (backtests) |

### Purpose

Take hypotheses in `testing` status (created by Signal Scientist or manually) and systematically train ML models using walk-forward validation. Identify the best model/feature combinations and update hypothesis status based on statistical rigor.

---

## Core Capabilities

### 1. Hypothesis Processing Pipeline

**What it does:** Processes hypotheses in `testing` status through the ML experimentation pipeline.

```python
# ML Scientist main loop
for hypothesis in get_hypotheses_in_testing():
    # Extract signal features from hypothesis
    features = parse_features_from_hypothesis(hypothesis)

    # Run walk-forward validation across model types
    results = []
    for model_type in MODEL_TYPES:
        result = walk_forward_validate(
            config=WalkForwardConfig(
                model_type=model_type,
                features=features,
                target='returns_20d',
                n_folds=5,
                window_type='expanding',
                n_jobs=-1,
            ),
            symbols=universe,
            log_to_mlflow=True,
        )
        results.append(result)

        # Track trial count (max 50 per hypothesis)
        trial_counter.log_trial(model_type, features, result.mean_ic)

    # Select best model and update hypothesis
    best = select_best_model(results)
    update_hypothesis_with_results(hypothesis, best)
```

**Inputs:**
- Hypotheses in `testing` status from hypothesis registry
- Feature data from feature store
- Price data for forward returns
- Universe symbols (S&P 500 filtered)

**Outputs:**
- Walk-forward validation results logged to MLflow
- Hypothesis status updates (`validated` or `rejected`)
- Best model configuration stored with hypothesis
- Lineage events for audit trail

### 2. Model Type Coverage

Test multiple model types to find best fit for each signal:

| Model Type | Use Case | Hyperparameters |
|------------|----------|-----------------|
| `ridge` | Linear with L2 regularization | `alpha: [0.1, 1.0, 10.0]` |
| `lasso` | Linear with L1 (sparse features) | `alpha: [0.001, 0.01, 0.1]` |
| `elasticnet` | Combined L1/L2 | `alpha, l1_ratio` |
| `random_forest` | Non-linear, feature importance | `n_estimators, max_depth` |
| `lightgbm` | Gradient boosting (fast) | `num_leaves, learning_rate` |
| `xgboost` | Gradient boosting (robust) | `max_depth, eta` |

**Default model set:** `['ridge', 'lasso', 'lightgbm']` (balanced speed/performance)

### 3. Walk-Forward Validation Strategy

**Configuration:**
```python
WalkForwardConfig(
    model_type='ridge',
    target='returns_20d',  # Predict 20-day forward returns
    features=['momentum_20d', 'volatility_60d', 'rsi_14d'],
    start_date=date(2015, 1, 1),
    end_date=date(2025, 12, 31),
    n_folds=5,
    window_type='expanding',  # Growing training window
    feature_selection=True,
    max_features=20,
    n_jobs=-1,  # Parallel fold processing
)
```

**Validation Criteria:**
- **Stability Score** ≤ 1.0 (IC std / mean IC)
- **Mean IC** ≥ 0.02 (weak) or ≥ 0.03 (moderate)
- **All folds positive** IC (no fold reversals)
- **Minimum samples** per fold: 500

### 4. Feature Combination Search

For each hypothesis, test feature combinations:

```python
# Base features from hypothesis
base_features = hypothesis.get_features()  # e.g., ['momentum_20d']

# Generate combinations with complementary features
COMPLEMENTARY_FEATURES = {
    'momentum_20d': ['volatility_60d', 'rsi_14d', 'volume_ratio'],
    'volatility_60d': ['momentum_20d', 'returns_252d', 'atr_14d'],
    'rsi_14d': ['momentum_20d', 'price_to_sma_200d', 'cci_20d'],
}

feature_combos = [
    base_features,  # Single feature
    base_features + [comp] for comp in COMPLEMENTARY_FEATURES.get(base_features[0], [])
]

# Test each combination
for features in feature_combos:
    result = walk_forward_validate(features=features, ...)
```

**Combination limits:**
- Max 3 features per model (avoid overfitting)
- Max 10 combinations per hypothesis
- Skip if trial counter exhausted (50 max)

### 5. Hyperparameter Optimization

**Strategy:** Grid search with early stopping

```python
# For each model type, test key hyperparameters
HYPERPARAMETER_GRIDS = {
    'ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
    'lasso': {'alpha': [0.001, 0.01, 0.1, 1.0]},
    'lightgbm': {
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
    },
}

for params in ParameterGrid(HYPERPARAMETER_GRIDS[model_type]):
    result = walk_forward_validate(
        model_type=model_type,
        model_params=params,
        ...
    )

    # Track with overfitting guard
    trial_counter.log_trial(
        model_type=model_type,
        hyperparameters=params,
        metric_name='mean_ic',
        metric_value=result.mean_ic,
    )

    # Early stopping if we find strong signal
    if result.mean_ic > 0.05 and result.is_stable:
        break
```

### 6. Trial Budget Management

Use existing `HyperparameterTrialCounter` to prevent overfitting:

```python
from hrp.risk.overfitting import HyperparameterTrialCounter

counter = HyperparameterTrialCounter(
    hypothesis_id=hypothesis.id,
    max_trials=50,
)

# Before each experiment
if counter.remaining_trials <= 0:
    logger.warning(f"Trial budget exhausted for {hypothesis.id}")
    break

# After each experiment
counter.log_trial(model_type, params, metric_value)

# Get best configuration
best_trial = counter.get_best_trial()
```

### 7. Model Selection Criteria

Rank models by composite score:

```python
def calculate_model_score(result: WalkForwardResult) -> float:
    """
    Score = IC * (1 / stability_score) * consistency_bonus

    Higher IC is better
    Lower stability_score is better (more consistent)
    Consistency bonus if all folds positive
    """
    ic_score = result.mean_ic
    stability_penalty = 1 / max(result.stability_score, 0.1)

    # Bonus if all folds have same sign IC
    all_positive = all(f.metrics['ic'] > 0 for f in result.fold_results)
    consistency_bonus = 1.2 if all_positive else 1.0

    return ic_score * stability_penalty * consistency_bonus
```

**Selection thresholds:**

| Outcome | Criteria | Action |
|---------|----------|--------|
| **Validated** | IC ≥ 0.03, stability ≤ 1.0, all folds positive | Update hypothesis to `validated` |
| **Promising** | IC ≥ 0.02, stability ≤ 1.5 | Keep in `testing`, flag for review |
| **Rejected** | IC < 0.02 or stability > 2.0 | Update hypothesis to `rejected` |

### 8. Results Storage

Store best model configuration with hypothesis:

```python
# Update hypothesis with ML results
api.update_hypothesis(
    hypothesis_id=hypothesis.id,
    status='validated' if meets_criteria else 'rejected',
    metadata={
        'ml_scientist_results': {
            'best_model_type': best.model_type,
            'best_features': best.features,
            'best_params': best.model_params,
            'mean_ic': best.mean_ic,
            'stability_score': best.stability_score,
            'n_trials': counter.total_trials,
            'mlflow_run_id': best.mlflow_run_id,
            'validated_at': datetime.now().isoformat(),
        }
    },
    actor='agent:ml-scientist',
)
```

### 9. MLflow Experiment Structure

```
Experiments:
├── ml-scientist/
│   ├── HYP-2026-001/
│   │   ├── ridge_momentum_20d_v1
│   │   ├── ridge_momentum_20d_volatility_60d_v1
│   │   ├── lasso_momentum_20d_v1
│   │   └── lightgbm_momentum_20d_v1
│   ├── HYP-2026-002/
│   │   └── ...
│   └── ...
```

**Logged metrics per run:**
- `mean_ic`, `ic_std`, `stability_score`
- `mse`, `mae`, `r2` (per fold)
- `n_samples`, `n_features`
- `train_time_seconds`

**Logged artifacts:**
- `fold_results.json` - Per-fold metrics
- `feature_importance.csv` - If tree-based model
- `model_config.json` - Full configuration

### 10. Email Notification

On completion, send summary:

```
Subject: [HRP] ML Scientist Complete - 3 hypotheses validated

ML Scientist Results
====================
Date: 2026-01-26
Duration: 45m 23s

HYPOTHESES PROCESSED
┌────────────────┬────────────┬──────────┬───────────┬─────────────┐
│ Hypothesis     │ Best Model │ Mean IC  │ Stability │ Status      │
├────────────────┼────────────┼──────────┼───────────┼─────────────┤
│ HYP-2026-001   │ ridge      │ 0.042    │ 0.71      │ ✅ Validated │
│ HYP-2026-002   │ lightgbm   │ 0.038    │ 0.85      │ ✅ Validated │
│ HYP-2026-003   │ ridge      │ 0.018    │ 1.82      │ ❌ Rejected  │
│ HYP-2026-004   │ lasso      │ 0.045    │ 0.62      │ ✅ Validated │
└────────────────┴────────────┴──────────┴───────────┴─────────────┘

COMPUTE SUMMARY
- Total trials: 127 / 200 budget
- Walk-forward runs: 48
- Total training time: 42m 15s

TOP MODEL: HYP-2026-004 (lasso)
- Features: momentum_20d, volatility_60d
- Mean IC: 0.045
- Stability: 0.62
- Ready for: Quant Developer backtest

MLflow Experiment: ml-scientist/2026-01-26

---
HRP ML Scientist | Automated Research Agent
```

---

## Implementation Design

### Class Structure

```python
# In hrp/agents/research_agents.py (extend existing file)

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
import time

from hrp.agents.jobs import JobStatus
from hrp.ml import WalkForwardConfig, WalkForwardResult, walk_forward_validate
from hrp.risk.overfitting import HyperparameterTrialCounter
from hrp.research.lineage import log_event, EventType


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
    Uses walk-forward validation to ensure statistical rigor.
    """

    DEFAULT_JOB_ID = "ml_scientist_training"
    ACTOR = "agent:ml-scientist"

    # Default model types to test
    DEFAULT_MODEL_TYPES = ['ridge', 'lasso', 'lightgbm']

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
        'ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
        'lasso': {'alpha': [0.001, 0.01, 0.1, 1.0]},
        'elasticnet': {
            'alpha': [0.01, 0.1, 1.0],
            'l1_ratio': [0.2, 0.5, 0.8],
        },
        'random_forest': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
        },
        'lightgbm': {
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200],
        },
        'xgboost': {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200],
        },
    }

    # Complementary features for combination search
    COMPLEMENTARY_FEATURES = {
        'momentum_20d': ['volatility_60d', 'rsi_14d', 'volume_ratio'],
        'momentum_60d': ['volatility_60d', 'returns_252d', 'adx_14d'],
        'momentum_252d': ['volatility_60d', 'price_to_sma_200d'],
        'volatility_60d': ['momentum_20d', 'returns_252d', 'atr_14d'],
        'volatility_20d': ['momentum_20d', 'rsi_14d'],
        'rsi_14d': ['momentum_20d', 'price_to_sma_200d', 'cci_20d'],
        'returns_252d': ['volatility_60d', 'momentum_20d'],
        'price_to_sma_200d': ['rsi_14d', 'momentum_20d', 'trend'],
        'volume_ratio': ['momentum_20d', 'obv'],
    }

    def __init__(
        self,
        hypothesis_ids: list[str] | None = None,
        model_types: list[str] | None = None,
        target: str = 'returns_20d',
        n_folds: int = 5,
        window_type: str = 'expanding',
        start_date: date | None = None,
        end_date: date | None = None,
        symbols: list[str] | None = None,
        max_trials_per_hypothesis: int | None = None,
        skip_hyperparameter_search: bool = False,
        parallel_folds: bool = True,
    ):
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=["signal_scientist_scan"],
        )
        self.hypothesis_ids = hypothesis_ids  # None = all in 'testing' status
        self.model_types = model_types or self.DEFAULT_MODEL_TYPES
        self.target = target
        self.n_folds = n_folds
        self.window_type = window_type
        self.start_date = start_date or date(2015, 1, 1)
        self.end_date = end_date or date.today()
        self.symbols = symbols  # None = all universe
        self.max_trials = max_trials_per_hypothesis or self.MAX_TRIALS_PER_HYPOTHESIS
        self.skip_hyperparameter_search = skip_hyperparameter_search
        self.parallel_folds = parallel_folds

    def execute(self) -> dict[str, Any]:
        """Run ML experimentation on hypotheses in testing status."""
        start_time = time.time()

        # 1. Get hypotheses to process
        hypotheses = self._get_hypotheses_to_process()
        if not hypotheses:
            return {"status": "no_hypotheses", "message": "No hypotheses in testing status"}

        # 2. Get universe symbols
        symbols = self.symbols or self._get_universe_symbols()

        # 3. Process each hypothesis
        results: list[ModelExperimentResult] = []
        validated_count = 0
        rejected_count = 0
        total_trials = 0

        for hypothesis in hypotheses:
            hyp_results = self._process_hypothesis(hypothesis, symbols)
            results.extend(hyp_results)

            # Update hypothesis status based on best result
            if hyp_results:
                best = max(hyp_results, key=lambda r: self._calculate_model_score(r))
                status = self._determine_status(best)
                self._update_hypothesis(hypothesis, best, status)

                if status == 'validated':
                    validated_count += 1
                elif status == 'rejected':
                    rejected_count += 1

                total_trials += len(hyp_results)

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
        self._send_email_notification(hypotheses, results, validated_count, rejected_count, duration)

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
            return [self.api.get_hypothesis(hid) for hid in self.hypothesis_ids]
        return self.api.list_hypotheses(status='testing')

    def _process_hypothesis(
        self,
        hypothesis: dict,
        symbols: list[str],
    ) -> list[ModelExperimentResult]:
        """Process a single hypothesis through ML pipeline."""
        results = []
        hypothesis_id = hypothesis['id']

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

                    result = self._run_experiment(
                        hypothesis_id=hypothesis_id,
                        model_type=model_type,
                        features=features,
                        model_params=params,
                        symbols=symbols,
                    )

                    if result:
                        results.append(result)
                        counter.log_trial(
                            model_type=model_type,
                            hyperparameters={'features': features, **params},
                            metric_name='mean_ic',
                            metric_value=result.mean_ic,
                        )

                        # Early stopping if we find excellent result
                        if result.mean_ic > 0.05 and result.is_stable:
                            break

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
                **model_params,
            )

            result = walk_forward_validate(
                config=config,
                symbols=symbols,
                log_to_mlflow=True,
                experiment_name=f"ml-scientist/{hypothesis_id}",
            )

            training_time = time.time() - start_time

            return ModelExperimentResult(
                hypothesis_id=hypothesis_id,
                model_type=model_type,
                features=features,
                model_params=model_params,
                mean_ic=result.mean_ic,
                ic_std=result.ic_std,
                stability_score=result.stability_score,
                is_stable=result.is_stable,
                n_folds=len(result.fold_results),
                fold_results=[f.metrics for f in result.fold_results],
                mlflow_run_id=result.mlflow_run_id,
                training_time_seconds=training_time,
            )

        except Exception as e:
            logger.error(f"Experiment failed: {model_type}/{features}: {e}")
            return None

    def _calculate_model_score(self, result: ModelExperimentResult) -> float:
        """Calculate composite score for model ranking."""
        ic_score = result.mean_ic
        stability_penalty = 1 / max(result.stability_score, 0.1)

        # Bonus if all folds have positive IC
        all_positive = all(f.get('ic', 0) > 0 for f in result.fold_results)
        consistency_bonus = 1.2 if all_positive else 1.0

        return ic_score * stability_penalty * consistency_bonus

    def _determine_status(self, result: ModelExperimentResult) -> str:
        """Determine hypothesis status based on best model result."""
        if (result.mean_ic >= self.IC_THRESHOLD_VALIDATED and
            result.stability_score <= self.STABILITY_THRESHOLD_VALIDATED and
            result.is_stable):
            return 'validated'
        elif (result.mean_ic >= self.IC_THRESHOLD_PROMISING and
              result.stability_score <= self.STABILITY_THRESHOLD_PROMISING):
            return 'testing'  # Keep in testing for further work
        else:
            return 'rejected'

    def _update_hypothesis(
        self,
        hypothesis: dict,
        best_result: ModelExperimentResult,
        status: str,
    ) -> None:
        """Update hypothesis with ML results."""
        self.api.update_hypothesis(
            hypothesis_id=hypothesis['id'],
            status=status,
            metadata={
                'ml_scientist_results': {
                    'best_model_type': best_result.model_type,
                    'best_features': best_result.features,
                    'best_params': best_result.model_params,
                    'mean_ic': best_result.mean_ic,
                    'ic_std': best_result.ic_std,
                    'stability_score': best_result.stability_score,
                    'is_stable': best_result.is_stable,
                    'mlflow_run_id': best_result.mlflow_run_id,
                    'validated_at': datetime.now().isoformat(),
                }
            },
            actor=self.ACTOR,
        )

    def _extract_features_from_hypothesis(self, hypothesis: dict) -> list[str]:
        """Extract feature names from hypothesis thesis/metadata."""
        # Check metadata first
        metadata = hypothesis.get('metadata', {})
        if 'features' in metadata:
            return metadata['features']

        # Parse from thesis text
        thesis = hypothesis.get('thesis', '')
        features = []
        for feature in self.ALL_FEATURES:
            if feature in thesis.lower():
                features.append(feature)

        return features or ['momentum_20d']  # Default fallback

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

        return unique[:self.MAX_FEATURE_COMBINATIONS]

    def _get_param_grid(self, model_type: str) -> list[dict]:
        """Get hyperparameter combinations for model type."""
        from sklearn.model_selection import ParameterGrid

        grid = self.HYPERPARAMETER_GRIDS.get(model_type, {})
        if not grid:
            return [{}]

        return list(ParameterGrid(grid))[:10]  # Limit combinations

    # Inherit _get_universe_symbols, _send_email_notification from base
    # or implement specific versions
```

### Scheduler Integration

```python
# In hrp/agents/scheduler.py

def setup_ml_scientist_schedule(
    self,
    run_time: str = "20:00",  # 8 PM ET (after Signal Scientist at 7 PM)
    day_of_week: str = "tue,wed",  # Tues-Wed for ML work
    model_types: list[str] | None = None,
    max_trials_per_hypothesis: int = 50,
) -> None:
    """Schedule ML Scientist to process hypotheses."""
    from hrp.agents.research_agents import MLScientist

    agent = MLScientist(
        model_types=model_types,
        max_trials_per_hypothesis=max_trials_per_hypothesis,
    )

    hour, minute = map(int, run_time.split(':'))

    trigger = CronTrigger(
        day_of_week=day_of_week,
        hour=hour,
        minute=minute,
        timezone=self.timezone,
    )

    self.scheduler.add_job(
        agent.run,
        trigger=trigger,
        id="ml_scientist_training",
        name="ML Scientist Training",
        replace_existing=True,
    )
```

### MCP Integration

```python
# In hrp/mcp/research_server.py

@mcp.tool()
def train_hypothesis(
    hypothesis_id: str,
    model_types: list[str] | None = None,
    skip_hyperparameter_search: bool = False,
) -> dict[str, Any]:
    """
    Run ML training on a specific hypothesis.

    Args:
        hypothesis_id: The hypothesis to train models for
        model_types: Models to test (default: ridge, lasso, lightgbm)
        skip_hyperparameter_search: Use default params only

    Returns:
        Training results including best model and validation metrics
    """
    from hrp.agents.research_agents import MLScientist

    agent = MLScientist(
        hypothesis_ids=[hypothesis_id],
        model_types=model_types,
        skip_hyperparameter_search=skip_hyperparameter_search,
    )

    return agent.run()


@mcp.tool()
def run_ml_scientist(
    model_types: list[str] | None = None,
    max_hypotheses: int | None = None,
) -> dict[str, Any]:
    """
    Run ML Scientist on all hypotheses in testing status.

    Args:
        model_types: Models to test (default: ridge, lasso, lightgbm)
        max_hypotheses: Limit number of hypotheses to process

    Returns:
        Summary of ML training results
    """
    from hrp.agents.research_agents import MLScientist

    agent = MLScientist(model_types=model_types)
    return agent.run()
```

---

## Integration with Existing Infrastructure

### Walk-Forward Validation

Uses existing `hrp/ml/walk_forward.py`:
- `WalkForwardConfig` for configuration
- `walk_forward_validate()` for execution
- `WalkForwardResult` for results

### Overfitting Guards

Uses existing `hrp/risk/overfitting.py`:
- `HyperparameterTrialCounter` - Tracks trials per hypothesis
- `SharpeDecayMonitor` - Can be used post-validation
- `FeatureCountValidator` - Validates feature count

### MLflow Integration

Uses existing `hrp/research/mlflow_utils.py`:
- Experiments organized by hypothesis ID
- Metrics, parameters, artifacts logged automatically

### Hypothesis Registry

Uses existing `hrp/api/platform.py`:
- `list_hypotheses(status='testing')` - Get hypotheses to process
- `update_hypothesis()` - Update status and metadata
- `get_hypothesis()` - Retrieve specific hypothesis

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `hrp/agents/research_agents.py` | MODIFY | Add `MLScientist` class, `ModelExperimentResult` dataclass |
| `hrp/agents/scheduler.py` | MODIFY | Add `setup_ml_scientist_schedule()` method |
| `hrp/agents/__init__.py` | MODIFY | Export `MLScientist` |
| `hrp/mcp/research_server.py` | MODIFY | Add `train_hypothesis`, `run_ml_scientist` tools |
| `tests/test_agents/test_ml_scientist.py` | CREATE | Unit tests for ML Scientist |
| `tests/test_agents/test_scheduler.py` | MODIFY | Add scheduler tests |

---

## Testing Strategy

### Unit Tests

```python
# tests/test_agents/test_ml_scientist.py

class TestMLScientistInit:
    def test_default_initialization(self):
        """MLScientist initializes with default model types."""

    def test_custom_model_types(self):
        """MLScientist accepts custom model types."""

    def test_hypothesis_filter(self):
        """MLScientist can filter to specific hypothesis IDs."""


class TestFeatureExtraction:
    def test_extract_features_from_metadata(self):
        """Features extracted from hypothesis metadata."""

    def test_extract_features_from_thesis(self):
        """Features parsed from hypothesis thesis text."""

    def test_fallback_to_default(self):
        """Falls back to momentum_20d if no features found."""


class TestFeatureCombinations:
    def test_generates_base_combination(self):
        """Base features included as first combination."""

    def test_adds_complementary_features(self):
        """Complementary features added to combinations."""

    def test_respects_max_features(self):
        """Combinations limited to MAX_FEATURES_PER_MODEL."""

    def test_limits_total_combinations(self):
        """Total combinations limited to MAX_FEATURE_COMBINATIONS."""


class TestModelScoring:
    def test_higher_ic_scores_higher(self):
        """Higher IC produces higher score."""

    def test_lower_stability_scores_higher(self):
        """Lower stability score produces higher score."""

    def test_consistency_bonus(self):
        """All positive folds get consistency bonus."""


class TestStatusDetermination:
    def test_validated_status(self):
        """High IC + low stability = validated."""

    def test_rejected_status(self):
        """Low IC or high stability = rejected."""

    def test_promising_stays_testing(self):
        """Moderate results keep hypothesis in testing."""


class TestTrialBudget:
    def test_respects_trial_limit(self):
        """Stops when trial budget exhausted."""

    def test_early_stopping_on_excellent(self):
        """Stops early if excellent result found."""


class TestMLflowIntegration:
    def test_logs_to_correct_experiment(self):
        """Results logged to ml-scientist/{hypothesis_id}."""

    def test_logs_all_metrics(self):
        """IC, stability, fold results logged."""


class TestEmailNotification:
    def test_sends_summary_email(self):
        """Summary email sent on completion."""

    def test_email_includes_best_models(self):
        """Email includes table of best models."""
```

### Integration Tests

```python
class TestMLScientistIntegration:
    def test_full_pipeline_with_mock_data(self):
        """End-to-end: hypothesis -> training -> update."""

    def test_scheduler_triggers_correctly(self):
        """ML Scientist runs on schedule."""

    def test_mcp_tool_invocation(self):
        """MCP tools trigger training correctly."""

    def test_lineage_tracking(self):
        """Agent events logged to lineage table."""
```

---

## Verification Plan

1. **Unit tests pass:** `pytest tests/test_agents/test_ml_scientist.py -v`
2. **Create test hypothesis:** Manually create hypothesis in `testing` status
3. **Run ML Scientist:** Execute on test hypothesis
4. **Verify MLflow:** Check experiment logged correctly
5. **Verify hypothesis update:** Check status updated with results
6. **Verify lineage:** Check agent events in lineage table
7. **Full test suite:** `pytest tests/ -v` (all tests pass)

---

## Dependencies on Existing Code

| Component | Location | Used For |
|-----------|----------|----------|
| `WalkForwardConfig` | `hrp/ml/walk_forward.py` | Validation configuration |
| `walk_forward_validate` | `hrp/ml/walk_forward.py` | Run validation |
| `HyperparameterTrialCounter` | `hrp/risk/overfitting.py` | Trial budget |
| `PlatformAPI` | `hrp/api/platform.py` | Hypothesis CRUD |
| `log_event` | `hrp/research/lineage.py` | Audit trail |
| `EmailNotifier` | `hrp/notifications/email.py` | Notifications |
| `ResearchAgent` | `hrp/agents/research_agents.py` | Base class |

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default models | ridge, lasso, lightgbm | Balance of simplicity and power |
| Max trials | 50 per hypothesis | Prevent overfitting via extensive search |
| Feature limit | 3 per model | Avoid curse of dimensionality |
| Early stopping | IC > 0.05 + stable | Don't waste compute on already-good results |
| Hyperparameter search | Grid search | Simple, predictable, easy to track |
| Status update | Automatic | Reduces manual overhead |

---

## Future Enhancements

1. **Ensemble models:** Combine top models for each hypothesis
2. **Feature selection:** Automatic feature importance-based selection
3. **Regime-aware training:** Train separate models for market regimes
4. **Cross-asset transfer:** Apply models trained on one universe to another
5. **Online learning:** Incremental updates as new data arrives
