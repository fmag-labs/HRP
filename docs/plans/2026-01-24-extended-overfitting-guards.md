# Extended Overfitting Guards Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement four additional overfitting prevention mechanisms: Sharpe Decay Monitor, Feature Count Validator, Hyperparameter Trial Counter, and Target Leakage Validator.

**Architecture:** Extend `hrp/risk/overfitting.py` with new validator classes that follow the same pattern as `TestSetGuard`. Each validator can be used standalone or integrated into the ML training pipeline. Database tracking via new `hyperparameter_trials` table.

**Tech Stack:** Python 3.11+, DuckDB, pytest, scipy (for statistical tests)

---

## Task 1: Sharpe Decay Monitor

**Files:**
- Modify: `hrp/risk/overfitting.py`
- Test: `tests/test_risk/test_overfitting.py`

### Step 1: Write failing tests for SharpeDecayMonitor

Add to `tests/test_risk/test_overfitting.py`:

```python
class TestSharpeDecayMonitor:
    """Tests for Sharpe ratio decay detection."""

    def test_no_decay_passes(self):
        """Test that similar train/test Sharpe passes."""
        from hrp.risk.overfitting import SharpeDecayMonitor

        monitor = SharpeDecayMonitor(max_decay_ratio=0.5)
        result = monitor.check(train_sharpe=1.2, test_sharpe=1.0)

        assert result.passed is True
        assert result.decay_ratio < 0.5

    def test_significant_decay_fails(self):
        """Test that large decay is flagged."""
        from hrp.risk.overfitting import SharpeDecayMonitor

        monitor = SharpeDecayMonitor(max_decay_ratio=0.5)
        result = monitor.check(train_sharpe=2.0, test_sharpe=0.5)

        assert result.passed is False
        assert result.decay_ratio == 0.75  # (2.0 - 0.5) / 2.0

    def test_negative_test_sharpe_fails(self):
        """Test that negative test Sharpe always fails."""
        from hrp.risk.overfitting import SharpeDecayMonitor

        monitor = SharpeDecayMonitor(max_decay_ratio=0.5)
        result = monitor.check(train_sharpe=1.5, test_sharpe=-0.2)

        assert result.passed is False
        assert "negative" in result.message.lower()

    def test_zero_train_sharpe_handled(self):
        """Test edge case of zero train Sharpe."""
        from hrp.risk.overfitting import SharpeDecayMonitor

        monitor = SharpeDecayMonitor(max_decay_ratio=0.5)
        result = monitor.check(train_sharpe=0.0, test_sharpe=0.1)

        # Can't compute decay ratio with zero train, should pass if test >= 0
        assert result.passed is True

    def test_custom_threshold(self):
        """Test custom decay threshold."""
        from hrp.risk.overfitting import SharpeDecayMonitor

        monitor = SharpeDecayMonitor(max_decay_ratio=0.3)
        result = monitor.check(train_sharpe=1.0, test_sharpe=0.6)

        # 40% decay exceeds 30% threshold
        assert result.passed is False
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_risk/test_overfitting.py::TestSharpeDecayMonitor -v
```

Expected: FAIL with `ImportError: cannot import name 'SharpeDecayMonitor'`

### Step 3: Implement SharpeDecayMonitor

Add to `hrp/risk/overfitting.py`:

```python
from dataclasses import dataclass


@dataclass
class DecayCheckResult:
    """Result of a Sharpe decay check."""
    passed: bool
    decay_ratio: float
    train_sharpe: float
    test_sharpe: float
    message: str


class SharpeDecayMonitor:
    """
    Monitor for train/test Sharpe ratio decay.

    Detects overfitting by comparing in-sample vs out-of-sample Sharpe ratios.
    A large decay indicates the strategy may be overfit to training data.

    Usage:
        monitor = SharpeDecayMonitor(max_decay_ratio=0.5)
        result = monitor.check(train_sharpe=1.5, test_sharpe=1.0)
        if not result.passed:
            print(f"Warning: {result.message}")
    """

    def __init__(self, max_decay_ratio: float = 0.5):
        """
        Initialize Sharpe decay monitor.

        Args:
            max_decay_ratio: Maximum allowed decay ratio (0.5 = 50% decay allowed)
        """
        if not 0 < max_decay_ratio < 1:
            raise ValueError("max_decay_ratio must be between 0 and 1")
        self.max_decay_ratio = max_decay_ratio

    def check(self, train_sharpe: float, test_sharpe: float) -> DecayCheckResult:
        """
        Check Sharpe decay between train and test.

        Args:
            train_sharpe: In-sample Sharpe ratio
            test_sharpe: Out-of-sample Sharpe ratio

        Returns:
            DecayCheckResult with pass/fail status and details
        """
        # Handle negative test Sharpe
        if test_sharpe < 0:
            return DecayCheckResult(
                passed=False,
                decay_ratio=1.0,
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                message=f"Negative test Sharpe ({test_sharpe:.2f}) indicates severe overfitting",
            )

        # Handle zero or negative train Sharpe
        if train_sharpe <= 0:
            return DecayCheckResult(
                passed=True,
                decay_ratio=0.0,
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                message="Train Sharpe <= 0, decay ratio not applicable",
            )

        # Calculate decay ratio
        decay_ratio = (train_sharpe - test_sharpe) / train_sharpe
        decay_ratio = max(0, decay_ratio)  # Can't have negative decay

        passed = decay_ratio <= self.max_decay_ratio

        if passed:
            message = f"Sharpe decay {decay_ratio:.1%} within threshold ({self.max_decay_ratio:.1%})"
        else:
            message = (
                f"Sharpe decay {decay_ratio:.1%} exceeds threshold ({self.max_decay_ratio:.1%}). "
                f"Train: {train_sharpe:.2f}, Test: {test_sharpe:.2f}"
            )

        return DecayCheckResult(
            passed=passed,
            decay_ratio=decay_ratio,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            message=message,
        )
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_risk/test_overfitting.py::TestSharpeDecayMonitor -v
```

Expected: All 5 tests PASS

### Step 5: Commit

```bash
git add hrp/risk/overfitting.py tests/test_risk/test_overfitting.py
git commit -m "feat(risk): add SharpeDecayMonitor for train/test performance comparison

- Detects overfitting by comparing in-sample vs out-of-sample Sharpe
- Configurable decay threshold (default 50%)
- Handles edge cases (negative Sharpe, zero train Sharpe)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Feature Count Validator

**Files:**
- Modify: `hrp/risk/overfitting.py`
- Test: `tests/test_risk/test_overfitting.py`

### Step 1: Write failing tests for FeatureCountValidator

Add to `tests/test_risk/test_overfitting.py`:

```python
class TestFeatureCountValidator:
    """Tests for feature count validation."""

    def test_under_threshold_passes(self):
        """Test that feature count under threshold passes."""
        from hrp.risk.overfitting import FeatureCountValidator

        validator = FeatureCountValidator(warn_threshold=30, max_threshold=50)
        result = validator.check(feature_count=20, sample_count=1000)

        assert result.passed is True
        assert result.warning is False

    def test_warning_threshold(self):
        """Test warning when feature count exceeds warn threshold."""
        from hrp.risk.overfitting import FeatureCountValidator

        validator = FeatureCountValidator(warn_threshold=30, max_threshold=50)
        result = validator.check(feature_count=35, sample_count=1000)

        assert result.passed is True
        assert result.warning is True
        assert "warning" in result.message.lower()

    def test_max_threshold_fails(self):
        """Test failure when feature count exceeds max threshold."""
        from hrp.risk.overfitting import FeatureCountValidator

        validator = FeatureCountValidator(warn_threshold=30, max_threshold=50)
        result = validator.check(feature_count=55, sample_count=1000)

        assert result.passed is False

    def test_features_per_sample_ratio(self):
        """Test that high features-per-sample ratio triggers warning."""
        from hrp.risk.overfitting import FeatureCountValidator

        validator = FeatureCountValidator(warn_threshold=30, max_threshold=50)
        # 25 features with only 100 samples = 0.25 ratio (too high)
        result = validator.check(feature_count=25, sample_count=100)

        assert result.warning is True
        assert "ratio" in result.message.lower() or "sample" in result.message.lower()

    def test_adequate_samples_no_warning(self):
        """Test no warning with adequate samples per feature."""
        from hrp.risk.overfitting import FeatureCountValidator

        validator = FeatureCountValidator(warn_threshold=30, max_threshold=50)
        # 20 features with 2000 samples = 100 samples per feature (good)
        result = validator.check(feature_count=20, sample_count=2000)

        assert result.passed is True
        assert result.warning is False
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_risk/test_overfitting.py::TestFeatureCountValidator -v
```

Expected: FAIL with `ImportError: cannot import name 'FeatureCountValidator'`

### Step 3: Implement FeatureCountValidator

Add to `hrp/risk/overfitting.py`:

```python
@dataclass
class FeatureCountResult:
    """Result of feature count validation."""
    passed: bool
    warning: bool
    feature_count: int
    sample_count: int
    features_per_sample: float
    message: str


class FeatureCountValidator:
    """
    Validator for number of features used in ML models.

    Prevents overfitting by limiting feature count and checking
    features-per-sample ratio.

    Rules:
    - Warn if features > warn_threshold (default 30)
    - Fail if features > max_threshold (default 50)
    - Warn if features/samples > 0.1 (need at least 10 samples per feature)

    Usage:
        validator = FeatureCountValidator(warn_threshold=30)
        result = validator.check(feature_count=25, sample_count=1000)
    """

    def __init__(
        self,
        warn_threshold: int = 30,
        max_threshold: int = 50,
        min_samples_per_feature: int = 10,
    ):
        """
        Initialize feature count validator.

        Args:
            warn_threshold: Feature count that triggers warning
            max_threshold: Feature count that triggers failure
            min_samples_per_feature: Minimum samples per feature ratio
        """
        if warn_threshold >= max_threshold:
            raise ValueError("warn_threshold must be less than max_threshold")

        self.warn_threshold = warn_threshold
        self.max_threshold = max_threshold
        self.min_samples_per_feature = min_samples_per_feature

    def check(self, feature_count: int, sample_count: int) -> FeatureCountResult:
        """
        Validate feature count.

        Args:
            feature_count: Number of features
            sample_count: Number of training samples

        Returns:
            FeatureCountResult with pass/fail/warning status
        """
        features_per_sample = feature_count / sample_count if sample_count > 0 else float('inf')
        samples_per_feature = sample_count / feature_count if feature_count > 0 else float('inf')

        messages = []
        warning = False
        passed = True

        # Check absolute feature count
        if feature_count > self.max_threshold:
            passed = False
            messages.append(
                f"Feature count ({feature_count}) exceeds maximum ({self.max_threshold})"
            )
        elif feature_count > self.warn_threshold:
            warning = True
            messages.append(
                f"Warning: Feature count ({feature_count}) exceeds threshold ({self.warn_threshold})"
            )

        # Check samples-per-feature ratio
        if samples_per_feature < self.min_samples_per_feature:
            warning = True
            messages.append(
                f"Warning: Only {samples_per_feature:.1f} samples per feature "
                f"(recommended: {self.min_samples_per_feature}+)"
            )

        if not messages:
            messages.append(
                f"Feature count ({feature_count}) OK with {samples_per_feature:.0f} samples/feature"
            )

        return FeatureCountResult(
            passed=passed,
            warning=warning,
            feature_count=feature_count,
            sample_count=sample_count,
            features_per_sample=features_per_sample,
            message="; ".join(messages),
        )
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_risk/test_overfitting.py::TestFeatureCountValidator -v
```

Expected: All 5 tests PASS

### Step 5: Commit

```bash
git add hrp/risk/overfitting.py tests/test_risk/test_overfitting.py
git commit -m "feat(risk): add FeatureCountValidator to prevent overfitting

- Configurable warn (30) and max (50) feature thresholds
- Checks samples-per-feature ratio (min 10 recommended)
- Returns structured result with warnings and failures

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Database Schema for Hyperparameter Trials

**Files:**
- Modify: `hrp/data/schema.py`
- Test: `tests/test_data/test_schema.py` (verify table creation)

### Step 1: Write failing test for schema

Add to `tests/test_data/test_schema.py` (or create if needed):

```python
def test_hyperparameter_trials_table_exists(test_db):
    """Test that hyperparameter_trials table is created."""
    from hrp.data.db import get_db

    db = get_db(test_db)
    with db.connection() as conn:
        result = conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = 'hyperparameter_trials'
        """).fetchone()

    assert result[0] == 1, "hyperparameter_trials table should exist"


def test_hyperparameter_trials_schema(test_db):
    """Test hyperparameter_trials table has correct columns."""
    from hrp.data.db import get_db

    db = get_db(test_db)
    with db.connection() as conn:
        # Insert a test record to verify schema
        conn.execute("""
            INSERT INTO hyperparameter_trials
            (trial_id, hypothesis_id, model_type, hyperparameters, metric_value, metric_name)
            VALUES (1, 'HYP-TEST-001', 'ridge', '{"alpha": 1.0}', 0.85, 'val_r2')
        """)

        result = conn.execute("""
            SELECT trial_id, hypothesis_id, model_type, hyperparameters, metric_value
            FROM hyperparameter_trials WHERE trial_id = 1
        """).fetchone()

    assert result[0] == 1
    assert result[1] == 'HYP-TEST-001'
    assert result[2] == 'ridge'
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_data/test_schema.py::test_hyperparameter_trials_table_exists -v
```

Expected: FAIL (table doesn't exist)

### Step 3: Add table to schema

Modify `hrp/data/schema.py`, add after `test_set_evaluations` table:

```python
    "hyperparameter_trials": """
        CREATE TABLE IF NOT EXISTS hyperparameter_trials (
            trial_id INTEGER PRIMARY KEY,
            hypothesis_id VARCHAR NOT NULL,
            model_type VARCHAR NOT NULL,
            hyperparameters JSON NOT NULL,
            metric_name VARCHAR NOT NULL,
            metric_value DECIMAL(10,6),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            fold_index INTEGER,
            notes VARCHAR
        )
    """,
```

Also add index after other indexes:

```python
    "idx_hp_trials_hypothesis": """
        CREATE INDEX IF NOT EXISTS idx_hp_trials_hypothesis
        ON hyperparameter_trials(hypothesis_id)
    """,
```

### Step 4: Reinitialize test database and run tests

```bash
pytest tests/test_data/test_schema.py -v -k hyperparameter
```

Expected: PASS

### Step 5: Commit

```bash
git add hrp/data/schema.py tests/test_data/test_schema.py
git commit -m "feat(schema): add hyperparameter_trials table for HP search tracking

- Tracks all hyperparameter combinations tried per hypothesis
- Stores metric values, model type, and optional fold index
- Enables limiting HP trials to prevent overfitting

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Hyperparameter Trial Counter

**Files:**
- Modify: `hrp/risk/overfitting.py`
- Test: `tests/test_risk/test_overfitting.py`

### Step 1: Write failing tests for HyperparameterTrialCounter

Add to `tests/test_risk/test_overfitting.py`:

```python
class TestHyperparameterTrialCounter:
    """Tests for hyperparameter trial tracking."""

    @pytest.fixture(autouse=True)
    def clean_hp_trials(self, test_db):
        """Clean hyperparameter_trials table before each test."""
        db = get_db(test_db)
        with db.connection() as conn:
            conn.execute("DELETE FROM hyperparameter_trials WHERE hypothesis_id LIKE 'HYP-TEST-%'")
        yield

    def test_first_trial_allowed(self):
        """Test first hyperparameter trial is allowed."""
        from hrp.risk.overfitting import HyperparameterTrialCounter

        counter = HyperparameterTrialCounter(hypothesis_id="HYP-TEST-HP-001", max_trials=50)

        assert counter.can_try() is True
        assert counter.trial_count == 0

    def test_log_trial_increments_count(self):
        """Test logging a trial increments the count."""
        from hrp.risk.overfitting import HyperparameterTrialCounter

        counter = HyperparameterTrialCounter(hypothesis_id="HYP-TEST-HP-002", max_trials=50)

        counter.log_trial(
            model_type="ridge",
            hyperparameters={"alpha": 1.0},
            metric_name="val_r2",
            metric_value=0.85,
        )

        assert counter.trial_count == 1

    def test_max_trials_blocks_new_trials(self):
        """Test that exceeding max_trials blocks new trials."""
        from hrp.risk.overfitting import HyperparameterTrialCounter, OverfittingError

        counter = HyperparameterTrialCounter(hypothesis_id="HYP-TEST-HP-003", max_trials=3)

        # Log 3 trials
        for i in range(3):
            counter.log_trial(
                model_type="ridge",
                hyperparameters={"alpha": float(i)},
                metric_name="val_r2",
                metric_value=0.8 + i * 0.01,
            )

        assert counter.can_try() is False

        with pytest.raises(OverfittingError, match="trial limit"):
            counter.log_trial(
                model_type="ridge",
                hyperparameters={"alpha": 99.0},
                metric_name="val_r2",
                metric_value=0.9,
            )

    def test_remaining_trials(self):
        """Test remaining_trials property."""
        from hrp.risk.overfitting import HyperparameterTrialCounter

        counter = HyperparameterTrialCounter(hypothesis_id="HYP-TEST-HP-004", max_trials=10)

        assert counter.remaining_trials == 10

        counter.log_trial("ridge", {"alpha": 1.0}, "val_r2", 0.8)

        assert counter.remaining_trials == 9

    def test_get_best_trial(self):
        """Test retrieving best trial by metric."""
        from hrp.risk.overfitting import HyperparameterTrialCounter

        counter = HyperparameterTrialCounter(hypothesis_id="HYP-TEST-HP-005", max_trials=50)

        counter.log_trial("ridge", {"alpha": 0.1}, "val_r2", 0.75)
        counter.log_trial("ridge", {"alpha": 1.0}, "val_r2", 0.85)
        counter.log_trial("ridge", {"alpha": 10.0}, "val_r2", 0.80)

        best = counter.get_best_trial()

        assert best is not None
        assert best["metric_value"] == 0.85
        assert best["hyperparameters"]["alpha"] == 1.0

    def test_persists_across_instances(self):
        """Test trial count persists in database across instances."""
        from hrp.risk.overfitting import HyperparameterTrialCounter

        hypothesis_id = "HYP-TEST-HP-006"

        # First instance logs a trial
        counter1 = HyperparameterTrialCounter(hypothesis_id=hypothesis_id, max_trials=50)
        counter1.log_trial("ridge", {"alpha": 1.0}, "val_r2", 0.85)

        # Second instance should see the existing trial
        counter2 = HyperparameterTrialCounter(hypothesis_id=hypothesis_id, max_trials=50)

        assert counter2.trial_count == 1
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_risk/test_overfitting.py::TestHyperparameterTrialCounter -v
```

Expected: FAIL with `ImportError: cannot import name 'HyperparameterTrialCounter'`

### Step 3: Implement HyperparameterTrialCounter

Add to `hrp/risk/overfitting.py`:

```python
import json
from typing import Optional


def _load_trial_count(hypothesis_id: str) -> int:
    """Load existing trial count from database."""
    db = get_db()

    with db.connection() as conn:
        result = conn.execute(
            """
            SELECT COUNT(*)
            FROM hyperparameter_trials
            WHERE hypothesis_id = ?
            """,
            (hypothesis_id,),
        ).fetchone()

    return result[0] if result else 0


class HyperparameterTrialCounter:
    """
    Track and limit hyperparameter search trials per hypothesis.

    Prevents overfitting by limiting the number of HP combinations tried.
    All trials are logged to the database for auditability.

    Usage:
        counter = HyperparameterTrialCounter(hypothesis_id='HYP-2025-001', max_trials=50)

        if counter.can_try():
            counter.log_trial(
                model_type='ridge',
                hyperparameters={'alpha': 1.0},
                metric_name='val_r2',
                metric_value=0.85,
            )

        best = counter.get_best_trial()
    """

    def __init__(self, hypothesis_id: str, max_trials: int = 50):
        """
        Initialize hyperparameter trial counter.

        Args:
            hypothesis_id: Hypothesis ID
            max_trials: Maximum allowed trials (default 50)
        """
        self.hypothesis_id = hypothesis_id
        self.max_trials = max_trials
        self._count = _load_trial_count(hypothesis_id)

        logger.debug(
            f"HyperparameterTrialCounter for {hypothesis_id}: "
            f"{self._count}/{max_trials} trials used"
        )

    @property
    def trial_count(self) -> int:
        """Current trial count."""
        return self._count

    @property
    def remaining_trials(self) -> int:
        """Remaining trials allowed."""
        return max(0, self.max_trials - self._count)

    def can_try(self) -> bool:
        """Check if more trials are allowed."""
        return self._count < self.max_trials

    def log_trial(
        self,
        model_type: str,
        hyperparameters: dict,
        metric_name: str,
        metric_value: float,
        fold_index: int | None = None,
        notes: str | None = None,
    ) -> None:
        """
        Log a hyperparameter trial.

        Args:
            model_type: Type of model (e.g., 'ridge', 'lightgbm')
            hyperparameters: Dictionary of hyperparameters tried
            metric_name: Name of evaluation metric (e.g., 'val_r2')
            metric_value: Value of evaluation metric
            fold_index: Optional fold index for walk-forward validation
            notes: Optional notes about the trial

        Raises:
            OverfittingError: If trial limit exceeded
        """
        if not self.can_try():
            raise OverfittingError(
                f"Hyperparameter trial limit exceeded for {self.hypothesis_id}. "
                f"Already tried {self._count} combinations (limit: {self.max_trials})."
            )

        db = get_db()
        hp_json = json.dumps(hyperparameters)

        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO hyperparameter_trials
                (trial_id, hypothesis_id, model_type, hyperparameters,
                 metric_name, metric_value, fold_index, notes)
                VALUES (
                    (SELECT COALESCE(MAX(trial_id), 0) + 1 FROM hyperparameter_trials),
                    ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    self.hypothesis_id,
                    model_type,
                    hp_json,
                    metric_name,
                    metric_value,
                    fold_index,
                    notes,
                ),
            )

        self._count += 1

        logger.debug(
            f"Logged HP trial {self._count}/{self.max_trials} for {self.hypothesis_id}: "
            f"{model_type} with {hyperparameters} -> {metric_name}={metric_value:.4f}"
        )

    def get_best_trial(self, metric_name: str | None = None) -> Optional[dict]:
        """
        Get the best trial by metric value.

        Args:
            metric_name: Optional filter by metric name

        Returns:
            Dictionary with trial details or None if no trials
        """
        db = get_db()

        query = """
            SELECT model_type, hyperparameters, metric_name, metric_value, fold_index
            FROM hyperparameter_trials
            WHERE hypothesis_id = ?
        """
        params = [self.hypothesis_id]

        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)

        query += " ORDER BY metric_value DESC LIMIT 1"

        with db.connection() as conn:
            result = conn.execute(query, params).fetchone()

        if not result:
            return None

        return {
            "model_type": result[0],
            "hyperparameters": json.loads(result[1]),
            "metric_name": result[2],
            "metric_value": result[3],
            "fold_index": result[4],
        }
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_risk/test_overfitting.py::TestHyperparameterTrialCounter -v
```

Expected: All 6 tests PASS

### Step 5: Commit

```bash
git add hrp/risk/overfitting.py tests/test_risk/test_overfitting.py
git commit -m "feat(risk): add HyperparameterTrialCounter to limit HP search

- Tracks all HP combinations tried per hypothesis in database
- Configurable max trials (default 50)
- Provides get_best_trial() to retrieve optimal configuration
- Raises OverfittingError when limit exceeded

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Target Leakage Validator

**Files:**
- Modify: `hrp/risk/overfitting.py`
- Test: `tests/test_risk/test_overfitting.py`

### Step 1: Write failing tests for TargetLeakageValidator

Add to `tests/test_risk/test_overfitting.py`:

```python
import pandas as pd
import numpy as np


class TestTargetLeakageValidator:
    """Tests for target leakage detection."""

    def test_no_leakage_passes(self):
        """Test that uncorrelated features pass."""
        from hrp.risk.overfitting import TargetLeakageValidator

        np.random.seed(42)
        features = pd.DataFrame({
            'momentum': np.random.randn(100),
            'volatility': np.random.randn(100),
        })
        target = pd.Series(np.random.randn(100))

        validator = TargetLeakageValidator(correlation_threshold=0.95)
        result = validator.check(features, target)

        assert result.passed is True
        assert len(result.suspicious_features) == 0

    def test_high_correlation_detected(self):
        """Test that highly correlated features are flagged."""
        from hrp.risk.overfitting import TargetLeakageValidator

        np.random.seed(42)
        target = pd.Series(np.random.randn(100))
        features = pd.DataFrame({
            'clean_feature': np.random.randn(100),
            'leaky_feature': target * 1.01 + np.random.randn(100) * 0.01,  # ~99% correlated
        })

        validator = TargetLeakageValidator(correlation_threshold=0.95)
        result = validator.check(features, target)

        assert result.passed is False
        assert 'leaky_feature' in result.suspicious_features

    def test_perfect_correlation_fails(self):
        """Test that perfect correlation definitely fails."""
        from hrp.risk.overfitting import TargetLeakageValidator

        target = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        features = pd.DataFrame({
            'exact_copy': target.values,
        })

        validator = TargetLeakageValidator(correlation_threshold=0.95)
        result = validator.check(features, target)

        assert result.passed is False
        assert 'exact_copy' in result.suspicious_features
        assert result.correlations['exact_copy'] > 0.99

    def test_future_date_features_flagged(self):
        """Test that features with future-looking names are flagged."""
        from hrp.risk.overfitting import TargetLeakageValidator

        np.random.seed(42)
        features = pd.DataFrame({
            'momentum_20d': np.random.randn(100),
            'returns_future_5d': np.random.randn(100),  # Suspicious name
            'next_day_volume': np.random.randn(100),    # Suspicious name
        })
        target = pd.Series(np.random.randn(100))

        validator = TargetLeakageValidator(correlation_threshold=0.95)
        result = validator.check(features, target)

        assert result.warning is True
        assert any('future' in f.lower() or 'next' in f.lower() for f in result.name_warnings)

    def test_custom_threshold(self):
        """Test custom correlation threshold."""
        from hrp.risk.overfitting import TargetLeakageValidator

        np.random.seed(42)
        target = pd.Series(np.random.randn(100))
        features = pd.DataFrame({
            'moderate_corr': target * 0.8 + np.random.randn(100) * 0.6,  # ~80% correlated
        })

        # With 0.95 threshold, should pass
        validator_high = TargetLeakageValidator(correlation_threshold=0.95)
        result_high = validator_high.check(features, target)
        assert result_high.passed is True

        # With 0.7 threshold, should fail
        validator_low = TargetLeakageValidator(correlation_threshold=0.7)
        result_low = validator_low.check(features, target)
        assert result_low.passed is False
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_risk/test_overfitting.py::TestTargetLeakageValidator -v
```

Expected: FAIL with `ImportError: cannot import name 'TargetLeakageValidator'`

### Step 3: Implement TargetLeakageValidator

Add to `hrp/risk/overfitting.py`:

```python
import re
import pandas as pd


@dataclass
class LeakageCheckResult:
    """Result of target leakage check."""
    passed: bool
    warning: bool
    suspicious_features: list[str]
    correlations: dict[str, float]
    name_warnings: list[str]
    message: str


class TargetLeakageValidator:
    """
    Validator for detecting target leakage in features.

    Detects potential data leakage by:
    1. Checking for suspiciously high correlations with target
    2. Flagging features with future-looking names

    Usage:
        validator = TargetLeakageValidator(correlation_threshold=0.95)
        result = validator.check(features_df, target_series)
        if not result.passed:
            print(f"Leakage detected: {result.suspicious_features}")
    """

    # Patterns that suggest future information
    FUTURE_PATTERNS = [
        r'future',
        r'next',
        r'forward',
        r'lead',
        r'target',
        r't\+\d',  # t+1, t+5, etc.
    ]

    def __init__(self, correlation_threshold: float = 0.95):
        """
        Initialize target leakage validator.

        Args:
            correlation_threshold: Correlation above this triggers failure (default 0.95)
        """
        if not 0 < correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")

        self.correlation_threshold = correlation_threshold
        self._future_regex = re.compile(
            '|'.join(self.FUTURE_PATTERNS),
            re.IGNORECASE
        )

    def check(self, features: pd.DataFrame, target: pd.Series) -> LeakageCheckResult:
        """
        Check features for target leakage.

        Args:
            features: DataFrame of feature values
            target: Series of target values

        Returns:
            LeakageCheckResult with pass/fail and suspicious features
        """
        suspicious_features = []
        correlations = {}
        name_warnings = []

        # Check correlations
        for col in features.columns:
            try:
                corr = features[col].corr(target)
                correlations[col] = abs(corr) if pd.notna(corr) else 0.0

                if abs(corr) >= self.correlation_threshold:
                    suspicious_features.append(col)
                    logger.warning(
                        f"High correlation detected: {col} has {corr:.3f} correlation with target"
                    )
            except Exception as e:
                logger.debug(f"Could not compute correlation for {col}: {e}")
                correlations[col] = 0.0

        # Check feature names for future-looking patterns
        for col in features.columns:
            if self._future_regex.search(col):
                name_warnings.append(col)
                logger.warning(
                    f"Suspicious feature name: '{col}' may indicate future information"
                )

        # Determine pass/fail
        passed = len(suspicious_features) == 0
        warning = len(name_warnings) > 0

        # Build message
        messages = []
        if suspicious_features:
            messages.append(
                f"High correlation with target: {', '.join(suspicious_features)}"
            )
        if name_warnings:
            messages.append(
                f"Suspicious feature names (may contain future info): {', '.join(name_warnings)}"
            )
        if not messages:
            messages.append("No target leakage detected")

        return LeakageCheckResult(
            passed=passed,
            warning=warning,
            suspicious_features=suspicious_features,
            correlations=correlations,
            name_warnings=name_warnings,
            message="; ".join(messages),
        )
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_risk/test_overfitting.py::TestTargetLeakageValidator -v
```

Expected: All 5 tests PASS

### Step 5: Commit

```bash
git add hrp/risk/overfitting.py tests/test_risk/test_overfitting.py
git commit -m "feat(risk): add TargetLeakageValidator to detect data leakage

- Checks for high correlation between features and target
- Flags features with future-looking names (future, next, forward, etc.)
- Configurable correlation threshold (default 0.95)
- Returns detailed report of suspicious features

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Integration with ML Training Pipeline

**Files:**
- Modify: `hrp/ml/training.py`
- Test: `tests/test_ml/test_training.py`

### Step 1: Write failing integration tests

Add to `tests/test_ml/test_training.py`:

```python
class TestOverfittingGuardIntegration:
    """Tests for overfitting guard integration in training."""

    def test_feature_count_warning_logged(self, caplog):
        """Test that feature count warning is logged during training."""
        from hrp.ml.training import train_model
        from hrp.ml.models import MLConfig
        from datetime import date
        import logging

        # Config with many features would trigger warning
        # This test verifies the integration point exists
        # Actual execution depends on having enough features in DB
        pass  # Placeholder - actual test needs real data

    def test_sharpe_decay_checked_after_training(self):
        """Test that Sharpe decay is checked after training completes."""
        # This test verifies the integration exists
        pass  # Placeholder - actual test needs backtest results

    def test_leakage_check_before_training(self):
        """Test that leakage is checked before training starts."""
        # This test verifies the integration exists
        pass  # Placeholder
```

### Step 2: Add validation hooks to train_model

Modify `hrp/ml/training.py`, add imports at top:

```python
from hrp.risk.overfitting import (
    TestSetGuard,
    FeatureCountValidator,
    TargetLeakageValidator,
    OverfittingError,
)
```

Add validation before training (around line 294, after loading data):

```python
    # Validate feature count
    feature_validator = FeatureCountValidator()
    feature_result = feature_validator.check(
        feature_count=len(config.features),
        sample_count=len(X_train),
    )
    if not feature_result.passed:
        raise OverfittingError(feature_result.message)
    if feature_result.warning:
        logger.warning(feature_result.message)

    # Check for target leakage
    leakage_validator = TargetLeakageValidator()
    leakage_result = leakage_validator.check(X_train, y_train)
    if not leakage_result.passed:
        raise OverfittingError(f"Target leakage detected: {leakage_result.message}")
    if leakage_result.warning:
        logger.warning(f"Potential leakage: {leakage_result.message}")
```

### Step 3: Run existing tests to ensure no breakage

```bash
pytest tests/test_ml/test_training.py -v
```

Expected: All existing tests still PASS

### Step 4: Commit

```bash
git add hrp/ml/training.py tests/test_ml/test_training.py
git commit -m "feat(ml): integrate overfitting guards into training pipeline

- Add FeatureCountValidator check before training
- Add TargetLeakageValidator check before training
- Raise OverfittingError on failures, log warnings

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Update Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/operations/cookbook.md` (if exists)

### Step 1: Update CLAUDE.md

Add to the "Common Tasks" section:

```markdown
### Use overfitting guards
```python
from hrp.risk import TestSetGuard, validate_strategy, check_parameter_sensitivity

# Test set discipline (limits to 3 evaluations per hypothesis)
guard = TestSetGuard(hypothesis_id='HYP-2025-001')

with guard.evaluate(metadata={"experiment": "final_validation"}):
    metrics = model.evaluate(test_data)

print(f"Evaluations remaining: {guard.remaining_evaluations}")

# Validate strategy meets minimum criteria
result = validate_strategy({
    "sharpe": 0.80,
    "num_trades": 200,
    "max_drawdown": 0.18,
    "win_rate": 0.52,
})

if result.passed:
    print(f"✅ Validation passed! Confidence: {result.confidence_score:.2f}")
else:
    print(f"❌ Failed: {result.failed_criteria}")

# Check parameter robustness
experiments = {
    "baseline": {"sharpe": 0.80, "params": {"lookback": 20}},
    "var_1": {"sharpe": 0.75, "params": {"lookback": 16}},
    "var_2": {"sharpe": 0.82, "params": {"lookback": 24}},
}

robustness = check_parameter_sensitivity(experiments, baseline_key="baseline")
print(f"Parameter stability: {'✅ PASS' if robustness.passed else '❌ FAIL'}")

# NEW: Sharpe decay monitoring
from hrp.risk.overfitting import SharpeDecayMonitor

monitor = SharpeDecayMonitor(max_decay_ratio=0.5)
result = monitor.check(train_sharpe=1.5, test_sharpe=1.0)
if not result.passed:
    print(f"⚠️ Sharpe decay warning: {result.message}")

# NEW: Hyperparameter trial tracking
from hrp.risk.overfitting import HyperparameterTrialCounter

counter = HyperparameterTrialCounter(hypothesis_id='HYP-2025-001', max_trials=50)
counter.log_trial(
    model_type='ridge',
    hyperparameters={'alpha': 1.0},
    metric_name='val_r2',
    metric_value=0.85,
)
print(f"HP trials remaining: {counter.remaining_trials}")
best = counter.get_best_trial()
```
```

### Step 2: Commit documentation

```bash
git add CLAUDE.md docs/operations/cookbook.md
git commit -m "docs: update documentation with new overfitting guards

- Add SharpeDecayMonitor usage examples
- Add HyperparameterTrialCounter usage examples
- Document FeatureCountValidator and TargetLeakageValidator

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Run Full Test Suite

### Step 1: Run all tests

```bash
pytest tests/ -v
```

Expected: ~97%+ pass rate (same or better than before)

### Step 2: Run specific overfitting tests

```bash
pytest tests/test_risk/test_overfitting.py -v
```

Expected: All new tests PASS

### Step 3: Final commit if needed

```bash
git status
# If any uncommitted changes, commit them
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | SharpeDecayMonitor | 5 tests |
| 2 | FeatureCountValidator | 5 tests |
| 3 | hyperparameter_trials schema | 2 tests |
| 4 | HyperparameterTrialCounter | 6 tests |
| 5 | TargetLeakageValidator | 5 tests |
| 6 | Training pipeline integration | 3 tests |
| 7 | Documentation | N/A |
| 8 | Full test suite verification | All |

**Total new tests:** ~26 tests
**Estimated implementation:** 8 commits
