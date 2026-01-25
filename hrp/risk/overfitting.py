"""
Overfitting prevention mechanisms.

Implements test set discipline and overfitting detection.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from hrp.data.db import get_db


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


class OverfittingError(Exception):
    """Raised when test set evaluation limit is exceeded."""
    pass


def _load_evaluation_count(hypothesis_id: str) -> int:
    """Load existing evaluation count from database."""
    db = get_db()
    
    with db.connection() as conn:
        result = conn.execute(
            """
            SELECT COUNT(*) 
            FROM test_set_evaluations 
            WHERE hypothesis_id = ?
            """,
            (hypothesis_id,),
        ).fetchone()
    
    return result[0] if result else 0


def _log_evaluation(
    hypothesis_id: str,
    override: bool,
    override_reason: str | None,
    metadata: dict[str, Any] | None,
):
    """Log test set evaluation to database."""
    db = get_db()

    # Convert metadata dict to JSON string if present
    import json
    metadata_json = json.dumps(metadata) if metadata else None

    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO test_set_evaluations
            (evaluation_id, hypothesis_id, evaluated_at, override, override_reason, metadata)
            VALUES (
                (SELECT COALESCE(MAX(evaluation_id), 0) + 1 FROM test_set_evaluations),
                ?, ?, ?, ?, ?
            )
            """,
            (
                hypothesis_id,
                datetime.utcnow(),
                override,
                override_reason,
                metadata_json,
            ),
        )


class TestSetGuard:
    """
    Guard against excessive test set evaluation.
    
    Enforces limit on number of times test set can be evaluated per hypothesis
    to prevent data snooping and overfitting.
    
    Usage:
        guard = TestSetGuard(hypothesis_id='HYP-2025-001')
        
        with guard.evaluate():
            metrics = model.evaluate(test_data)
    
    Raises:
        OverfittingError: If evaluation limit exceeded without explicit override
    """

    def __init__(self, hypothesis_id: str, max_evaluations: int = 3):
        """
        Initialize test set guard.
        
        Args:
            hypothesis_id: Hypothesis ID
            max_evaluations: Maximum allowed evaluations (default 3)
        """
        self.hypothesis_id = hypothesis_id
        self.max_evaluations = max_evaluations
        self._count = _load_evaluation_count(hypothesis_id)
        
        logger.debug(
            f"TestSetGuard for {hypothesis_id}: "
            f"{self._count}/{max_evaluations} evaluations used"
        )

    @property
    def evaluation_count(self) -> int:
        """Current evaluation count."""
        return self._count

    @property
    def remaining_evaluations(self) -> int:
        """Remaining evaluations allowed."""
        return max(0, self.max_evaluations - self._count)

    @contextmanager
    def evaluate(
        self,
        override: bool = False,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Context manager for test set evaluation.
        
        Args:
            override: Explicitly override limit (requires reason)
            reason: Reason for override (required if override=True)
            metadata: Optional metadata to log with evaluation
            
        Raises:
            OverfittingError: If limit exceeded without override
            ValueError: If override=True but reason not provided
        """
        if override and not reason:
            raise ValueError("Override requires a reason")

        if not override and self._count >= self.max_evaluations:
            raise OverfittingError(
                f"Test set evaluation limit exceeded for {self.hypothesis_id}. "
                f"Already evaluated {self._count} times (limit: {self.max_evaluations}). "
                f"Use override=True with justification if needed."
            )

        if override:
            logger.warning(
                f"Test set evaluation override for {self.hypothesis_id}: {reason}"
            )

        # Log the evaluation
        _log_evaluation(self.hypothesis_id, override, reason, metadata)
        self._count += 1

        try:
            yield
        except Exception:
            # Evaluation failed, but still counts toward limit
            logger.error(f"Test set evaluation failed for {self.hypothesis_id}")
            raise
