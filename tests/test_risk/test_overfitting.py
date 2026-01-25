"""Tests for overfitting prevention mechanisms."""

import pytest
from unittest.mock import MagicMock, patch

from hrp.risk.overfitting import TestSetGuard, OverfittingError
from hrp.data.db import get_db


@pytest.fixture(autouse=True)
def clean_test_evaluations(test_db):
    """Clean test_set_evaluations table before each test."""
    db = get_db(test_db)
    with db.connection() as conn:
        conn.execute("DELETE FROM test_set_evaluations WHERE hypothesis_id LIKE 'HYP-TEST-%'")
    yield


class TestTestSetGuard:
    """Tests for TestSetGuard class."""

    def test_first_evaluation_allowed(self):
        """Test first test set evaluation is allowed."""
        guard = TestSetGuard(hypothesis_id="HYP-TEST-001")
        
        # Should not raise
        with guard.evaluate():
            pass
        
        assert guard.evaluation_count == 1

    def test_multiple_evaluations_allowed_under_limit(self):
        """Test multiple evaluations allowed under limit."""
        guard = TestSetGuard(hypothesis_id="HYP-TEST-002")
        
        for _ in range(3):
            with guard.evaluate():
                pass
        
        assert guard.evaluation_count == 3

    def test_fourth_evaluation_raises_error(self):
        """Test fourth evaluation raises OverfittingError."""
        guard = TestSetGuard(hypothesis_id="HYP-TEST-003", max_evaluations=3)
        
        # Use up 3 evaluations
        for _ in range(3):
            with guard.evaluate():
                pass
        
        # Fourth should fail
        with pytest.raises(OverfittingError, match="limit exceeded"):
            with guard.evaluate():
                pass

    def test_explicit_override_allows_evaluation(self):
        """Test explicit override bypasses limit."""
        guard = TestSetGuard(hypothesis_id="HYP-TEST-004", max_evaluations=3)
        
        # Use up 3 evaluations
        for _ in range(3):
            with guard.evaluate():
                pass
        
        # Override should work
        with guard.evaluate(override=True, reason="Final validation after bug fix"):
            pass
        
        assert guard.evaluation_count == 4

    def test_evaluation_logged_to_database(self):
        """Test evaluations are logged with timestamp and metadata."""
        with patch("hrp.risk.overfitting._load_evaluation_count", return_value=0):
            with patch("hrp.risk.overfitting.get_db") as mock_db:
                mock_conn = MagicMock()
                mock_db.return_value.connection.return_value.__enter__.return_value = mock_conn
                
                guard = TestSetGuard(hypothesis_id="HYP-2025-001")
                
                with guard.evaluate(metadata={"model_type": "ridge"}):
                    pass
                
                # Should have logged to database
                assert mock_conn.execute.called

    def test_load_existing_count(self):
        """Test guard loads existing evaluation count from database."""
        with patch("hrp.risk.overfitting._load_evaluation_count") as mock_load:
            mock_load.return_value = 2
            
            guard = TestSetGuard(hypothesis_id="HYP-2025-001")
            assert guard.evaluation_count == 2

    def test_override_requires_reason(self):
        """Test override without reason raises ValueError."""
        guard = TestSetGuard(hypothesis_id="HYP-2025-001")
        
        with pytest.raises(ValueError, match="Override requires a reason"):
            with guard.evaluate(override=True):
                pass

    def test_remaining_evaluations(self):
        """Test remaining_evaluations property."""
        guard = TestSetGuard(hypothesis_id="HYP-TEST-008", max_evaluations=3)
        
        assert guard.remaining_evaluations == 3
        
        with guard.evaluate():
            pass
        
        assert guard.remaining_evaluations == 2


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
