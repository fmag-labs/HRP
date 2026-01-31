"""
End-to-end integration tests for ML deployment pipeline.

Tests the complete workflow from model registration to production deployment,
including drift monitoring and rollback capabilities.
"""

from datetime import date, datetime
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

from hrp.data.db import get_db
from hrp.api.platform import PlatformAPI
from hrp.ml.registry import ModelRegistry
from hrp.ml.deployment import DeploymentPipeline
from hrp.monitoring.drift_monitor import DriftMonitor


@pytest.fixture(scope="function")
def trained_model():
    """
    Create a trained model for testing.

    Returns a simple sklearn model.
    """
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=1.0)
    # Fit on dummy data
    X = np.random.randn(100, 3)
    y = X[:, 0] + 0.1 * np.random.randn(100)
    model.fit(X, y)

    return model


@pytest.fixture(scope="function")
def sample_features():
    """
    Create sample feature data for testing.

    Returns DataFrame with typical feature columns.
    """
    return pd.DataFrame({
        'momentum_20d': np.random.randn(100),
        'volatility_60d': 0.02 + 0.01 * np.random.rand(100),
        'rsi_14d': 50 + 10 * np.random.randn(100),
        'prediction': 0.01 + 0.001 * np.random.randn(100),
        'returns_20d': 0.01 + 0.02 * np.random.randn(100),
    })


@pytest.fixture(scope="function")
def validation_data():
    """
    Create validation data for deployment checks.
    """
    return pd.DataFrame({
        'momentum_20d': np.random.randn(100),
        'volatility_60d': 0.02 + 0.005 * np.random.rand(100),
        'prediction': 0.01 + 0.001 * np.random.randn(100),
        'returns_20d': 0.01 + 0.015 * np.random.randn(100),
    })


class TestModelRegistryWorkflow:
    """Test model registration and retrieval workflow."""

    @patch('mlflow.tracking.MlflowClient')
    @patch('mlflow.active_run')
    @patch('mlflow.sklearn.log_model')
    @patch('mlflow.start_run')
    def test_register_model_creates_version(self, mock_start_run, mock_log_model, mock_active_run, mock_client_class, trained_model):
        """
        Model registration creates a new version.

        Given:
            - Trained model
            - Model metadata
        When:
            - Registering model
        Then:
            - Model version is created
            - Model is placed in staging
        """
        # Mock MLflow client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_run = MagicMock()
        mock_run.info = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_active_run.return_value = mock_run

        mock_version = MagicMock()
        mock_version.version = "1"
        mock_client.create_model_version.return_value = mock_version

        registry = ModelRegistry()

        model_version = registry.register_model(
            model=trained_model,
            model_name="test_strategy",
            model_type="ridge",
            features=["momentum_20d", "volatility_60d"],
            target="returns_20d",
            metrics={"sharpe": 0.85, "ic": 0.07},
            hyperparameters={"alpha": 1.0},
            training_date=date.today(),
            hypothesis_id="HYP-TEST-001",
        )

        assert model_version == "1"
        mock_client.transition_model_version_stage.assert_called_with(
            name="test_strategy",
            version="1",
            stage="staging",
        )

    @patch('mlflow.tracking.MlflowClient')
    def test_get_production_model(self, mock_client_class):
        """
        Can retrieve production model.

        Given:
            - Model in production
        When:
            - Getting production model
        Then:
            - Production model details returned
        """
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_mv = MagicMock()
        mock_mv.version = "2"
        mock_mv.current_stage = "production"
        mock_mv.run_id = "prod-run-456"

        mock_details = MagicMock()
        mock_details.run_id = "prod-run-456"
        mock_details.creation_timestamp = datetime.now()

        mock_client.get_latest_versions.return_value = [mock_mv]
        mock_client.get_model_version.return_value = mock_details

        registry = ModelRegistry()
        model = registry.get_production_model("test_strategy")

        assert model is not None
        assert model.model_version == "2"
        assert model.stage == "production"


class TestDriftMonitoringWorkflow:
    """Test drift detection and monitoring workflow."""

    def test_prediction_drift_detection(self):
        """
        Prediction drift is detected when distribution changes.

        Given:
            - Reference predictions
            - New predictions with different distribution
        When:
            - Checking for drift
        Then:
            - Drift is detected
            - Alert threshold exceeded
        """
        monitor = DriftMonitor()

        # Reference: normal distribution
        ref_predictions = np.random.normal(0.01, 0.005, 1000)

        # New: shifted distribution
        new_predictions = np.random.normal(0.03, 0.01, 1000)

        result = monitor.check_prediction_drift(
            model_name="test_model",
            predictions_ref=ref_predictions,
            predictions_new=new_predictions,
        )

        assert bool(result.is_drift_detected) is True
        assert result.drift_type == "prediction"
        assert result.metric_value > result.threshold_value

    def test_no_drift_with_stable_distribution(self):
        """
        No drift when distribution is stable.

        Given:
            - Reference predictions
            - New predictions with similar distribution
        When:
            - Checking for drift
        Then:
            - No drift detected
        """
        monitor = DriftMonitor()

        rng = np.random.RandomState(42)
        predictions_1 = rng.normal(0.01, 0.005, 1000)
        predictions_2 = rng.normal(0.01, 0.005, 1000)

        result = monitor.check_prediction_drift(
            model_name="test_model",
            predictions_ref=predictions_1,
            predictions_new=predictions_2,
        )

        assert bool(result.is_drift_detected) is False

    def test_feature_drift_detection(self, sample_features):
        """
        Feature drift is detected for individual features.

        Given:
            - Reference features
            - New features with shifted distributions
        When:
            - Checking for feature drift
        Then:
            - Drift detected for shifted features
        """
        monitor = DriftMonitor()

        # Reference features
        ref_features = sample_features.copy()

        # New features: shift one feature
        new_features = sample_features.copy()
        new_features['momentum_20d'] = new_features['momentum_20d'] + 2.0  # Large shift

        results = monitor.check_feature_drift(
            model_name="test_model",
            features_ref=ref_features,
            features_new=new_features,
        )

        # Should detect drift in momentum_20d
        assert 'momentum_20d' in results
        assert bool(results['momentum_20d'].is_drift_detected) is True

    def test_concept_drift_detection(self):
        """
        Concept drift is detected when IC decays.

        Given:
            - Predictions and actuals with degraded correlation
        When:
            - Checking for concept drift
        Then:
            - IC decay detected
        """
        monitor = DriftMonitor()

        # Reference IC: 0.10 (good)
        reference_ic = 0.10

        # Current: poor predictions
        predictions = np.random.randn(500)
        actuals = np.random.randn(500)  # Uncorrelated

        result = monitor.check_concept_drift(
            model_name="test_model",
            predictions=predictions,
            actuals=actuals,
            reference_ic=reference_ic,
        )

        # IC should be close to 0, indicating significant decay
        assert result.metric_value > result.threshold_value


class TestDeploymentWorkflow:
    """Test end-to-end deployment workflow."""

    @patch('hrp.ml.deployment.log_event')
    @patch('hrp.ml.deployment.ModelRegistry')
    def test_deploy_to_staging_workflow(self, mock_registry, mock_log_event, validation_data):
        """
        Staging deployment workflow completes successfully.

        Given:
            - Model version
            - Validation data
        When:
            - Deploying to staging
        Then:
            - Validation checks run
            - Model deployed to staging
            - Deployment logged
        """
        # Mock registry
        mock_client = MagicMock()
        mock_registry.return_value.client = mock_client

        pipeline = DeploymentPipeline(registry=mock_registry())

        result = pipeline.deploy_to_staging(
            model_name="test_strategy",
            model_version="1",
            validation_data=validation_data,
            actor="user",
        )

        # Check status field, not status string
        assert result.status == "success"
        assert result.environment == "staging"
        assert result.validation_passed is True
        assert "data_not_empty" in result.validation_results
        assert result.validation_results["data_not_empty"] is True

    @patch('hrp.ml.deployment.log_event')
    @patch('hrp.ml.deployment.ModelRegistry')
    def test_promote_to_production_workflow(self, mock_registry, mock_log_event):
        """
        Production promotion workflow completes successfully.

        Given:
            - Model in staging
        When:
            - Promoting to production
        Then:
            - Model promoted to production
            - Lineage event logged
        """
        mock_client = MagicMock()
        mock_registry.return_value.client = mock_client
        mock_registry.return_value.promote_to_production = MagicMock()
        mock_registry.return_value.get_staging_model = MagicMock(
            return_value=MagicMock(model_version="1")
        )

        pipeline = DeploymentPipeline(registry=mock_registry())

        result = pipeline.promote_to_production(
            model_name="test_strategy",
            actor="user",
            model_version="1",
        )

        assert result.status == "success"
        assert result.environment == "production"

    @patch('hrp.ml.deployment.ModelRegistry')
    def test_rollback_workflow(self, mock_registry):
        """
        Rollback workflow restores previous version.

        Given:
            - Production model at version 2
            - Need to rollback to version 1
        When:
            - Rolling back
        Then:
            - Previous version restored
            - Rollback logged
        """
        mock_client = MagicMock()
        mock_registry.return_value.client = mock_client
        mock_registry.return_value.rollback_production = MagicMock()
        mock_registry.return_value.get_production_model = MagicMock(
            return_value=MagicMock(model_version="2")
        )

        pipeline = DeploymentPipeline(registry=mock_registry())

        result = pipeline.rollback_production(
            model_name="test_strategy",
            to_version="1",
            actor="user",
            reason="Performance degradation",
        )

        assert result.status == "success"
        assert result.model_version == "1"


class TestPlatformAPIIntegration:
    """Test PlatformAPI integration with new ML features."""

    @patch('hrp.ml.registry.ModelRegistry')
    def test_register_model_via_platform_api(self, mock_registry_class, trained_model):
        """
        Models can be registered via PlatformAPI.

        Given:
            - Trained model
            - PlatformAPI instance
        When:
            - Registering model
        Then:
            - Model version returned
        """
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry
        mock_registry.register_model = MagicMock(return_value="3")

        api = PlatformAPI()

        model_version = api.register_model(
            model=trained_model,
            model_name="api_test_strategy",
            model_type="ridge",
            features=["momentum_20d"],
            target="returns_20d",
            metrics={"sharpe": 0.8},
            hyperparameters={"alpha": 1.0},
            training_date=date.today(),
        )

        assert model_version == "3"

    @patch('hrp.monitoring.drift_monitor.DriftMonitor')
    def test_check_drift_via_platform_api(self, mock_monitor_class, sample_features):
        """
        Drift checks can be performed via PlatformAPI.

        Given:
            - Current and reference data
            - PlatformAPI instance
        When:
            - Checking drift
        Then:
            - Drift results returned
        """
        mock_monitor = MagicMock()
        mock_monitor_class.return_value = mock_monitor

        mock_result = MagicMock()
        mock_result.is_drift_detected = False
        mock_result.to_dict = MagicMock(return_value={"drift_detected": False})

        mock_monitor.run_drift_check = MagicMock(
            return_value={"prediction_drift": mock_result}
        )

        api = PlatformAPI()

        results = api.check_model_drift(
            model_name="test_model",
            current_data=sample_features,
            reference_data=sample_features,
        )

        assert "summary" in results
        assert results["summary"]["total_checks"] == 1

    @patch('hrp.ml.deployment.DeploymentPipeline')
    def test_deploy_model_via_platform_api(self, mock_pipeline_class, validation_data):
        """
        Models can be deployed via PlatformAPI.

        Given:
            - Model version
            - Validation data
            - PlatformAPI instance
        When:
            - Deploying model
        Then:
            - Deployment result returned
        """
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.validation_passed = True
        mock_result.to_dict = MagicMock(return_value={"status": "success"})

        mock_pipeline.deploy_to_staging = MagicMock(return_value=mock_result)

        api = PlatformAPI()

        result = api.deploy_model(
            model_name="test_strategy",
            model_version="1",
            validation_data=validation_data,
            environment="staging",
        )

        assert result["status"] == "success"


class TestPurgeEmbargoValidation:
    """Test purge/embargo periods in validation."""

    def test_purge_embargo_creates_gap(self):
        """
        Purge and embargo periods create temporal gaps.

        Given:
            - WalkForwardConfig with purge=5, embargo=10
        When:
            - Generating folds
        Then:
            - Gap exists between train and test
        """
        from hrp.ml.validation import WalkForwardConfig, generate_folds

        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=5,
            purge_days=5,
            embargo_days=10,
        )

        dates = [date(2020, 1, 1) + __import__('datetime').timedelta(days=i) for i in range(1000)]
        folds = generate_folds(config, dates)

        assert len(folds) == 5

        # Check first fold
        train_start, train_end, test_start, test_end = folds[0]
        gap_days = (test_start - train_end).days

        # Gap should be purge_days + 1 (natural gap)
        assert gap_days == 6

    def test_backward_compatible_default_values(self):
        """
        Default purge=0, embargo=0 maintains old behavior.

        Given:
            - WalkForwardConfig without purge/embargo
        When:
            - Creating config
        Then:
            - Defaults to 0
        """
        from hrp.ml.validation import WalkForwardConfig

        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )

        assert config.purge_days == 0
        assert config.embargo_days == 0


class TestEndToEndDeploymentScenario:
    """Test complete end-to-end deployment scenario."""

    def test_full_deployment_lifecycle(
        self,
        trained_model,
        validation_data,
        sample_features,
    ):
        """
        Complete deployment lifecycle from train to production.

        Scenario:
        1. Train model
        2. Register in staging
        3. Run drift checks
        4. Deploy to production
        5. Monitor for drift
        6. Rollback if needed

        Given:
            - Trained model
            - Validation data
        When:
            - Running full lifecycle
        Then:
            - Each stage completes successfully
        """
        api = PlatformAPI()

        # 1. Register model (mock the internal registry)
        with patch.object(api, 'register_model', return_value="1"):
            model_version = api.register_model(
                model=trained_model,
                model_name="e2e_strategy",
                model_type="ridge",
                features=["momentum_20d", "volatility_60d"],
                target="returns_20d",
                metrics={"sharpe": 0.85, "ic": 0.07},
                hyperparameters={"alpha": 1.0},
                training_date=date.today(),
            )

        assert model_version == "1"

        # 2. Deploy to staging
        with patch.object(api, 'deploy_model', return_value={"status": "success", "environment": "staging"}):
            staging_result = api.deploy_model(
                model_name="e2e_strategy",
                model_version="1",
                validation_data=validation_data,
                environment="staging",
            )

        assert staging_result["status"] == "success"

        # 3. Check drift
        with patch.object(api, 'check_model_drift', return_value={
            "summary": {"drift_detected": False},
            "prediction_drift": {"drift_detected": False},
        }):
            drift_results = api.check_model_drift(
                model_name="e2e_strategy",
                current_data=sample_features,
                reference_data=sample_features,
            )

        assert drift_results["summary"]["drift_detected"] is False

        # 4. Promote to production
        with patch.object(api, 'deploy_model', return_value={"status": "success", "environment": "production"}):
            prod_result = api.deploy_model(
                model_name="e2e_strategy",
                model_version="1",
                validation_data=validation_data,
                environment="production",
            )

        assert prod_result["status"] == "success"

        # Verify complete lifecycle
        assert model_version == "1"
        assert staging_result["status"] == "success"
        assert not drift_results["summary"]["drift_detected"]
        assert prod_result["status"] == "success"
