"""
MLflow Model Registry for HRP.

Provides model lifecycle management including registration, versioning,
staging/production promotion, and rollback capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from loguru import logger

from hrp.research.lineage import log_event, EventType
from hrp.research.mlflow_utils import setup_mlflow


@dataclass
class RegisteredModel:
    """
    Represents a registered model version.

    Attributes:
        model_name: Name of the registered model
        model_version: Version string (e.g., "1", "2", "3")
        model_type: Type of model (e.g., "ridge", "lightgbm")
        stage: Current stage ("staging", "production", "archived")
        features: List of feature names used for training
        target: Target variable name
        metrics: Dictionary of performance metrics
        hyperparameters: Model hyperparameters
        training_date: Date when model was trained
        hypothesis_id: Associated hypothesis ID if applicable
        run_id: MLflow run ID for this model
        registered_at: Timestamp when model was registered
    """

    model_name: str
    model_version: str
    model_type: str
    stage: str
    features: list[str]
    target: str
    metrics: dict[str, float]
    hyperparameters: dict[str, Any]
    training_date: date
    hypothesis_id: Optional[str]
    run_id: str
    registered_at: Optional[str] = None


class ModelRegistry:
    """
    MLflow Model Registry integration for model lifecycle management.

    Provides methods for:
    - Registering trained models
    - Promoting models between stages (staging → production)
    - Rolling back to previous versions
    - Querying model history

    Example:
        ```python
        from hrp.ml.registry import ModelRegistry

        registry = ModelRegistry()

        # Register a trained model
        model_version = registry.register_model(
            model=trained_model,
            model_name="momentum_strategy",
            model_type="ridge",
            features=["momentum_20d", "volatility_60d"],
            target="returns_20d",
            metrics={"sharpe": 0.85, "ic": 0.07},
            hyperparameters={"alpha": 1.0},
            training_date=date.today(),
            hypothesis_id="HYP-2026-001",
        )

        # Promote to production
        registry.promote_to_production(
            model_name="momentum_strategy",
            model_version=model_version,
            actor="user",
        )
        ```
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize Model Registry.

        Args:
            tracking_uri: MLflow tracking URI (default: local sqlite)
        """
        setup_mlflow(tracking_uri)
        self._client = None

    @property
    def client(self):
        """Lazy-load MLflow client."""
        if self._client is None:
            import mlflow

            self._client = mlflow.tracking.MlflowClient()
        return self._client

    def register_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        features: list[str],
        target: str,
        metrics: dict[str, float],
        hyperparameters: dict[str, Any],
        training_date: date,
        hypothesis_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> str:
        """
        Register a trained model in the Model Registry.

        Creates a new model version in MLflow Model Registry with
        associated metadata. The model is initially placed in "staging" stage.

        Args:
            model: Trained model object (scikit-learn, XGBoost, LightGBM)
            model_name: Name for the registered model
            model_type: Type of model (e.g., "ridge", "lightgbm")
            features: List of feature names used for training
            target: Target variable name
            metrics: Dictionary of performance metrics (sharpe, ic, etc.)
            hyperparameters: Model hyperparameters
            training_date: Date when model was trained
            hypothesis_id: Associated hypothesis ID if applicable
            experiment_id: MLflow experiment ID for the run

        Returns:
            model_version: Version string of the registered model

        Raises:
            ValueError: If model_name is empty or model cannot be serialized
        """
        import mlflow
        import mlflow.sklearn

        if not model_name:
            raise ValueError("model_name cannot be empty")

        logger.info(f"Registering model: {model_name} (type: {model_type})")

        # Create or get model in registry
        try:
            model_uri = f"runs:/{experiment_id}/model" if experiment_id else None
            if model_uri is None:
                # Log model to MLflow first
                with mlflow.start_run(nested=True):
                    mlflow.sklearn.log_model(model, "model")
                    run_id = mlflow.active_run().info.run_id
                    model_uri = f"runs:/{run_id}/model"
            else:
                run_id = experiment_id

            # Register model
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
            )

            version_str = str(model_version.version)

            # Transition to staging
            self.client.transition_model_version_stage(
                name=model_name,
                version=version_str,
                stage="staging",
            )

            # Log model registration to lineage
            log_event(
                event_type=EventType.EXPERIMENT_LINKED
                if hypothesis_id
                else EventType.EXPERIMENT_RUN,
                actor="user",
                details={
                    "model_name": model_name,
                    "model_version": version_str,
                    "model_type": model_type,
                    "features": features,
                    "target": target,
                    "metrics": metrics,
                    "hyperparameters": hyperparameters,
                    "training_date": str(training_date),
                },
                hypothesis_id=hypothesis_id,
                experiment_id=run_id,
            )

            logger.info(
                f"Model registered: {model_name} version {version_str} -> staging"
            )

            return version_str

        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            raise ValueError(f"Model registration failed: {e}") from e

    def get_production_model(self, model_name: str) -> Optional[RegisteredModel]:
        """
        Get the current production model version.

        Args:
            model_name: Name of the registered model

        Returns:
            RegisteredModel if production version exists, None otherwise
        """
        try:
            versions = self.client.get_latest_versions(
                name=model_name, stages=["production"]
            )

            if not versions:
                logger.warning(f"No production version found for {model_name}")
                return None

            latest = versions[0]
            model_version_details = self.client.get_model_version(
                name=model_name, version=latest.version
            )

            return RegisteredModel(
                model_name=model_name,
                model_version=latest.version,
                model_type="",  # Would need to fetch from run metadata
                stage="production",
                features=[],
                target="",
                metrics={},
                hyperparameters={},
                training_date=None,
                hypothesis_id=None,
                run_id=model_version_details.run_id,
                registered_at=str(model_version_details.creation_timestamp),
            )

        except Exception as e:
            logger.error(f"Failed to get production model {model_name}: {e}")
            return None

    def promote_to_production(
        self,
        model_name: str,
        model_version: str,
        actor: str,
        validation_checks: Optional[dict[str, bool]] = None,
    ) -> None:
        """
        Promote a staging model to production.

        Args:
            model_name: Name of the registered model
            model_version: Version string to promote
            actor: Who is promoting (e.g., "user", "agent:ml_scientist")
            validation_checks: Optional dict of validation check results

        Raises:
            ValueError: If model version not found or already in production
        """
        logger.info(f"Promoting {model_name} version {model_version} to production")

        try:
            # Transition to production
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="production",
                archive_existing_versions=True,
            )

            # Log deployment to lineage
            log_event(
                event_type=EventType.DEPLOYMENT_APPROVED,
                actor=actor,
                details={
                    "model_name": model_name,
                    "model_version": model_version,
                    "stage": "production",
                    "validation_checks": validation_checks or {},
                },
            )

            logger.info(f"Model promoted: {model_name} version {model_version} -> production")

        except Exception as e:
            logger.error(f"Failed to promote model {model_name} v{model_version}: {e}")
            raise ValueError(f"Model promotion failed: {e}") from e

    def rollback_production(
        self,
        model_name: str,
        to_version: str,
        actor: str,
        reason: str,
    ) -> None:
        """
        Rollback production to a previous model version.

        Args:
            model_name: Name of the registered model
            to_version: Version string to rollback to
            actor: Who is rolling back
            reason: Reason for rollback

        Raises:
            ValueError: If target version not found
        """
        logger.info(f"Rolling back {model_name} to version {to_version}")

        try:
            # Get current production version
            current = self.get_production_model(model_name)

            # Demote current production to archived
            if current:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=current.model_version,
                    stage="archived",
                )

            # Promote target version to production
            self.client.transition_model_version_stage(
                name=model_name,
                version=to_version,
                stage="production",
                archive_existing_versions=False,  # Already archived above
            )

            # Log rollback to lineage
            log_event(
                event_type=EventType.VALIDATION_FAILED,
                actor=actor,
                details={
                    "model_name": model_name,
                    "from_version": current.model_version if current else None,
                    "to_version": to_version,
                    "reason": reason,
                },
            )

            logger.info(f"Rollback complete: {model_name} -> version {to_version}")

        except Exception as e:
            logger.error(f"Failed to rollback model {model_name}: {e}")
            raise ValueError(f"Rollback failed: {e}") from e

    def get_model_history(self, model_name: str) -> list[RegisteredModel]:
        """
        Get the version history for a model.

        Args:
            model_name: Name of the registered model

        Returns:
            List of RegisteredModel objects (ordered by version, newest first)
        """
        try:
            versions = self.client.get_latest_versions(name=model_name)

            history = []
            for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
                model_version_details = self.client.get_model_version(
                    name=model_name, version=v.version
                )
                history.append(
                    RegisteredModel(
                        model_name=model_name,
                        model_version=v.version,
                        model_type="",
                        stage=v.current_stage,
                        features=[],
                        target="",
                        metrics={},
                        hyperparameters={},
                        training_date=None,
                        hypothesis_id=None,
                        run_id=model_version_details.run_id,
                        registered_at=str(model_version_details.creation_timestamp),
                    )
                )

            return history

        except Exception as e:
            logger.error(f"Failed to get model history for {model_name}: {e}")
            return []

    def get_staging_model(self, model_name: str) -> Optional[RegisteredModel]:
        """
        Get the current staging model version.

        Args:
            model_name: Name of the registered model

        Returns:
            RegisteredModel if staging version exists, None otherwise
        """
        try:
            versions = self.client.get_latest_versions(name=model_name, stages=["staging"])

            if not versions:
                return None

            latest = versions[0]
            return RegisteredModel(
                model_name=model_name,
                model_version=latest.version,
                model_type="",
                stage="staging",
                features=[],
                target="",
                metrics={},
                hyperparameters={},
                training_date=None,
                hypothesis_id=None,
                run_id=latest.run_id,
                registered_at=None,
            )

        except Exception as e:
            logger.error(f"Failed to get staging model {model_name}: {e}")
            return None


def get_model_registry(tracking_uri: Optional[str] = None) -> ModelRegistry:
    """
    Get a ModelRegistry instance.

    Args:
        tracking_uri: MLflow tracking URI (default: local sqlite)

    Returns:
        ModelRegistry instance
    """
    return ModelRegistry(tracking_uri=tracking_uri)
