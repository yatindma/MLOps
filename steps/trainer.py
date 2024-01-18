from typing import Optional

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from typing_extensions import Annotated
from zenml import ArtifactConfig, step, get_step_context, log_artifact_metadata
from zenml.logger import get_logger
import mlflow

logger = get_logger(__name__)


@step
def trainer(
        train_df: pd.DataFrame,
        model_type: str = "RandomForestRegressor",
        target: Optional[str] = "target",
) -> tuple[
    Annotated[RegressorMixin, "trained_model"],
    Annotated[float, "training_acc"],
]:
    """
    Train a machine learning model based on the given training data.
    """
    # Check if the target column is in the DataFrame
    if target not in train_df.columns:
        raise ValueError(f"Target column '{target}' not found in training data.")

    # Initialize the model based on the model_type parameter
    if model_type == "RandomForestRegressor":
        model = RandomForestRegressor()
        logger.info("Random Forest Regressor selected as the model.")
    else:
        raise ValueError(f"Unknown model type '{model_type}'")

    # Separate the features and the target
    features = train_df.drop(columns=[target])
    target_data = train_df[target]

    # Fit the model
    logger.info("Starting model training...")
    model.fit(features, target_data)
    logger.info("Model training completed.")

    # get model accuracy
    accuracy = model.score(features, target_data)

    mlflow.log_metric("accuracy", accuracy)
    return model, accuracy
