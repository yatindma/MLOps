from typing import Any

import pandas as pd
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def inference_predict(
        model: Any,
        dataset_inf: pd.DataFrame,
) -> Annotated[pd.Series, "predictions"]:
    """
    Generate predictions using the trained model on the inference dataset.

    Parameters:
    model (Any): The trained model.
    dataset_inf (pd.DataFrame): The dataset for inference.

    Returns:
    Annotated[pd.Series, "predictions"]: A pandas Series of predictions.
    """
    try:
        logger.info("Starting prediction...")
        predictions = model.predict(dataset_inf)
        predictions = pd.Series(predictions, name="predicted")
        logger.info("Prediction completed.")
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        raise

    return predictions
