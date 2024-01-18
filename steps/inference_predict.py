from typing import Any
import pandas as pd
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def inference_predict(model: Any, dataset_inf: pd.DataFrame) -> pd.Series:
    logger.info("Starting prediction...")
    try:
        predictions = model.predict(dataset_inf)
        logger.info("Prediction completed.")
        return pd.Series(predictions, name="predicted")
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise
