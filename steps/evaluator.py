import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def evaluator(model: RegressorMixin, test_df: pd.DataFrame, target: str) -> float:
    logger.info("Evaluating the model...")
    try:
        features = test_df.drop(columns=[target])
        accuracy = model.score(features, test_df[target])
        logger.info(f"Model evaluation completed with accuracy: {accuracy}")
        return accuracy
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
        raise
