# model_evaluation/evaluator.py
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def evaluator(
        model: RegressorMixin,
        test_df: pd.DataFrame,
        target: str
) -> float:
    """
    Evaluate the model's performance on the test dataset.

    Parameters:
    model (RandomForestClassifier): The trained model.
    test_df (pd.DataFrame): The test dataset.
    target (str): The name of the target variable.

    Returns:
    float: The accuracy of the model.
    """
    try:
        logger.info("Evaluating the model...")
        accuracy = model.score(test_df.drop(columns=[target], axis=1), test_df[target])
        logger.info(f"Model evaluation completed with accuracy: {accuracy}")
    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {e}")
        raise

    return accuracy
