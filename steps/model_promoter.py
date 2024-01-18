from zenml import get_step_context, step
from zenml.logger import get_logger
from zenml.client import Client
logger = get_logger(__name__)


@step
def model_promoter(accuracy: float, stage: str = "production") -> bool:
    """
    Promotes a model to a specified stage based on its accuracy.

    Args:
        accuracy (float): The accuracy of the current model.
        stage (str): The stage to which the model will be promoted, defaults to 'production'.

    Returns:
        bool: True if the model is promoted, False otherwise.
    """
    promotion_threshold = 0.8
    if accuracy < promotion_threshold:
        logger.info(f"Model not promoted due to lower accuracy: {accuracy:.2f}")
        return False
    try:
        current_model_version = get_step_context().model_version
        current_model_version.set_stage(stage, force=True)
    except Exception as e:
        logger.error("Failed to get previous model version", exc_info=True)
    logger.info(f"Model promoted with accuracy: {accuracy:.2f}")
    return True
