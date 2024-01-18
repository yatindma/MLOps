import pandas as pd
from sklearn.pipeline import Pipeline
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)
@step
def inference_preprocessor(dataset_inf: pd.DataFrame, preprocess_pipeline: Pipeline, target: str) -> pd.DataFrame:
    logger.info("Preprocessing inference dataset...")
    try:
        dataset_inf[target] = [0] * len(dataset_inf)
        processed_inf = preprocess_pipeline.transform(dataset_inf)
        processed_inf.drop(columns=[target], inplace=True)
        return processed_inf
    except Exception as e:
        logger.error(f"Error during inference preprocessing: {e}", exc_info=True)
        raise
