from typing import Optional, List
import pandas as pd
from zenml import pipeline
from zenml.logger import get_logger
from steps import data_loader, data_preprocessor, data_splitter

logger = get_logger(__name__)

@pipeline
def feature_engineering(test_size: float = 0.2, drop_na: Optional[bool] = None, drop_columns: Optional[List[str]] = None, target: Optional[str] = "target", random_state: int = 17) -> (pd.DataFrame, pd.DataFrame):
    """
    Feature engineering pipeline.
    """
    logger.info("Starting the feature engineering pipeline.")
    try:
        raw_data = data_loader(random_state=random_state, target=target)
        dataset_trn, dataset_tst = data_splitter(dataset=raw_data, test_size=test_size)
        dataset_trn, dataset_tst, _ = data_preprocessor(dataset_trn=dataset_trn, dataset_tst=dataset_tst, drop_na=drop_na, drop_columns=drop_columns)
        logger.info("Feature engineering pipeline completed.")
        return dataset_trn, dataset_tst
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {e}", exc_info=True)
        raise
