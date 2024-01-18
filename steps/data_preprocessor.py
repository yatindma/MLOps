from typing import List, Optional, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from zenml import step
from zenml.logger import get_logger
from utils.preprocess import ColumnsDropper, NADropper

logger = get_logger(__name__)

@step
def data_preprocessor(dataset_trn: pd.DataFrame, dataset_tst: pd.DataFrame, drop_na: Optional[bool] = None, drop_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    logger.info("Starting data preprocessing...")
    try:
        preprocess_pipeline = Pipeline([("passthrough", "passthrough")])
        if drop_na:
            preprocess_pipeline.steps.append(("drop_na", NADropper()))
        if drop_columns:
            preprocess_pipeline.steps.append(("drop_columns", ColumnsDropper(drop_columns)))

        dataset_trn = preprocess_pipeline.fit_transform(dataset_trn)
        dataset_tst = preprocess_pipeline.transform(dataset_tst)
        return dataset_trn, dataset_tst, preprocess_pipeline
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}", exc_info=True)
        raise
