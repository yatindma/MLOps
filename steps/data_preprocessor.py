from typing import List, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step
from zenml.logger import get_logger

from utils.preprocess import ColumnsDropper, NADropper

logger = get_logger(__name__)


@step
def data_preprocessor(
        random_state: int,
        dataset_trn: pd.DataFrame,
        dataset_tst: pd.DataFrame,
        drop_na: Optional[bool] = None,
        drop_columns: Optional[List[str]] = None,
        target: Optional[str] = "Class",
) -> Tuple[
    Annotated[pd.DataFrame, "dataset_trn"],
    Annotated[pd.DataFrame, "dataset_tst"],
    Annotated[Pipeline, "preprocess_pipeline"],
]:
    """
    Preprocesses the training and testing datasets.

    Parameters:
    random_state (int): Seed for random number generation.
    dataset_trn (pd.DataFrame): Training dataset.
    dataset_tst (pd.DataFrame): Testing dataset.
    drop_na (Optional[bool]): Whether to drop rows with N/A values.
    drop_columns (Optional[List[str]]): List of columns to drop.
    target (Optional[str]): Target column name.

    Returns:
    Tuple: Preprocessed training and testing datasets, and the preprocessing pipeline.
    """
    try:
        logger.info("Initializing preprocessing pipeline...")
        preprocess_pipeline = Pipeline([("passthrough", "passthrough")])
        if drop_na:
            preprocess_pipeline.steps.append(("drop_na", NADropper()))
        if drop_columns:
            preprocess_pipeline.steps.append(("drop_columns", ColumnsDropper(drop_columns)))

        logger.info("Applying preprocessing to training dataset...")
        dataset_trn = preprocess_pipeline.fit_transform(dataset_trn)
        logger.info("Applying preprocessing to testing dataset...")
        dataset_tst = preprocess_pipeline.transform(dataset_tst)

        log_artifact_metadata(
            artifact_name="dataset_trn",
            metadata={
                "random_state": random_state,
                "target": target,
            },
        )
    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {e}")
        raise

    return dataset_trn, dataset_tst, preprocess_pipeline
