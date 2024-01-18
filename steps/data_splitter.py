from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def data_splitter(dataset: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Splitting the dataset...")
    try:
        dataset_trn, dataset_tst = train_test_split(dataset, test_size=test_size, random_state=42)
        return dataset_trn, dataset_tst
    except Exception as e:
        logger.error(f"Error during dataset splitting: {e}", exc_info=True)
        raise
