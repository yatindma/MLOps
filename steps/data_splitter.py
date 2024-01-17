from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def data_splitter(
        dataset: pd.DataFrame,
        test_size: float = 0.2
) -> Tuple[
    Annotated[pd.DataFrame, "raw_dataset_trn"],
    Annotated[pd.DataFrame, "raw_dataset_tst"]
]:
    """
    Split the dataset into training and testing sets.

    Parameters:
    dataset (pd.DataFrame): The full dataset to split.
    test_size (float): Proportion of the dataset to include in the test split.

    Returns:
    Tuple: The training and testing datasets.
    """
    try:
        logger.info("Splitting the dataset...")
        dataset_trn, dataset_tst = train_test_split(
            dataset,
            test_size=test_size,
            random_state=42,
            shuffle=True
        )
        dataset_trn = pd.DataFrame(dataset_trn, columns=dataset.columns)
        dataset_tst = pd.DataFrame(dataset_tst, columns=dataset.columns)
        logger.info("Dataset splitting completed.")
    except Exception as e:
        logger.error(f"An error occurred during dataset splitting: {e}")
        raise

    return dataset_trn, dataset_tst
