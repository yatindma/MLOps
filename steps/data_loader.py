import pandas as pd
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self):
        pass

    def get_data(self):
        try:
            data = pd.read_csv("./creditcard.csv")
            return data.head(200)
        except Exception as e:
            logger.error(f"An error occurred while ingesting data: {e}")
            raise


@step
def data_loader(
        random_state: int, is_inference: bool = False, target: str = "target"
) -> pd.DataFrame:
    """
    Load dataset for training or inference.

    Parameters:
    random_state (int): Seed for random number generation.
    is_inference (bool): Whether the data is for inference.
    target (str): The name of the target column.

    Returns:
    pd.DataFrame: The loaded dataset.
    """
    try:
        logger.info("Initializing data ingestion...")
        ing_obj = IngestData()
        dataset = ing_obj.get_data()

        inference_size = int(len(dataset) * 0.05)
        inference_subset = dataset.sample(inference_size, random_state=random_state)

        if is_inference:
            dataset = inference_subset
        else:
            dataset.drop(inference_subset.index, inplace=True)
        dataset.reset_index(drop=True, inplace=True)

        logger.info(f"Dataset with {len(dataset)} records loaded for {'inference' if is_inference else 'training'}.")
    except Exception as e:
        logger.error(f"An error occurred during data loading: {e}")
        raise

    return dataset
