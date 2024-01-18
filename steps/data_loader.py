import pandas as pd
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

class IngestData:
    def get_data(self):
        try:
            data = pd.read_csv("./creditcard.csv")
            return data.head(200)
        except Exception as e:
            logger.error(f"Error ingesting data: {e}", exc_info=True)
            raise

@step
def data_loader(random_state: int, is_inference: bool = False, target: str = "target") -> pd.DataFrame:
    logger.info(f"Loading data for {'inference' if is_inference else 'training'}...")
    ing_obj = IngestData()
    dataset = ing_obj.get_data()
    inference_size = int(len(dataset) * 0.05)
    inference_subset = dataset.sample(inference_size, random_state=random_state)
    if is_inference:
        dataset = inference_subset
    else:
        dataset.drop(inference_subset.index, inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    logger.info(f"Dataset loaded with {len(dataset)} records.")
    return dataset
