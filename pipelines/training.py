from typing import Optional
from zenml import pipeline
from zenml.logger import get_logger
from pipelines import feature_engineering
from steps import trainer, evaluator, model_promoter

logger = get_logger(__name__)

@pipeline
def training(target: Optional[str] = "target", model_type: Optional[str] = "RandomForestRegressor"):
    logger.info("Initiating the training pipeline.")
    try:
        dataset_trn, dataset_tst = feature_engineering()
        model, train_acc = trainer(train_df=dataset_trn, target=target, model_type=model_type)
        acc = evaluator(model=model, test_df=dataset_tst, target=target)
        is_promoted = model_promoter(accuracy=acc)
        logger.info(f"Model {'promoted' if is_promoted else 'not promoted'} with accuracy {acc}")
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}", exc_info=True)
        raise
