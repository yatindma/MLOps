# {% include 'template/license_header' %}

from typing import Optional

from zenml import pipeline
from zenml.logger import get_logger

from pipelines import (
    feature_engineering,
)
from steps import trainer, evaluator, model_promoter

logger = get_logger(__name__)


@pipeline
def training(
        target: Optional[str] = "target",
        model_type: Optional[str] = "RandomForestRegressor"
):
    dataset_trn, dataset_tst = feature_engineering()

    model = trainer(
        train_df=dataset_trn,
        target=target,
        model_type=model_type
    )

    acc = evaluator(
        model=model,
        test_df=dataset_tst,
        target=target,
    )
    is_promoted = model_promoter(accuracy=acc)
    if is_promoted:
        logger.info(f"Model promoted with accuracy {acc}")
    return is_promoted
