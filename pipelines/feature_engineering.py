from typing import Optional, List

import pandas as pd
from zenml import pipeline
from zenml.logger import get_logger

from steps import (
    data_loader,
    data_preprocessor,
    data_splitter,
)

logger = get_logger(__name__)


@pipeline
def feature_engineering(
        test_size: float = 0.2,
        drop_na: Optional[bool] = None,
        drop_columns: Optional[List[str]] = None,
        target: Optional[str] = "target",
        random_state: int = 17,
) -> (pd.DataFrame, pd.DataFrame):
    print("target", target)
    print("test_size", test_size)
    """
    Feature engineering pipeline.
    :param test_size:
    :param drop_na:
    :param drop_columns:
    :param target:
    :param random_state:
    :return:
    """
    raw_data = data_loader(random_state=random_state, target=target)
    dataset_trn, dataset_tst = data_splitter(
        dataset=raw_data,
        test_size=test_size,
    )
    dataset_trn, dataset_tst, _ = data_preprocessor(
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        drop_na=drop_na,
        drop_columns=drop_columns,
        target=target,
        random_state=random_state,
    )

    return dataset_trn, dataset_tst
