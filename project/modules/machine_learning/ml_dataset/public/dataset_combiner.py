import pandas as pd
from typing import Iterable, Tuple

from .ml_dataset import MLDataset
from utils.timeseries import Duration


def merge_datasets_by_period(
    datasets: Iterable[Tuple[MLDataset, Duration]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """複数の ``MLDataset`` を期間指定で結合するヘルパー関数。

    Args:
        datasets: 各 ``MLDataset`` とその適用期間 ``Duration`` を ``(dataset, duration)``
            のタプルにして並べたイテラブル。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            0番目に ``pred_result_df``、1番目に ``raw_returns_df`` を、
            それぞれ時系列順に結合した ``DataFrame`` を返します。
    """
    pred_list = []
    raw_list = []

    for ds, period in datasets:
        pred_df = period.extract_from_df(ds.pred_result_df, ds.date_column)
        pred_list.append(pred_df)

        raw_df = period.extract_from_df(ds.raw_returns_df, ds.date_column)
        raw_list.append(raw_df)

    combined_pred = pd.concat(pred_list).sort_index()
    combined_raw = pd.concat(raw_list).sort_index()

    return combined_pred, combined_raw
