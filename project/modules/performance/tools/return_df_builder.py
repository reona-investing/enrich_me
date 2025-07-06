from __future__ import annotations

from datetime import datetime
import pandas as pd
from machine_learning.ml_dataset import MLDataset

class ReturnDataFrameBuilder:
    """Datasetからリターン計算用DataFrameを構築するクラス"""

    def __init__(self, dataset_root: str, start_date: datetime, end_date: datetime) -> None:
        self.dataset_root = dataset_root
        self.start_date = start_date
        self.end_date = end_date

    def build(self) -> pd.DataFrame:
        """予測値と生の目的変数を結合し期間で抽出したDataFrameを返す"""
        ml_dataset = MLDataset.from_files(dataset_path=self.dataset_root)
        pred_df = ml_dataset.pred_result_df
        raw_df = ml_dataset.raw_returns_df
        df = pd.merge(raw_df, pred_df[["Pred"]], how="right", left_index=True, right_index=True)
        df = df[(df.index.get_level_values("Date") >= self.start_date) &
                (df.index.get_level_values("Date") <= self.end_date)]
        return df
