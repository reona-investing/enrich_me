from __future__ import annotations

from datetime import datetime
import pandas as pd
from models.machine_learning.loaders import DatasetLoader

class ReturnDataFrameBuilder:
    """Datasetからリターン計算用DataFrameを構築するクラス"""

    def __init__(self, dataset_root: str, start_date: datetime, end_date: datetime) -> None:
        self.dataset_root = dataset_root
        self.start_date = start_date
        self.end_date = end_date

    def build(self) -> pd.DataFrame:
        """予測値と生の目的変数を結合し期間で抽出したDataFrameを返す"""
        loader = DatasetLoader(dataset_root=self.dataset_root)
        pred_df = loader.load_pred_results()
        raw_df = loader.load_raw_targets()
        df = pd.merge(raw_df, pred_df[["Pred"]], how="right", left_index=True, right_index=True)
        df = df[(df.index.get_level_values("Date") >= self.start_date) &
                (df.index.get_level_values("Date") <= self.end_date)]
        return df
