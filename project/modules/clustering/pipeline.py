from __future__ import annotations

import pandas as pd

from .reducer import UMAPReducer
from .hdbscan_cluster import HDBSCANCluster


class SectorClusteringPipeline:
    """UMAP -> HDBSCAN -> 距離解析 をまとめたパイプライン"""

    def __init__(self, stock_list_df: pd.DataFrame) -> None:
        self.reducer = UMAPReducer()
        self.cluster = HDBSCANCluster()
        self.stock_list_df = stock_list_df

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """パイプライン全体を実行し、最終的なクラスタ付与結果を返す"""
        reduced = self.reducer.fit_transform(df)
        labels = self.cluster.fit_recursive(reduced)
        merged = (
            labels
            .merge(self.stock_list_df, how="left", left_index=True, right_on="Code")
            .set_index("Code")
        )
        info_cols = [
            "CompanyName",
            "MarketCodeName",
            "Sector33CodeName",
            "Sector17CodeName",
            "ScaleCategory",
            "Listing",
        ]
        return merged[info_cols + list(labels.columns)]
