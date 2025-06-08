from __future__ import annotations

import pandas as pd

from .reducer import UMAPReducer
from .hdbscan_cluster import HDBSCANCluster
from .distance_assigner import EuclideanClusterAssigner


class SectorClusteringPipeline:
    """UMAP -> HDBSCAN -> 距離解析 をまとめたパイプライン"""

    def __init__(self, stock_list_df: pd.DataFrame) -> None:
        self.reducer = UMAPReducer()
        self.cluster = HDBSCANCluster()
        self.assigner = EuclideanClusterAssigner(stock_list_df)

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """パイプライン全体を実行し、最終的なクラスタ付与結果を返す"""
        reduced = self.reducer.fit_transform(df)
        clustered = self.cluster.fit(reduced)
        return self.assigner.assign(clustered)
