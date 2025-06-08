from __future__ import annotations

import pandas as pd

from .reducer import UMAPReducer
from .hdbscan_cluster import HDBSCANCluster
from .distance_assigner import EuclideanClusterAssigner


class SectorClusterer:
    """セクタークラスタリングを補助する互換用クラス"""

    def __init__(self, stock_list_df: pd.DataFrame) -> None:
        self.reducer = UMAPReducer()
        self.cluster = HDBSCANCluster()
        self.assigner = EuclideanClusterAssigner(stock_list_df)

    def apply_umap(
        self,
        df: pd.DataFrame,
        n_components: int = 15,
        n_neighbors: int = 5,
        min_dist: float = 0.1,
    ) -> pd.DataFrame:
        return self.reducer.fit_transform(df, n_components, n_neighbors, min_dist)

    def apply_hdbscan(
        self,
        df: pd.DataFrame,
        min_cluster_sizes: list[int] | None = None,
    ) -> pd.DataFrame:
        return self.cluster.fit(df, min_cluster_sizes)

    def determine_cluster_from_euclidean(
        self,
        df: pd.DataFrame,
        threshold: float = 0.03,
    ) -> pd.DataFrame:
        return self.assigner.assign(df, threshold)

    def analyze_cluster_distances(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        return self.assigner.analyze_distances(df, labels)
