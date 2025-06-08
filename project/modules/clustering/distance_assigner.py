from __future__ import annotations

import pandas as pd
import networkx as nx
from sklearn.metrics import pairwise_distances


class EuclideanClusterAssigner:
    """距離行列を解析してクラスタ割り当てを行うクラス"""

    def __init__(self, stock_list_df: pd.DataFrame) -> None:
        self.stock_list_df = stock_list_df

    def assign(
        self,
        df: pd.DataFrame,
        threshold: float = 0.03,
    ) -> pd.DataFrame:
        """ユークリッド距離の閾値に基づきクラスタを付与する"""
        dist = pairwise_distances(df.values, metric="euclidean")
        codes = df.index.tolist()
        graph = nx.Graph()
        graph.add_nodes_from(codes)

        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                if dist[i, j] <= threshold:
                    graph.add_edge(codes[i], codes[j])

        clusters = {code: idx for idx, comp in enumerate(nx.connected_components(graph)) for code in comp}
        cluster_df = pd.DataFrame({"Code": codes, "Cluster": [clusters.get(c, -1) for c in codes]}).set_index("Code")

        merged = (
            cluster_df
            .merge(self.stock_list_df, how="left", left_index=True, right_on="Code")
            .set_index("Code")
            .sort_values(["Cluster", "Code"])
        )

        return merged[
            [
                "CompanyName",
                "MarketCodeName",
                "Sector33CodeName",
                "Sector17CodeName",
                "ScaleCategory",
                "Listing",
                "Cluster",
            ]
        ]
