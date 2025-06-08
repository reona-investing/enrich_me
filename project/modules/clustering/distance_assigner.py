from __future__ import annotations

import pandas as pd
import numpy as np
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

    def analyze_distances(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        """各クラスタ内の平均距離を算出する"""
        dist = pairwise_distances(df.values, metric="euclidean")
        dist_df = pd.DataFrame(dist, index=df.index, columns=df.index)
        results = []
        for cluster in sorted(labels.unique()):
            members = labels.index[labels == cluster]
            if len(members) <= 1:
                mean_dist = 0.0
            else:
                sub = dist_df.loc[members, members]
                mean_dist = float(sub.where(~np.eye(len(sub), dtype=bool)).mean().mean())
            results.append({"Cluster": cluster, "MeanDistance": mean_dist})
        return pd.DataFrame(results).set_index("Cluster")
