from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import hdbscan


class HDBSCANCluster:
    """HDBSCAN を用いたクラスタリング処理を担うクラス"""

    def fit(
        self,
        df: pd.DataFrame,
        min_cluster_sizes: list[int] | None = None,
    ) -> pd.DataFrame:
        """複数パラメータでクラスタリングを実行し、シルエット係数で最良の結果を返す"""
        if min_cluster_sizes is None:
            min_cluster_sizes = [5, 10, 15]

        best_labels: np.ndarray | None = None
        best_score = -1.0

        for size in min_cluster_sizes:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=size)
            labels = clusterer.fit_predict(df)
            if len(np.unique(labels)) <= 1:
                continue
            score = silhouette_score(df, labels)
            if score > best_score:
                best_score = score
                best_labels = labels

        if best_labels is None:
            best_labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_sizes[0]).fit_predict(df)

        return pd.DataFrame({"Cluster": best_labels}, index=df.index)

    def fit_recursive(
        self,
        df: pd.DataFrame,
        min_cluster_sizes: list[int] | None = None,
    ) -> pd.DataFrame:
        """再帰的にクラスタリングを実行し、各段階のラベルを保持する"""

        result = pd.DataFrame(index=df.index)

        def _recurse(indexes: pd.Index, depth: int) -> None:
            sub_df = df.loc[indexes]
            labels = self.fit(sub_df, min_cluster_sizes)["Cluster"]
            result.loc[indexes, f"Level{depth}"] = labels
            if labels.nunique() <= 1:
                return
            for lbl in labels.unique():
                members = indexes[labels == lbl]
                if len(members) <= 1:
                    continue
                _recurse(members, depth + 1)

        _recurse(df.index, 0)
        final = pd.factorize(result.fillna(-1).apply(tuple, axis=1))[0]
        result["Cluster"] = final
        return result

