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
        """再帰的にクラスタリングを実行し、分割不能になるまで細分化する"""

        final_labels = pd.Series(index=df.index, dtype=int)
        next_label = 0

        def _split(sub_df: pd.DataFrame) -> None:
            nonlocal next_label
            result = self.fit(sub_df, min_cluster_sizes)["Cluster"]
            unique_labels = sorted(result.unique())

            if len(unique_labels) <= 1:
                final_labels.loc[sub_df.index] = next_label
                next_label += 1
                return

            for lbl in unique_labels:
                members = sub_df.index[result == lbl]
                if len(members) <= 1:
                    final_labels.loc[members] = next_label
                    next_label += 1
                    continue
                sub_df_slice = sub_df.loc[members]
                sub_res = self.fit(sub_df_slice, min_cluster_sizes)["Cluster"]
                if len(np.unique(sub_res)) <= 1:
                    final_labels.loc[members] = next_label
                    next_label += 1
                else:
                    _split(sub_df_slice)

        _split(df)
        return pd.DataFrame({"Cluster": final_labels})
