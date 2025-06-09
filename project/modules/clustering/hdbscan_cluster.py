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
        metric: str = 'euclidean',
        **kwargs,
    ) -> pd.DataFrame:
        """複数パラメータでクラスタリングを実行し、シルエット係数で最良の結果を返す"""
        if min_cluster_sizes is None:
            min_cluster_sizes = [5, 10, 15]

        best_labels: np.ndarray | None = None
        best_score = -1.0

        for size in min_cluster_sizes:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=size, metric=metric, **kwargs)
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
        metric: str = 'euclidean',
        **kwargs,
    ) -> pd.DataFrame:
        """再帰的にクラスタリングを実行し、各段階のラベルを保持する"""

        result = pd.DataFrame(index=df.index)

        def _recurse(indexes: pd.Index, depth: int, metric: str, **kwargs) -> None:
            sub_df = df.loc[indexes]
            labels = self.fit(sub_df, min_cluster_sizes, metric, **kwargs)["Cluster"]
            result.loc[indexes, f"Level{depth}"] = labels
            if labels.nunique() <= 1:
                return
            for lbl in labels.unique():
                members = indexes[labels == lbl]
                if len(members) <= 1:
                    continue
                _recurse(members, depth + 1, metric, **kwargs)

        _recurse(df.index, 0, metric, **kwargs)
        
        # 各Level列を昇順に並び替え
        level_cols = [col for col in result.columns if col.startswith('Level')]
        sorted_result = result[level_cols].copy()
        
        # 各Level列について、NaNを適切に処理しながら昇順ソート
        for col in level_cols:
            # 各列の値を取得し、NaNでない値のみでソート用のマッピングを作成
            unique_vals = sorted_result[col].dropna().unique()
            unique_vals_sorted = sorted(unique_vals)
            
            # 値のマッピング辞書を作成（NaNは-1のまま保持）
            value_mapping = {val: i for i, val in enumerate(unique_vals_sorted)}
            value_mapping[-1] = -1  # ノイズラベルは-1のまま保持
            
            # マッピングを適用
            sorted_result[col] = sorted_result[col].map(value_mapping).fillna(-1)
        
        # ソート済みの結果でCluster列を生成
        sorted_result = sorted_result.sort_values(level_cols)
        final = pd.factorize(sorted_result.fillna(-1).apply(tuple, axis=1))[0]
        
        # 元のresultにソート済みの列とCluster列を設定
        sorted_result["Cluster"] = final
        
        return sorted_result

