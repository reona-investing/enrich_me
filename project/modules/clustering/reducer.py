from __future__ import annotations

import pandas as pd
import umap


class UMAPReducer:
    """UMAP を用いた次元削減処理を行うクラス"""

    def fit_transform(
        self,
        df: pd.DataFrame,
        n_components: int = 15,
        n_neighbors: int = 5,
        min_dist: float = 0.1,
    ) -> pd.DataFrame:
        """UMAP により次元削減を行い、結果の DataFrame を返す"""
        model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        embedding = model.fit_transform(df)
        return pd.DataFrame(
            embedding,
            index=df.index,
            columns=[f"Feature{i}" for i in range(n_components)],
        )
