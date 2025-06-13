"""Stepwise sector clustering using built modules"""
import pandas as pd
from pathlib import Path

from project.modules.utils.paths import Paths
from project.modules.clustering import (
    PCAResidualExtractor,
    UMAPReducer,
    HDBSCANCluster,
)
from sklearn.decomposition import PCA


def compare_rows(df: pd.DataFrame) -> pd.DataFrame:
    """各行同士を比較し一致する列の数をカウント"""
    result = pd.DataFrame(index=df.index, columns=df.index, dtype=int)
    for idx1 in df.index:
        for idx2 in df.index:
            result.loc[idx1, idx2] = (df.loc[idx1] == df.loc[idx2]).sum()
    return result


def main() -> None:
    # 入力データ
    pca_residue_csv = Path(Paths.EXECUTION_SCRIPTS_FOLDER) / "clustering" / "pca_residue.csv"
    stock_list_path = Path(Paths.STOCK_LIST_PARQUET)

    residue_df = pd.read_csv(pca_residue_csv, index_col="Code")
    stock_list_df = pd.read_parquet(stock_list_path)

    residue_df = residue_df.dropna(axis=1)

    dfs = []
    for n_comp in range(10, 26):
        # PCAで次元を圧縮・復元
        pca = PCA(n_components=n_comp)
        pca_array = pca.fit_transform(residue_df)
        inversed = pca.inverse_transform(pca_array)
        reconstructed = pd.DataFrame(inversed, index=residue_df.index, columns=residue_df.columns)
        extractor = PCAResidualExtractor(remove_components=0)
        residuals = extractor.fit_transform(reconstructed)

        # UMAP次元削減
        reducer = UMAPReducer()
        reduced = reducer.fit_transform(residuals, n_components=2, n_neighbors=2, min_dist=0.01)

        # HDBSCANクラスタリング
        clusterer = HDBSCANCluster()
        labels = clusterer.fit(reduced, min_cluster_sizes=[2], metric="euclidean")
        dfs.append(labels.rename(columns={"Cluster": f"Cluster{n_comp}"}))

    cluster_df = pd.concat(dfs, axis=1)
    merged = (
        cluster_df
        .merge(stock_list_df, how="left", left_index=True, right_on="Code")
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
    merged[info_cols + list(cluster_df.columns)].to_csv("Clusters.csv")

    result = compare_rows(cluster_df)
    result.to_csv("CompareResult.csv")


if __name__ == "__main__":
    main()
