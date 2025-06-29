import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
from models.machine_learning import MLDatasets, SingleMLDataset


class DatasetLoader:
    """``MLDatasets`` の作成・読み込みをまとめたユーティリティクラス."""

    def __init__(self, dataset_root: str, *, group_level: str = "Sector") -> None:
        self.dataset_root = dataset_root
        self.group_level = group_level

    def create_grouped_datasets(
        self,
        target_df: pd.DataFrame,
        features_df: pd.DataFrame,
        train_start_day: datetime,
        train_end_day: datetime,
        test_start_day: datetime,
        test_end_day: datetime,
        *,
        outlier_threshold: float = 0.0,
        raw_target_df: Optional[pd.DataFrame] = None,
        order_price_df: Optional[pd.DataFrame] = None,
        no_shift_features: Optional[List[str]] = None,
        reuse_features_df: bool = False,
    ) -> MLDatasets:
        """グループ単位で ``SingleMLDataset`` を作成し ``MLDatasets`` として保存する。"""
        if no_shift_features is None:
            no_shift_features = []

        ml_datasets = MLDatasets()
        os.makedirs(self.dataset_root, exist_ok=True)

        groups = target_df.index.get_level_values(self.group_level).unique()
        for group in groups:
            target_single = target_df[target_df.index.get_level_values(self.group_level) == group]
            features_single = features_df[features_df.index.get_level_values(self.group_level) == group]
            single_path = os.path.join(self.dataset_root, str(group))
            single_ds = SingleMLDataset(single_path, str(group), init_load=False)
            single_ds.archive_train_test_data(
                target_df=target_single,
                features_df=features_single,
                train_start_day=train_start_day,
                train_end_day=train_end_day,
                test_start_day=test_start_day,
                test_end_day=test_end_day,
                outlier_threshold=outlier_threshold,
                no_shift_features=no_shift_features,
                reuse_features_df=reuse_features_df,
            )
            if raw_target_df is not None:
                single_ds.archive_raw_target(raw_target_df)
            if order_price_df is not None:
                single_ds.archive_order_price(order_price_df)
            ml_datasets.append_model(single_ds)

        ml_datasets.save_all()
        return ml_datasets

    def load_datasets(self, names: Optional[List[str]] = None) -> MLDatasets:
        """保存済み ``SingleMLDataset`` 群から ``MLDatasets`` を復元する."""
        ml_datasets = MLDatasets()
        if not os.path.isdir(self.dataset_root):
            raise FileNotFoundError(f"{self.dataset_root} が存在しません")

        for name in sorted(os.listdir(self.dataset_root)):
            if names is not None and name not in names:
                continue
            path = os.path.join(self.dataset_root, name)
            if os.path.isdir(path):
                ml_datasets.append_model(SingleMLDataset(path, name))
        return ml_datasets

    def _load_post_processing_dfs(self, filename: str, data_name: str) -> pd.DataFrame:
        """post_processing_data配下の指定ファイルを結合して返す内部ユーティリティ."""
        if not os.path.isdir(self.dataset_root):
            raise FileNotFoundError(f"{self.dataset_root} が存在しません")

        results = []
        for name in sorted(os.listdir(self.dataset_root)):
            file_path = os.path.join(self.dataset_root, name, "post_processing_data", filename)
            if os.path.isfile(file_path):
                try:
                    df = pd.read_parquet(file_path)
                    results.append(df)
                except Exception as e:
                    print(f"{file_path} の読み込みに失敗しました。: {e}")
            else:
                print(f"警告: {file_path} が存在しません。スキップします。")

        if not results:
            raise ValueError(f"有効な{data_name}データが見つかりませんでした。")

        combined_df = pd.concat(results, axis=0)
        if 'Date' in combined_df.index.names:
            combined_df = combined_df.sort_index()
        return combined_df.dropna().drop_duplicates(keep='last')

    def load_pred_results(self) -> pd.DataFrame:
        """dataset_root内に保存された全モデルの ``pred_result_df`` を結合して返す."""
        return self._load_post_processing_dfs(
            filename="pred_result_df.parquet",
            data_name="予測結果"
        )

    def load_raw_targets(self) -> pd.DataFrame:
        """dataset_root内の全モデルの ``raw_target_df`` を読み込み結合して返す."""
        return self._load_post_processing_dfs(
            filename="raw_target_df.parquet",
            data_name="生の目的変数",
        )


if __name__ == "__main__":
    from utils.paths import Paths
    test_path = f'{Paths.ML_DATASETS_FOLDER}/48sectors_LASSO_learned_in_250607'
    loader = DatasetLoader(test_path)
    ml_datasets = loader.load_datasets()
    print(ml_datasets.get_model_names())
    print(ml_datasets.get_pred_result())