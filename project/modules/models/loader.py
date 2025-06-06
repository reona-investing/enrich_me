import os
from datetime import datetime
from typing import List, Optional

import pandas as pd

from .machine_learning import MLDatasets, SingleMLDataset


def create_grouped_datasets(
    target_df: pd.DataFrame,
    features_df: pd.DataFrame,
    dataset_root: str,
    train_start_day: datetime,
    train_end_day: datetime,
    test_start_day: datetime,
    test_end_day: datetime,
    *,
    outlier_threshold: float = 0.0,
    group_level: str = "Sector",
    raw_target_df: Optional[pd.DataFrame] = None,
    order_price_df: Optional[pd.DataFrame] = None,
    no_shift_features: Optional[List[str]] = None,
    reuse_features_df: bool = False,
) -> MLDatasets:
    """グループ単位で ``SingleMLDataset`` を作成し ``MLDatasets`` として保存する。

    Args:
        target_df: 目的変数データ。 ``group_level`` でグループ化できる MultiIndex を想定。
        features_df: 特徴量データ。 ``group_level`` でグループ化できる MultiIndex を想定。
        dataset_root: 出力先フォルダ。
        train_start_day: 学習期間開始日。
        train_end_day: 学習期間終了日。
        test_start_day: テスト期間開始日。
        test_end_day: テスト期間終了日。
        outlier_threshold: 外れ値除去の閾値。
        group_level: グループ化に使用するインデックス名。
        raw_target_df: 生の目的変数データを保存する場合に指定。
        order_price_df: 発注価格データを保存する場合に指定。
        no_shift_features: シフトしない特徴量名のリスト。
        reuse_features_df: True の場合は特徴量のシフト処理をスキップする。

    Returns:
        MLDatasets: 生成されたデータセット群。
    """
    if no_shift_features is None:
        no_shift_features = []

    ml_datasets = MLDatasets()
    os.makedirs(dataset_root, exist_ok=True)

    groups = target_df.index.get_level_values(group_level).unique()
    for group in groups:
        target_single = target_df[target_df.index.get_level_values(group_level) == group]
        features_single = features_df[features_df.index.get_level_values(group_level) == group]
        single_path = os.path.join(dataset_root, str(group))
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


def load_datasets(dataset_root: str, names: Optional[List[str]] = None) -> MLDatasets:
    """フォルダに保存された ``SingleMLDataset`` 群を読み込み ``MLDatasets`` を復元する。"""
    ml_datasets = MLDatasets()
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"{dataset_root} が存在しません")

    for name in sorted(os.listdir(dataset_root)):
        if names is not None and name not in names:
            continue
        path = os.path.join(dataset_root, name)
        if os.path.isdir(path):
            ml_datasets.append_model(SingleMLDataset(path, name))
    return ml_datasets
