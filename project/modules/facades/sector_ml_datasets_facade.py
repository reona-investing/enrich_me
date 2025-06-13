import os
from datetime import datetime
from typing import Dict, Optional, List, Literal

import pandas as pd

from calculation.sector_index import SectorIndex
from calculation.target.target_calculator import TargetCalculator
from calculation.features.implementation import IndexFeatures, PriceFeatures
from calculation.features.integration.features_set import FeaturesSet
from models.machine_learning.ml_dataset.single_ml_dataset import SingleMLDataset
from models.machine_learning.ml_dataset.ml_datasets import MLDatasets

class SectorMLDatasetsFacade:
    """業種単位での学習用データセット作成をまとめて実行するファサード。"""

    def create_datasets(
        self,
        stock_dfs_dict: Dict,
        sector_redefinitions_csv: str,
        sector_index_parquet: str,
        dataset_root: str,
        train_start_day: datetime,
        train_end_day: datetime,
        test_start_day: datetime,
        test_end_day: datetime,
        outlier_threshold: float = 0.0,
        groups_setting: Optional[Dict] = None,
        names_setting: Optional[Dict] = None,
        currencies_type: Literal['relative', 'raw'] = 'relative',
        commodity_type: Literal['JPY', 'raw'] = 'raw',
        adopt_1d_return: bool = True,
        mom_duration: Optional[List[int]] = None,
        vola_duration: Optional[List[int]] = None,
        adopt_size_factor: bool = True,
        adopt_eps_factor: bool = True,
        adopt_sector_categorical: bool = True,
        add_rank: bool = True,
    ) -> MLDatasets:
        """業種ごとの ``SingleMLDataset`` を生成し ``MLDatasets`` として保存する。

        各種パラメータは ``FeaturesSet`` や ``TargetCalculator`` へ渡され、
        特徴量計算からデータセットの保存までを一括で行う。
        """
        if mom_duration is None:
            mom_duration = [5, 21]
        if vola_duration is None:
            vola_duration = [5, 21]
        if groups_setting is None:
            groups_setting = {}
        if names_setting is None:
            names_setting = {}

        sic = SectorIndex(
            stock_dfs_dict=stock_dfs_dict,
            sector_redefinitions_csv=sector_redefinitions_csv,
            sector_index_parquet=sector_index_parquet,
        )
        sector_index_df, order_price_df = sic.calc_sector_index()
        sector_index_dict, order_price_dict = sic.get_sector_index_dict()

        sector_list_df = pd.read_csv(sector_redefinitions_csv)

        # 特徴量計算
        index_calc = IndexFeatures()
        index_calc.calculate_features(
            groups_setting=groups_setting,
            names_setting=names_setting,
            currencies_type=currencies_type,
            commodity_type=commodity_type,
        )
        price_calc = PriceFeatures()
        price_calc.calculate_features(
            new_sector_price=sector_index_df,
            new_sector_list=sector_list_df,
            stock_dfs_dict=stock_dfs_dict,
            adopt_1d_return=adopt_1d_return,
            mom_duration=mom_duration,
            vola_duration=vola_duration,
            adopt_size_factor=adopt_size_factor,
            adopt_eps_factor=adopt_eps_factor,
            adopt_sector_categorical=adopt_sector_categorical,
            add_rank=add_rank,
        )
        features_df = FeaturesSet().combine_features(index_calc, price_calc)

        ml_datasets = MLDatasets()
        os.makedirs(dataset_root, exist_ok=True)

        for sector, sector_df in sector_index_dict.items():
            target_df = TargetCalculator.daytime_return(sector_df)
            features_sector_df = features_df.xs(sector, level='Sector')
            single_path = os.path.join(dataset_root, sector)
            single_ds = SingleMLDataset(single_path, sector, init_load=False)
            single_ds.archive_train_test_data(
                target_df=target_df,
                features_df=features_sector_df,
                train_start_day=train_start_day,
                train_end_day=train_end_day,
                test_start_day=test_start_day,
                test_end_day=test_end_day,
                outlier_threshold=outlier_threshold,
            )
            single_ds.archive_raw_target(target_df)
            single_ds.archive_order_price(order_price_dict[sector])
            ml_datasets.append_model(single_ds)

        ml_datasets.save_all()
        return ml_datasets
