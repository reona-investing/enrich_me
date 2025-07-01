from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd

from calculation import TargetCalculator, FeaturesCalculator, SectorIndex
from utils.timeseries import Duration
from machine_learning.ml_dataset.core import PreprocessingConfig, MLDataset
from machine_learning.ml_dataset.creators import MLDatasetCreator
from machine_learning.ml_dataset.components import MachineLearningAsset
from machine_learning.models import BaseTrainer


class MachineLearningFacadeNew:
    """MLDataset と新しい前処理パイプラインを利用した学習ファサード"""

    def __init__(
        self,
        stock_dfs_dict: Dict[str, pd.DataFrame],
        sector_redef_csv_path: str,
        sector_index_parquet_path: str,
        dataset_path: str,
        train_duration: Duration,
        test_duration: Duration,
        date_column: str,
        model_division_column: Optional[str] = None,
        preprocessing_config: PreprocessingConfig | None = None,
    ) -> None:
        self.stock_dfs_dict = stock_dfs_dict
        self.sector_redef_csv_path = sector_redef_csv_path
        self.sector_index_parquet_path = sector_index_parquet_path
        self.dataset_path = dataset_path
        self.train_duration = train_duration
        self.test_duration = test_duration
        self.date_column = date_column
        self.model_division_column = model_division_column
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.ml_dataset: MLDataset | None = None

        self.target_df: pd.DataFrame | None = None
        self.raw_target_df: pd.DataFrame | None = None
        self.features_df: pd.DataFrame | None = None
        self.order_price_df: pd.DataFrame | None = None
        self.new_sector_price_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # データセット作成 / 読み込み
    # ------------------------------------------------------------------
    def create_dataset(
        self,
        adopt_features_price: bool = True,
        adopt_size_factor: bool = True,
        adopt_eps_factor: bool = True,
        adopt_sector_categorical: bool = True,
        add_rank: bool = True,
        mom_duration: Optional[List[int]] = None,
        vola_duration: Optional[List[int]] = None,
    ) -> None:
        """原データからMLDatasetを生成"""
        self._get_necessary_dfs()
        self.features_df = self._get_features_df(
            adopt_features_price=adopt_features_price,
            adopt_size_factor=adopt_size_factor,
            adopt_eps_factor=adopt_eps_factor,
            adopt_sector_categorical=adopt_sector_categorical,
            add_rank=add_rank,
            mom_duration=mom_duration,
            vola_duration=vola_duration,
        )

        dummy_asset = MachineLearningAsset(name="init", model=None, scaler=None)
        self.ml_dataset = MLDatasetCreator.create(
            dataset_path=self.dataset_path,
            target_df=self.target_df,
            features_df=self.features_df,
            raw_returns_df=self.raw_target_df,
            pred_result_df=pd.DataFrame(),
            train_duration=self.train_duration,
            test_duration=self.test_duration,
            date_column=self.date_column,
            ml_asset=dummy_asset,
            model_division_column=self.model_division_column,
            preprocessing_config=self.preprocessing_config,
        )

    def load_dataset(self) -> None:
        """保存済みデータセットを読み込む"""
        self.ml_dataset = MLDatasetCreator.load(self.dataset_path)

    # ------------------------------------------------------------------
    # データ生成
    # ------------------------------------------------------------------
    def _get_necessary_dfs(self) -> None:
        """目的変数計算に必要なデータフレームを生成する"""
        sic = SectorIndex(
            self.stock_dfs_dict,
            self.sector_redef_csv_path,
            self.sector_index_parquet_path,
        )
        new_sector_price_df, order_price_df = sic.calc_sector_index()
        raw_target_df, target_df = TargetCalculator.daytime_return_PCAresiduals(
            new_sector_price_df,
            reduce_components=1,
            train_start_day=self.train_duration.start,
            train_end_day=self.train_duration.end,
        )
        self.target_df = target_df
        self.raw_target_df = raw_target_df
        self.order_price_df = order_price_df
        self.new_sector_price_df = new_sector_price_df

    def _get_features_df(
        self,
        adopt_features_price: bool,
        adopt_size_factor: bool,
        adopt_eps_factor: bool,
        adopt_sector_categorical: bool,
        add_rank: bool,
        mom_duration: Optional[List[int]] = None,
        vola_duration: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        new_sector_list = pd.read_csv(self.sector_redef_csv_path)
        return FeaturesCalculator.calculate_features(
            new_sector_price=self.new_sector_price_df,
            new_sector_list=new_sector_list,
            stock_dfs_dict=self.stock_dfs_dict,
            adopts_features_indices=True,
            adopts_features_price=adopt_features_price,
            groups_setting=None,
            names_setting=None,
            currencies_type="relative",
            adopt_1d_return=True,
            mom_duration=mom_duration,
            vola_duration=vola_duration,
            adopt_size_factor=adopt_size_factor,
            adopt_eps_factor=adopt_eps_factor,
            adopt_sector_categorical=adopt_sector_categorical,
            add_rank=add_rank,
        )

    # ------------------------------------------------------------------
    # 学習・予測
    # ------------------------------------------------------------------
    def train(self, trainer: BaseTrainer, **kwargs) -> None:
        """データセットを学習"""
        if self.ml_dataset is None:
            self.load_dataset()
        self.ml_dataset.train(trainer, **kwargs)
        self.ml_dataset._save()

    def predict(self) -> pd.DataFrame:
        """データセットで予測を実施"""
        if self.ml_dataset is None:
            self.load_dataset()
        self.ml_dataset.predict()
        self.ml_dataset._save()
        return self.ml_dataset.pred_result_df
