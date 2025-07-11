from __future__ import annotations

from typing import Literal, Dict, List
import pandas as pd
import os
from datetime import datetime

from calculation import SectorIndex, FeaturesCalculator
from timeseries_data.facades import Pc1RemovedIntradayReturn
from machine_learning.ml_dataset import MLDataset
from machine_learning.models import LassoTrainer
from utils.notifier import SlackNotifier
from utils.timeseries import Duration


class LassoLearningFacade:
    """LASSOモデルによる学習・予測を担当するファサード"""

    def __init__(
        self,
        mode: Literal["train_and_predict", "predict_only", "load_only", "none"],
        stock_dfs_dict: Dict[pd.DataFrame],
        dataset_path: str,
        sector_redef_csv_path: str,
        sector_index_parquet_path: str,
        train_start_day: datetime | None = None,
        train_end_day: datetime | None = None,
        test_start_day: datetime | None = None,
        test_end_day: datetime | None = None,
    ) -> None:
            
        self.mode = mode
        self.stock_dfs_dict = stock_dfs_dict
        self.dataset_path = dataset_path
        self.sector_redef_csv_path = sector_redef_csv_path
        self.sector_index_parquet_path = sector_index_parquet_path
        self.train_duration = Duration(start=train_start_day, end=train_end_day)
        self.test_duration = Duration(start=test_start_day, end=test_end_day)
        self.ml_dataset: MLDataset | None = None

        # Slack通知用
        self.slack = SlackNotifier(program_name=os.path.basename(__file__))

        #クラス内で計算するプロパティ
        self.target_df = None
        self.raw_returns_df = None
        self.order_price_df = None
        self.new_sector_price_df = None

    def execute(self) -> MLDataset | None:
        if self.mode == "none":
            return None
        self._get_necessary_dfs()
        self._get_features_df(
            adopt_features_price = False, adopt_size_factor = False,
            adopt_eps_factor = False, adopt_sector_categorical = False, add_rank = False
            )
        if self.mode == "train_and_predict":
            self._create_dataset()
            self._train()
            self._predict()
        elif self.mode == "predict_only":
            self._update_dataset()
            self._predict()
        elif self.mode == "load_only":
            self._load_dataset()
        else:
            raise NotImplementedError

        if self.ml_dataset is not None:
            self._notify_latest_prediction_date(self.ml_dataset)
        return self.ml_dataset

    def _get_necessary_dfs(self):
        sic = SectorIndex(self.stock_dfs_dict, self.sector_redef_csv_path, self.sector_index_parquet_path)
        new_sector_price_df, order_price_df = sic.calc_sector_index()
        pc1_return = Pc1RemovedIntradayReturn(original_timeseries=new_sector_price_df)
        pc1_return.calculate(fit_duration=self.train_duration)
        raw_returns_df = pc1_return.raw_return
        target_df = pc1_return.processed_return
        self.target_df = target_df
        self.raw_returns_df = raw_returns_df
        self.order_price_df = order_price_df
        self.new_sector_price_df = new_sector_price_df

    def _get_features_df(self, adopt_features_price: bool, adopt_size_factor: bool, adopt_eps_factor: bool,
                         adopt_sector_categorical: bool, add_rank: bool,
                         mom_duration: List[int] | None = None, 
                         vola_duration: List[int] | None = None) -> pd.DataFrame:
        self.features_df = \
              FeaturesCalculator.calculate_features(
                new_sector_price = self.new_sector_price_df,
                new_sector_list = pd.read_csv(self.sector_redef_csv_path),
                stock_dfs_dict = self.stock_dfs_dict,
                adopts_features_indices=True,
                adopts_features_price=adopt_features_price, #TODO LASSO: False, LightGBM: True
                groups_setting=None,
                names_setting=None,
                currencies_type='relative',
                adopt_1d_return=True,
                mom_duration=mom_duration, #TODO LightGBM [5, 21]
                vola_duration=vola_duration, #TODO LightGBM [5, 21]
                adopt_size_factor=adopt_size_factor, #TODO LASSO: False, LightGBM: True
                adopt_eps_factor=adopt_eps_factor, #TODO LASSO: False, LightGBM: True
                adopt_sector_categorical=adopt_sector_categorical, #TODO LASSO: False, LightGBM: True
                add_rank=add_rank, #TODO LASSO: False, LightGBM: True
                )

    def _create_dataset(self):
            self.ml_dataset = MLDataset.from_raw(
                dataset_path=self.dataset_path,
                target_df=self.target_df,
                features_df=self.features_df,
                raw_returns_df=self.raw_returns_df,
                order_price_df=self.order_price_df,
                pred_return_df=None,
                train_duration=self.train_duration,
                test_duration=self.test_duration,
                date_column='Date',
                sector_column='Sector',
                is_model_divided=True,
                ml_assets=None,
                no_shift_features=[],
                outlier_threshold=3
            )

    def _update_dataset(self):
        self.ml_dataset = MLDataset.from_files(self.dataset_path)
        self.ml_dataset.update_data(self.target_df, self.features_df, self.raw_returns_df, self.order_price_df)

    def _load_dataset(self):
        self.ml_dataset = MLDataset.from_files(self.dataset_path)

    def _train(self):
        # 学習
        self.ml_dataset.train(trainer=LassoTrainer())
    
    def _predict(self):
        self.ml_dataset.predict()

    def _notify_latest_prediction_date(self, ml_dataset: MLDataset) -> None:
        latest_date = ml_dataset.pred_result_df.index.get_level_values('Date')[-1]
        self.slack.send_message(f'最新予測日: {latest_date}')
