from __future__ import annotations

from typing import Dict, Literal
import pandas as pd
from datetime import datetime
import os

from models.machine_learning.loaders.loader import DatasetLoader
from models.machine_learning.models.lasso_model import LassoModel
from models.machine_learning.ml_dataset.ml_datasets import MLDatasets
from calculation import SectorIndex, TargetCalculator, FeaturesCalculator
from utils.notifier import SlackNotifier


class LassoLearningFacade:
    """LASSOモデルの学習・予測を担当するシンプルなファサード"""

    def __init__(
        self,
        mode: Literal["train_and_predict", "predict_only", "load_only", "none"],
        dataset_path: str,
        stock_dfs_dict: Dict[str, pd.DataFrame] | None = None,
        sector_redef_csv_path: str | None = None,
        sector_index_parquet_path: str | None = None,
        train_start_day: datetime | None = None,
        train_end_day: datetime | None = None,
        test_start_day: datetime | None = None,
        test_end_day: datetime | None = None,
    ) -> None:
        self.mode = mode
        self.dataset_path = dataset_path
        self.stock_dfs_dict = stock_dfs_dict
        self.sector_redef_csv_path = sector_redef_csv_path
        self.sector_index_parquet_path = sector_index_parquet_path
        self.train_start_day = train_start_day
        self.train_end_day = train_end_day
        self.test_start_day = test_start_day
        self.test_end_day = test_end_day
        # Slack通知用
        self.slack = SlackNotifier(program_name=os.path.basename(__file__))
        self.target_df: pd.DataFrame | None = None
        self.raw_target_df: pd.DataFrame | None = None
        self.order_price_df: pd.DataFrame | None = None
        self.new_sector_price_df: pd.DataFrame | None = None
        self.features_df: pd.DataFrame | None = None

    def _get_necessary_dfs(self) -> None:
        if self.stock_dfs_dict is None:
            return
        sic = SectorIndex(
            self.stock_dfs_dict,
            self.sector_redef_csv_path,
            self.sector_index_parquet_path,
        )
        new_sector_price_df, order_price_df = sic.calc_sector_index()
        raw_target_df, target_df = TargetCalculator.daytime_return_PCAresiduals(
            new_sector_price_df,
            reduce_components=1,
            train_start_day=self.train_start_day,
            train_end_day=self.train_end_day,
        )
        self.target_df = target_df
        self.raw_target_df = raw_target_df
        self.order_price_df = order_price_df
        self.new_sector_price_df = new_sector_price_df

    def _get_features_df(self) -> pd.DataFrame:
        return FeaturesCalculator.calculate_features(
            new_sector_price=self.new_sector_price_df,
            new_sector_list=pd.read_csv(self.sector_redef_csv_path),
            stock_dfs_dict=self.stock_dfs_dict,
            adopts_features_indices=True,
            adopts_features_price=False,
            groups_setting=None,
            names_setting=None,
            currencies_type="relative",
            adopt_1d_return=True,
            adopt_size_factor=False,
            adopt_eps_factor=False,
            adopt_sector_categorical=False,
            add_rank=False,
        )

    def _refresh_test_data(self, ml_datasets: MLDatasets) -> None:
        if self.target_df is None or self.features_df is None:
            return
        for name, single_ml in ml_datasets.items():
            target_single = self.target_df.xs(name, level="Sector")
            features_single = self.features_df.xs(name, level="Sector")
            single_ml.archive_train_test_data(
                target_df=target_single,
                features_df=features_single,
                train_start_day=self.train_start_day,
                train_end_day=self.train_end_day,
                test_start_day=self.test_start_day,
                test_end_day=self.test_end_day,
                outlier_threshold=3,
            )
            single_ml.archive_raw_target(self.raw_target_df)
            single_ml.archive_order_price(self.order_price_df)
            single_ml.save()
            ml_datasets.replace_model(single_ml_dataset=single_ml)

    def execute(self) -> MLDatasets | None:
        if self.mode == "none":
            return None

        loader = DatasetLoader(self.dataset_path)
        ml_datasets = loader.load_datasets()

        if self.mode in ("train_and_predict", "predict_only"):
            self._get_necessary_dfs()
            self.features_df = self._get_features_df()
            self._refresh_test_data(ml_datasets)

        if self.mode == "train_and_predict":
            self._train(ml_datasets)
            self._predict(ml_datasets)
        elif self.mode == "predict_only":
            self._predict(ml_datasets)
        elif self.mode == "load_only":
            pass
        else:
            raise NotImplementedError

        self._notify_latest_prediction_date(ml_datasets)
        return ml_datasets

    def _train(self, ml_datasets: MLDatasets) -> None:
        model = LassoModel()
        for _, single_ml in ml_datasets.items():
            trainer_outputs = model.train(
                single_ml.train_test_materials.target_train_df,
                single_ml.train_test_materials.features_train_df,
            )
            single_ml.archive_ml_objects(trainer_outputs.model, trainer_outputs.scaler)
            single_ml.save()
            ml_datasets.replace_model(single_ml_dataset=single_ml)

    def _predict(self, ml_datasets: MLDatasets) -> None:
        model = LassoModel()
        for _, single_ml in ml_datasets.items():
            pred_df = model.predict(
                single_ml.train_test_materials.target_test_df,
                single_ml.train_test_materials.features_test_df,
                single_ml.ml_object_materials.model,
                single_ml.ml_object_materials.scaler,
            )
            single_ml.archive_pred_result(pred_df)
            single_ml.save()
            ml_datasets.replace_model(single_ml_dataset=single_ml)

    def _notify_latest_prediction_date(self, ml_datasets: MLDatasets) -> None:
        df = ml_datasets.get_pred_result()
        latest_date = df.index.get_level_values('Date')[-1]
        self.slack.send_message(f'最新予測日: {latest_date}')
