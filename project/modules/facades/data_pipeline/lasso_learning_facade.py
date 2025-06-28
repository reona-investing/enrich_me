from __future__ import annotations

from typing import Literal
import pandas as pd
import os
from datetime import datetime

from models.machine_learning.loaders.loader import DatasetLoader
from models.machine_learning.models.lasso_model import LassoModel
from models.machine_learning.ml_dataset.ml_datasets import MLDatasets
from utils.notifier import SlackNotifier


class LassoLearningFacade:
    """LASSOモデルによる学習・予測を担当するファサード"""

    def __init__(
        self,
        mode: Literal["train_and_predict", "predict_only", "load_only", "none"],
        dataset_path: str,
        target_df: pd.DataFrame | None = None,
        features_df: pd.DataFrame | None = None,
        train_start_day: datetime | None = None,
        train_end_day: datetime | None = None,
        test_start_day: datetime | None = None,
        test_end_day: datetime | None = None,
    ) -> None:
        self.mode = mode
        self.dataset_path = dataset_path
        self.target_df = target_df
        self.features_df = features_df
        self.train_start_day = train_start_day
        self.train_end_day = train_end_day
        self.test_start_day = test_start_day
        self.test_end_day = test_end_day
        self.ml_datasets: MLDatasets | None = None

        # Slack通知用
        self.slack = SlackNotifier(program_name=os.path.basename(__file__))

    def execute(self) -> MLDatasets | None:
        if self.mode == "none":
            return None

        loader = DatasetLoader(self.dataset_path)

        if self.mode == "train_and_predict":
            if self.target_df is not None and self.features_df is not None:
                self.ml_datasets = loader.create_grouped_datasets(
                    target_df=self.target_df,
                    features_df=self.features_df,
                    train_start_day=self.train_start_day,
                    train_end_day=self.train_end_day,
                    test_start_day=self.test_start_day,
                    test_end_day=self.test_end_day,
                    raw_target_df=None,
                    order_price_df=None,
                    outlier_threshold=3,
                )
            else:
                self.ml_datasets = loader.load_datasets()
            self._train(self.ml_datasets)
            self._predict(self.ml_datasets)
        elif self.mode == "predict_only":
            self.ml_datasets = loader.load_datasets()
            self._update_test_data(self.ml_datasets)
            self._predict(self.ml_datasets)
        elif self.mode == "load_only":
            self.ml_datasets = loader.load_datasets()
        else:
            raise NotImplementedError

        if self.ml_datasets is not None:
            self._notify_latest_prediction_date(self.ml_datasets)
        return self.ml_datasets

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

    def _update_test_data(self, ml_datasets: MLDatasets) -> None:
        if self.target_df is None or self.features_df is None:
            return
        for name, single_ml in ml_datasets.items():
            sector_target = self.target_df[self.target_df.index.get_level_values('Sector') == name]
            sector_features = self.features_df[self.features_df.index.get_level_values('Sector') == name]
            if self.test_start_day is not None and self.test_end_day is not None:
                sector_target = sector_target[(sector_target.index.get_level_values('Date') >= self.test_start_day) &
                                             (sector_target.index.get_level_values('Date') <= self.test_end_day)]
                sector_features = sector_features[(sector_features.index.get_level_values('Date') >= self.test_start_day) &
                                                 (sector_features.index.get_level_values('Date') <= self.test_end_day)]

            # データが空のときは更新しない
            if sector_target.empty or sector_features.empty:
                continue

            single_ml.train_test_data._target_test_df = sector_target
            single_ml.train_test_data._features_test_df = sector_features
            single_ml.save()
            ml_datasets.replace_model(single_ml_dataset=single_ml)

    def _notify_latest_prediction_date(self, ml_datasets: MLDatasets) -> None:
        df = ml_datasets.get_pred_result()
        latest_date = df.index.get_level_values('Date')[-1]
        self.slack.send_message(f'最新予測日: {latest_date}')
