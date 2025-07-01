from __future__ import annotations

from typing import Optional
import pandas as pd

from utils.timeseries import Duration
from machine_learning.ml_dataset.core import PreprocessingConfig, MLDataset
from machine_learning.ml_dataset.creators import MLDatasetCreator
from machine_learning.ml_dataset.components import MachineLearningAsset
from machine_learning.models import BaseTrainer


class MachineLearningFacadeNew:
    """MLDataset と新しい前処理パイプラインを利用した学習ファサード"""

    def __init__(
        self,
        dataset_path: str,
        train_duration: Duration,
        test_duration: Duration,
        date_column: str,
        model_division_column: Optional[str] = None,
        preprocessing_config: PreprocessingConfig | None = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.train_duration = train_duration
        self.test_duration = test_duration
        self.date_column = date_column
        self.model_division_column = model_division_column
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.ml_dataset: MLDataset | None = None

    # ------------------------------------------------------------------
    # データセット作成 / 読み込み
    # ------------------------------------------------------------------
    def create_dataset(
        self,
        target_df: pd.DataFrame,
        features_df: pd.DataFrame,
        raw_returns_df: pd.DataFrame,
    ) -> None:
        """MLDataset を新規作成"""
        dummy_asset = MachineLearningAsset(name="init", model=None, scaler=None)
        self.ml_dataset = MLDatasetCreator.create(
            dataset_path=self.dataset_path,
            target_df=target_df,
            features_df=features_df,
            raw_returns_df=raw_returns_df,
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
