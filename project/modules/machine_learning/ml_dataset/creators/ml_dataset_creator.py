import os
import yaml
import pandas as pd
from typing import Optional
from datetime import datetime
import pickle

from utils.timeseries import Duration
from machine_learning.ml_dataset.core import MLDataset, MLDatasetStorage, PreprocessingConfig
from machine_learning.ml_dataset.components import MachineLearningAsset

# MLDataset, Duration, MachineLearningAssets は適切なパスからインポート

class MLDatasetCreator:

    @staticmethod
    def create(dataset_path: str,
               target_df: pd.DataFrame,
               features_df: pd.DataFrame,
               raw_returns_df: pd.DataFrame,
               pred_result_df: pd.DataFrame,
               train_duration: Duration,
               test_duration: Duration,
               date_column: str,
               ml_asset: MachineLearningAsset,
               model_division_column: Optional[str] = None,
               preprocessing_config: PreprocessingConfig | None = None) -> MLDataset:
        """
        既存のDataFrameと期間情報からMLDatasetインスタンスを生成します。
        """
        return MLDataset(
            dataset_path=dataset_path,
            target_df=target_df,
            features_df=features_df,
            raw_returns_df=raw_returns_df,
            pred_result_df=pred_result_df,
            train_duration=train_duration,
            test_duration=test_duration,
            date_column=date_column,
            model_division_column=model_division_column,
            ml_assets=ml_asset,
            preprocessing_config=preprocessing_config,
        )

    @staticmethod
    def load(dataset_path: str) -> MLDataset:
        
        ml_dataset_storage = MLDatasetStorage(base_path=dataset_path)
        target_df = pd.read_parquet(ml_dataset_storage.target_df_parquet)
        features_df = pd.read_parquet(ml_dataset_storage.features_df_parquet)
        raw_returns_df = pd.read_parquet(ml_dataset_storage.raw_returns_df_parquet)
        pred_result_df = pd.read_parquet(ml_dataset_storage.pred_result_df_parquet)
        with open(ml_dataset_storage.train_duration, 'rb') as f:
            train_duration = pickle.load(f)
        with open(ml_dataset_storage.test_duration, 'rb') as f:
            test_duration = pickle.load(f)
        with open(ml_dataset_storage.date_column, 'rb') as f:
            date_column = pickle.load(f)
        with open(ml_dataset_storage.model_division_column, 'rb') as f:
            model_division_column = pickle.load(f)
        with open(ml_dataset_storage.ml_assets, 'rb') as f:
            ml_asset = pickle.load(f)
        with open(ml_dataset_storage.preprocessing_config, 'rb') as f:
            preprocessing_config = pickle.load(f)
        
        ml_dataset = MLDataset(
            dataset_path=dataset_path,
            target_df=target_df,
            features_df=features_df,
            raw_returns_df=raw_returns_df,
            pred_result_df=pred_result_df,
            train_duration=train_duration,
            test_duration=test_duration,
            date_column=date_column,
            model_division_column=model_division_column,
            ml_assets=ml_asset,
            preprocessing_config=preprocessing_config
        )

        return ml_dataset
