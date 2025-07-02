from abc import ABC, abstractmethod
import pandas as pd
from machine_learning.ml_dataset.components import MachineLearningAsset


class BaseTrainer(ABC):
    """機械学習モデルのトレーナーの抽象基底クラス"""
    
    def __init__(self):
        pass
    
    @abstractmethod
    def train(self, model_name: str, target_df: pd.DataFrame, features_df: pd.DataFrame, **kwargs) -> MachineLearningAsset:
        """
        モデルの学習を行う抽象メソッド
        
        Args:
            model_name (str): モデルの名称
            target_df (pd.DataFrame): 目的変数のdf
            featurse_df (pd.DataFrame): 特徴量のdf
            **kwargs: モデル固有のハイパーパラメータ
        """
        pass