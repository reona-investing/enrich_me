from abc import ABC, abstractmethod
from typing import Union, Dict, Any
import pandas as pd
from machine_learning.params.hyperparams import HyperParams

class MachineLearningModelBase(ABC):
    """機械学習モデルの基底クラス"""
    
    def __init__(self):
        self._model = None
        self._scaler = None
        self._prediction_df = None
        self._feature_importance_df = None
        self._metadata = {}
    
    @abstractmethod
    def train(self, y: Union[pd.DataFrame, pd.Series], X: pd.DataFrame, params: HyperParams) -> None:
        """モデルを学習する"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """予測を行う"""
        pass
    
    @property
    def model(self):
        """モデルを取得する"""
        return self._model
    
    @property
    def scaler(self):
        """モデルを取得する"""
        return self._scaler
    
    @property
    def prediction(self) -> pd.DataFrame:
        """予測結果を取得する"""
        return self._prediction_df
    
    @property
    def feature_importance(self) -> pd.DataFrame:
        """特徴量重要度を取得する"""
        return self._feature_importance_df
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """メタデータを取得する"""
        return self._metadata