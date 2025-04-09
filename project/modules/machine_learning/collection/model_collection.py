from typing import Union, List, Dict, Optional, Any, TypeVar
import pandas as pd
import pickle
import os

from machine_learning.models.ml_model_base import MachineLearningModelBase
from machine_learning.params.hyperparams import HyperParams
from machine_learning.factory.model_factory import ModelFactory


class ModelCollection:
    """複数の機械学習モデルを管理するクラス"""
    
    def __init__(self, model_type: str = None):
        self._models: Dict[str, MachineLearningModelBase] = {}
        self._model_type = model_type
        self._metadata = {}
    
    def add_model(self, sector_name: str, model: MachineLearningModelBase) -> None:
        """モデルを追加する"""
        self._models[sector_name] = model
    
    def get_model(self, sector_name: str) -> MachineLearningModelBase:
        """指定したセクターのモデルを取得する"""
        return self._models.get(sector_name)
    
    def train_all(self, sector_data: Dict[str, Dict[str, Union[pd.DataFrame, pd.Series]]], 
                 params: HyperParams) -> None:
        """全てのセクターのモデルを学習する"""
        for sector_name, data in sector_data.items():
            if sector_name not in self._models:
                # モデルがなければ作成
                self._models[sector_name] = ModelFactory.create_model(self._model_type)
            
            # 学習実行
            self._models[sector_name].train(data['y'], data['X'], params)
    
    def predict_all(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """全てのセクターで予測を行う"""
        predictions = {}
        for sector_name, X in sector_data.items():
            if sector_name in self._models:
                predictions[sector_name] = self._models[sector_name].predict(X)
        return predictions
    
    def save(self, filepath: str) -> None:
        """モデルコレクションをファイルに保存する"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelCollection':
        """ファイルからモデルコレクションを読み込む"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @property
    def sectors(self) -> List[str]:
        """管理しているセクターのリストを取得する"""
        return list(self._models.keys())
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """メタデータを取得する"""
        return self._metadata