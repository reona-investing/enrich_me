from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Tuple, List

from machine_learning.models import BaseModel
from machine_learning.params import BaseParams

class BaseCollection(ABC):
    """機械学習モデルのコレクション基底クラス"""
    
    def __init__(self):
        self.models = {}  # Dict[str, BaseModel]
        
    def get_model(self, name: str) -> BaseModel:
        """名前を指定してモデルを取得する"""
        if name not in self.models:
            raise KeyError(f"モデル '{name}' はコレクションに存在しません。")
        return self.models[name]
    
    def set_model(self, model: BaseModel) -> None:
        """モデルをコレクションに追加/更新する"""
        self.models[model.name] = model
    
    def get_models(self) -> List[Tuple[str, BaseModel]]:
        """全モデルの名前とインスタンスのリストを取得する"""
        return [(name, model) for name, model in self.models.items()]
    
    @abstractmethod
    def generate_model(self, name: str, type: str, params: Optional[BaseParams] = None) -> BaseModel:
        """新しいモデルを生成してコレクションに追加する"""
        pass
    
    def train_all(self) -> None:
        """全てのモデルを学習する"""
        for model in self.models.values():
            model.train()
    
    def predict_all(self) -> None:
        """全てのモデルで予測を実行する"""
        for model in self.models.values():
            model.predict()
    
    def get_result_df(self) -> pd.DataFrame:
        """全てのモデルの予測結果を結合する"""
        result_dfs = []
        for model in self.models.values():
            if model.pred_result_df is not None:
                result_dfs.append(model.pred_result_df)
        
        if not result_dfs:
            raise ValueError("予測結果が存在しません。predict_all()を先に実行してください。")
        
        return pd.concat(result_dfs)
    
    @abstractmethod
    def save(self, path: str) -> None:
        """コレクションをファイルに保存する"""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseCollection':
        """ファイルからコレクションを読み込む"""
        pass