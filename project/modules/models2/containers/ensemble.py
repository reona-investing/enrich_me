"""
複数のモデルやコンテナの予測結果をアンサンブルするクラス
"""
from typing import Dict, List, Union, Optional, Tuple, Any
import pandas as pd

from models2.base.base_model import BaseModel
from models2.containers.model_container import ModelContainer


class EnsembleModel:
    """
    複数のモデルの予測結果をアンサンブルするクラス
    """
    
    def __init__(self, name: str = "Ensemble"):
        """
        初期化
        
        Args:
            name: アンサンブルモデルの名前
        """
        self.name = name
        self.models = {}  # 名前→モデルまたはコンテナのマッピング
        self.weights = {}  # 名前→重みのマッピング
    
    def add_model(self, name: str, model: Union[BaseModel, ModelContainer], weight: float = 1.0):
        """
        アンサンブルにモデルを追加
        
        Args:
            name: モデルの名前
            model: 追加するモデルまたはコンテナ
            weight: アンサンブルにおける重み
        """
        self.models[name] = model
        self.weights[name] = weight
    
    def remove_model(self, name: str):
        """
        アンサンブルからモデルを削除
        
        Args:
            name: 削除するモデルの名前
        """
        if name in self.models:
            del self.models[name]
        if name in self.weights:
            del self.weights[name]
    
    def set_weight(self, name: str, weight: float):
        """
        特定のモデルの重みを設定
        
        Args:
            name: モデルの名前
            weight: 新しい重み
        """
        if name not in self.models:
            raise KeyError(f"モデル '{name}' はアンサンブルに存在しません")
        self.weights[name] = weight
    
    def predict(self, X: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        アンサンブルで予測を行い、重み付け平均を返す
        
        Args:
            X: 特徴量データ
            
        Returns:
            pd.DataFrame: 重み付けアンサンブル予測結果
        """
        # 各モデルの予測を収集
        predictions = {}
        for name, model in self.models.items():
            # モデルのタイプに応じて予測
            if isinstance(model, ModelContainer):
                pred_df = model.predict(X)
            else:  # BaseModel
                pred_df = pd.DataFrame({'Pred': model.predict(X)}, index=X.index)
            
            predictions[name] = pred_df
        
        # 重み付けアンサンブル
        # 注: 各モデルが同じインデックスで予測結果を返すと仮定
        if not predictions:
            return pd.DataFrame()
        
        # 結果を初期化
        first_pred = next(iter(predictions.values()))
        result = pd.DataFrame(index=first_pred.index)
        
        # 重み付けアンサンブル
        result['Pred'] = 0.0
        total_weight = sum(self.weights.values())
        
        for name, pred_df in predictions.items():
            norm_weight = self.weights[name] / total_weight
            result['Pred'] += pred_df['Pred'] * norm_weight
        
        return result
    
    def rank_ensemble(self, X: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        ランクベースのアンサンブルを実行
        
        Args:
            X: 特徴量データ
            
        Returns:
            pd.DataFrame: ランクベースのアンサンブル予測結果
        """
        from models import ensemble
        
        # 各モデルの予測を収集して重み付けの形式に変換
        ensemble_inputs = []
        for name, model in self.models.items():
            # モデルのタイプに応じて予測
            if isinstance(model, ModelContainer):
                pred_df = model.predict(X)
            else:  # BaseModel
                pred_df = pd.DataFrame({'Pred': model.predict(X)}, index=X.index)
            
            # (予測結果, 重み) の形式でリストに追加
            ensemble_inputs.append((pred_df, self.weights[name]))
        
        # ensemble.by_rankを使用
        result = ensemble.by_rank(ensemble_inputs)
        
        return result