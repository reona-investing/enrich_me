"""
複数のモデルやコンテナの予測結果をアンサンブルするクラス
"""
from typing import Dict, List, Union, Optional, Tuple, Any
import pandas as pd
import os
import pickle

from models2.base.base_model import BaseModel
from models2.containers.model_container import ModelContainer
from models2.base.base_container import BaseContainer


class EnsembleModel(BaseContainer):
    """
    複数のモデルの予測結果をアンサンブルするクラス
    """
    
    def __init__(self, name: str = "Ensemble"):
        """
        初期化
        
        Args:
            name: アンサンブルモデルの名前
        """
        super().__init__(name=name)
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
            if isinstance(model, (ModelContainer, BaseContainer)):
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
            if isinstance(model, (ModelContainer, BaseContainer)):
                pred_df = model.predict(X)
            else:  # BaseModel
                pred_df = pd.DataFrame({'Pred': model.predict(X)}, index=X.index)
            
            # (予測結果, 重み) の形式でリストに追加
            ensemble_inputs.append((pred_df, self.weights[name]))
        
        # ensemble.by_rankを使用
        result = ensemble.by_rank(ensemble_inputs)
        
        return result
    
    def _get_state_dict(self) -> Dict[str, Any]:
        """コンテナの状態を辞書形式で取得する"""
        state = super()._get_state_dict()
        
        # モデル名とその重みを保存
        state['model_names'] = list(self.models.keys())
        state['weights'] = {name: weight for name, weight in self.weights.items()}
        
        return state
    
    def _set_state_dict(self, state_dict: Dict[str, Any]):
        """辞書からコンテナの状態を復元する"""
        super()._set_state_dict(state_dict)
        
        # 重みを復元
        if 'weights' in state_dict:
            for name, weight in state_dict['weights'].items():
                if name in self.models:
                    self.weights[name] = weight
    
    def _export_models(self, models_dir: str):
        """コンテナ内のモデルをエクスポートする"""
        for name, model in self.models.items():
            # モデル固有のディレクトリを作成
            model_dir = os.path.join(models_dir, name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # モデルの種類に応じたエクスポート
            if isinstance(model, BaseContainer):
                # 他のコンテナの場合はそのエクスポート機能を使用
                sub_dir = os.path.join(model_dir, "container")
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                model.export(sub_dir)
                
                # コンテナの種類を記録
                container_type_path = os.path.join(model_dir, "container_type.txt")
                with open(container_type_path, 'w') as f:
                    f.write(model.__class__.__name__)
            else:
                # 通常のモデルの場合は直接シリアライズ
                model_path = os.path.join(model_dir, "model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
    
    def _import_models(self, models_dir: str):
        """保存されたモデルをインポートする"""
        # ディレクトリ内の各モデルをロード
        for model_name in os.listdir(models_dir):
            model_dir = os.path.join(models_dir, model_name)
            
            # コンテナの場合
            container_type_path = os.path.join(model_dir, "container_type.txt")
            if os.path.exists(container_type_path):
                with open(container_type_path, 'r') as f:
                    container_type = f.read().strip()
                
                # コンテナタイプに応じてロード
                container_dir = os.path.join(model_dir, "container")
                
                # コンテナタイプに基づいてインポート
                if container_type == "ModelContainer":
                    from models2.containers.model_container import ModelContainer
                    model = ModelContainer.load(container_dir)
                elif container_type == "EnsembleModel":
                    model = EnsembleModel.load(container_dir)
                elif container_type == "PeriodSwitchingModel":
                    from models2.containers.period_model import PeriodSwitchingModel
                    model = PeriodSwitchingModel.load(container_dir)
                else:
                    raise ValueError(f"不明なコンテナタイプ: {container_type}")
                
                self.models[model_name] = model
                
            else:
                # 通常のモデルの場合
                model_path = os.path.join(model_dir, "model.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        self.models[model_name] = model