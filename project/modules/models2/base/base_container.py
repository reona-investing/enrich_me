"""
すべてのコンテナクラスの基底クラス
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import os
import pickle
import pandas as pd


class BaseContainer(ABC):
    """
    すべてのコンテナクラス（ModelContainer、EnsembleModel、PeriodSwitchingModel）の
    基底クラス。共通のインターフェースとエクスポート/インポート機能を提供します。
    """
    
    def __init__(self, name: str = ""):
        """
        初期化
        
        Args:
            name: コンテナの名前（任意）
        """
        self.name = name
    
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        コンテナ内のモデルで予測を行い、結果を返す
        
        Args:
            X: 特徴量データ
            
        Returns:
            pd.DataFrame: 予測結果
        """
        pass
    
    def export(self, export_dir: str) -> str:
        """
        コンテナデータをディスクに保存する
        
        Args:
            export_dir: エクスポート先ディレクトリパス
            
        Returns:
            str: 保存された親ディレクトリのパス
        """
        # ディレクトリが存在しない場合は作成
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
            
        # コンテナタイプ名でサブディレクトリを作成
        container_type = self.__class__.__name__
        container_dir = os.path.join(export_dir, f"{container_type}_{self.name}")
        
        if not os.path.exists(container_dir):
            os.makedirs(container_dir)
            
        # 状態をdict形式で取得
        state_dict = self._get_state_dict()
        
        # 状態をファイルに保存
        state_path = os.path.join(container_dir, "container_state.pkl")
        with open(state_path, 'wb') as f:
            pickle.dump(state_dict, f)
            
        # モデルデータを保存
        models_dir = os.path.join(container_dir, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        self._export_models(models_dir)
        
        print(f"コンテナを {container_dir} に保存しました")
        return container_dir
    
    @classmethod
    def load(cls, import_dir: str) -> 'BaseContainer':
        """
        ディスクから保存されたコンテナをロードする
        
        Args:
            import_dir: インポート元ディレクトリパス
            
        Returns:
            BaseContainer: ロードされたコンテナインスタンス
        """
        # 状態をロード
        state_path = os.path.join(import_dir, "container_state.pkl")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"コンテナ状態ファイルが見つかりません: {state_path}")
            
        with open(state_path, 'rb') as f:
            state_dict = pickle.load(f)
            
        # インスタンスを作成
        instance = cls(name=state_dict.get('name', ''))
        
        # 状態を復元
        instance._set_state_dict(state_dict)
        
        # モデルデータをロード
        models_dir = os.path.join(import_dir, "models")
        if os.path.exists(models_dir):
            instance._import_models(models_dir)
            
        print(f"コンテナを {import_dir} からロードしました")
        return instance
    
    @abstractmethod
    def _get_state_dict(self) -> Dict[str, Any]:
        """
        コンテナの状態を辞書形式で取得する
        
        Returns:
            Dict[str, Any]: コンテナの状態を表す辞書
        """
        # 基本情報
        return {
            'name': self.name,
            'type': self.__class__.__name__,
        }
    
    @abstractmethod
    def _set_state_dict(self, state_dict: Dict[str, Any]):
        """
        辞書からコンテナの状態を復元する
        
        Args:
            state_dict: コンテナの状態を表す辞書
        """
        self.name = state_dict.get('name', self.name)
    
    @abstractmethod
    def _export_models(self, models_dir: str):
        """
        コンテナ内のモデルをエクスポートする
        
        Args:
            models_dir: モデルの保存先ディレクトリ
        """
        pass
    
    @abstractmethod
    def _import_models(self, models_dir: str):
        """
        保存されたモデルをインポートする
        
        Args:
            models_dir: モデルのインポート元ディレクトリ
        """
        pass