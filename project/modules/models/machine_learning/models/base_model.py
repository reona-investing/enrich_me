from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, List, Optional
from models.machine_learning.trainers.outputs import TrainerOutputs

class BaseModel(ABC):
    """機械学習モデルの抽象基底クラス"""
    
    @abstractmethod
    def train(self, target_train_df: pd.DataFrame, features_train_df: pd.DataFrame, **kwargs) -> TrainerOutputs:
        """
        学習を実行する抽象メソッド
        
        Args:
            target_train_df (pd.DataFrame): 訓練用の目的変数データフレーム
            features_train_df (pd.DataFrame): 訓練用の特徴量データフレーム
            **kwargs: モデル固有のハイパーパラメータ
            
        Returns:
            TrainerOutputs: 学習済みモデルとスケーラーを格納したデータクラス
        """
        pass
    
    @abstractmethod
    def predict(self, target_test_df: pd.DataFrame, features_test_df: pd.DataFrame, 
                models: List[Any], scalers: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        予測を実行する抽象メソッド
        
        Args:
            target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
            features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
            models (List[Any]): 学習済みモデルのリスト
            scalers (Optional[List[Any]]): スケーラーのリスト（必要な場合）
            
        Returns:
            pd.DataFrame: 予測結果のデータフレーム
        """
        pass