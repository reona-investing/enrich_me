from abc import ABC, abstractmethod
import pandas as pd
from models.machine_learning.outputs import TrainerOutputs


class BaseTrainer(ABC):
    """機械学習モデルのトレーナーの抽象基底クラス"""
    
    def __init__(self, target_train_df: pd.DataFrame, features_train_df: pd.DataFrame):
        """
        Args:
            target_train_df (pd.DataFrame): 訓練用の目的変数データフレーム
            features_train_df (pd.DataFrame): 訓練用の特徴量データフレーム
        """
        self.target_train_df = target_train_df
        self.features_train_df = features_train_df
    
    @abstractmethod
    def train(self, **kwargs) -> TrainerOutputs:
        """
        モデルの学習を行う抽象メソッド
        
        Args:
            **kwargs: モデル固有のハイパーパラメータ
            
        Returns:
            TrainerOutputs: 学習済みモデルとスケーラーを格納したデータクラス
        """
        pass
    
    def _is_multi_sector(self) -> bool:
        """マルチセクターかどうかを判定"""
        return self.target_train_df.index.nlevels > 1
    
    def _get_sectors(self) -> pd.Index:
        """セクター一覧を取得"""
        if self._is_multi_sector():
            return self.target_train_df.index.get_level_values('Sector').unique()
        return pd.Index([None])  # シングルセクターの場合はNoneを返す