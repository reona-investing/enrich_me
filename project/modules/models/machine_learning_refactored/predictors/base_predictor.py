from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, List, Optional


class BasePredictor(ABC):
    """機械学習モデルの予測器の抽象基底クラス"""
    
    def __init__(self, target_test_df: pd.DataFrame, features_test_df: pd.DataFrame, 
                 models: List[Any], scalers: Optional[List[Any]] = None):
        """
        Args:
            target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
            features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
            models (List[Any]): 学習済みモデルのリスト
            scalers (Optional[List[Any]]): スケーラーのリスト（必要な場合）
        """
        self.target_test_df = target_test_df
        self.features_test_df = features_test_df
        self.models = models
        self.scalers = scalers
        
        self._validate_inputs()
    
    def _validate_inputs(self):
        """入力の妥当性をチェック"""
        assert len(self.models) > 0, '予測のためには1つ以上のモデルが必要です。'
        if self.scalers is not None:
            assert len(self.models) == len(self.scalers), 'モデルとスケーラーには同じ数を設定してください。'
    
    @abstractmethod
    def predict(self) -> pd.DataFrame:
        """
        予測を行う抽象メソッド
        
        Returns:
            pd.DataFrame: 予測結果を格納したデータフレーム
        """
        pass
    
    def _is_multi_sector(self) -> bool:
        """マルチセクターかどうかを判定"""
        return self.target_test_df.index.nlevels > 1
    
    def _get_sectors(self) -> pd.Index:
        """セクター一覧を取得"""
        if self._is_multi_sector():
            return self.target_test_df.index.get_level_values('Sector').unique()
        return pd.Index([None])  # シングルセクターの場合はNoneを返す