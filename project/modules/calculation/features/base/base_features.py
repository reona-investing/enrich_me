from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from preprocessing import PreprocessingPipeline


class BaseFeatures(ABC):
    """特徴量計算の基底クラス"""
    
    def __init__(self):
        """共通の初期化処理"""
        self.features_df: Optional[pd.DataFrame] = None
    
    @abstractmethod
    def calculate_features(self, **kwargs) -> pd.DataFrame:
        """
        特徴量を計算する抽象メソッド
        実装クラスはこのメソッド内でself.features_dfを更新する必要がある
        """
        pass
    
    def apply_preprocessing(self, pipeline: Optional[PreprocessingPipeline] = None) -> pd.DataFrame:
        """
        前処理パイプラインを適用し、self.features_dfを更新
        
        Args:
            pipeline: 前処理パイプライン
            
        Returns:
            前処理後の特徴量データフレーム
        """
        if self.features_df is None:
            raise ValueError("特徴量が計算されていません。先にcalculate_features()を実行してください。")
        
        if pipeline is not None:
            self.features_df = pipeline.fit_transform(self.features_df)
        
        return self.features_df.copy()
    
    def get_features(self) -> pd.DataFrame:
        """
        計算済み特徴量の安全なコピーを取得
        
        Returns:
            特徴量データフレームのコピー
            
        Raises:
            ValueError: 特徴量が計算されていない場合
        """
        if self.features_df is None:
            raise ValueError("特徴量が計算されていません。先にcalculate_features()を実行してください。")
        
        return self.features_df.copy()
    
    def has_features(self) -> bool:
        """
        特徴量が計算済みかどうかを確認
        
        Returns:
            特徴量が存在するかどうか
        """
        return self.features_df is not None and not self.features_df.empty
    
    def clear_features(self) -> None:
        """特徴量データをクリア"""
        self.features_df = None
