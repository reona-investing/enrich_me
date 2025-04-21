from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List

class ModelBase(ABC):
    """すべてのモデルの基底クラス"""
    
    def __init__(self, name: str):
        """
        Args:
            name: モデル名
        """
        self.name = name
        self.model = None
        self.scaler = None
        self.feature_importances = None
        self.trained = False
        self.pred_result_df = None
        
        # 学習・テストデータ
        self.target_train_df = None
        self.features_train_df = None
        self.target_test_df = None
        self.features_test_df = None
        
        # 評価用・取引用
        self.raw_target_df = None
        self.order_price_df = None
        
    def load_dataset(self, 
                    target_df: pd.DataFrame, 
                    features_df: pd.DataFrame,
                    train_start_date: datetime,
                    train_end_date: datetime,
                    test_start_date: Optional[datetime] = None,
                    test_end_date: Optional[datetime] = None,
                    outlier_threshold: float = 0,
                    no_shift_features: List[str] = None,
                    reuse_features_df: bool = False) -> None:
        """
        データセットを読み込み、前処理し、学習用とテスト用に分割する
        
        Args:
            target_df: 目的変数のデータフレーム
            features_df: 特徴量のデータフレーム
            train_start_date: 学習データの開始日
            train_end_date: 学習データの終了日
            test_start_date: テストデータの開始日（省略時はtrain_end_date）
            test_end_date: テストデータの終了日（省略時はデータの最終日）
            outlier_threshold: 外れ値除去の閾値（±何σ、0の場合は除去なし）
            no_shift_features: シフトしない特徴量のリスト
            reuse_features_df: 特徴量を他の業種から再利用するか
        """
        from machine_learning.data.processor import DataProcessor
        
        # 設定値の保存
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date if test_start_date else train_end_date
        self.test_end_date = test_end_date if test_end_date else target_df.index.get_level_values('Date').max()
        
        # データのコピー
        target_df = target_df.copy()
        features_df = features_df.copy()
        
        # データの前処理
        if hasattr(target_df.index, 'levels') and 'Date' in target_df.index.names:
            # 次の営業日を追加
            from utils.jquants_api_utils import get_next_open_date
            target_df = DataProcessor.append_next_business_day_row(target_df, get_next_open_date)
            if not reuse_features_df:
                features_df = DataProcessor.append_next_business_day_row(features_df, get_next_open_date)
                
        # 特徴量のシフトと目的変数との整合性確保
        no_shift_features = no_shift_features or []
        features_df = DataProcessor.shift_features(features_df, no_shift_features)
        features_df = DataProcessor.align_index(features_df, target_df)
        
        # 学習データとテストデータに分割
        self.target_train_df = DataProcessor.narrow_period(target_df, train_start_date, train_end_date)
        self.target_test_df = DataProcessor.narrow_period(target_df, self.test_start_date, self.test_end_date)
        self.features_train_df = DataProcessor.narrow_period(features_df, train_start_date, train_end_date)
        self.features_test_df = DataProcessor.narrow_period(features_df, self.test_start_date, self.test_end_date)
        
        # 外れ値除去（学習データのみ）
        if outlier_threshold > 0:
            self.target_train_df, self.features_train_df = DataProcessor.remove_outliers(
                self.target_train_df, self.features_train_df, outlier_threshold
            )
    
    def set_raw_target(self, raw_target_df: pd.DataFrame) -> None:
        """生の目的変数を設定する"""
        self.raw_target_df = raw_target_df
    
    def get_raw_target(self) -> Optional[pd.DataFrame]:
        """生の目的変数を取得する"""
        return self.raw_target_df
    
    def set_order_price(self, order_price_df: pd.DataFrame) -> None:
        """発注価格情報を設定する"""
        self.order_price_df = order_price_df
        
    def get_order_price(self) -> Optional[pd.DataFrame]:
        """発注価格情報を取得する"""
        return self.order_price_df
    
    @abstractmethod
    def train(self) -> None:
        """モデルを学習する"""
        pass
    
    @abstractmethod
    def predict(self) -> pd.DataFrame:
        """予測を実行する"""
        pass
    
    def save(self, path: str) -> None:
        """モデルを保存する"""
        from machine_learning.utils.serialization import save_model
        save_model(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'ModelBase':
        """モデルを読み込む"""
        from machine_learning.utils.serialization import load_model
        return load_model(path)