import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, List

from machine_learning.data.processor import DataProcessor


class Dataset:
    """データセットの管理を行うクラス"""
    
    def __init__(self, 
                target_df: pd.DataFrame = None, 
                features_df: pd.DataFrame = None,
                raw_target_df: pd.DataFrame = None,
                order_price_df: pd.DataFrame = None):
        """
        Args:
            target_df: 目的変数のデータフレーム
            features_df: 特徴量のデータフレーム
            raw_target_df: 生の目的変数のデータフレーム
            order_price_df: 発注価格のデータフレーム
        """
        self.target_df = target_df
        self.features_df = features_df
        self.raw_target_df = raw_target_df
        self.order_price_df = order_price_df
        
        # 分割後のデータ
        self.target_train_df = None
        self.target_test_df = None
        self.features_train_df = None
        self.features_test_df = None
        
    def split_train_test(self, 
                        train_start_date: datetime,
                        train_end_date: datetime,
                        test_start_date: Optional[datetime] = None,
                        test_end_date: Optional[datetime] = None,
                        outlier_threshold: float = 0,
                        no_shift_features: List[str] = None,
                        reuse_features_df: bool = False) -> None:
        """
        データセットを学習用とテスト用に分割する
        
        Args:
            train_start_date: 学習データの開始日
            train_end_date: 学習データの終了日
            test_start_date: テストデータの開始日（省略時はtrain_end_dateの翌日）
            test_end_date: テストデータの終了日（省略時はデータの最終日）
            outlier_threshold: 外れ値除去の閾値（±何σ、0の場合は除去なし）
            no_shift_features: シフトしない特徴量のリスト
            reuse_features_df: 特徴量を他の業種から再利用するか
        """
        if self.target_df is None or self.features_df is None:
            raise ValueError("データフレームがセットされていません。")
        
        # パラメータのデフォルト値設定
        test_start_date = test_start_date or train_end_date
        test_end_date = test_end_date or self.target_df.index.get_level_values('Date').max()
        no_shift_features = no_shift_features or []
        
        # データのコピー
        target_df = self.target_df.copy()
        features_df = self.features_df.copy()
        
        # データの前処理
        if hasattr(target_df.index, 'levels') and 'Date' in target_df.index.names:
            # 次の営業日を追加（実際の実装ではget_next_open_dateを渡す必要がある）
            from utils.jquants_api_utils import get_next_open_date
            target_df = DataProcessor.append_next_business_day_row(target_df, get_next_open_date)
            if not reuse_features_df:
                features_df = DataProcessor.append_next_business_day_row(features_df, get_next_open_date)
        
        # 特徴量のシフトと目的変数との整合性確保
        features_df = DataProcessor.shift_features(features_df, no_shift_features)
        features_df = DataProcessor.align_index(features_df, target_df)
        
        # 学習データとテストデータに分割
        self.target_train_df = DataProcessor.narrow_period(target_df, train_start_date, train_end_date)
        self.target_test_df = DataProcessor.narrow_period(target_df, test_start_date, test_end_date)
        self.features_train_df = DataProcessor.narrow_period(features_df, train_start_date, train_end_date)
        self.features_test_df = DataProcessor.narrow_period(features_df, test_start_date, test_end_date)
        
        # 外れ値除去（学習データのみ）
        if outlier_threshold > 0:
            self.target_train_df, self.features_train_df = DataProcessor.remove_outliers(
                self.target_train_df, self.features_train_df, outlier_threshold
            )
    
    def get_sectors(self) -> List[str]:
        """セクターの一覧を取得する"""
        if self.target_df is None:
            return []
        
        if 'Sector' in self.target_df.index.names:
            return self.target_df.index.get_level_values('Sector').unique().tolist()
        elif 'Sector' in self.target_df.columns:
            return self.target_df['Sector'].unique().tolist()
        else:
            return []
    
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """データの日付範囲を取得する"""
        if self.target_df is None:
            return None, None
        
        if 'Date' in self.target_df.index.names:
            dates = self.target_df.index.get_level_values('Date')
            return dates.min(), dates.max()
        elif 'Date' in self.target_df.columns:
            return self.target_df['Date'].min(), self.target_df['Date'].max()
        else:
            return None, None
    
    def filter_by_sector(self, sector: str) -> 'Dataset':
        """特定のセクターのデータセットを取得する"""
        if sector is None or self.target_df is None:
            return self
        
        if 'Sector' not in self.target_df.index.names and 'Sector' not in self.target_df.columns:
            return self
        
        filtered_dataset = Dataset()
        
        # 目的変数のフィルタリング
        if 'Sector' in self.target_df.index.names:
            filtered_dataset.target_df = self.target_df[
                self.target_df.index.get_level_values('Sector') == sector
            ]
        elif 'Sector' in self.target_df.columns:
            filtered_dataset.target_df = self.target_df[self.target_df['Sector'] == sector]
        
        # 特徴量のフィルタリング
        if self.features_df is not None:
            if 'Sector' in self.features_df.index.names:
                filtered_dataset.features_df = self.features_df[
                    self.features_df.index.get_level_values('Sector') == sector
                ]
            elif 'Sector' in self.features_df.columns:
                filtered_dataset.features_df = self.features_df[self.features_df['Sector'] == sector]
            else:
                filtered_dataset.features_df = self.features_df
        
        # 生の目的変数のフィルタリング
        if self.raw_target_df is not None:
            if 'Sector' in self.raw_target_df.index.names:
                filtered_dataset.raw_target_df = self.raw_target_df[
                    self.raw_target_df.index.get_level_values('Sector') == sector
                ]
            elif 'Sector' in self.raw_target_df.columns:
                filtered_dataset.raw_target_df = self.raw_target_df[self.raw_target_df['Sector'] == sector]
            else:
                filtered_dataset.raw_target_df = self.raw_target_df
        
        # 発注価格のフィルタリング
        filtered_dataset.order_price_df = self.order_price_df
        
        return filtered_dataset
    
    @classmethod
    def from_dict(cls, data_dict: dict) -> 'Dataset':
        """辞書からDatasetを作成する"""
        return cls(
            target_df=data_dict.get('target_df'),
            features_df=data_dict.get('features_df'),
            raw_target_df=data_dict.get('raw_target_df'),
            order_price_df=data_dict.get('order_price_df')
        )
    
    def to_dict(self) -> dict:
        """辞書に変換する"""
        return {
            'target_df': self.target_df,
            'features_df': self.features_df,
            'raw_target_df': self.raw_target_df,
            'order_price_df': self.order_price_df,
            'target_train_df': self.target_train_df,
            'target_test_df': self.target_test_df,
            'features_train_df': self.features_train_df,
            'features_test_df': self.features_test_df
        }