from datetime import datetime
from typing import Tuple, Dict, Optional

from machine_learning.data.dataset import Dataset


class DataSplitter:
    """データセットの分割を担当するクラス"""
    
    @staticmethod
    def split_by_sector(dataset: Dataset) -> Dict[str, Dataset]:
        """
        セクターごとにデータセットを分割する
        
        Args:
            dataset: 分割対象のデータセット
            
        Returns:
            セクター名をキーとするデータセットの辞書
        """
        sectors = dataset.get_sectors()
        if not sectors:
            return {'all': dataset}
        
        return {sector: dataset.filter_by_sector(sector) for sector in sectors}
    
    @staticmethod
    def split_time_series(dataset: Dataset,
                          train_start_date: datetime,
                          train_end_date: datetime,
                          test_start_date: Optional[datetime] = None,
                          test_end_date: Optional[datetime] = None,
                          valid_ratio: float = 0.0) -> Tuple[Dataset, Optional[Dataset], Dataset]:
        """
        時系列に基づいてデータセットを分割する
        
        Args:
            dataset: 分割対象のデータセット
            train_start_date: 学習データの開始日
            train_end_date: 学習データの終了日
            test_start_date: テストデータの開始日（省略時はtrain_end_dateの翌日）
            test_end_date: テストデータの終了日（省略時はデータの最終日）
            valid_ratio: 検証データの割合（0の場合は検証データを作成しない）
            
        Returns:
            (学習用データセット, 検証用データセット, テスト用データセット)のタプル
            検証用データセットは valid_ratio=0 の場合は None
        """
        # コピーを作成
        train_dataset = Dataset(
            target_df=dataset.target_df.copy() if dataset.target_df is not None else None,
            features_df=dataset.features_df.copy() if dataset.features_df is not None else None,
            raw_target_df=dataset.raw_target_df.copy() if dataset.raw_target_df is not None else None,
            order_price_df=dataset.order_price_df.copy() if dataset.order_price_df is not None else None
        )
        
        # 学習・テストデータに分割
        train_dataset.split_train_test(
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            outlier_threshold=0,  # 外れ値除去は別途行う
            no_shift_features=[]  # シフトは別途行う
        )
        
        # 学習用データセット
        train_ds = Dataset(
            target_df=train_dataset.target_train_df,
            features_df=train_dataset.features_train_df,
            raw_target_df=dataset.raw_target_df,
            order_price_df=dataset.order_price_df
        )
        
        # テスト用データセット
        test_ds = Dataset(
            target_df=train_dataset.target_test_df,
            features_df=train_dataset.features_test_df,
            raw_target_df=dataset.raw_target_df,
            order_price_df=dataset.order_price_df
        )
        
        # 検証用データセットの作成（必要な場合）
        valid_ds = None
        if valid_ratio > 0:
            # 学習データの長さから検証データのサイズを計算
            if train_ds.target_df is not None and 'Date' in train_ds.target_df.index.names:
                dates = train_ds.target_df.index.get_level_values('Date').unique().sort_values()
                valid_size = int(len(dates) * valid_ratio)
                
                if valid_size > 0:
                    # 検証データの期間を決定（学習データの末尾から）
                    valid_start_date = dates[-valid_size]
                    valid_end_date = dates[-1]
                    train_end_date_new = dates[-valid_size-1]
                    
                    # 学習データと検証データに再分割
                    valid_mask = (train_ds.target_df.index.get_level_values('Date') >= valid_start_date) & \
                                (train_ds.target_df.index.get_level_values('Date') <= valid_end_date)
                    
                    valid_ds = Dataset(
                        target_df=train_ds.target_df[valid_mask],
                        features_df=train_ds.features_df[valid_mask] if train_ds.features_df is not None else None,
                        raw_target_df=dataset.raw_target_df,
                        order_price_df=dataset.order_price_df
                    )
                    
                    # 学習データを更新
                    train_mask = train_ds.target_df.index.get_level_values('Date') <= train_end_date_new
                    train_ds.target_df = train_ds.target_df[train_mask]
                    if train_ds.features_df is not None:
                        train_ds.features_df = train_ds.features_df[train_mask]
        
        return train_ds, valid_ds, test_ds