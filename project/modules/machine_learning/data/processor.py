import pandas as pd
from datetime import datetime
from typing import Tuple, List, Callable


class DataProcessor:
    """データ前処理を担当するユーティリティクラス"""
    
    @staticmethod
    def append_next_business_day_row(df: pd.DataFrame, get_next_open_date_func: Callable) -> pd.DataFrame:
        """
        次の営業日の行を追加する
        
        Args:
            df: 処理対象のデータフレーム
            get_next_open_date_func: 次の営業日を取得する関数
            
        Returns:
            次の営業日の行を追加したデータフレーム
        """
        if df is None or len(df) == 0:
            return df
            
        # セクターレベルがあるかチェック
        if 'Sector' not in df.index.names:
            return df
            
        # 次の営業日を取得
        latest_date = df.index.get_level_values('Date').max()
        next_open_date = get_next_open_date_func(latest_date=latest_date)
        
        # セクターのリストを取得
        sectors = df.index.get_level_values('Sector').unique()
        
        # 新しい行のインデックスを作成
        new_rows = [
            [next_open_date for _ in range(len(sectors))],
            [sector for sector in sectors]
        ]

        # 新しい行をデータフレームとして作成
        data_to_add = pd.DataFrame(index=new_rows, columns=df.columns).dropna(axis=1, how='all')
        data_to_add.index.names = ['Date', 'Sector']

        # 既存のデータと結合
        df = pd.concat([df, data_to_add], axis=0).reset_index(drop=False)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.set_index(['Date', 'Sector'], drop=True)

    @staticmethod
    def shift_features(features_df: pd.DataFrame, no_shift_features: List[str]) -> pd.DataFrame:
        """
        特徴量を1日シフトする
        
        Args:
            features_df: 特徴量のデータフレーム
            no_shift_features: シフトの対象外とする特徴量のリスト
            
        Returns:
            シフト後の特徴量データフレーム
        """
        if features_df is None or len(features_df) == 0:
            return features_df
            
        # コピーを作成して元のデータフレームを変更しないようにする
        features_df = features_df.copy()
        
        # インデックスにSectorがあるか確認
        if features_df.index.nlevels > 1 and 'Sector' in features_df.index.names:
            # シフト対象の特徴量を選択
            shift_features = [col for col in features_df.columns if col not in no_shift_features]
            
            # セクターごとにシフト
            features_df[shift_features] = features_df.groupby('Sector')[shift_features].shift(1)
        else:
            # Sectorがない場合は単純にシフト
            shift_features = [col for col in features_df.columns if col not in no_shift_features]
            features_df[shift_features] = features_df[shift_features].shift(1)
        
        return features_df

    @staticmethod
    def align_index(features_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量データフレームのインデックスを目的変数データフレームと揃える
        
        Args:
            features_df: 特徴量のデータフレーム
            target_df: 目的変数のデータフレーム
            
        Returns:
            インデックスを揃えた特徴量データフレーム
        """
        if features_df is None or target_df is None or len(features_df) == 0 or len(target_df) == 0:
            return features_df
            
        # インデックスを揃える
        common_indices = target_df.index.intersection(features_df.index)
        return features_df.loc[common_indices, :]

    @staticmethod
    def narrow_period(df: pd.DataFrame, start_day: datetime, end_day: datetime) -> pd.DataFrame:
        """
        指定した期間のデータを抽出する
        
        Args:
            df: 処理対象のデータフレーム
            start_day: 開始日
            end_day: 終了日
            
        Returns:
            指定期間のデータ
        """
        if df is None or len(df) == 0:
            return df
            
        # Dateがインデックスにあるか確認
        if df.index.nlevels > 1 and 'Date' in df.index.names:
            date_index = df.index.get_level_values('Date')
            return df[(date_index >= start_day) & (date_index <= end_day)]
        elif df.index.nlevels == 1 and isinstance(df.index, pd.DatetimeIndex):
            return df[(df.index >= start_day) & (df.index <= end_day)]
        else:
            # Dateがインデックスにない場合
            if 'Date' in df.columns:
                return df[(df['Date'] >= start_day) & (df['Date'] <= end_day)]
            else:
                return df  # 何もフィルタリングしない

    @staticmethod
    def remove_outliers(target_df: pd.DataFrame, features_df: pd.DataFrame, outlier_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        目的変数、特徴量の各dfから標準偏差のthreshold倍を超えるデータの外れ値を除去する
        
        Args:
            target_df: 目的変数のデータフレーム
            features_df: 特徴量のデータフレーム
            outlier_threshold: 外れ値除去の閾値（±何σ）
            
        Returns:
            外れ値を除去した目的変数と特徴量のデータフレームのタプル
        """
        if target_df is None or features_df is None or len(target_df) == 0 or len(features_df) == 0 or outlier_threshold == 0:
            return target_df, features_df
        
        # Sectorがインデックスにあるか確認
        if target_df.index.nlevels > 1 and 'Sector' in target_df.index.names:
            # 目的変数の外れ値を除去
            target_df = target_df.groupby('Sector').apply(
                DataProcessor._filter_outliers, column_name='Target', coef=outlier_threshold
            ).droplevel(0, axis=0)
            
            target_df = target_df.sort_index()
            
            # 特徴量を目的変数と同じインデックスに制限
            features_df = features_df.loc[features_df.index.isin(target_df.index), :]
        else:
            # Sectorがない場合は単純に外れ値除去
            target_df = DataProcessor._filter_outliers(target_df, 'Target', outlier_threshold)
            features_df = features_df.loc[features_df.index.isin(target_df.index), :]
        
        return target_df, features_df

    @staticmethod
    def _filter_outliers(group: pd.DataFrame, column_name: str, coef: float = 3) -> pd.DataFrame:
        """
        標準偏差のcoef倍を超えるデータの外れ値を除去する
        
        Args:
            group: 除去対象のデータ群
            column_name: 閾値を計算するデータ列の名称
            coef: 閾値計算に使用する係数
            
        Returns:
            外れ値を除去したデータフレーム
        """
        if len(group) <= 1:
            return group
            
        mean = group[column_name].mean()
        std = group[column_name].std()
        
        if std == 0:
            return group
            
        lower_bound = mean - coef * std
        upper_bound = mean + coef * std
        
        return group[(group[column_name] >= lower_bound) & (group[column_name] <= upper_bound)]