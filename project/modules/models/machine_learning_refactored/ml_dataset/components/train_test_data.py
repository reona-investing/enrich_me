import pandas as pd
from utils.jquants_api_utils import get_next_open_date
from datetime import datetime

from models.machine_learning_refactored.ml_dataset.components.base_data_component import BaseDataComponent
from models.machine_learning_refactored.outputs import TrainTestDatasets

class TrainTestData(BaseDataComponent):
    """訓練・テストデータの管理と前処理を担当"""
    
    instance_vars = {
        'target_train_df': '.parquet',
        'target_test_df': '.parquet',
        'features_train_df': '.parquet',
        'features_test_df': '.parquet',
    }

    def archive(self, 
                target_df: pd.DataFrame, features_df: pd.DataFrame,
                train_start_day: datetime, train_end_day: datetime,
                test_start_day: datetime, test_end_day: datetime,
                outlier_theshold: float = 0, no_shift_features: list = [],
                reuse_features_df: bool = False):
        """データの前処理と分割を実行"""
        
        # データの前処理
        target_df = self._prepare_target_data(target_df)
        features_df = self._prepare_features_data(features_df, target_df, no_shift_features, reuse_features_df)
        
        # 学習・テストデータに分割
        self._split_data(target_df, features_df, train_start_day, train_end_day, test_start_day, test_end_day)
        
        # 外れ値除去
        if outlier_theshold != 0:
            self._remove_outliers(outlier_theshold)

    def _prepare_target_data(self, target_df: pd.DataFrame) -> pd.DataFrame:
        """目的変数データの前処理"""
        return self._append_next_business_day_row(target_df)
    
    def _prepare_features_data(self, features_df: pd.DataFrame, target_df: pd.DataFrame, 
                              no_shift_features: list, reuse_features_df: bool) -> pd.DataFrame:
        """特徴量データの前処理"""
        if not reuse_features_df:
            features_df = self._append_next_business_day_row(features_df)
        
        features_df = self._shift_features(features_df, no_shift_features)
        features_df = self._align_index(features_df, target_df)
        return features_df
    
    def _split_data(self, target_df: pd.DataFrame, features_df: pd.DataFrame,
                   train_start_day: datetime, train_end_day: datetime,
                   test_start_day: datetime, test_end_day: datetime):
        """データを学習・テストに分割"""
        self._target_train_df = self._narrow_period(target_df, train_start_day, train_end_day)
        self._target_test_df = self._narrow_period(target_df, test_start_day, test_end_day)
        self._features_train_df = self._narrow_period(features_df, train_start_day, train_end_day)
        self._features_test_df = self._narrow_period(features_df, test_start_day, test_end_day)
    
    def _remove_outliers(self, outlier_theshold: float):
        """外れ値除去"""
        self._target_train_df, self._features_train_df = self._filter_outliers_from_datasets(
            self._target_train_df, self._features_train_df, outlier_theshold
        )

    # === データ処理のコアメソッド群 ===
    
    def _append_next_business_day_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """次の営業日の行を追加"""
        next_open_date = get_next_open_date(latest_date=df.index.get_level_values('Date')[-1])
        sectors = df.index.get_level_values('Sector').unique()
        new_rows = [[next_open_date for _ in range(len(sectors))], [sector for sector in sectors]]

        data_to_add = pd.DataFrame(index=new_rows, columns=df.columns).dropna(axis=1, how='all')
        data_to_add.index.names = ['Date', 'Sector']

        df = pd.concat([df, data_to_add], axis=0).reset_index(drop=False)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.set_index(['Date', 'Sector'], drop=True)

    def _shift_features(self, features_df: pd.DataFrame, no_shift_features: list) -> pd.DataFrame:
        """特徴量を1日シフト"""
        shift_features = [col for col in features_df.columns if col not in no_shift_features]
        features_df[shift_features] = features_df.groupby('Sector')[shift_features].shift(1)
        return features_df

    def _align_index(self, features_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        """特徴量のインデックスを目的変数と揃える"""
        return features_df.loc[target_df.index, :]

    def _narrow_period(self, df: pd.DataFrame, start_day: datetime, end_day: datetime) -> pd.DataFrame:
        """指定期間でデータを絞り込み"""
        return df[(df.index.get_level_values('Date') >= start_day) & 
                 (df.index.get_level_values('Date') <= end_day)]

    def _filter_outliers_from_datasets(self, target_train: pd.DataFrame, features_train: pd.DataFrame,
                                      outlier_theshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """外れ値除去の実行"""
        target_train = target_train.groupby('Sector').apply(
            self._filter_outliers_by_group, column_name='Target', coef=outlier_theshold
        ).droplevel(0, axis=0)
        target_train = target_train.sort_index()
        features_train = features_train.loc[features_train.index.isin(target_train.index), :]
        return target_train, features_train

    def _filter_outliers_by_group(self, group: pd.DataFrame, column_name: str, coef: float = 3) -> pd.DataFrame:
        """グループ単位での外れ値除去"""
        mean = group[column_name].mean()
        std = group[column_name].std()
        lower_bound = mean - coef * std
        upper_bound = mean + coef * std
        return group[(group[column_name] >= lower_bound) & (group[column_name] <= upper_bound)]

    def getter(self) -> TrainTestDatasets:
        """データクラスとして返却"""
        return TrainTestDatasets(
            target_train_df=self._target_train_df,
            target_test_df=self._target_test_df,
            features_train_df=self._features_train_df,
            features_test_df=self._features_test_df
        )