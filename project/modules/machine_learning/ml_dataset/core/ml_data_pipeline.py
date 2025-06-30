from __future__ import annotations

from typing import List
import pandas as pd

from utils.jquants_api_utils import get_next_open_date

from .preprocessing_config import PreprocessingConfig


class MLDataPipeline:
    """MLDataset用の前処理パイプライン"""

    def __init__(self, ml_dataset: 'MLDataset'):
        self.ml_dataset = ml_dataset
        self._preprocessing_done = False

    def prepare_for_training(self, config: PreprocessingConfig | None = None):
        """学習用の前処理を実行"""
        if self._preprocessing_done:
            return
        config = config or self.ml_dataset.preprocessing_config
        self._prepare_data_structure(config.no_shift_features)
        if config.outlier_threshold and config.outlier_threshold > 0:
            self.remove_outliers(config.outlier_threshold)
        if config.remove_missing_data:
            self._handle_missing_data()
        self._align_dataframes()
        self._preprocessing_done = True

    def prepare_for_prediction(self):
        """予測用の前処理を実行"""
        if self._preprocessing_done:
            return
        config = self.ml_dataset.preprocessing_config
        self._prepare_data_structure(config.no_shift_features)
        if config.remove_missing_data:
            self._handle_missing_data()
        self._align_dataframes()
        self._preprocessing_done = True

    def _prepare_data_structure(self, no_shift_features: List[str]):
        """翌営業日追加→特徴量シフトを順次実行"""
        self.ml_dataset.target_df = self._append_next_business_day_row(self.ml_dataset.target_df)
        self.ml_dataset.features_df = self._append_next_business_day_row(self.ml_dataset.features_df)
        self.ml_dataset.raw_returns_df = self._append_next_business_day_row(self.ml_dataset.raw_returns_df)

        shift_features = [c for c in self.ml_dataset.features_df.columns if c not in no_shift_features]
        if shift_features:
            self.ml_dataset.features_df[shift_features] = (
                self.ml_dataset.features_df.groupby('Sector')[shift_features].shift(1)
            )

    def remove_outliers(self, threshold: float):
        """外れ値除去（セクター別、統計的手法）"""
        target_df = self.ml_dataset.target_df
        filtered = target_df.groupby('Sector').apply(
            self._filter_outliers_by_group, column_name='Target', coef=threshold
        ).droplevel(0)
        filtered = filtered.sort_index()
        self.ml_dataset.target_df = filtered
        self.ml_dataset.features_df = self.ml_dataset.features_df.loc[self.ml_dataset.features_df.index.isin(filtered.index)]
        self.ml_dataset.raw_returns_df = self.ml_dataset.raw_returns_df.loc[self.ml_dataset.raw_returns_df.index.isin(filtered.index)]

    def _align_dataframes(self):
        """3つのDataFrameのインデックスを揃える"""
        idx = self.ml_dataset.target_df.index
        idx = idx.intersection(self.ml_dataset.features_df.index)
        idx = idx.intersection(self.ml_dataset.raw_returns_df.index)
        self.ml_dataset.target_df = self.ml_dataset.target_df.loc[idx]
        self.ml_dataset.features_df = self.ml_dataset.features_df.loc[idx]
        self.ml_dataset.raw_returns_df = self.ml_dataset.raw_returns_df.loc[idx]

    def _handle_missing_data(self):
        """欠損値の処理"""
        merged = pd.concat(
            [self.ml_dataset.target_df, self.ml_dataset.features_df, self.ml_dataset.raw_returns_df],
            axis=1,
            join='inner'
        ).dropna()
        t_cols = self.ml_dataset.target_df.columns
        f_cols = self.ml_dataset.features_df.columns
        r_cols = self.ml_dataset.raw_returns_df.columns
        self.ml_dataset.target_df = merged[t_cols]
        self.ml_dataset.features_df = merged[f_cols]
        self.ml_dataset.raw_returns_df = merged[r_cols]

    @staticmethod
    def _filter_outliers_by_group(group: pd.DataFrame, column_name: str, coef: float = 3) -> pd.DataFrame:
        mean = group[column_name].mean()
        std = group[column_name].std()
        lower_bound = mean - coef * std
        upper_bound = mean + coef * std
        return group[(group[column_name] >= lower_bound) & (group[column_name] <= upper_bound)]

    @staticmethod
    def _append_next_business_day_row(df: pd.DataFrame) -> pd.DataFrame:
        next_open_date = get_next_open_date(latest_date=df.index.get_level_values('Date')[-1])
        sectors = df.index.get_level_values('Sector').unique()
        new_rows = [[next_open_date for _ in range(len(sectors))], [s for s in sectors]]
        data_to_add = pd.DataFrame(index=new_rows, columns=df.columns).dropna(axis=1, how='all')
        data_to_add.index.names = ['Date', 'Sector']
        df = pd.concat([df, data_to_add], axis=0).reset_index(drop=False)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.set_index(['Date', 'Sector'], drop=True)
