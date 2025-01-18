import os
import pickle
import pandas as pd
from typing import Tuple
from utils.jquants_api_utils import get_next_open_date
from datetime import datetime

class MLDataset:
    def __init__(self, dataset_folder_path: str, init_load: bool = True):
        self.dataset_folder_path = dataset_folder_path
        self._initialize_instance_vars(init_load)

    def _initialize_instance_vars(self, init_load: bool):
        """インスタンス変数を初期化"""
        for attr_name, ext in FileHandler.instance_vars.items():
            file_path = f"{self.dataset_folder_path}/{attr_name}{ext}" if init_load else None
            setattr(self, attr_name, FileHandler.load(file_path))

    def copy_from_other_dataset(self, copy_from: 'MLDataset'):
        """
        他のデータセットからすべてのインスタンス変数をコピー
        copy_from (MLDataset): コピー元のデータセット
        """
        for attr_name in FileHandler.instance_vars.keys():
            setattr(self, attr_name, getattr(copy_from, attr_name))

    def archive_dfs(self, target_df: pd.DataFrame, features_df: pd.DataFrame,
                    train_start_day: datetime, train_end_day: datetime,
                    test_start_day: datetime, test_end_day: datetime,
                    raw_target_df: pd.DataFrame, order_price_df: pd.DataFrame,
                    outlier_theshold: float = 0, no_shift_features: list = [],
                    reuse_features_df_of_others: bool = False):
        """
        目的変数と特徴量を格納
        :param target_df: 目的変数のデータフレーム
        :param features_df: 特徴量のデータフレーム
        :param train_start_day: 学習データの開始日
        :param train_end_day: 学習データの終了日
        :param test_start_day: テストデータの開始日
        :param test_end_day: テストデータの終了日
        :param raw_target_df: 生の目的変数データフレーム
        :param order_price_df: 発注価格データフレーム
        :param outlier_theshold: 外れ値除去の閾値（±何σ）
        :param no_shift_features: シフトしない特徴量のリスト
        :param reuse_features_df_of_others: 特徴量を他の業種から再利用するか
        """
        # 必要に応じて次の営業日を追加
        target_df = DataProcessor._append_next_business_day_row(target_df)
        if not reuse_features_df_of_others:
            features_df = DataProcessor._append_next_business_day_row(features_df)
        
        features_df = DataProcessor._shift_features(features_df, no_shift_features)
        features_df = DataProcessor._align_index(features_df, target_df)

        # 学習データとテストデータに分割
        self.target_train_df = DataProcessor._narrow_period(target_df, train_start_day, train_end_day)
        self.target_test_df = DataProcessor._narrow_period(target_df, test_start_day, test_end_day)
        self.features_train_df = DataProcessor._narrow_period(features_df, train_start_day, train_end_day)
        self.features_test_df = DataProcessor._narrow_period(features_df, test_start_day, test_end_day)

        # 外れ値除去
        if outlier_theshold != 0:
            self.target_train_df, self.features_train_df = \
                DataProcessor._remove_outliers(self.target_train_df, self.features_train_df, outlier_theshold)

        # その他データの格納
        self.raw_target_df = raw_target_df
        self.order_price_df = order_price_df

    def archive_ml_objects(self, ml_models:list, ml_scalers:list):
        self.ml_models = ml_models
        self.ml_scalers = ml_scalers

    def archive_raw_target(self, raw_target_df:pd.DataFrame):
        self.raw_target_df = raw_target_df

    def archive_pred_result(self, pred_result_df:pd.DataFrame):
        self.pred_result_df = pred_result_df

    def save_instance(self, dataset_folder_path:str):
        """インスタンスを保存"""
        # 各インスタンス変数を格納していく
        if not os.path.exists(dataset_folder_path):
            os.makedirs(dataset_folder_path)
        for attr_name, ext in FileHandler.instance_vars.items():
            attr = getattr(self, attr_name)
            if attr is not None:
                file_path = f"{self.dataset_folder_path}/{attr_name}{ext}"
                FileHandler.save(file_path, attr)

    def retrieve_target_and_features(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        Returns:
            pd.DataFrame: 目的変数・学習用
            pd.DataFrame: 目的変数・テスト用
            pd.DataFrame: 特徴量・学習用
            pd.DataFrame: 特徴量・テスト用
        '''
        return self.target_train_df, self.target_test_df, self.features_train_df, self.features_test_df

    def retrieve_ml_objects(self) -> Tuple[list, list]:
        '''
        Returns:
            list: 機械学習のモデルをリストとして格納
            list: 機械学習のスケーラーをリストとして格納
        '''
        return self.ml_models, self.ml_scalers

    def retrieve_pred_result(self) -> pd.DataFrame:
        '''
        Return:
            pd.DataFrame: 予測結果
        '''
        return self.pred_result_df

class DataProcessor:
    @staticmethod
    def _append_next_business_day_row(df:pd.DataFrame) -> pd.DataFrame:
        '''次の営業日の行を追加'''
        next_open_date = get_next_open_date(latest_date=df.index.get_level_values('Date')[-1])
        sectors = df.index.get_level_values('Sector').unique()
        new_rows = [[next_open_date for _ in range(len(sectors))],[sector for sector in sectors]]

        data_to_add = pd.DataFrame(index=new_rows, columns=df.columns).dropna(axis=1, how='all')
        data_to_add.index.names = ['Date', 'Sector']

        df = pd.concat([df, data_to_add], axis=0).reset_index(drop=False)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.set_index(['Date', 'Sector'], drop=True)

    @staticmethod
    def _shift_features(features_df: pd.DataFrame, no_shift_features: list) -> pd.DataFrame:
        '''
        特徴量を1日シフトします。
        Args:
            features_df (DataFrame): 特徴量データフレーム
            no_shift_features(list): シフトの対象外とする特徴量
        Return:
            DataFrame: シフト後の特徴量データフレーム
        '''
        shift_features = [col for col in features_df.columns if col not in no_shift_features]
        features_df[shift_features] = features_df.groupby('Sector')[shift_features].shift(1)
        return features_df
    
    @staticmethod
    def _align_index(features_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        '''
        特徴量データフレームのインデックスを目的変数データフレームと揃える
        Args:
            features_df (DataFrame): 特徴量データフレーム
            target_df (DataFrame): 目的変数データフレーム
        Return:
            DataFrame: 特徴量データフレーム
        '''
        return features_df.loc[target_df.index, :]  # インデックスを揃える

    @staticmethod
    def _narrow_period(df:pd.DataFrame, 
                          start_day:datetime, end_day:datetime) -> pd.DataFrame:
        '''訓練データとテストデータに分ける'''
        return df[(df.index.get_level_values('Date')>=start_day)&(df.index.get_level_values('Date')<=end_day)]

    @staticmethod
    def _remove_outliers(target_train: pd.DataFrame,
                         features_train: pd.DataFrame,
                         outlier_theshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        目的変数、特徴量の各dfから、標準偏差のcoef倍を超えるデータの外れ値を除去します。
        Args:
            target_train (pd.DataFrame): 目的変数
            features_train (pd.DataFrame): 特徴量
            outlier_theshold (float): 外れ値除去の閾値（±何σ）
        '''
        target_train = target_train.groupby('Sector').apply(
            DataProcessor._filter_outliers, column_name = 'Target', coef = outlier_theshold
        ).droplevel(0, axis=0)
        target_train = target_train.sort_index()
        features_train = features_train.loc[
            features_train.index.isin(target_train.index), :
        ]
        return target_train, features_train

    @staticmethod
    def _filter_outliers(group:pd.DataFrame, column_name: str, coef: float = 3) -> pd.DataFrame:
        '''
        標準偏差のcoef倍を超えるデータの外れ値を除去します。
        Args:
            group (pd.DataFrame): 除去対象のデータ群
            column_name (str): 閾値を計算するデータ列の名称
            coef (float): 閾値計算に使用する係数
        '''
        mean = group[column_name].mean() 
        std = group[column_name].std()
        lower_bound = mean - coef * std
        upper_bound = mean + coef * std
        return group[(group[column_name] >= lower_bound) & (group[column_name] <= upper_bound)]


class FileHandler:
    instance_vars = {
        'target_train_df': '.parquet',
        'target_test_df': '.parquet',
        'features_train_df': '.parquet',
        'features_test_df': '.parquet',
        'raw_target_df': '.parquet',
        'order_price_df': '.parquet',
        'pred_result_df': '.parquet',
        'ml_models': '.pkl',
        'ml_scalers': '.pkl',
    }

    @staticmethod
    def load(file_path):
        """ファイルをロード"""
        try:
            if not file_path or not os.path.exists(file_path):
                return None
            if file_path.endswith('.parquet'):
                return pd.read_parquet(file_path)
            if file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"{file_path} の読み込みに失敗しました。: {e}")
            return None

    @staticmethod
    def save(file_path, data):
        """ファイルを保存"""
        if file_path.endswith('.parquet'):
            data.to_parquet(file_path)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

if __name__ == '__main__':
    from utils import Paths
    dataset_path = f'{Paths.ML_DATASETS_FOLDER}/New48sectors_mock'
    ml_dataset = MLDataset(dataset_path)
    ml_dataset.save_instance(dataset_path)