#%% モジュールのインポート
# 自作モジュール
import jquants_api_fetcher as fetcher
import data_pickler
# 既存モジュール
from typing import Tuple
import pandas as pd
import pickle
import os
from datetime import datetime

#%% 関数群 
class MLDataset:
    '''インスタンス変数の一覧と出力時の拡張子を，クラス変数として定義'''
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

    def __init__(self, dataset_folder_path:str=None):
        # データセット用フォルダが存在しない場合，新規作成
        if dataset_folder_path is None:
            os.mkdir(dataset_folder_path)
        # 各インスタンス変数を格納していく
        for attr_name, ext in MLDataset.instance_vars.items():
            file_path = f'{dataset_folder_path}/{attr_name}{ext}'
            obj = self._load(file_path)
            setattr(self, attr_name, obj)
        print('インスタンス情報を読み込みました。')

    def _load(self, file_path):
        '''ファイルが存在する場合のみ，適した方法で開く'''
        if os.path.exists(file_path):
            if file_path.endswith('.parquet'):
                return pd.read_parquet(file_path)
            elif file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    ml_list = pickle.load(f)
                return ml_list
        else:
            return None

    def _dump(self, attr_name, file_path):
        '''インスタンス変数をファイルとして出力する．'''
        attr = getattr(self, attr_name)
        if file_path.endswith('.parquet'):
            attr.to_parquet(file_path)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'wb') as f:
                pickle.dump(attr, f)

    def _append_next_business_day_row(self, target_df:pd.DataFrame, features_df:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''target, featuresの各データフレームに、次の営業日の行を追加'''
        next_open_date = fetcher.get_next_open_date(latest_date=features_df.index.get_level_values('Date')[-1])
        sectors = target_df.index.get_level_values('Sector').unique()
        new_rows = [[next_open_date for _ in range(len(sectors))],[sector for sector in sectors]]
        target_to_add = pd.DataFrame(index=new_rows, columns=target_df.columns)
        target_to_add.index.names = ['Date', 'Sector']
        features_to_add = pd.DataFrame(index=new_rows, columns=features_df.columns)
        features_to_add.index.names = ['Date', 'Sector']
        target_to_add = target_to_add.dropna(axis=1, how='all') # すべてがNAの列を削除
        target_df = pd.concat([target_df, target_to_add], axis=0).reset_index(drop=False)
        target_df['Date'] = pd.to_datetime(target_df['Date'])
        target_df = target_df.set_index(['Date', 'Sector'], drop=True)
        features_to_add = features_to_add.dropna(axis=1, how='all') # すべてがNAの列を削除
        features_df = pd.concat([features_df, features_to_add], axis=0).reset_index(drop=False)
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        features_df = features_df.set_index(['Date', 'Sector'], drop=True)
        return target_df, features_df

    def _split_train_test(self, df:pd.DataFrame, train_start_day:datetime, train_end_day:datetime, test_start_day:datetime, test_end_day:datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''訓練データとテストデータに分ける'''
        train_df = df[(df.index.get_level_values('Date')>=train_start_day)&(df.index.get_level_values('Date')<=train_end_day)]
        test_df = df[(df.index.get_level_values('Date')>=test_start_day)&(df.index.get_level_values('Date')<=test_end_day)]
        return train_df, test_df

    def _filter_outliers(self, group:pd.DataFrame, coef:float=3) -> pd.DataFrame:
        '''外れ値の除去（デフォルト：±3σの範囲外を除去）'''
        mean = group['Target'].mean() 
        std = group['Target'].std()
        lower_bound = mean - coef * std
        upper_bound = mean + coef * std
        return group[(group['Target'] >= lower_bound) & (group['Target'] <= upper_bound)]

    def archive_dfs(self, target_df:pd.DataFrame, features_df:pd.DataFrame,
                    train_start_day:datetime, train_end_day:datetime,
                    test_start_day:datetime, test_end_day:datetime,
                    raw_target_df: pd.DataFrame, order_price_df: pd.DataFrame,
                    outlier_theshold:float = 0, no_shift_features:list = [], 
                    add_next_business_day:bool = True):
        '''
        目的変数と特徴量を格納 :param float outlier_theshold: 外れ値除去の閾値（±何σ）？
        '''
        # 次の営業日の行を追加して、featuresの行を1日ぶん後ろにずらす
        if add_next_business_day:
            target_df, features_df = self._append_next_business_day_row(target_df, features_df)
        shift_features = [col for col in features_df.columns if col not in no_shift_features]
        features_df[shift_features] = features_df.groupby('Sector')[shift_features].shift(1) 
        features_df = features_df.loc[target_df.index, :] # features_dfのインデックスをtarget_dfにそろえる
        # trainとtestに分割する
        self.target_train_df, self.target_test_df = self._split_train_test(target_df, train_start_day, train_end_day, test_start_day, test_end_day)
        self.features_train_df, self.features_test_df = self._split_train_test(features_df, train_start_day, train_end_day, test_start_day, test_end_day)
        # 外れ値除去の閾値が設定されている場合は処理
        if outlier_theshold != 0:
            #target_trainから外れ値の行を除去
            self.target_train_df = self.target_train_df.groupby('Sector').apply(self._filter_outliers, coef=outlier_theshold).droplevel(0, axis=0)
            self.target_train_df = self.target_train_df.sort_index()
            self.features_train_df = self.features_train_df.loc[self.features_train_df.index.isin(self.target_train_df.index), :]
            # raw_target_dfとorder_price_dfも格納
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
        # 各インスタンス変数を格納していく
        existing_var = self.__dict__.keys()
        for attr_name, ext in MLDataset.instance_vars.items():
            if getattr(self, attr_name) is not None:
                file_path = f'{dataset_folder_path}/{attr_name}{ext}'
                self._dump(attr_name, file_path)
        print('インスタンス情報を保存しました。')

    def retrieve_target_and_features(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.target_train_df, self.target_test_df, self.features_train_df, self.features_test_df

    def retrieve_ml_objects(self) -> Tuple[list, list]:
        return self.ml_models, self.ml_scalers

    def retrieve_pred_result(self) -> pd.DataFrame:
        return self.pred_result_df

#%% デバッグ
if __name__ == '__main__':
    import paths 
    from IPython.display import display
    ml_dataset = MLDataset(f'{paths.ML_DATASETS_FOLDER}/New48sectors')
    ml_dataset.archive_dfs(ml_dataset.target_test_df, ml_dataset.features_test_df,
                           datetime(2014,1,1), datetime(2021,12,31), datetime(2022,1,1), datetime.today(),
                           outlier_theshold=3, raw_target_df=ml_dataset.raw_target_df, order_price_df=ml_dataset.order_price_df)
    display(ml_dataset.pred_result_df)