import os
import pickle
import pandas as pd
from dataclasses import dataclass
from utils.jquants_api_utils import get_next_open_date
from datetime import datetime


@dataclass
class TrainTestMaterials:
    target_train_df: pd.DataFrame
    target_test_df: pd.DataFrame
    features_train_df: pd.DataFrame
    features_test_df: pd.DataFrame

@dataclass
class MLObjectMaterials:
    models: list[object]
    scalers: list[object]

@dataclass
class EvaluationMaterials:
    pred_result_df: pd.DataFrame
    raw_target_df: pd.DataFrame

@dataclass
class StockSelectionMaterials:
    order_price_df: pd.DataFrame
    pred_result_df: pd.DataFrame



class _FileHandler:
    @staticmethod
    def load_all_instances(ins: object, folder_path: str | os.PathLike[str], instance_vars: dict[str]):
        '''
        全てのインスタンス変数を一括でロードします。
        Args:
            ins: 変数を読み込むインスタンス
            dataset_folder_path: インスタンス変数として読み込むデータの格納先フォルダパス
            instance_vars: キー：読み込むインスタンス変数の名称、値：読み込むデータのファイル形式（拡張子）
        '''
        for attr_name, ext in instance_vars.items():
            file_path = f"{folder_path}/{attr_name}{ext}"
            hided_attr_name = f"_{attr_name}"
            setattr(ins, hided_attr_name, _FileHandler.load(file_path))

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
    def save_all_instances(ins: object, folder_path: str | os.PathLike[str], instance_vars: dict[str]):
        '''
        全てのインスタンス変数を一括で保存します。
        Args:
            ins: 変数を読み込むインスタンス
            dataset_folder_path: インスタンス変数に格納されたデータの保存先フォルダパス
            instance_vars: キー：保存するインスタンス変数の名称、値：保存するデータのファイル形式（拡張子）
        '''
        # 各インスタンス変数を格納していく
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for attr_name, ext in instance_vars.items():
            hided_attr_name = f"_{attr_name}"
            attr = getattr(ins, hided_attr_name)
            if attr is not None:
                file_path = f"{folder_path}/{attr_name}{ext}"
                _FileHandler.save(file_path, attr)

    @staticmethod
    def save(file_path, data):
        """ファイルを保存"""
        if file_path.endswith('.parquet'):
            data.to_parquet(file_path)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)



class TrainTestData:
    instance_vars = {
        'target_train_df': '.parquet',
        'target_test_df': '.parquet',
        'features_train_df': '.parquet',
        'features_test_df': '.parquet',
    }
    def __init__(self, folder_path: str|os.PathLike[str], init_load: bool = True):
        '''
        目的変数と特徴量のデータフレームを格納・管理します。
        '''
        self.folder_path = folder_path
        if init_load:
            _FileHandler.load_all_instances(self, self.folder_path, TrainTestData.instance_vars)

 
    def archive(self, 
                target_df: pd.DataFrame, features_df: pd.DataFrame,
                train_start_day: datetime, train_end_day: datetime,
                test_start_day: datetime, test_end_day: datetime,
                outlier_theshold: float = 0, no_shift_features: list = [],
                reuse_features_df: bool = False):
        """
        目的変数と特徴量を格納
        Params:
            target_df: 目的変数のデータフレーム
            features_df: 特徴量のデータフレーム
            train_start_day: 学習データの開始日
            train_end_day: 学習データの終了日
            test_start_day: テストデータの開始日
            test_end_day: テストデータの終了日
            outlier_theshold: 外れ値除去の閾値（±何σ）
            no_shift_features: シフトしない特徴量のリスト
            reuse_features_df: 特徴量を他の業種から再利用するか
        """
        # 必要に応じて次の営業日を追加
        target_df = _DataProcessor._append_next_business_day_row(target_df)
        if not reuse_features_df:
            features_df = _DataProcessor._append_next_business_day_row(features_df)
        
        features_df = _DataProcessor._shift_features(features_df, no_shift_features)
        features_df = _DataProcessor._align_index(features_df, target_df)

        # 学習データとテストデータに分割
        self._target_train_df = _DataProcessor._narrow_period(target_df, train_start_day, train_end_day)
        self._target_test_df = _DataProcessor._narrow_period(target_df, test_start_day, test_end_day)
        self._features_train_df = _DataProcessor._narrow_period(features_df, train_start_day, train_end_day)
        self._features_test_df = _DataProcessor._narrow_period(features_df, test_start_day, test_end_day)

        # 外れ値除去
        if outlier_theshold != 0:
            self._target_train_df, self._features_train_df = \
                _DataProcessor._remove_outliers(self._target_train_df, self._features_train_df, outlier_theshold)


    def save_instance(self):
        """インスタンスを保存"""
        _FileHandler.save_all_instances(self, self.folder_path, TrainTestData.instance_vars)


    def getter(self) -> TrainTestMaterials:
        '''
        Returns:
            TrainTestMaterials: 目的変数と特徴量のデータフレームをまとめたデータクラス
        '''
        return TrainTestMaterials(
            target_train_df = self._target_train_df, 
            target_test_df = self._target_test_df, 
            features_train_df = self._features_train_df, 
            features_test_df = self._features_test_df
        )
    

class _DataProcessor:
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
        特徴量データフレームのインデックスを目的変数データフレームと揃えます。

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
            _DataProcessor._filter_outliers, column_name = 'Target', coef = outlier_theshold
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
    


class MLObjects:
    instance_vars = {
        'models': '.pkl',
        'scalers': '.pkl',
    }
    def __init__(self, folder_path: str | os.PathLike[str], init_load: bool = True):
        '''
        目的変数と特徴量のデータフレームを格納・管理します。
        '''
        self.folder_path = folder_path
        if init_load:
            _FileHandler.load_all_instances(self, self.folder_path, MLObjects.instance_vars) 


    def archive_ml_objects(self, models:list[object], scalers:list[object]):
        """
        機械学習のモデルとスケーラーを格納します。
        Args:
            models: 機械学習モデルを格納したリスト
            scalers: 機械学習モデルに対応するスケーラーを格納したリスト
        """
        self._models = models
        self._scalers = scalers


    def save_instance(self):
        """インスタンスを保存"""
        _FileHandler.save_all_instances(self, self.folder_path, MLObjects.instance_vars)


    def getter(self) -> MLObjectMaterials:
        '''
        Returns:
            MLObjectMaterials: 機械学習モデルとスケーラーをリストとしてまとめたデータクラス
        '''
        return MLObjectMaterials(
            models = self._models,
            scalers = self._scalers
        )



class PostProcessingData:
    instance_vars = {
        'raw_target_df': '.parquet',
        'order_price_df': '.parquet',
        'pred_result_df': '.parquet',
    }
    def __init__(self, folder_path: str | os.PathLike[str], init_load: bool = True):
        '''
        目的変数と特徴量のデータフレームを格納・管理します。
        '''
        self.folder_path = folder_path
        if init_load:
            _FileHandler.load_all_instances(self, self.folder_path, PostProcessingData.instance_vars) 


    def archive_raw_target(self, raw_target_df:pd.DataFrame):
        """
        生の目的変数を格納します。
        Args:
            raw_target_df: 生の目的変数（リターン）を格納したdf
        """
        self._raw_target_df = raw_target_df


    def archive_order_price(self, order_price_df:pd.DataFrame):
        """
        個別銘柄の発注価格を格納します。
        Args:
            raw_target_df: 個別銘柄の発注価格を格納したdf
        """
        self._order_price_df = order_price_df


    def archive_pred_result(self, pred_result_df:pd.DataFrame):
        """
        予測結果を格納します。
        Args:
            raw_target_df: 予測結果を格納したdf
        """
        self._pred_result_df = pred_result_df


    def save_instance(self):
        """インスタンスを保存"""
        _FileHandler.save_all_instances(self, self.folder_path, PostProcessingData.instance_vars)


    def getter_stock_selection(self) -> StockSelectionMaterials:
        '''
        Returns:
            StockSelectionMaterials: 発注銘柄と単位数選択に用いるデータフレーム群
        '''
        return StockSelectionMaterials(
            order_price_df = self._order_price_df,
            pred_result_df = self._pred_result_df
        )


    def getter_evaluation(self) -> EvaluationMaterials:
        '''
        Returns:
            StockSelectionMaterials: 発注銘柄と単位数選択に用いるデータフレーム群
        '''
        return EvaluationMaterials(
            pred_result_df = self._pred_result_df,
            raw_target_df = self._raw_target_df
        )


    def getter_stock_selection(self) -> StockSelectionMaterials:
        '''
        Returns:
            StockSelectionMaterials: 発注銘柄と単位数選択に用いるデータフレーム群
        '''
        return StockSelectionMaterials(
            order_price_df = self._order_price_df,
            pred_result_df = self._pred_result_df
        )


class MLDataset:

    def __init__(self, dataset_folder_path: str, init_load: bool = True):
        self.dataset_folder_path = dataset_folder_path
        TRAIN_TEST_DATA_PATH = f'{dataset_folder_path}/train_test_data'
        ML_OBJECTS_PATH = f'{dataset_folder_path}/ml_objects'
        POST_PROCESSING_DATA_PATH = f'{dataset_folder_path}/post_processing_data'
        self.train_test_data = TrainTestData(folder_path = TRAIN_TEST_DATA_PATH, init_load = init_load)
        self.ml_objects = MLObjects(folder_path = ML_OBJECTS_PATH, init_load = init_load)
        self.post_processing_data = PostProcessingData(folder_path = POST_PROCESSING_DATA_PATH, init_load = init_load)


    def save(self):
        """全体を保存"""
        self.train_test_data.save_instance()
        self.ml_objects.save_instance()
        self.post_processing_data.save_instance()


    def archive_train_test_data(self, 
                                target_df: pd.DataFrame, features_df: pd.DataFrame,
                                train_start_day: datetime, train_end_day: datetime,
                                test_start_day: datetime, test_end_day: datetime,
                                outlier_threshold: float = 0, no_shift_features: list = [],
                                reuse_features_df: bool = False):
        """
        TrainTestData の archive メソッドを実行
        """
        self.train_test_data.archive(
            target_df=target_df,
            features_df=features_df,
            train_start_day=train_start_day,
            train_end_day=train_end_day,
            test_start_day=test_start_day,
            test_end_day=test_end_day,
            outlier_theshold=outlier_threshold,
            no_shift_features=no_shift_features,
            reuse_features_df=reuse_features_df
        )


    def archive_ml_objects(self, models: list[object], scalers: list[object]):
        """
        MLObjects の archive メソッドを実行
        """
        self.ml_objects.archive_ml_objects(
            models = models, 
            scalers = scalers
        )


    def archive_post_processing_data(self, 
                                     raw_target_df: pd.DataFrame, 
                                     order_price_df: pd.DataFrame, 
                                     pred_result_df: pd.DataFrame):
        """
        PostProcessingData の archive メソッドを実行
        """
        self.post_processing_data.archive_raw_target(raw_target_df=raw_target_df)
        self.post_processing_data.archive_order_price(order_price_df=order_price_df)
        self.post_processing_data.archive_pred_result(pred_result_df=pred_result_df)        


    def archive_raw_target(self, raw_target_df: pd.DataFrame):
        """
        raw_target_dfをアーカイブ
        """
        self.post_processing_data.archive_raw_target(raw_target_df=raw_target_df)        


    def archive_order_price(self, order_price_df: pd.DataFrame):
        """
        order_price_dfをアーカイブ
        """
        self.post_processing_data.archive_order_price(order_price_df=order_price_df)


    def archive_pred_result(self, pred_result_df: pd.DataFrame):
        """
        pred_result_dfをアーカイブ
        """
        self.post_processing_data.archive_pred_result(pred_result_df=pred_result_df)


    @property
    def train_test_materials(self) -> TrainTestMaterials:
        return self.train_test_data.getter()

    @property
    def ml_object_materials(self) -> MLObjectMaterials:
        return self.ml_objects.getter()

    @property
    def evaluation_materials(self) -> EvaluationMaterials:
        return self.post_processing_data.getter_evaluation()

    @property
    def stock_selection_materials(self) -> StockSelectionMaterials:
        return self.post_processing_data.getter_stock_selection()


    def copy_from_other_dataset(self, copy_from: 'MLDataset'):
        """
        他のデータセットからすべてのインスタンス変数をコピー
        copy_from (MLDataset): コピー元のデータセット
        """
        if not isinstance(copy_from, MLDataset):
            raise TypeError("copy_fromにはMLDatasetインスタンスを指定してください。")
    
        for attrname, attrvalue in vars(copy_from.train_test_data).items():
            if attrname != 'folder_path':
                setattr(self.train_test_data, attrname, attrvalue)
        for attrname, attrvalue in vars(copy_from.ml_objects).items():
            if attrname != 'folder_path':
                setattr(self.ml_objects, attrname, attrvalue)
        for attrname, attrvalue in vars(copy_from.post_processing_data).items():
            if attrname != 'folder_path':
                setattr(self.post_processing_data, attrname, attrvalue)


if __name__ == '__main__':
    from utils.paths import Paths
    dataset_path = f'{Paths.ML_DATASETS_FOLDER}/New48sectors_mock'
    override_path =f'{Paths.ML_DATASETS_FOLDER}/New48sectors_to2023'
    ml_dataset = MLDataset(dataset_path)
    print(ml_dataset.train_test_materials.target_train_df)
    override_dataset = MLDataset(override_path)
    ml_dataset.copy_from_other_dataset(override_dataset)
    print(ml_dataset.train_test_materials.target_train_df)