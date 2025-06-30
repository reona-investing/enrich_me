from dataclasses import dataclass, field
from typing import Optional, Union, Iterable, List
import pandas as pd
import pickle # モデルやスケーラーの保存用
import os # ファイル存在チェック用

from utils.timeseries import Duration
from machine_learning.ml_dataset.components import MachineLearningAsset
from machine_learning.models import BaseTrainer
from .ml_data_pipeline import MLDataPipeline
from .preprocessing_config import PreprocessingConfig

@dataclass
class MLDataset:
    dataset_path: str

    target_df: pd.DataFrame
    features_df: pd.DataFrame
    raw_returns_df: pd.DataFrame
    pred_result_df: Optional[pd.DataFrame] = None

    train_duration: Duration
    test_duration: Duration
    
    date_column: str
    model_division_column: Optional[str] = None #モデルを分割する場合、どの列の値をキーに分割するか？

    ml_assets: Union[MachineLearningAsset, List[MachineLearningAsset]] = field(default_factory=list)
    preprocessing_config: PreprocessingConfig | None = None
    data_pipeline: MLDataPipeline = field(init=False)

    def __post_init__(self):
        if self.preprocessing_config is None:
            self.preprocessing_config = PreprocessingConfig()
        self.data_pipeline = MLDataPipeline(self)

    def update_dataframes(self, new_target_data: pd.DataFrame, new_features_data: pd.DataFrame, new_raw_returns_data: pd.DataFrame):
        """
        内部のデータフレームを新しい日次データで更新します。
        このメソッドは、外部から取得した新しいデータをソートします。
        """
        self.target_df = new_target_data.sort_index()
        self.features_df = new_features_data.sort_index()
        self.raw_returns_df = new_raw_returns_data.sort_index()
        print("MLDataset内のデータフレームが更新されました。")
        # ここで更新されたDataFrameをファイルに保存するロジックを追加することも可能
        # 例: self.target_df.to_parquet("path/to/updated_target.parquet")

    def apply_preprocessing(self, config: PreprocessingConfig | None = None):
        """手動で前処理を実行"""
        self.data_pipeline.prepare_for_training(config)

    def train(self, trainer: BaseTrainer, **kwargs):
        """
        学習期間のデータを使用してモデルを学習させます。
        学習済みモデルとスケーラーはml_assetsに格納され、指定されたパスに保存されます。

        Args:
            trainer (BaseTrainer): 任意の機械学習トレーナークラス
        """
        # 前処理を自動実行
        self.data_pipeline.prepare_for_training()

        if self.model_division_column not in self.target_df.columns:
            raise ValueError(f"モデル分割列 '{self.model_division_column}' が学習データに存在しません。")
        
        print("学習を開始します...")
        # 内部のDataFrameから学習データを抽出
        target_train = self.train_duration.extract_from_df(self.target_df, datetime_column=self.date_column)
        features_train = self.train_duration.extract_from_df(self.features_df, datetime_column=self.date_column)

        # 日付とセクター（あれば）をインデックスに設定。
        if self.model_division_column:
            index_cols = [self.date_column, self.model_division_column]
        else:
            index_cols = [self.date_column]
        target_train = target_train.reset_index(drop=False).set_index(index_cols, drop=True)
        features_train = features_train.reset_index(drop=False).set_index(index_cols, drop=True)

        # 目的変数と特徴量のインデックスを揃える
        target_col_names = target_train.columns.names
        features_col_names = features_train.columns.names
        train_data = pd.merge(features_train, target_train, left_index=True, right_index=True, how='inner', suffixes=('_features', '_target'))
        target_train = train_data[target_col_names]
        features_train = train_data[features_col_names]

        if self.model_division_column:
            # セクター別モデルの場合
            self.ml_assets = [] # リストを初期化
            sectors = train_data[self.model_division_column].unique()

            for sector in sectors:
                print(f"セクター '{sector}' のモデルを学習中...")
                sector_target = target_train[target_train.index.get_level_values(self.model_division_column) == sector].copy()
                sector_features = features_train[features_train.index.get_level_values(self.model_division_column) == sector].copy()
                ml_asset = trainer.train(model_name=sector, target_df=sector_target, features_df=sector_features, **kwargs)
                self.ml_assets.append(ml_asset)
        else:
            # 単一モデルの場合
            print("単一モデルを学習中...")
            ml_asset = trainer.train(model_name='Grobal', target_df=target_train, features_df=features_train, **kwargs)
            self.ml_assets = ml_asset
        print("学習が完了しました。")


    def predict(self):
        """
        テスト期間のデータを使用して予測を実行します。
        model_load_pathが指定されていれば、そこからモデルをロードします。
        予測結果はpred_result_dfに格納され、pred_result_save_pathが指定されていればファイルに保存されます。
        """
        # 予測前の前処理を実行
        self.data_pipeline.prepare_for_prediction()

        print("予測を開始します...")
        
        # モデルがml_assetsにロードされていない場合、または明示的にパスが指定された場合にロードを試みる
        if not self.ml_assets:
            raise ValueError("モデルが学習されていません。またはロードパスが指定されていません。")

        # 内部のDataFrameからテストデータを抽出
        target_test = self.test_duration.extract_from_df(self.target_df, datetime_column=self.date_column)
        features_test = self.test_duration.extract_from_df(self.features_df, datetime_column=self.date_column)

        # 日付とセクター（あれば）をインデックスに設定。
        if self.model_division_column:
            index_cols = [self.date_column, self.model_division_column]
        else:
            index_cols = [self.date_column]
        target_test = target_test.reset_index(drop=False).set_index(index_cols, drop=True)
        features_test = features_test.reset_index(drop=False).set_index(index_cols, drop=True)

        # 目的変数と特徴量のインデックスを揃える
        target_col_names = target_test.columns.names
        features_col_names = features_test.columns.names
        test_data = pd.merge(features_test, target_test, left_index=True, right_index=True, how='inner', suffixes=('_features', '_target'))
        target_test = test_data[target_col_names]
        features_test = test_data[features_col_names]


        # ml_assetsがリスト型（複数モデル）か、単一のMachineLearningAssetsオブジェクトかを確認
        if isinstance(self.ml_assets, list): # 複数モデルの場合
            print("複数モデルで予測中...")
            all_predictions_df = []
            for ml_asset_item in self.ml_assets:
                print(f"セクター '{ml_asset_item.name}' で予測中...")
                # モデル分割列が存在する場合、そのセクターのデータのみを抽出
                target_sector = target_test[target_test[self.model_division_column] == ml_asset_item.name].copy()
                features_sector = features_test[features_test[self.model_division_column] == ml_asset_item.name].copy()

                # 予測実行
                predictions_sector = ml_asset_item.predict(features_sector)
                predictions_sector = pd.concat([target_sector, predictions_sector], axis=1)
                all_predictions_df.append(predictions_sector)
            
            # 全ての予測結果を結合
            self.pred_result_df = pd.concat(all_predictions_df, axis=0).sort_index()

        else: # 単一モデルの場合 (self.ml_assetsがMachineLearningAssetsオブジェクトの場合)
            print("単一モデルで予測中...")
            predictions = self.ml_assets.predict(target_test)
            self.pred_result_df = pd.concat([target_test, predictions], axis=1).sort_index()

        print("予測が完了しました。")

    def _save(self):
        ml_dataset_storage = MLDatasetStorage(base_path=self.dataset_path)
        self.target_df.to_parquet(ml_dataset_storage.target_df_parquet)
        self.features_df.to_parquet(ml_dataset_storage.features_df_parquet)
        self.raw_returns_df.to_parquet(ml_dataset_storage.raw_returns_df_parquet)
        self.pred_result_df.to_parquet(ml_dataset_storage.pred_result_df_parquet)
        with open(ml_dataset_storage.train_duration, 'wb') as f:
            pickle.dump(self.train_duration, f)
        with open(ml_dataset_storage.test_duration, 'wb') as f:
            pickle.dump(self.test_duration, f)
        with open(ml_dataset_storage.date_column, 'wb') as f:
            pickle.dump(self.date_column, f)
        with open(ml_dataset_storage.model_division_column, 'wb') as f:
            pickle.dump(self.model_division_column, f)
        with open(ml_dataset_storage.ml_assets, 'wb') as f:
            pickle.dump(self.ml_assets, f) 


@dataclass
class MLDatasetStorage:
    base_path: str

    def __post_init__(self):
        os.makedirs(self.base_path, exist_ok=True)
        
    @property
    def target_df_parquet(self):
        return f'{self.base_path}/target_df.parquet'
        
    @property
    def features_df_parquet(self):
        return f'{self.base_path}/features_df.parquet'

    @property
    def raw_returns_df_parquet(self):
        return f'{self.base_path}/raw_returns_df.parquet'    
        
    @property
    def pred_result_df_parquet(self):
        return f'{self.base_path}/pred_result_df.parquet'
    
    @property
    def train_duration(self):
        return f'{self.base_path}/train_duration.pickle'

    @property
    def test_duration(self):
        return f'{self.base_path}/test_duration.pickle'
        
    @property
    def date_column(self):
        return f'{self.base_path}/date_column.pickle'
    
    @property
    def model_division_column(self):
        return f'{self.base_path}/model_division_column.pickle'
        
    @property
    def ml_assets(self):
        return f'{self.base_path}/ml_assets.pickle'