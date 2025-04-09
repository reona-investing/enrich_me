from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime


class DatasetManager:
    """
    機械学習用データセットを管理するクラス
    models2パッケージ用に最適化されたデータ管理コンテナ
    """
    
    def __init__(self, dataset_path: Optional[str] = None, init_load: bool = True):
        """
        初期化
        
        Args:
            dataset_path: データセット保存パス
            init_load: 初期ロード実行フラグ
        """
        self.dataset_path = dataset_path
        
        # 主要データフレーム
        self.target_train = None
        self.target_test = None
        self.features_train = None
        self.features_test = None
        
        # 評価・分析用データ
        self.raw_target = None
        self.order_price = None
        self.pred_result = None
        
        # モデル関連オブジェクト
        self.models = []
        self.scalers = []
        self.model_metadata = {}  # セクターとモデルのマッピング情報
        
        # 初期ロード
        if dataset_path and os.path.exists(dataset_path) and init_load:
            self.load()
    
    def set_train_test_data(self, 
                            target_train: pd.DataFrame, 
                            target_test: pd.DataFrame,
                            features_train: pd.DataFrame, 
                            features_test: pd.DataFrame):
        """訓練・テストデータを直接設定"""
        self.target_train = target_train
        self.target_test = target_test
        self.features_train = features_train
        self.features_test = features_test
    
    def prepare_dataset(self, 
                        target_df: pd.DataFrame, 
                        features_df: pd.DataFrame,
                        train_start_day: datetime, 
                        train_end_day: datetime,
                        test_start_day: datetime, 
                        test_end_day: datetime,
                        outlier_threshold: float = 0, 
                        no_shift_features: List[str] = None,
                        reuse_features_df: bool = False):
        """
        原データから訓練・テストデータを生成・準備する
        
        Args:
            target_df: 目的変数データフレーム
            features_df: 特徴量データフレーム
            train_start_day: 訓練開始日
            train_end_day: 訓練終了日
            test_start_day: テスト開始日
            test_end_day: テスト終了日
            outlier_threshold: 外れ値除去閾値
            no_shift_features: シフトしない特徴量リスト
            reuse_features_df: 特徴量を他の業種から再利用するか
        """
        if no_shift_features is None:
            no_shift_features = []
            
        # 必要に応じて次の営業日を追加
        target_df = self._append_next_business_day_row(target_df)
        if not reuse_features_df:
            features_df = self._append_next_business_day_row(features_df)
        
        features_df = self._shift_features(features_df, no_shift_features)
        features_df = self._align_index(features_df, target_df)

        # 学習データとテストデータに分割
        self.target_train = self._narrow_period(target_df, train_start_day, train_end_day)
        self.target_test = self._narrow_period(target_df, test_start_day, test_end_day)
        self.features_train = self._narrow_period(features_df, train_start_day, train_end_day)
        self.features_test = self._narrow_period(features_df, test_start_day, test_end_day)
        
        # 外れ値除去
        if outlier_threshold > 0:
            self.target_train, self.features_train = \
                self._remove_outliers(self.target_train, self.features_train, outlier_threshold)
        
        # 生の目的変数を保存
        self.raw_target = target_df
    
    def save(self):
        """データセットをディスクに保存"""
        if not self.dataset_path:
            raise ValueError("保存先パスが指定されていません")
            
        # ディレクトリ構造の作成
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            
        train_test_path = os.path.join(self.dataset_path, "train_test_data")
        post_processing_path = os.path.join(self.dataset_path, "post_processing_data")
        ml_objects_path = os.path.join(self.dataset_path, "ml_objects")
        
        if not os.path.exists(train_test_path):
            os.makedirs(train_test_path)
        if not os.path.exists(post_processing_path):
            os.makedirs(post_processing_path)
        if not os.path.exists(ml_objects_path):
            os.makedirs(ml_objects_path)
        
        # 訓練・テストデータの保存
        self._save_dataframe(os.path.join(train_test_path, "target_train_df.parquet"), self.target_train)
        self._save_dataframe(os.path.join(train_test_path, "target_test_df.parquet"), self.target_test)
        self._save_dataframe(os.path.join(train_test_path, "features_train_df.parquet"), self.features_train)
        self._save_dataframe(os.path.join(train_test_path, "features_test_df.parquet"), self.features_test)
        
        # 後処理データの保存
        self._save_dataframe(os.path.join(post_processing_path, "raw_target_df.parquet"), self.raw_target)
        self._save_dataframe(os.path.join(post_processing_path, "order_price_df.parquet"), self.order_price)
        self._save_dataframe(os.path.join(post_processing_path, "pred_result_df.parquet"), self.pred_result)
        
        # モデルとスケーラーの保存
        self._save_model_objects(os.path.join(ml_objects_path, "models.pkl"), self.models)
        self._save_model_objects(os.path.join(ml_objects_path, "scalers.pkl"), self.scalers)
        
        # モデルメタデータの保存
        self._save_model_objects(os.path.join(ml_objects_path, "model_metadata.pkl"), self.model_metadata)
    
    def load(self):
        """保存済みデータセットを読み込み"""
        if not self.dataset_path:
            raise ValueError("読み込み元パスが指定されていません")
        
        train_test_path = os.path.join(self.dataset_path, "train_test_data")
        post_processing_path = os.path.join(self.dataset_path, "post_processing_data")
        ml_objects_path = os.path.join(self.dataset_path, "ml_objects")
        
        # 訓練・テストデータの読み込み
        self.target_train = self._load_dataframe(os.path.join(train_test_path, "target_train_df.parquet"))
        self.target_test = self._load_dataframe(os.path.join(train_test_path, "target_test_df.parquet"))
        self.features_train = self._load_dataframe(os.path.join(train_test_path, "features_train_df.parquet"))
        self.features_test = self._load_dataframe(os.path.join(train_test_path, "features_test_df.parquet"))
        
        # 後処理データの読み込み
        self.raw_target = self._load_dataframe(os.path.join(post_processing_path, "raw_target_df.parquet"))
        self.order_price = self._load_dataframe(os.path.join(post_processing_path, "order_price_df.parquet"))
        self.pred_result = self._load_dataframe(os.path.join(post_processing_path, "pred_result_df.parquet"))
        
        # モデルとスケーラーの読み込み
        self.models = self._load_model_objects(os.path.join(ml_objects_path, "models.pkl"))
        self.scalers = self._load_model_objects(os.path.join(ml_objects_path, "scalers.pkl"))
        
        # モデルメタデータの読み込み
        metadata_path = os.path.join(ml_objects_path, "model_metadata.pkl")
        if os.path.exists(metadata_path):
            self.model_metadata = self._load_model_objects(metadata_path)
        else:
            self.model_metadata = {}
    
    def get_sectors(self) -> List[str]:
        """データセット内のセクター一覧を取得"""
        if self.target_train is None or self.target_train.index.nlevels <= 1:
            return []
        
        return self.target_train.index.get_level_values('Sector').unique().tolist()
    
    def get_sector_data(self, sector: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        特定のセクターのデータを取得
        
        Args:
            sector: セクター名
            
        Returns:
            Tuple: (target_train, target_test, features_train, features_test)
        """
        if self.target_train is None or self.target_train.index.nlevels <= 1:
            raise ValueError("マルチインデックスのデータセットではありません")
        
        # セクターでフィルタリング
        target_train = self.target_train[self.target_train.index.get_level_values('Sector') == sector]
        target_test = self.target_test[self.target_test.index.get_level_values('Sector') == sector]
        features_train = self.features_train[self.features_train.index.get_level_values('Sector') == sector]
        features_test = self.features_test[self.features_test.index.get_level_values('Sector') == sector]
        
        return target_train, target_test, features_train, features_test
    
    def set_pred_result(self, pred_result: pd.DataFrame):
        """予測結果を設定"""
        self.pred_result = pred_result
    
    def set_order_price(self, order_price: pd.DataFrame):
        """発注価格を設定"""
        self.order_price = order_price
    
    def set_models(self, models: List[Any]):
        """モデルを設定"""
        self.models = models
    
    def set_scalers(self, scalers: List[Any]):
        """スケーラーを設定"""
        self.scalers = scalers
    
    def archive_ml_objects(self, models: List[Any], scalers: List[Any] = None):
        """モデルとスケーラーをアーカイブ（MLDatasetとの互換性用）"""
        self.models = models
        if scalers is not None:
            self.scalers = scalers
        
    def archive_pred_result(self, pred_result: pd.DataFrame):
        """予測結果をアーカイブ（MLDatasetとの互換性用）"""
        self.pred_result = pred_result
        
    def archive_order_price(self, order_price: pd.DataFrame):
        """発注価格をアーカイブ（MLDatasetとの互換性用）"""
        self.order_price = order_price
        
    def archive_raw_target(self, raw_target: pd.DataFrame):
        """生ターゲットをアーカイブ（MLDatasetとの互換性用）"""
        self.raw_target = raw_target
    
    # 以下、内部ヘルパーメソッド
    def _save_dataframe(self, path: str, df: Optional[pd.DataFrame]):
        """DataFrameを保存"""
        if df is not None:
            df.to_parquet(path)
    
    def _load_dataframe(self, path: str) -> Optional[pd.DataFrame]:
        """DataFrameを読み込み"""
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None
    
    def _save_model_objects(self, path: str, obj: Any):
        """モデルオブジェクトを保存"""
        if obj is not None:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
    
    def _load_model_objects(self, path: str) -> Any:
        """モデルオブジェクトを読み込み"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return []
    
    def _append_next_business_day_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """次の営業日の行を追加"""
        from utils.jquants_api_utils import get_next_open_date
        
        next_open_date = get_next_open_date(latest_date=df.index.get_level_values('Date')[-1])
        sectors = df.index.get_level_values('Sector').unique()
        new_rows = [[next_open_date for _ in range(len(sectors))],[sector for sector in sectors]]

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
        """特徴量データフレームのインデックスを目的変数データフレームと揃える"""
        return features_df.loc[target_df.index, :]
    
    def _narrow_period(self, df: pd.DataFrame, start_day: datetime, end_day: datetime) -> pd.DataFrame:
        """期間でデータを絞り込み"""
        return df[(df.index.get_level_values('Date') >= start_day) & 
                 (df.index.get_level_values('Date') <= end_day)]
    
    def _remove_outliers(self, target_df: pd.DataFrame, features_df: pd.DataFrame, 
                        threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """外れ値を除去"""
        target_filtered = target_df.copy()
        
        # セクターごとに外れ値を除去
        sectors = target_df.index.get_level_values('Sector').unique()
        for sector in sectors:
            sector_data = target_df[target_df.index.get_level_values('Sector') == sector]
            
            mean = sector_data['Target'].mean()
            std = sector_data['Target'].std()
            
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            valid_mask = (sector_data['Target'] >= lower_bound) & (sector_data['Target'] <= upper_bound)
            valid_indices = sector_data[valid_mask].index
            
            # セクター内の有効なデータのみを残す
            sector_mask = target_filtered.index.get_level_values('Sector') == sector
            # pd.concatを使用して行を結合
            target_filtered = pd.concat([
                target_filtered.loc[~sector_mask], 
                target_filtered.loc[valid_indices]
            ])
        
        # 特徴量も同じインデックスにフィルタリング
        features_filtered = features_df.loc[features_df.index.isin(target_filtered.index)]
        
        return target_filtered, features_filtered
        
    # 互換性のためのプロパティ
    @property
    def train_test_materials(self):
        """訓練・テストデータを取得（MLDatasetとの互換性用）"""
        from models.dataset import TrainTestMaterials
        return TrainTestMaterials(
            target_train_df=self.target_train,
            target_test_df=self.target_test,
            features_train_df=self.features_train,
            features_test_df=self.features_test
        )
    
    @property
    def ml_object_materials(self):
        """モデルとスケーラーを取得（MLDatasetとの互換性用）"""
        from models.dataset import MLObjectMaterials
        return MLObjectMaterials(
            models=self.models,
            scalers=self.scalers
        )
    
    @property
    def evaluation_materials(self):
        """評価用データを取得（MLDatasetとの互換性用）"""
        from models.dataset import EvaluationMaterials
        return EvaluationMaterials(
            pred_result_df=self.pred_result,
            raw_target_df=self.raw_target
        )
    
    @property
    def stock_selection_materials(self):
        """銘柄選択用データを取得（MLDatasetとの互換性用）"""
        from models.dataset import StockSelectionMaterials
        return StockSelectionMaterials(
            order_price_df=self.order_price,
            pred_result_df=self.pred_result
        )


class ModelDatasetConnector:
    """
    DatasetManagerとモデルを連携させるユーティリティクラス
    """
    
    @staticmethod
    def train_models(dataset: DatasetManager, model_container, params: Optional[Dict] = None):
        """
        データセットを使用してモデルコンテナを訓練
        
        Args:
            dataset: データセット
            model_container: ModelContainerオブジェクト
            params: 訓練パラメータ
        """
        # パラメータの準備
        if params is None:
            params = {}
            
        # 訓練データが存在するか確認
        if dataset.target_train is None or dataset.features_train is None:
            raise ValueError("訓練データが設定されていません")
            
        models = []
        scalers = []
            
        # マルチセクターの場合
        if dataset.target_train.index.nlevels > 1:
            sectors = dataset.get_sectors()
            
            for sector in sectors:
                model = model_container.get_model(sector)
                if model is None:
                    continue
                    
                # セクターのデータを取得
                target_train, _, features_train, _ = dataset.get_sector_data(sector)
                
                # モデルを訓練
                model.train(features_train, target_train['Target'], **params)
                
                # モデルとスケーラーを保存用リストに追加
                models.append(model.model if hasattr(model, 'model') else model)
                if hasattr(model, 'scaler'):
                    scalers.append(model.scaler)
        else:
            # シングルセクターの場合
            if len(model_container.models) > 0:
                key = next(iter(model_container.models.keys()))
                model = model_container.models[key]
                model.train(dataset.features_train, dataset.target_train['Target'], **params)
                
                # モデルとスケーラーを保存用リストに追加
                models.append(model.model if hasattr(model, 'model') else model)
                if hasattr(model, 'scaler'):
                    scalers.append(model.scaler)
        
        # 学習したモデルとスケーラーをデータセットに保存
        dataset.archive_ml_objects(models, scalers)
    
    @staticmethod
    def predict_with_models(dataset: DatasetManager, model_container):
        """
        データセットとモデルコンテナを使用して予測を実行
        
        Args:
            dataset: データセット
            model_container: ModelContainerオブジェクト
            
        Returns:
            pd.DataFrame: 予測結果 (Target列を含む)
        """
        # テストデータが存在するか確認
        if dataset.features_test is None:
            raise ValueError("テストデータが設定されていません")
            
        # コンテナで予測
        pred_df = model_container.predict(dataset.features_test)
        
        # 予測結果にターゲット値を追加
        if dataset.target_test is not None and 'Target' in dataset.target_test.columns:
            pred_df = pd.merge(
                pred_df,
                dataset.target_test[['Target']],
                left_index=True,
                right_index=True,
                how='left'
            )
        
        # 予測結果をデータセットに格納
        dataset.set_pred_result(pred_df)
        
        return pred_df