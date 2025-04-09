from typing import Dict, Optional
import pandas as pd

from models2.datasets.dataset_manager import DatasetManager

class ModelDatasetConnector:
    """
    DatasetManagerとモデルを連携させるユーティリティクラス
    """
    
    @staticmethod
    def train_models(dataset: DatasetManager, model_container, params: Optional[Dict] = None, 
                    force_learn: bool = False):
        """
        データセットを使用してモデルコンテナを訓練または読み込み
        
        Args:
            dataset: データセット
            model_container: ModelContainerオブジェクト
            params: 訓練パラメータ
            force_learn: 強制的に学習するかどうか (Trueなら常に学習、Falseならファイルがあれば読み込み)
        """
        # パラメータの準備
        if params is None:
            params = {}
        
        # モデルを読み込むか学習するか決定
        should_train = force_learn
        
        # 学習しない場合は既存のモデルを読み込む
        if not should_train and dataset.models and len(dataset.models) > 0:
            print("既存のモデルをロードします")
            
            # モデルが存在する場合は読み込み済みモデルをコンテナに設定
            ModelDatasetConnector._load_models_to_container(dataset, model_container)
            return
            
        # ここから先は学習が必要な場合の処理
        print("モデルの学習を開始します")
            
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
        
        # モデルメタデータ（セクターとモデルの対応）を保存
        ModelDatasetConnector._save_model_metadata(dataset, model_container)
    
    @staticmethod
    def _load_models_to_container(dataset: DatasetManager, model_container):
        """
        保存済みモデルをモデルコンテナに読み込む
        
        Args:
            dataset: データセット
            model_container: モデルコンテナ
        """
        # セクター一覧を取得
        sectors = dataset.get_sectors()
        
        # モデルメタデータが存在する場合
        if hasattr(dataset, 'model_metadata') and dataset.model_metadata:
            for sector, model_idx in dataset.model_metadata.items():
                if sector in model_container.models and model_idx < len(dataset.models):
                    # メタデータを使ってモデルを割り当て
                    model_obj = model_container.get_model(sector)
                    model_obj._model = dataset.models[model_idx]
                    
                    # スケーラーがある場合
                    if hasattr(model_obj, '_scaler') and model_idx < len(dataset.scalers):
                        model_obj._scaler = dataset.scalers[model_idx]
        else:
            # メタデータがない場合はセクター順に割り当て
            for i, sector in enumerate(sectors):
                if i < len(dataset.models) and sector in model_container.models:
                    model_obj = model_container.get_model(sector)
                    model_obj._model = dataset.models[i]
                    
                    # スケーラーがある場合
                    if hasattr(model_obj, '_scaler') and i < len(dataset.scalers):
                        model_obj._scaler = dataset.scalers[i]
    
    @staticmethod
    def _save_model_metadata(dataset: DatasetManager, model_container):
        """
        モデルのメタデータを保存
        
        Args:
            dataset: データセット
            model_container: モデルコンテナ
        """
        # セクターとモデルインデックスの対応マップを作成
        model_metadata = {}
        sectors = dataset.get_sectors()
        
        for i, sector in enumerate(sectors):
            if sector in model_container.models:
                model_metadata[sector] = i
        
        # データセットにメタデータを保存
        dataset.model_metadata = model_metadata
        
        # メタデータをファイルに保存
        if dataset.dataset_path:
            import os
            import pickle
            
            ml_objects_path = os.path.join(dataset.dataset_path, "ml_objects")
            if not os.path.exists(ml_objects_path):
                os.makedirs(ml_objects_path)
                
            metadata_path = os.path.join(ml_objects_path, "model_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(model_metadata, f)
    
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
            
        # 保存済みモデルがあるか確認し、必要に応じて読み込む
        if not any(hasattr(model, '_model') and model._model is not None 
                  for model in model_container.models.values()):
            ModelDatasetConnector._load_models_to_container(dataset, model_container)
            
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