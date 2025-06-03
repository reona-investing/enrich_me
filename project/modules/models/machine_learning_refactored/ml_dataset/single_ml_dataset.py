import pandas as pd
from datetime import datetime
from typing import Optional

from models.machine_learning_refactored.ml_dataset.components import MLObjects, PostProcessingData, TrainTestData
from models.machine_learning_refactored.outputs import TrainerOutputs, TrainTestDatasets, EvaluationMaterials, StockSelectionMaterials

class SingleMLDataset:
    """単体機械学習データセットの統合管理"""
    
    def __init__(self, dataset_folder_path: str, name: str, init_load: bool = True):
        self.dataset_folder_path = dataset_folder_path
        self.name = name
        
        # 各コンポーネントの初期化
        self.train_test_data = TrainTestData(
            folder_path=f'{dataset_folder_path}/train_test_data', 
            init_load=init_load
        )
        self.ml_objects = MLObjects(
            folder_path=f'{dataset_folder_path}/ml_objects', 
            init_load=init_load
        )
        self.post_processing_data = PostProcessingData(
            folder_path=f'{dataset_folder_path}/post_processing_data', 
            init_load=init_load
        )

    def get_name(self) -> str:
        """オブジェクトの名称を取得"""
        return self.name

    def save(self):
        """全体を保存"""
        self.train_test_data.save_instance()
        self.ml_objects.save_instance()
        self.post_processing_data.save_instance()

    # === データアーカイブメソッド群 ===
    
    def archive_train_test_data(self, target_df: pd.DataFrame, features_df: pd.DataFrame,
                                train_start_day: datetime, train_end_day: datetime,
                                test_start_day: datetime, test_end_day: datetime,
                                outlier_threshold: float = 0, no_shift_features: list = [],
                                reuse_features_df: bool = False):
        """TrainTestData の archive メソッドを実行"""
        self.train_test_data.archive(
            target_df=target_df, features_df=features_df,
            train_start_day=train_start_day, train_end_day=train_end_day,
            test_start_day=test_start_day, test_end_day=test_end_day,
            outlier_theshold=outlier_threshold, no_shift_features=no_shift_features,
            reuse_features_df=reuse_features_df
        )

    def archive_ml_objects(self, model: object, scaler: Optional[object] = None):
        """MLObjects の archive メソッドを実行"""
        self.ml_objects.archive_ml_objects(model=model, scaler=scaler)

    def archive_post_processing_data(self, raw_target_df: pd.DataFrame, 
                                     order_price_df: pd.DataFrame, pred_result_df: pd.DataFrame):
        """PostProcessingData の archive メソッドを実行"""
        self.post_processing_data.archive_raw_target(raw_target_df=raw_target_df)
        self.post_processing_data.archive_order_price(order_price_df=order_price_df)
        self.post_processing_data.archive_pred_result(pred_result_df=pred_result_df)

    def archive_raw_target(self, raw_target_df: pd.DataFrame):
        """raw_target_dfをアーカイブ"""
        self.post_processing_data.archive_raw_target(raw_target_df=raw_target_df)

    def archive_order_price(self, order_price_df: pd.DataFrame):
        """order_price_dfをアーカイブ"""
        self.post_processing_data.archive_order_price(order_price_df=order_price_df)

    def archive_pred_result(self, pred_result_df: pd.DataFrame):
        """pred_result_dfをアーカイブ"""
        self.post_processing_data.archive_pred_result(pred_result_df=pred_result_df)

    # === プロパティでのデータアクセス ===
    
    @property
    def train_test_materials(self) -> TrainTestDatasets:
        return self.train_test_data.getter()

    @property
    def ml_object_materials(self) -> TrainerOutputs:
        return self.ml_objects.getter()

    @property
    def evaluation_materials(self) -> EvaluationMaterials:
        return self.post_processing_data.getter_evaluation()

    @property
    def stock_selection_materials(self) -> StockSelectionMaterials:
        return self.post_processing_data.getter_stock_selection()

    def copy_from_other_dataset(self, copy_from: 'SingleMLDataset'):
        """他のデータセットからすべてのインスタンス変数をコピー"""
        if not isinstance(copy_from, SingleMLDataset):
            raise TypeError("copy_fromにはSingleMLDatasetインスタンスを指定してください。")

        for component_name in ['train_test_data', 'ml_objects', 'post_processing_data']:
            source_component = getattr(copy_from, component_name)
            target_component = getattr(self, component_name)
            
            for attr_name, attr_value in vars(source_component).items():
                if attr_name != 'folder_path':
                    setattr(target_component, attr_name, attr_value)