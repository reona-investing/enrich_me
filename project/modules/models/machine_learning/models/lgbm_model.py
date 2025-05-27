import pandas as pd
import lightgbm as lgb
from typing import List, Optional
from models.machine_learning.trainers.outputs import TrainerOutputs
from models.machine_learning.models.base_model import BaseModel
from models.machine_learning.trainers.lgbm_trainer import LgbmTrainer
from models.machine_learning.predictors.lgbm_predictor import LgbmPredictor

class LgbmModel(BaseModel):
    """LightGBMモデルのファサードクラス"""
    
    def train(self, target_train_df: pd.DataFrame, features_train_df: pd.DataFrame, 
              categorical_features: Optional[List[str]] = None, **kwargs) -> TrainerOutputs:
        """
        lightGBMによる学習を管理します。
        
        Args:
            target_train_df (pd.DataFrame): 訓練用の目的変数データフレーム
            features_train_df (pd.DataFrame): 訓練用の特徴量データフレーム
            categorical_features (Optional[List[str]]): カテゴリ変数として使用する特徴量
            **kwargs: lightGBMのハイパーパラメータを任意で設定可能
            
        Returns:
            TrainerOutputs: モデルのリストとスケーラーのリストを格納したデータクラス
        """
        trainer = LgbmTrainer(target_train_df, features_train_df)
        trainer_outputs = trainer.train(categorical_features, **kwargs)
        return trainer_outputs
    
    def predict(self, target_test_df: pd.DataFrame, features_test_df: pd.DataFrame, 
                models: List[lgb.train], scalers: Optional[List] = None) -> pd.DataFrame:
        """
        lightGBMによる予測を管理します。
        
        Args:
            target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
            features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
            models (List[lgb.train]): LightGBMモデルを格納したリスト
            scalers (Optional[List]): 使用しない（LightGBMでは不要）
            
        Returns:
            pd.DataFrame: 予測結果のデータフレーム
        """
        predictor = LgbmPredictor(target_test_df, features_test_df, models)
        pred_result_df = predictor.predict()
        return pred_result_df