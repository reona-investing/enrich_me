import pandas as pd
from typing import List
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from models.machine_learning.trainers.outputs import TrainerOutputs
from models.machine_learning.models import BaseModel
from models.machine_learning.trainers import LassoTrainer
from models.machine_learning.predictors import LassoPredictor

class LassoModel(BaseModel):
    """LASSOモデルのファサードクラス"""
    
    def train(self, target_train_df: pd.DataFrame, features_train_df: pd.DataFrame, 
              max_features: int = 5, min_features: int = 3, **kwargs) -> TrainerOutputs:
        """
        LASSOによる学習を管理します。
        
        Args:
            target_train_df (pd.DataFrame): 訓練用の目的変数データフレーム
            features_train_df (pd.DataFrame): 訓練用の特徴量データフレーム
            max_features (int): 採用する特徴量の最大値
            min_features (int): 採用する特徴量の最小値
            **kwargs: LASSOのハイパーパラメータを任意で設定可能
            
        Returns:
            TrainerOutputs: モデルのリストとスケーラーのリストを格納したデータクラス
        """
        trainer = LassoTrainer(target_train_df, features_train_df)
        trainer_outputs = trainer.train(max_features, min_features, **kwargs)
        return trainer_outputs
    
    def predict(self, target_test_df: pd.DataFrame, features_test_df: pd.DataFrame, 
                models: List[Lasso], scalers: List[StandardScaler]) -> pd.DataFrame:
        """
        LASSOによる予測を管理します。
        
        Args:
            target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
            features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
            models (List[Lasso]): LASSOモデルを格納したリスト
            scalers (List[StandardScaler]): LASSOモデルに対応したスケーラーを格納したリスト
            
        Returns:
            pd.DataFrame: 予測結果のデータフレーム
        """
        predictor = LassoPredictor(target_test_df, features_test_df, models, scalers)
        pred_result_df = predictor.predict()
        return pred_result_df