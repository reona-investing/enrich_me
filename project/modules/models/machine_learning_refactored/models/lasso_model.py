import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from typing import Optional
from models.machine_learning_refactored.outputs import TrainerOutputs
from models.machine_learning_refactored.models import BaseModel
from models.machine_learning_refactored.trainers import LassoTrainer
from models.machine_learning_refactored.predictors import LassoPredictor

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
            TrainerOutputs: モデルとスケーラーを格納したデータクラス
        """
        trainer = LassoTrainer(target_train_df, features_train_df)
        trainer_outputs = trainer.train(max_features, min_features, **kwargs)
        return trainer_outputs
    
    def predict(self, target_test_df: pd.DataFrame, features_test_df: pd.DataFrame, 
                model: Lasso, scaler: StandardScaler) -> pd.DataFrame:
        """
        LASSOによる予測を管理します。
        
        Args:
            target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
            features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
            model (Lasso): LASSOモデル
            scaler (StandardScaler): LASSOモデルに対応したスケーラー
            
        Returns:
            pd.DataFrame: 予測結果のデータフレーム
        """
        predictor = LassoPredictor(target_test_df, features_test_df, [model], [scaler])
        pred_result_df = predictor.predict()
        return pred_result_df