import pandas as pd
import lightgbm as lgb
from typing import List, Optional
from models.machine_learning.predictors.base_predictor import BasePredictor

class LgbmPredictor(BasePredictor):
    """LightGBMモデルの予測器クラス"""
    
    def __init__(self, target_test_df: pd.DataFrame, features_test_df: pd.DataFrame, 
                 models: List[lgb.train], categorical_features: Optional[List[str]] = None):
        """
        lightGBMによる予測を管理します。
        
        Args:
            target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
            features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
            models (List[lgb.train]): lightGBMモデルを格納したリスト
            categorical_features (Optional[List[str]]): カテゴリ変数の一覧
        """
        # LightGBMはスケーラーを使わないのでNoneを渡す
        super().__init__(target_test_df, features_test_df, models, scalers=None)
        self.categorical_features = categorical_features

    def predict(self) -> pd.DataFrame:
        """
        lightGBMによる予測を行います。シングルセクターとマルチセクターの双方に対応しています。
        
        Returns:
            pd.DataFrame: 予測結果を格納したデータフレーム
        """
        X_test = self.features_test_df.copy()

        # 予測時にも学習時と同じカテゴリ情報を保持するため、
        # "_cat"で終わる列はカテゴリ型に変換しておく
        categorical_cols = [col for col in X_test.columns if col.lower().endswith('_cat')]
        for col in categorical_cols:
            X_test[col] = X_test[col].astype('category')

        pred_result_df = self.target_test_df.copy()
        pred_result_df['Pred'] = self.models[0].predict(
            X_test, 
            num_iteration=self.models[0].best_iteration,
            categorical_feature=self.categorical_features
        )
        
        return pred_result_df