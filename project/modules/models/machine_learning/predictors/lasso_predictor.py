import pandas as pd
from typing import List
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from models.machine_learning.predictors import BasePredictor

class LassoPredictor(BasePredictor):
    """LASSOモデルの予測器クラス"""
    
    def __init__(self, target_test_df: pd.DataFrame, features_test_df: pd.DataFrame,
                 models: List[Lasso], scalers: List[StandardScaler]):
        """
        LASSOによる予測を管理します。
        
        Args:
            target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
            features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
            models (List[Lasso]): LASSOモデルを格納したリスト
            scalers (List[StandardScaler]): LASSOモデルに対応したスケーラーを格納したリスト
        """
        super().__init__(target_test_df, features_test_df, models, scalers)

    def predict(self) -> pd.DataFrame:
        """
        LASSOによる予測を行います。シングルセクターとマルチセクターの双方に対応しています。
        
        Returns:
            pd.DataFrame: 予測結果を格納したデータフレーム
        """
        if self._is_multi_sector():
            pred_result_df = self._pred_multi_sectors()
        else:
            pred_result_df = self._pred_single_sector(
                self.target_test_df, self.features_test_df, 
                self.models[0], self.scalers[0]
            )
        
        return pred_result_df

    def _pred_single_sector(self, y_test: pd.DataFrame, X_test: pd.DataFrame, 
                          model: Lasso, scaler: StandardScaler) -> pd.DataFrame:
        """
        LASSOモデルで予測して予測結果を返す関数
        """
        y_test = y_test.loc[X_test.dropna(how='any').index, :]
        X_test = X_test.loc[X_test.dropna(how='any').index, :]
        X_test_scaled = scaler.transform(X_test)  # 標準化
        y_test['Pred'] = model.predict(X_test_scaled)  # 予測

        return y_test

    def _pred_multi_sectors(self) -> pd.DataFrame:
        """
        複数セクターに関して、LASSOモデルで予測して予測結果を返す関数
        """
        y_tests = []
        sectors = self._get_sectors()
        
        # セクターごとに予測する
        for i, sector in enumerate(sectors):
            y_test = self.target_test_df[self.target_test_df.index.get_level_values('Sector') == sector]
            X_test = self.features_test_df[self.features_test_df.index.get_level_values('Sector') == sector]
            y_test = self._pred_single_sector(y_test, X_test, self.models[i], self.scalers[i])
            y_tests.append(y_test)

        pred_result_df = pd.concat(y_tests, axis=0).sort_index()
        return pred_result_df