import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import scipy
from IPython.display import display

from machine_learning.ml_dataset.components import MachineLearningAsset
from machine_learning.models import BaseTrainer


class LassoTrainer(BaseTrainer):
    """LASSOモデルのトレーナークラス"""
    
    def train(self, model_name: str, 
              target_df: pd.DataFrame, features_df: pd.DataFrame, 
              max_features: int = 5, min_features: int = 3, **kwargs) -> MachineLearningAsset:
        """
        LASSOによる学習を行います。シングルセクターとマルチセクターの双方に対応しています。
        
        Args:
            model_name (str): モデルの名称
            target_df (pd.DataFrame): 目的変数データフレーム
            features_df (pd.DataFrame): 特徴量データフレーム
            max_features (int): 採用する特徴量の最大値
            min_features (int): 採用する特徴量の最小値
            **kwargs: LASSOのハイパーパラメータを任意で設定可能
            
        Returns:
            MachineLearningAsset: 機械学習のデータセットを格納したデータクラス
        """
        # 欠損値のある行を削除
        not_na_indices = target_df.dropna(how='any').index
        X = features_df.loc[not_na_indices, :]
        y = target_df.loc[not_na_indices, :]

        # 特徴量の標準化
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        # ランダムサーチで適切なアルファを探索
        alpha = self._search_alpha(X_scaled, y, max_features, min_features)

        # 確定したモデルで学習
        model = Lasso(alpha=alpha, max_iter=100000, tol=0.00001, **kwargs)
        model.fit(X_scaled, y[['Target']])

        # 特徴量重要度のデータフレームを返す
        feature_importances_df = self._get_feature_importances_df(model, feature_names=X.columns)
        print(alpha)
        display(feature_importances_df)

        return MachineLearningAsset(name=model_name, model=model, scaler=scaler)
    

    def _search_alpha(self, X: np.array, y: pd.DataFrame, max_features: int, min_features: int) -> float:
        """
        適切なalphaの値をサーチする。
        残る特徴量の数が、min_features以上、max_feartures以下となるように
        """
        # alphaの探索範囲の初期値を事前指定しておく
        min_alpha = 0.000005
        max_alpha = 0.005
        is_searching = True
        
        while is_searching:
            # ランダムサーチの準備
            model = Lasso(max_iter=100000, tol=0.00001)
            param_grid = {'alpha': scipy.stats.uniform(min_alpha, max_alpha - min_alpha)}
            random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=3, cv=5, random_state=42)

            # ランダムサーチを実行
            random_search.fit(X, y)

            # 最適なalphaを取得
            alpha = random_search.best_params_['alpha']

            # Lassoモデルを作成し、特徴量の数を確認
            model = Lasso(alpha=alpha, max_iter=100000, tol=0.00001)
            model.fit(X, y[['Target']])
            num_features = len(model.coef_[model.coef_ != 0])

            # 特徴量の数が範囲内に収まるか判定
            if num_features < min_features and max_alpha > alpha:
                max_alpha = alpha
            elif num_features > max_features and min_alpha < alpha:
                min_alpha = alpha
            else:
                is_searching = False

        return alpha

    def _get_feature_importances_df(self, model: Lasso, feature_names: pd.core.indexes.base.Index) -> pd.DataFrame:
        """
        feature importancesをdf化して返す
        """
        feature_importances_df = pd.DataFrame(model.coef_, index=feature_names, columns=['FI'])
        feature_importances_df = feature_importances_df[feature_importances_df['FI'] != 0]
        feature_importances_df['abs'] = abs(feature_importances_df['FI'])
        return feature_importances_df.sort_values(by='abs', ascending=False)[['FI']]