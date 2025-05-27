import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import scipy
from IPython.display import display
from models.machine_learning.trainers.outputs import TrainerOutputs
from models.machine_learning.trainers.base_trainer import BaseTrainer


class LassoTrainer(BaseTrainer):
    """LASSOモデルのトレーナークラス"""
    
    def train(self, max_features: int = 5, min_features: int = 3, **kwargs) -> TrainerOutputs:
        """
        LASSOによる学習を行います。シングルセクターとマルチセクターの双方に対応しています。
        
        Args:
            max_features (int): 採用する特徴量の最大値
            min_features (int): 採用する特徴量の最小値
            **kwargs: LASSOのハイパーパラメータを任意で設定可能
            
        Returns:
            TrainerOutputs: モデルのリストとスケーラーのリストを設定したデータクラス
        """
        if self._is_multi_sector():
            models, scalers = self._train_multi_sectors(max_features, min_features, **kwargs)
        else:
            model, scaler = self._train_single_sector(
                self.target_train_df, self.features_train_df, 
                max_features, min_features, **kwargs
            )
            models = [model]
            scalers = [scaler]
        
        return TrainerOutputs(models=models, scalers=scalers)
    
    def _train_single_sector(self, y: pd.DataFrame, X: pd.DataFrame, 
                           max_features: int, min_features: int, **kwargs) -> Tuple[Lasso, StandardScaler]:
        """
        LASSOで学習して，モデルとスケーラーを返す関数
        """
        # 欠損値のある行を削除
        not_na_indices = X.dropna(how='any').index
        y = y.loc[not_na_indices, :]
        X = X.loc[not_na_indices, :]

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

        return model, scaler

    def _train_multi_sectors(self, max_features: int, min_features: int, **kwargs) -> Tuple[List[Lasso], List[StandardScaler]]:
        """
        複数セクターに関して、LASSOで学習してモデルとスケーラーを返す関数
        """
        models = []
        scalers = []
        sectors = self._get_sectors()

        # セクターごとに学習する
        for sector in sectors:
            print(sector)
            y = self.target_train_df[self.target_train_df.index.get_level_values('Sector') == sector]
            X = self.features_train_df[self.features_train_df.index.get_level_values('Sector') == sector]
            model, scaler = self._train_single_sector(y, X, max_features, min_features, **kwargs)
            models.append(model)
            scalers.append(scaler)

        return models, scalers

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
