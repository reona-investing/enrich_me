import pandas as pd
import numpy as np
from typing import Optional
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import scipy
from machine_learning.params import LassoParams
from machine_learning.models import BaseModel


class LassoModel(BaseModel):
    """LASSO回帰モデル"""
    
    def __init__(self, name: str, params: Optional[LassoParams] = None):
        """
        Args:
            name: モデル名
            params: モデルパラメータ。指定しない場合はデフォルトパラメータが使用される。
        """
        super().__init__(name, params or LassoParams())
        self.model = None
        self.scaler = None
        self.feature_importances = None
    
    def train(self) -> None:
        """モデルを学習する"""
        if self.target_train_df is None or self.features_train_df is None:
            raise ValueError("訓練データがセットされていません。load_dataset()を先に実行してください。")
        
        # 欠損値のある行を削除
        not_na_indices = self.features_train_df.dropna(how='any').index
        y_train = self.target_train_df.loc[not_na_indices, :]
        X_train = self.features_train_df.loc[not_na_indices, :]
        
        # 特徴量の標準化
        self.scaler = StandardScaler().fit(X_train)
        X_scaled = self.scaler.transform(X_train)
        
        # パラメータの取得
        params = self.params
        
        # ランダムサーチで適切なアルファを探索
        alpha = self._search_alpha(
            X_scaled, 
            y_train, 
            params.max_features, 
            params.min_features,
            params.min_alpha,
            params.max_alpha,
            params.alpha_search_iterations
        )
        
        # 確定したモデルで学習
        self.model = Lasso(
            alpha=alpha, 
            max_iter=params.max_iter, 
            tol=params.tol, 
            random_state=params.random_seed,
            **params.extra_params
        )
        self.model.fit(X_scaled, y_train[['Target']])
        
        # 特徴量重要度を計算
        self.feature_importances = self._get_feature_importances(self.model, X_train.columns)
        
        print(f"Model {self.name} trained with alpha={alpha}")
        print(f"Selected features: {len(self.feature_importances)}")
    
    def predict(self) -> pd.DataFrame:
        """予測を実行する"""
        if self.model is None or self.scaler is None:
            raise ValueError("モデルが学習されていません。train()を先に実行してください。")
        
        if self.target_test_df is None or self.features_test_df is None:
            raise ValueError("テストデータがセットされていません。load_dataset()を先に実行してください。")
        
        # 欠損値のある行を処理
        valid_indices = self.features_test_df.dropna(how='any').index
        y_test = self.target_test_df.loc[valid_indices, :].copy()
        X_test = self.features_test_df.loc[valid_indices, :]
        
        # 特徴量の標準化
        X_scaled = self.scaler.transform(X_test)
        
        # 予測の実行
        y_test['Pred'] = self.model.predict(X_scaled)
        
        # 予測結果を保存
        self.pred_result_df = y_test
        
        return self.pred_result_df
    
    def _search_alpha(self, 
                      X: np.ndarray, 
                      y: pd.DataFrame, 
                      max_features: int, 
                      min_features: int,
                      min_alpha: float,
                      max_alpha: float,
                      n_iter: int) -> float:
        """
        適切なalphaの値をサーチする。
        残る特徴量の数が、min_features以上、max_features以下となるように調整する。
        
        Args:
            X: 標準化済み特徴量の配列
            y: 目的変数のデータフレーム
            max_features: 採用する特徴量の最大値
            min_features: 採用する特徴量の最小値
            min_alpha: 探索範囲の最小値
            max_alpha: 探索範囲の最大値
            n_iter: ランダム探索の回数
            
        Returns:
            選択された適切なalpha値
        """
        is_searching = True
        current_min_alpha = min_alpha
        current_max_alpha = max_alpha
        
        while is_searching:
            # ランダムサーチの準備
            model = Lasso(max_iter=self.params.max_iter, tol=self.params.tol)
            param_grid = {'alpha': scipy.stats.uniform(current_min_alpha, current_max_alpha - current_min_alpha)}
            random_search = RandomizedSearchCV(
                model, 
                param_distributions=param_grid, 
                n_iter=n_iter, 
                cv=5, 
                random_state=self.params.random_seed
            )

            # ランダムサーチを実行
            random_search.fit(X, y)

            # 最適なalphaを取得
            alpha = random_search.best_params_['alpha']

            # Lassoモデルを作成し、特徴量の数を確認
            model = Lasso(alpha=alpha, max_iter=self.params.max_iter, tol=self.params.tol)
            model.fit(X, y[['Target']])
            num_features = len(model.coef_[model.coef_ != 0])

            # 特徴量の数が範囲内に収まるか判定
            if num_features < min_features and current_max_alpha > alpha:
                current_max_alpha = alpha
            elif num_features > max_features and current_min_alpha < alpha:
                current_min_alpha = alpha
            else:
                is_searching = False

        return alpha
    
    def _get_feature_importances(self, model: Lasso, feature_names: pd.Index) -> pd.DataFrame:
        """
        特徴量重要度をデータフレーム化して返す
        
        Args:
            model: 学習済みLASSOモデル
            feature_names: 特徴量の名前のインデックス
            
        Returns:
            特徴量とその重要度を格納したデータフレーム
        """
        feature_importances = pd.DataFrame(model.coef_, index=feature_names, columns=['importance'])
        feature_importances = feature_importances[feature_importances['importance'] != 0]
        feature_importances['abs_importance'] = feature_importances['importance'].abs()
        
        return feature_importances.sort_values(by='abs_importance', ascending=False)
    
    def get_feature_importances(self) -> pd.DataFrame:
        """学習済みモデルの特徴量重要度を取得する"""
        if self.feature_importances is None:
            raise ValueError("モデルが学習されていません。train()を先に実行してください。")
        return self.feature_importances[['importance']]