import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import scipy

from machine_learning.core.model_base import ModelBase


class LassoModel(ModelBase):
    """LASSOモデルの実装"""
    
    def __init__(self, name: str, alpha: float = 0.001, max_iter: int = 100000, tol: float = 0.00001, 
                 max_features: int = 5, min_features: int = 3, random_state: int = 42, **kwargs):
        """
        Args:
            name: モデル名
            alpha: 正則化の強さ
            max_iter: 最大反復回数
            tol: 収束条件の閾値
            max_features: 採用する特徴量の最大数
            min_features: 採用する特徴量の最小数
            random_state: 乱数シード
            **kwargs: その他のLASSOパラメータ
        """
        super().__init__(name)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.max_features = max_features
        self.min_features = min_features
        self.random_state = random_state
        self.extra_params = kwargs
        
        # モデル用のオブジェクト
        self.model = None
        self.scaler = None
    
    def train(self) -> None:
        """LASSOモデルを学習する"""
        if self.target_train_df is None or self.features_train_df is None:
            raise ValueError("訓練データがセットされていません。load_dataset()を先に実行してください。")
        
        # 欠損値のある行を削除
        not_na_indices = self.features_train_df.dropna(how='any').index
        y_train = self.target_train_df.loc[not_na_indices, :]
        X_train = self.features_train_df.loc[not_na_indices, :]
        
        # 特徴量の標準化
        self.scaler = StandardScaler().fit(X_train)
        X_scaled = self.scaler.transform(X_train)
        
        # ランダムサーチで適切なアルファを探索
        alpha = self._search_alpha(X_scaled, y_train)
        
        # 確定したモデルで学習
        self.model = Lasso(
            alpha=alpha, 
            max_iter=self.max_iter, 
            tol=self.tol, 
            random_state=self.random_state,
            **self.extra_params
        )
        self.model.fit(X_scaled, y_train[['Target']])
        
        # 特徴量重要度を計算
        self.feature_importances = self._get_feature_importances(self.model, X_train.columns)
        self.trained = True
        
        print(f"Model {self.name} trained with alpha={alpha}")
        print(f"Selected features: {len(self.feature_importances)}")
    
    def predict(self) -> pd.DataFrame:
        """学習済みモデルで予測を実行する"""
        if not self.trained or self.model is None or self.scaler is None:
            raise ValueError("モデルが学習されていません。train()を先に実行してください。")
        
        if self.target_test_df is None or self.features_test_df is None:
            raise ValueError("テストデータがセットされていません。load_dataset()を先に実行してください。")
        
        # 欠損値のある行を処理
        valid_indices = self.features_test_df.dropna(how='any').index
        X_test = self.features_test_df.loc[valid_indices, :]
        y_test = self.target_test_df.loc[valid_indices, :].copy()
        
        # 特徴量の標準化
        X_scaled = self.scaler.transform(X_test)
        
        # 予測を実行
        y_test['Pred'] = self.model.predict(X_scaled)
        
        # 予測結果を保存
        self.pred_result_df = y_test
        
        return self.pred_result_df
    
    def _search_alpha(self, X: np.ndarray, y: pd.DataFrame) -> float:
        """
        適切なalphaの値をサーチする
        残る特徴量の数が、min_features以上、max_features以下となるように調整する
        
        Args:
            X: 標準化済み特徴量の配列
            y: 目的変数のデータフレーム
            
        Returns:
            選択された適切なalpha値
        """
        min_alpha = 0.000005
        max_alpha = 0.005
        is_searching = True
        
        while is_searching:
            # ランダムサーチの準備
            model = Lasso(max_iter=self.max_iter, tol=self.tol, random_state=self.random_state)
            param_grid = {'alpha': scipy.stats.uniform(min_alpha, max_alpha - min_alpha)}
            random_search = RandomizedSearchCV(
                model, 
                param_distributions=param_grid, 
                n_iter=3, 
                cv=5, 
                random_state=self.random_state
            )

            # ランダムサーチを実行
            random_search.fit(X, y)

            # 最適なalphaを取得
            alpha = random_search.best_params_['alpha']

            # Lassoモデルを作成し、特徴量の数を確認
            model = Lasso(alpha=alpha, max_iter=self.max_iter, tol=self.tol, random_state=self.random_state)
            model.fit(X, y[['Target']])
            num_features = sum(model.coef_ != 0)

            # 特徴量の数が範囲内に収まるか判定
            if num_features < self.min_features and max_alpha > alpha:
                max_alpha = alpha
            elif num_features > self.max_features and min_alpha < alpha:
                min_alpha = alpha
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
        # 0でない重要度のみを保持
        feature_importances = feature_importances[feature_importances['importance'] != 0]
        feature_importances['abs_importance'] = feature_importances['importance'].abs()
        
        return feature_importances.sort_values(by='abs_importance', ascending=False)
    
    def get_feature_importances(self) -> pd.DataFrame:
        """学習済みモデルの特徴量重要度を取得する"""
        if self.feature_importances is None:
            raise ValueError("モデルが学習されていません。train()を先に実行してください。")
        return self.feature_importances[['importance']]