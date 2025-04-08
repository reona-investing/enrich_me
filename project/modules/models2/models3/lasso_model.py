"""
単一のLassoモデルを管理するクラス
"""
from typing import Optional, List, Union, Dict, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import scipy

from models2.base.base_model import BaseModel
from models2.base.params import LassoParams


class LassoModel(BaseModel):
    """
    単一のLassoモデルを管理するクラス
    """
    
    def __init__(self):
        """初期化"""
        self._model: Optional[Lasso] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: Optional[List[str]] = None
        self._feature_importances_df: Optional[pd.DataFrame] = None
    
    def train(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], params: LassoParams | None = None, **kwargs):
        """
        Lassoモデルを学習する
        
        Args:
            X: 特徴量DataFrame
            y: 目的変数Series（または単一列DataFrame）
            params: LassoParams (ハイパーパラメータをまとめたデータクラス)
            **kwargs: パラメータ辞書
        """
        # パラメータの取得と整理
        if not params:
            # 直接kwargs から LassoParams を構築
            max_features = kwargs.get("max_features", 5)
            min_features = kwargs.get("min_features", 3)
            alpha = kwargs.get("alpha", None)
            max_iter = kwargs.get("max_iter", 100000)
            tol = kwargs.get("tol", 0.00001)
            additional_params = {k: v for k, v in kwargs.items() 
                                if k not in ["max_features", "min_features", "alpha", "max_iter", "tol"]}
            
            params = LassoParams(
                max_features=max_features,
                min_features=min_features,
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
                additional_params=additional_params
            )
        
        # 特徴量名を保存
        self._feature_names = X.columns.tolist()
        
        # 欠損値のある行を削除
        not_na_indices = X.dropna(how='any').index
        X_filtered = X.loc[not_na_indices, :]
        
        if isinstance(y, pd.DataFrame):
            y_filtered = y.loc[not_na_indices, :]
            if y_filtered.shape[1] == 1:
                # 単一列DataFrameをSeriesに変換
                y_filtered = y_filtered.iloc[:, 0]
        else:
            y_filtered = y.loc[not_na_indices]
        
        # 特徴量の標準化
        self._scaler = StandardScaler().fit(X_filtered)
        X_scaled = self._scaler.transform(X_filtered)
        
        # alphaの自動探索（必要な場合）
        if params.alpha is None:
            alpha = self._search_alpha(
                X_scaled, y_filtered, 
                params.max_features, params.min_features
            )
        else:
            alpha = params.alpha
        
        # モデルの構築と学習
        model_params = params.get_model_params()
        self._model = Lasso(alpha=alpha, **model_params)
        
        if isinstance(y_filtered, pd.DataFrame) and y_filtered.shape[1] == 1:
            # DataFrameの場合は.valuesで配列に変換
            self._model.fit(X_scaled, y_filtered.values)
        else:
            self._model.fit(X_scaled, y_filtered)
        
        # 特徴量重要度の計算
        self._calculate_feature_importances()
        
        return self
    
    def _search_alpha(self, X: np.ndarray, y: Union[pd.Series, np.ndarray], 
                     max_features: int, min_features: int) -> float:
        """
        適切なalphaの値を探索する
        残る特徴量の数が、min_features以上、max_feartures以下となるように調整
        
        Args:
            X: 標準化済み特徴量行列
            y: 目的変数
            max_features: 最大特徴量数
            min_features: 最小特徴量数
            
        Returns:
            float: 最適なalpha値
        """
        # alphaの探索範囲の初期値
        min_alpha = 0.000005
        max_alpha = 0.005
        
        is_searching = True
        while is_searching:
            # ランダムサーチの準備
            model = Lasso(max_iter=100000, tol=0.00001)
            param_grid = {'alpha': scipy.stats.uniform(min_alpha, max_alpha - min_alpha)}
            random_search = RandomizedSearchCV(model, param_distributions=param_grid, 
                                              n_iter=3, cv=5, random_state=42)
            
            # ランダムサーチを実行
            if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
                random_search.fit(X, y.values)
            else:
                random_search.fit(X, y)
            
            # 最適なalphaを取得
            alpha = random_search.best_params_['alpha']
            
            # Lassoモデルを作成し、特徴量の数を確認
            model = Lasso(alpha=alpha, max_iter=100000, tol=0.00001)
            if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
                model.fit(X, y.values)
            else:
                model.fit(X, y)
            
            num_features = len(model.coef_[model.coef_ != 0])
            
            # 特徴量の数が範囲内に収まるか判定
            if num_features < min_features and max_alpha > alpha:
                max_alpha = alpha
            elif num_features > max_features and min_alpha < alpha:
                min_alpha = alpha
            else:
                is_searching = False
        
        return alpha
    
    def _calculate_feature_importances(self):
        """特徴量重要度を計算してDataFrameに格納"""
        if self._model is None or self._feature_names is None:
            return
        
        # 係数をデータフレームに変換
        self._feature_importances_df = pd.DataFrame({
            'Feature': self._feature_names,
            'Importance': self._model.coef_
        })
        
        # 重要度が0でない特徴量のみ抽出
        self._feature_importances_df = self._feature_importances_df[
            self._feature_importances_df['Importance'] != 0
        ]
        
        # 絶対値列を追加して降順ソート
        self._feature_importances_df['AbsImportance'] = self._feature_importances_df['Importance'].abs()
        self._feature_importances_df = self._feature_importances_df.sort_values(
            'AbsImportance', ascending=False
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        学習済みモデルを使用して予測を行う
        
        Args:
            X: 特徴量DataFrame
        
        Returns:
            np.ndarray: 予測値の配列
        """
        if self._model is None or self._scaler is None:
            raise ValueError("モデルが学習されていません。先にtrainメソッドを呼び出してください。")
        
        # 欠損値のある行を除外
        not_na_indices = X.dropna(how='any').index
        X_filtered = X.loc[not_na_indices, :]
        
        # 特徴量を標準化
        X_scaled = self._scaler.transform(X_filtered)
        
        # 予測を実行
        predictions = self._model.predict(X_scaled)
        
        # 元のインデックスに合わせた予測結果配列を作成
        full_predictions = np.full(X.shape[0], np.nan)
        full_predictions[X.index.get_indexer(not_na_indices)] = predictions
        
        return full_predictions
    
    @property
    def feature_importances(self) -> Optional[pd.DataFrame]:
        """特徴量重要度を取得"""
        return self._feature_importances_df
    
    @property
    def model(self) -> Optional[Lasso]:
        """内部モデルを取得"""
        return self._model
    
    @property
    def scaler(self) -> Optional[StandardScaler]:
        """スケーラーを取得"""
        return self._scaler