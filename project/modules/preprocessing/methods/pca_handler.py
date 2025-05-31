import pandas as pd
import numpy as np
from typing import Optional, Union
from sklearn.decomposition import PCA
import warnings

from preprocessing.methods.base_preprocessor import BasePreprocessor


class PCAHandler(BasePreprocessor):
    """
    汎用PCA処理クラス - データ形式に依存しない実装
    
    numpy配列またはDataFrameの数値データに対してPCAを適用し、
    主成分の抽出、残差の取得、またはPCA変換結果の取得を行う。
    
    Parameters
    ----------
    n_components : int
        抽出する主成分の数
    mode : str, default='components'
        'components': 主成分を抽出（逆変換して元空間に戻す）
        'residuals': 残差を抽出
        'transform': PCA変換結果を直接取得
    copy : bool, default=True
        データをコピーするかどうか
    random_state : int, optional
        乱数シード
    """
    
    def __init__(self, 
                 n_components: int,
                 mode: str = 'components',
                 copy: bool = True,
                 random_state: Optional[int] = None):
        super().__init__(copy=copy)
        self.n_components = n_components
        self.mode = mode
        self.random_state = random_state
        
        self._validate_params()
    
    def _validate_params(self) -> None:
        """パラメータの妥当性をチェック"""
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        
        if self.mode not in ['components', 'residuals', 'transform']:
            raise ValueError("mode must be 'components', 'residuals', or 'transform'")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[any] = None) -> 'PCAHandler':
        """
        PCAのパラメータを学習
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            学習用データ（2次元配列）
        y : ignored
            sklearn互換のため
            
        Returns
        -------
        self : PCAHandler
        """
        # データを数値配列に変換
        X_array = self._to_numeric_array(X)
        self._validate_array(X_array)
        
        # n_componentsの妥当性チェック
        max_components = min(X_array.shape)
        if self.n_components > max_components:
            warnings.warn(f"n_components ({self.n_components}) is greater than "
                         f"max possible components ({max_components}). "
                         f"Using {max_components} components instead.")
            self.n_components = max_components
        
        # PCAを学習
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca_.fit(X_array)
        
        # メタデータを保存
        self.n_features_in_ = X_array.shape[1]
        self.fit_data_shape_ = X_array.shape
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        PCA変換を実行
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            変換対象データ
            
        Returns
        -------
        X_transformed : np.ndarray
            変換後のデータ
        """
        
        self._check_is_fitted()
        
        # データを数値配列に変換
        X_array = self._to_numeric_array(X)
        self._validate_array(X_array)
        
        if self.mode == 'transform':
            return self.pca_.transform(X_array)
        elif self.mode == 'components':
            return self._extract_components(X_array)
        elif self.mode == 'residuals':
            return self._extract_residuals(X_array)
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[any] = None) -> np.ndarray:
        """
        学習と変換を同時実行
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            学習・変換用データ
        y : ignored
            sklearn互換のため
            
        Returns
        -------
        X_transformed : np.ndarray
            変換後のデータ
        """
        return self.fit(X, y).transform(X)
    
    def _to_numeric_array(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """データを数値配列に変換"""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, np.ndarray):
            return X
        else:
            return np.array(X)
    
    def _validate_array(self, X: np.ndarray) -> None:
        """配列の妥当性をチェック"""
        if X.ndim != 2:
            raise ValueError("Input must be a 2D array")
        
        if np.isnan(X).any():
            raise ValueError("Input contains NaN values. Please handle them before PCA.")
        
        if hasattr(self, 'n_features_in_'):
            if X.shape[1] != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
    
    def _extract_components(self, X: np.ndarray) -> np.ndarray:
        """主成分を抽出（逆変換して元空間に戻す）"""
        # PCA変換
        components_array = self.pca_.transform(X)
        # 逆変換して元の特徴量空間に戻す
        return self.pca_.inverse_transform(components_array)
    
    def _extract_residuals(self, X: np.ndarray) -> np.ndarray:
        """残差を抽出"""
        # 主成分を取得
        components_array = self._extract_components(X)
        # 残差を計算
        return X - components_array
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """各主成分の寄与率を取得"""
        self._check_is_fitted()
        return self.pca_.explained_variance_ratio_
    
    def get_cumulative_variance_ratio(self) -> np.ndarray:
        """累積寄与率を取得"""
        return np.cumsum(self.get_explained_variance_ratio())
    
    def get_components(self) -> np.ndarray:
        """主成分ベクトルを取得"""
        self._check_is_fitted()
        return self.pca_.components_