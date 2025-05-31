import pandas as pd
import numpy as np
from typing import Optional, Union

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
        """PCAのパラメータを学習"""
        # 基本検証
        self._validate_input(X)
        
        # 共通メタデータを保存
        self._store_fit_metadata(X)
        
        # PCA固有の処理...
        from sklearn.decomposition import PCA
        
        X_array = self._to_numeric_array(X)
        max_components = min(X_array.shape)
        self.n_components_fitted_ = min(self.n_components, max_components)
        
        self.pca_ = PCA(n_components=self.n_components_fitted_, random_state=self.random_state)
        self.pca_.fit(X_array)
        
        # 重要: fit完了をマーク
        self._mark_as_fitted(
            pca_=self.pca_,
            n_components_fitted_=self.n_components_fitted_
        )
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """PCA変換を実行"""
        # シンプルなfit状態チェック
        self._check_is_fitted()
        
        # 基本検証
        self._validate_input(X)
        
        # 変換処理...
        X_array = self._to_numeric_array(X)
        
        if self.mode == 'transform':
            result_array = self.pca_.transform(X_array)
        elif self.mode == 'components':
            result_array = self._extract_components(X_array)
        elif self.mode == 'residuals':
            result_array = self._extract_residuals(X_array)
        
        # 結果の整形...
        if isinstance(X, pd.DataFrame):
            if self.mode == 'transform':
                columns = [f'PC{i+1}' for i in range(result_array.shape[1])]
            else:
                columns = X.columns[:result_array.shape[1]]
            return pd.DataFrame(result_array, index=X.index, columns=columns)
        
        return result_array
    
    def _to_numeric_array(self, X):
        """ヘルパーメソッド"""
        if isinstance(X, pd.DataFrame):
            return X.select_dtypes(include=[np.number]).values
        return np.array(X)
    
    def _extract_components(self, X):
        """ヘルパーメソッド"""
        return self.pca_.inverse_transform(self.pca_.transform(X))
    
    def _extract_residuals(self, X):
        """ヘルパーメソッド"""
        return X - self._extract_components(X)