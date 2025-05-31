import pandas as pd
import numpy as np
from typing import Optional, Union, List
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
    
    # fitの確認に必要な属性を明示的に定義
    _FIT_REQUIRED_ATTRIBUTES = ['pca_', 'n_components_fitted_']
    
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
        # 基本検証
        self._validate_input(X)
        
        # データを数値配列に変換
        X_array = self._to_numeric_array(X)
        self._validate_array(X_array)
        
        # 共通メタデータを保存（これにより_check_is_fitted()が動作する）
        self._store_fit_metadata(X)
        
        # n_componentsの妥当性チェック
        max_components = min(X_array.shape)
        self.n_components_fitted_ = self.n_components  # 実際に使用されるcomponent数
        
        if self.n_components > max_components:
            warnings.warn(f"n_components ({self.n_components}) is greater than "
                         f"max possible components ({max_components}). "
                         f"Using {max_components} components instead.")
            self.n_components_fitted_ = max_components
        
        # PCAを学習
        self.pca_ = PCA(n_components=self.n_components_fitted_, random_state=self.random_state)
        self.pca_.fit(X_array)
        
        # 追加のメタデータ
        self.fit_data_shape_ = X_array.shape
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        PCA変換を実行
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            変換対象データ
            
        Returns
        -------
        X_transformed : np.ndarray or pd.DataFrame
            変換後のデータ（入力と同じ型）
        """
        # fit状態をチェック（改善された方法を使用）
        self._check_is_fitted()
        
        # 基本検証
        self._validate_input(X)
        
        # データを数値配列に変換
        X_array = self._to_numeric_array(X)
        self._validate_array(X_array)
        
        # 変換実行
        if self.mode == 'transform':
            result_array = self.pca_.transform(X_array)
        elif self.mode == 'components':
            result_array = self._extract_components(X_array)
        elif self.mode == 'residuals':
            result_array = self._extract_residuals(X_array)
        
        # 入力がDataFrameの場合は、適切な列名でDataFrameとして返す
        if isinstance(X, pd.DataFrame):
            if self.mode == 'transform':
                # PCA変換の場合は主成分名
                columns = [f'PC{i+1}' for i in range(result_array.shape[1])]
            else:
                # components/residualsの場合は元の列名を保持
                columns = X.columns[:result_array.shape[1]]
            
            return pd.DataFrame(result_array, index=X.index, columns=columns)
        else:
            return result_array
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[any] = None) -> Union[np.ndarray, pd.DataFrame]:
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
        X_transformed : np.ndarray or pd.DataFrame
            変換後のデータ
        """
        return self.fit(X, y).transform(X)
    
    def _to_numeric_array(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """データを数値配列に変換"""
        if isinstance(X, pd.DataFrame):
            # 数値列のみを抽出
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < len(X.columns):
                non_numeric = set(X.columns) - set(numeric_cols)
                warnings.warn(f"Non-numeric columns will be ignored: {non_numeric}")
            return X[numeric_cols].values
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
            expected_features = self.n_features_in_
            # DataFrameの場合、数値列のみを使用するので調整
            if hasattr(self, 'feature_names_in_') and isinstance(self.feature_names_in_, list):
                # 元のデータがDataFrameの場合を考慮
                pass  # より柔軟な検証
            elif X.shape[1] != expected_features:
                raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}")
    
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
        self._check_is_fitted()
        return np.cumsum(self.get_explained_variance_ratio())
    
    def get_components(self) -> np.ndarray:
        """主成分ベクトルを取得"""
        self._check_is_fitted()
        return self.pca_.components_
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """変換後の特徴量名を取得（sklearn互換）"""
        self._check_is_fitted()
        
        if self.mode == 'transform':
            # PCA変換の場合は主成分名
            return [f'PC{i+1}' for i in range(self.n_components_fitted_)]
        else:
            # components/residualsの場合は元の特徴量名
            if hasattr(self, 'feature_names_in_'):
                return self.feature_names_in_
            return input_features if input_features is not None else []


# 使用例とテスト
if __name__ == '__main__':
    def demonstrate_improved_usage():
        print("=== PCAHandler使用例 ===\n")
        
        # サンプルデータ作成
        np.random.seed(42)
        
        # 1. DataFrameでのテスト
        print("1. DataFrameでのテスト:")
        df = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randn(100),
            'text_col': ['text'] * 100  # 非数値列
        })
        
        pca_handler = PCAHandler(n_components=2, mode='components')
        
        # fit状態の確認（エラーになるはず）
        try:
            pca_handler.get_explained_variance_ratio()
        except ValueError as e:
            print(f"Expected error before fit: {e}")
        
        # fit実行
        df_transformed = pca_handler.fit_transform(df)
        print(f"Transformed DataFrame shape: {df_transformed.shape}")
        print(f"Columns: {df_transformed.columns.tolist()}")
        print(f"Explained variance ratio: {pca_handler.get_explained_variance_ratio()}")
        print()
        
        # 2. numpy arrayでのテスト
        print("2. Numpy arrayでのテスト:")
        X_array = np.random.randn(50, 4)
        
        pca_handler2 = PCAHandler(n_components=2, mode='transform')
        X_transformed = pca_handler2.fit_transform(X_array)
        
        print(f"Original shape: {X_array.shape}")
        print(f"Transformed shape: {X_transformed.shape}")
        print(f"Feature names out: {pca_handler2.get_feature_names_out()}")
        print()
        
        # 3. エラーハンドリングのテスト
        print("3. エラーハンドリングテスト:")
        try:
            # まだfitしていないインスタンス
            pca_handler3 = PCAHandler(n_components=2)
            pca_handler3.transform(X_array)
        except ValueError as e:
            print(f"Expected error: {e}")

    demonstrate_improved_usage()