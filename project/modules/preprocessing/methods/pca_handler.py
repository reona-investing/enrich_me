"""
PCAハンドラー - scikit-learn互換版
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple
from sklearn.decomposition import PCA
import warnings

from preprocessing.methods.base_preprocessor import BasePreprocessor


class PCAHandler(BasePreprocessor):
    """
    PCA（主成分分析）を用いた次元削減・残差抽出を行うTransformer
    
    二階層インデックスを持つDataFrameに対してPCAを適用し、
    主成分の抽出または残差の取得を行う。
    
    Parameters
    ----------
    n_components : int
        抽出する主成分の数
    fit_start : datetime, optional
        学習期間の開始日。Noneの場合は全期間を使用
    fit_end : datetime, optional
        学習期間の終了日。Noneの場合は全期間を使用
    target_column : str, default='Target'
        対象となる列名
    mode : str, default='components'
        'components': 主成分を抽出
        'residuals': 残差を抽出
    copy : bool, default=True
        データをコピーするかどうか
    random_state : int, optional
        乱数シード
    """
    
    def __init__(self, 
                 n_components: int,
                 fit_start: Optional[datetime] = None,
                 fit_end: Optional[datetime] = None,
                 target_column: str = 'Target',
                 mode: str = 'components',
                 copy: bool = True,
                 random_state: Optional[int] = None):
        super().__init__(copy=copy)
        self.n_components = n_components
        self.fit_start = fit_start
        self.fit_end = fit_end
        self.target_column = target_column
        self.mode = mode
        self.random_state = random_state
        
        self._validate_params()
    
    def _validate_params(self) -> None:
        """パラメータの妥当性をチェック"""
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        
        if self.mode not in ['components', 'residuals']:
            raise ValueError("mode must be 'components' or 'residuals'")
        
        if self.fit_start is not None and self.fit_end is not None:
            if self.fit_start > self.fit_end:
                raise ValueError("fit_start must be earlier than fit_end")
    
    def fit(self, X: pd.DataFrame, y: Optional[any] = None) -> 'PCAHandler':
        """
        PCAのパラメータを学習
        
        Parameters
        ----------
        X : pd.DataFrame
            学習用データ（二階層インデックス必須）
        y : ignored
            sklearn互換のため
            
        Returns
        -------
        self : PCAHandler
        """
        self._validate_input(X)
        self._validate_index_structure(X)
        
        # 学習用データの準備
        df_for_pca, df_for_fit = self._prepare_fit_and_transform_dfs(X)
        
        # n_componentsの妥当性チェック
        max_components = min(df_for_fit.shape)
        if self.n_components > max_components:
            warnings.warn(f"n_components ({self.n_components}) is greater than "
                         f"max possible components ({max_components}). "
                         f"Using {max_components} components instead.")
            self.n_components = max_components
        
        # PCAを学習
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca_.fit(df_for_fit)
        
        # メタデータを保存
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        self.fit_data_shape_ = df_for_fit.shape
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        PCA変換を実行
        
        Parameters
        ----------
        X : pd.DataFrame
            変換対象データ
            
        Returns
        -------
        X_transformed : pd.DataFrame
            変換後のデータ
        """
        self._check_is_fitted()
        self._validate_input(X)
        self._validate_index_structure(X)
        
        # 変換用データの準備
        df_for_pca, _ = self._prepare_fit_and_transform_dfs(X)
        
        if self.mode == 'components':
            return self._extract_components(df_for_pca)
        elif self.mode == 'residuals':
            return self._extract_residuals(df_for_pca)
    
    def _validate_index_structure(self, X: pd.DataFrame) -> None:
        """インデックス構造の妥当性をチェック"""
        if X.index.nlevels != 2:
            raise ValueError('DataFrame must have a 2-level MultiIndex')
        
        if self.target_column not in X.columns:
            raise ValueError(f"Column '{self.target_column}' not found in DataFrame")
    
    def _prepare_fit_and_transform_dfs(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """学習・変換用データフレームを準備"""
        # Target列をアンスタック
        df_for_pca = X[self.target_column].unstack(-1)
        
        # 期間フィルタリング
        if self.fit_start is not None:
            df_for_pca = df_for_pca[df_for_pca.index >= self.fit_start]
        
        # 学習用データの準備
        df_for_fit = df_for_pca.copy()
        if self.fit_end is not None:
            df_for_fit = df_for_fit[df_for_fit.index <= self.fit_end]
        
        # NaNチェック
        if df_for_fit.isnull().any().any():
            warnings.warn("NaN values found in training data. They will be forward-filled.")
            df_for_fit = df_for_fit.fillna(method='ffill').fillna(method='bfill')
        
        if df_for_pca.isnull().any().any():
            warnings.warn("NaN values found in transform data. They will be forward-filled.")
            df_for_pca = df_for_pca.fillna(method='ffill').fillna(method='bfill')
        
        return df_for_pca, df_for_fit
    
    def _extract_components(self, df_for_pca: pd.DataFrame) -> pd.DataFrame:
        """主成分を抽出"""
        # PCA変換
        components_array = self.pca_.transform(df_for_pca)
        
        # 逆変換して元の特徴量空間に戻す
        reconstructed_array = self.pca_.inverse_transform(components_array)
        
        # DataFrameに変換
        reconstructed_df = pd.DataFrame(
            reconstructed_array,
            index=df_for_pca.index,
            columns=df_for_pca.columns
        ).sort_index(ascending=True)
        
        # 元の形式に戻す
        result = reconstructed_df.stack().to_frame(self.target_column)
        result.index.names = ['Date', 'Sector']  # インデックス名を設定
        
        return result
    
    def _extract_residuals(self, df_for_pca: pd.DataFrame) -> pd.DataFrame:
        """残差を抽出"""
        # 主成分を取得
        components_df = self._get_components_dataframe(df_for_pca)
        
        # 残差を計算
        residuals = (df_for_pca - components_df).stack()
        residuals_df = residuals.to_frame(self.target_column)
        residuals_df.index.names = ['Date', 'Sector']
        
        return residuals_df
    
    def _get_components_dataframe(self, df_for_pca: pd.DataFrame) -> pd.DataFrame:
        """主成分のDataFrameを取得（内部使用）"""
        components_array = self.pca_.transform(df_for_pca)
        reconstructed_array = self.pca_.inverse_transform(components_array)
        
        return pd.DataFrame(
            reconstructed_array,
            index=df_for_pca.index,
            columns=df_for_pca.columns
        )
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """各主成分の寄与率を取得"""
        self._check_is_fitted()
        return self.pca_.explained_variance_ratio_
    
    def get_cumulative_variance_ratio(self) -> np.ndarray:
        """累積寄与率を取得"""
        return np.cumsum(self.get_explained_variance_ratio())
    
    def get_components(self) -> pd.DataFrame:
        """主成分ベクトルを取得"""
        self._check_is_fitted()
        if not hasattr(self, 'pca_'):
            raise ValueError("PCA has not been fitted yet")
        
        # 元のアンスタック形状を再現するために列名を推定
        # これは完全ではないが、一般的なケースに対応
        n_sectors = self.pca_.components_.shape[1]
        component_names = [f'PC_{i:02d}' for i in range(self.n_components)]
        sector_names = [f'Sector_{i}' for i in range(n_sectors)]
        
        return pd.DataFrame(
            self.pca_.components_,
            index=component_names,
            columns=sector_names
        )
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """変換後の特徴量名を取得"""
        if hasattr(self, 'feature_names_in_'):
            return [self.target_column]  # 常にTarget列を返す
        return [self.target_column] if input_features is None else input_features


# 使用例とテスト用の関数
def create_sample_multiindex_data():
    """サンプルの二階層インデックスデータを作成"""
    import numpy as np
    
    np.random.seed(42)
    
    # 日付範囲
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    sectors = ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer']
    
    # マルチインデックス作成
    index = pd.MultiIndex.from_product([dates, sectors], names=['Date', 'Sector'])
    
    # 相関のあるランダムデータ生成
    n_samples = len(index)
    base_signal = np.sin(np.arange(n_samples) * 2 * np.pi / 365) * 10
    
    # セクター固有の変動を追加
    target_values = []
    for i, (date, sector) in enumerate(index):
        sector_effect = hash(sector) % 10 - 5  # セクター固有のバイアス
        noise = np.random.randn() * 2
        value = base_signal[i] + sector_effect + noise
        target_values.append(value)
    
    df = pd.DataFrame({'Target': target_values}, index=index)
    
    return df

def demonstrate_pca_usage():
    """PCAHandlerの使用例"""
    print("=== PCAHandler使用例 ===\n")
    
    # サンプルデータ作成
    df = create_sample_multiindex_data()
    print(f"データ形状: {df.shape}")
    print(f"インデックス: {df.index.names}")
    print(f"期間: {df.index.get_level_values('Date').min()} - {df.index.get_level_values('Date').max()}")
    print()
    
    # 学習期間設定
    fit_start = datetime(2020, 1, 1)
    fit_end = datetime(2022, 12, 31)
    
    # 1. 主成分抽出モード
    print("1. 主成分抽出:")
    pca_components = PCAHandler(
        n_components=3,
        fit_start=fit_start,
        fit_end=fit_end,
        mode='components'
    )
    
    df_components = pca_components.fit_transform(df)
    print(f"変換後データ形状: {df_components.shape}")
    print(f"寄与率: {pca_components.get_explained_variance_ratio()}")
    print(f"累積寄与率: {pca_components.get_cumulative_variance_ratio()}")
    print()
    
    # 2. 残差抽出モード
    print("2. 残差抽出:")
    pca_residuals = PCAHandler(
        n_components=2,
        fit_start=fit_start,
        fit_end=fit_end,
        mode='residuals'
    )
    
    df_residuals = pca_residuals.fit_transform(df)
    print(f"残差データ形状: {df_residuals.shape}")
    print(f"残差の統計:")
    print(df_residuals.describe())

if __name__ == "__main__":
    demonstrate_pca_usage()