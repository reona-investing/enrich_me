import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple
import warnings

from preprocessing.methods.base_preprocessor import BasePreprocessor
from preprocessing.methods import PCAHandler

class PCAforMultiSectorTarget(BasePreprocessor):
    """
    機械学習用途特化PCAハンドラー（ファサード）
    
    二階層インデックスDataFrameの目的変数前処理に特化した実装。
    内部で汎用PCAHandlerを使用し、ML用途に必要な前後処理を提供。
    
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
    mode : str, default='residuals'
        'residuals': 残差を抽出
        'components': 主成分を抽出
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
                 mode: str = 'residuals',
                 copy: bool = True,
                 random_state: Optional[int] = None):
        super().__init__(copy=copy)
        
        # 汎用PCAハンドラーを内包
        self._generic_pca = PCAHandler(
            n_components=n_components,
            mode=mode,
            copy=copy,
            random_state=random_state
        )
        
        # ML特化パラメータ
        self.fit_start = fit_start
        self.fit_end = fit_end
        self.target_column = target_column
        
        self._validate_ml_params()
    
    def _validate_ml_params(self) -> None:
        """ML特化パラメータの妥当性をチェック"""
        if self.fit_start is not None and self.fit_end is not None:
            if self.fit_start > self.fit_end:
                raise ValueError("fit_start must be earlier than fit_end")
    
    def fit(self, X: pd.DataFrame, y: Optional[any] = None) -> 'PCAforMultiSectorTarget':
        """
        ML用途特化の前処理 + 汎用PCA学習
        
        Parameters
        ----------
        X : pd.DataFrame
            学習用データ（二階層インデックス必須）
        y : ignored
            sklearn互換のため
            
        Returns
        -------
        self : PCAforMultiSectorTarget
        """
        self._validate_input(X)
        self._validate_multiindex_structure(X)
        
        # ML特化の前処理
        df_for_pca, df_for_fit = self._prepare_fit_and_transform_dfs(X)
        
        # 汎用PCAに委譲して学習
        self._generic_pca.fit(df_for_fit.values)
        
        # メタデータを保存（ML特化）
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        self._original_sectors = df_for_pca.columns.tolist()
        
        # 重要: fit完了をマーク
        # 内包するPCAHandlerと、このクラス固有の重要な属性を指定
        self._mark_as_fitted(
            _generic_pca=self._generic_pca,
            _original_sectors=self._original_sectors,
            fit_start=self.fit_start,
            fit_end=self.fit_end,
            target_column=self.target_column
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        ML用途特化の前処理 + 汎用PCA変換 + 後処理
        
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
        self._validate_multiindex_structure(X)
        
        # ML特化の前処理
        df_for_pca, _ = self._prepare_fit_and_transform_dfs(X)
        
        # 汎用PCAで変換
        transformed_array = self._generic_pca.transform(df_for_pca.values)
        
        # ML特化の後処理（DataFrame復元）
        return self._restore_dataframe_format(transformed_array, df_for_pca)
    
    def _validate_multiindex_structure(self, X: pd.DataFrame) -> None:
        """マルチインデックス構造の妥当性をチェック"""
        if X.index.nlevels != 2:
            raise ValueError('DataFrame must have a 2-level MultiIndex')
        
        if self.target_column not in X.columns:
            raise ValueError(f"Column '{self.target_column}' not found in DataFrame")
    
    def _prepare_fit_and_transform_dfs(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """学習・変換用データフレームを準備（ML特化処理）"""
        # Target列をアンスタック
        df_for_pca = X[self.target_column].unstack(-1)
        
        # 期間フィルタリング
        if self.fit_start is not None:
            df_for_pca = df_for_pca[df_for_pca.index >= self.fit_start]
        
        # 学習用データの準備
        df_for_fit = df_for_pca.copy()
        if self.fit_end is not None:
            df_for_fit = df_for_fit[df_for_fit.index <= self.fit_end]
        
        # NaNチェックと処理
        if df_for_fit.isnull().any().any():
            warnings.warn("NaN values found in training data. They will be forward-filled.")
            df_for_fit = df_for_fit.fillna(method='ffill').fillna(method='bfill')
        
        if df_for_pca.isnull().any().any():
            warnings.warn("NaN values found in transform data. They will be forward-filled.")
            df_for_pca = df_for_pca.fillna(method='ffill').fillna(method='bfill')
        
        return df_for_pca, df_for_fit
    
    def _restore_dataframe_format(self, transformed_array: np.ndarray, 
                                 original_df: pd.DataFrame) -> pd.DataFrame:
        """変換後の配列をDataFrame形式に復元"""
        # 変換後の配列をDataFrameに変換
        result_df = pd.DataFrame(
            transformed_array,
            index=original_df.index,
            columns=original_df.columns
        ).sort_index(ascending=True)
        
        # 元のマルチインデックス形式に戻す
        result = result_df.stack().to_frame(self.target_column)
        result.index.names = ['Date', 'Sector']  # インデックス名を設定
        
        return result
    
    # 汎用PCAHandlerのメソッドを委譲
    def get_explained_variance_ratio(self) -> np.ndarray:
        """各主成分の寄与率を取得"""
        self._check_is_fitted()
        return self._generic_pca.get_explained_variance_ratio()
    
    def get_cumulative_variance_ratio(self) -> np.ndarray:
        """累積寄与率を取得"""
        self._check_is_fitted()
        return self._generic_pca.get_cumulative_variance_ratio()
    
    def get_components(self) -> pd.DataFrame:
        """主成分ベクトルをDataFrame形式で取得"""
        self._check_is_fitted()
        components_array = self._generic_pca.get_components()
        
        # ML特化：セクター名を使用してDataFrame化
        component_names = [f'PC_{i:02d}' for i in range(components_array.shape[0])]
        sector_names = getattr(self, '_original_sectors', 
                             [f'Sector_{i}' for i in range(components_array.shape[1])])
        
        return pd.DataFrame(
            components_array,
            index=component_names,
            columns=sector_names
        )
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """変換後の特徴量名を取得"""
        return [self.target_column]
    
    def get_fit_info(self) -> dict:
        """fit状態と設定情報を取得（オーバーライド）"""
        # 親クラスの基本情報を取得
        base_info = super().get_fit_info()
        
        # このクラス固有の情報を追加
        if self._is_fitted:
            ml_specific_info = {
                'fit_start': self.fit_start,
                'fit_end': self.fit_end,
                'target_column': self.target_column,
                'n_sectors': len(getattr(self, '_original_sectors', [])),
                'original_sectors': getattr(self, '_original_sectors', []),
                'pca_mode': self._generic_pca.mode,
                'n_components': self._generic_pca.n_components,
                'explained_variance_ratio': self._generic_pca.get_explained_variance_ratio().tolist() if hasattr(self._generic_pca, 'pca_') else None
            }
            base_info.update(ml_specific_info)
        
        return base_info