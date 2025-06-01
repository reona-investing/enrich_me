import pandas as pd
import numpy as np
from typing import Optional, Union
from sklearn.decomposition import PCA

from .base_preprocessor import BasePreprocessor


class PCAHandler(BasePreprocessor):
    """
    汎用PCA処理クラス
    
    numpy配列またはDataFrameの数値データに対してPCAを適用し、
    主成分の抽出、残差の取得、またはPCA変換結果の取得を行う。
    指定期間でfitし、全期間でtransformすることが可能。
    
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
    fit_start : str, pd.Timestamp, or None, default=None
        fitに使用する開始日時
    fit_end : str, pd.Timestamp, or None, default=None
        fitに使用する終了日時
    time_column : str or None, default='Date'
        時間列の名前
    """
    
    def __init__(self, 
                 n_components: int,
                 mode: str = 'components',
                 copy: bool = True,
                 random_state: Optional[int] = None,
                 fit_start: Union[str, pd.Timestamp, None] = None,
                 fit_end: Union[str, pd.Timestamp, None] = None,
                 time_column: Optional[str] = 'Date'):
        super().__init__(copy=copy, fit_start=fit_start, fit_end=fit_end, time_column=time_column)
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
        """PCAのパラメータを学習（指定期間のデータを使用）"""
        # 基本検証
        self._validate_input(X)
        
        # 指定期間でデータをフィルタリング
        X_fit = self._filter_data_by_time(X, self.fit_start, self.fit_end)
        
        # 共通メタデータを保存（元のデータで）
        self._store_fit_metadata(X)
        
        # PCA固有の処理...
        X_array = self._to_numeric_array(X_fit)
        max_components = min(X_array.shape)
        self.n_components_fitted_ = min(self.n_components, max_components)
        
        # 欠損値の処理
        if np.isnan(X_array).any():
            # 欠損値がある行を除外してPCAを学習
            mask = ~np.isnan(X_array).any(axis=1)
            if mask.sum() == 0:
                raise ValueError("All rows contain NaN values in the fit period")
            X_clean = X_array[mask]
        else:
            X_clean = X_array
        
        self.pca_ = PCA(n_components=self.n_components_fitted_, random_state=self.random_state)
        self.pca_.fit(X_clean)
        
        # fit期間のデータ統計を保存（transform時の欠損値処理用）
        self.fit_mean_ = np.nanmean(X_array, axis=0)
        self.fit_std_ = np.nanstd(X_array, axis=0)
        
        # 重要: fit完了をマーク
        self._mark_as_fitted(
            pca_=self.pca_,
            n_components_fitted_=self.n_components_fitted_,
            fit_mean_=self.fit_mean_,
            fit_std_=self.fit_std_
        )
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """PCA変換を実行（全期間のデータに適用）"""
        # シンプルなfit状態チェック
        self._check_is_fitted()
        
        # 基本検証
        self._validate_input(X)
        
        # 変換処理...
        X_array = self._to_numeric_array(X)
        
        # 欠損値の処理（fit時の統計で補完）
        X_processed = self._handle_missing_values(X_array)
        
        if self.mode == 'transform':
            result_array = self.pca_.transform(X_processed)
        elif self.mode == 'components':
            result_array = self._extract_components(X_processed)
        elif self.mode == 'residuals':
            result_array = self._extract_residuals(X_processed)
        
        # 結果の整形...
        if isinstance(X, pd.DataFrame):
            if self.mode == 'transform':
                columns = [f'PC{i+1}' for i in range(result_array.shape[1])]
            else:
                # 時間列を除外した数値列のみを対象とする
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                if self.time_column and self.time_column in numeric_cols:
                    numeric_cols.remove(self.time_column)
                columns = numeric_cols[:result_array.shape[1]]
            
            return pd.DataFrame(result_array, index=X.index, columns=columns)
        
        return result_array
    
    def _to_numeric_array(self, X):
        """ヘルパーメソッド - 数値データを配列に変換"""
        if isinstance(X, pd.DataFrame):
            # 数値列のみを取得
            numeric_df = X.select_dtypes(include=[np.number])
            # 時間列があれば除外
            if self.time_column and self.time_column in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=[self.time_column])
            return numeric_df.values
        return np.array(X)
    
    def _handle_missing_values(self, X_array):
        """欠損値を処理（fit時の統計で補完）"""
        if not np.isnan(X_array).any():
            return X_array
        
        X_filled = X_array.copy()
        for col_idx in range(X_array.shape[1]):
            mask = np.isnan(X_array[:, col_idx])
            if mask.any():
                # fit時の平均値で補完
                X_filled[mask, col_idx] = self.fit_mean_[col_idx]
        
        return X_filled
    
    def _extract_components(self, X):
        """ヘルパーメソッド - 主成分を抽出"""
        return self.pca_.inverse_transform(self.pca_.transform(X))
    
    def _extract_residuals(self, X):
        """ヘルパーメソッド - 残差を抽出"""
        return X - self._extract_components(X)
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """寄与率を取得"""
        self._check_is_fitted()
        return self.pca_.explained_variance_ratio_
    
    def get_cumulative_explained_variance_ratio(self) -> np.ndarray:
        """累積寄与率を取得"""
        self._check_is_fitted()
        return np.cumsum(self.pca_.explained_variance_ratio_)
    
    def get_components(self) -> np.ndarray:
        """主成分ベクトルを取得"""
        self._check_is_fitted()
        return self.pca_.components_