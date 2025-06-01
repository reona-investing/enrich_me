import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple
import warnings

from preprocessing.methods import PCAHandler


class PCAforMultiSectorTarget:
    """
    機械学習用途特化PCA前処理ファサード
    
    特定用途（ML目的変数前処理）に特化したシンプルなファサードパターン。
    内部でPCAHandlerを使用し、ML特化の前後処理を提供。
    
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
        
        # ML特化パラメータ
        self.fit_start = fit_start
        self.fit_end = fit_end
        self.target_column = target_column
        self.copy = copy
        
        # パイプラインを初期化（PCAHandlerのみを含む）
        self._pipeline = [
            PCAHandler(
                n_components=n_components,
                mode=mode,
                copy=copy,
                random_state=random_state
            )
        ]
        
        # 内部状態管理
        self._is_fitted = False
        self._original_sectors = None
        
        if self.fit_start is not None and self.fit_end is not None:
            if self.fit_start > self.fit_end:
                raise ValueError("fit_start must be earlier than fit_end")
    
    def apply_pca(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        ML用途特化のPCA前処理を実行
        
        初回実行時は学習も同時に行い、2回目以降は学習済みパラメータで変換のみ実行。
        
        Parameters
        ----------
        X : pd.DataFrame
            処理対象データ（二階層インデックス必須）
            
        Returns
        -------
        X_transformed : pd.DataFrame
            PCA処理後のデータ
        """
        # ML特化の前処理
        df_for_pca, df_for_fit = self._prepare_data(X)
        
        if not self._is_fitted:
            # 初回実行：学習 + 変換
            current_data = df_for_fit.values
            for processor in self._pipeline:
                processor.fit(current_data)
                current_data = processor.transform(current_data)
            
            # メタデータを保存
            self._original_sectors = df_for_pca.columns.tolist()
            self._is_fitted = True
            
            # 全データに対して変換を実行
            current_data = df_for_pca.values
            for processor in self._pipeline:
                current_data = processor.transform(current_data)
        else:
            # 2回目以降：変換のみ
            current_data = df_for_pca.values
            for processor in self._pipeline:
                current_data = processor.transform(current_data)
        
        # ML特化の後処理（DataFrame復元）
        return self._restore_dataframe_format(current_data, df_for_pca)
    
    def _prepare_data(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """学習・変換用データフレームを準備（ML特化処理）"""
        # データをコピー（必要に応じて）
        if self.copy:
            X = X.copy()
        
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
    
    # PCAHandlerへの委譲メソッド（パイプラインの最初の要素を使用）
    def get_explained_variance_ratio(self) -> np.ndarray:
        """各主成分の寄与率を取得"""
        if not self._is_fitted:
            raise ValueError("PCA has not been applied yet. Call apply_pca() first.")
        return self._pipeline[0].get_explained_variance_ratio()
    
    def get_cumulative_variance_ratio(self) -> np.ndarray:
        """累積寄与率を取得"""
        if not self._is_fitted:
            raise ValueError("PCA has not been applied yet. Call apply_pca() first.")
        return self._pipeline[0].get_cumulative_variance_ratio()
    
    def get_components(self) -> pd.DataFrame:
        """主成分ベクトルをDataFrame形式で取得"""
        if not self._is_fitted:
            raise ValueError("PCA has not been applied yet. Call apply_pca() first.")
        components_array = self._pipeline[0].get_components()
        
        # ML特化：セクター名を使用してDataFrame化
        component_names = [f'PC_{i:02d}' for i in range(components_array.shape[0])]
        sector_names = self._original_sectors or [f'Sector_{i}' for i in range(components_array.shape[1])]
        
        return pd.DataFrame(
            components_array,
            index=component_names,
            columns=sector_names
        )
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """変換後の特徴量名を取得"""
        return [self.target_column]
    
    def get_fit_info(self) -> dict:
        """fit状態と設定情報を取得"""
        base_info = {
            'is_fitted': self._is_fitted,
            'fit_start': self.fit_start,
            'fit_end': self.fit_end,
            'target_column': self.target_column,
        }
        
        if self._is_fitted:
            pca_handler = self._pipeline[0]  # PCAHandlerは最初の要素
            ml_specific_info = {
                'n_sectors': len(self._original_sectors) if self._original_sectors else 0,
                'original_sectors': self._original_sectors,
                'pca_mode': pca_handler.mode,
                'n_components': pca_handler.n_components,
                'explained_variance_ratio': pca_handler.get_explained_variance_ratio().tolist()
            }
            base_info.update(ml_specific_info)
        
        return base_info
    
    @property
    def is_fitted(self) -> bool:
        """fit状態を確認"""
        return self._is_fitted
    
    @property
    def pipeline(self) -> List:
        """内部のパイプラインにアクセス（デバッグ用）"""
        return self._pipeline