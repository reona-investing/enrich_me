import pandas as pd
import numpy as np
from typing import Optional, Union, List
from utils.timeseries import Duration
from sklearn.preprocessing import StandardScaler

from .base_preprocessor import BasePreprocessor


class Standardizer(BasePreprocessor):
    """
    指定期間でfitする標準化Transformer
    
    BasePreprocessorを継承し、統一されたインターフェースを提供。
    指定期間でStandardScalerをfitし、全期間でtransformすることが可能。
    
    Parameters
    ----------
    with_mean : bool, default=True
        平均を0にするかどうか（標準化）
    with_std : bool, default=True
        標準偏差を1にするかどうか（標準化）
    target_columns : list of str or None, default=None
        標準化対象の列名。Noneの場合は全ての数値列を対象
    copy : bool, default=True
        データをコピーするかどうか
    fit_duration : Duration or None, default=None
        fitに使用する期間
    time_column : str
        時間列の名前
        
    Attributes
    ----------
    scaler_ : StandardScaler
        fit済みのStandardScaler
    target_columns_ : list of str
        実際に標準化対象となった列名
    mean_ : array-like
        fit期間の平均値
    scale_ : array-like
        fit期間のスケール（標準偏差）
    var_ : array-like
        fit期間の分散
    """
    
    def __init__(self,
                 *,
                 with_mean: bool = True,
                 with_std: bool = True,
                 target_columns: Optional[List[str]] = None,
                 copy: bool = True,
                 fit_duration: Optional[Duration] = None,
                 time_column: str = 'Date'):
        super().__init__(copy=copy, fit_duration=fit_duration, time_column=time_column)
        self.with_mean = with_mean
        self.with_std = with_std
        self.target_columns = target_columns
    
    def fit(self, X: pd.DataFrame, y: Optional[any] = None) -> 'Standardizer':
        """標準化のパラメータを学習（指定期間のデータを使用）"""
        # 基本検証
        self._validate_input(X)
        
        # 指定期間でデータをフィルタリング
        if self.fit_duration is not None:
            X_fit = self.fit_duration.extract_from_df(X, self.time_column)
            if X_fit.empty:
                raise ValueError(
                    f"指定された期間({self.fit_duration.start}～{self.fit_duration.end})のデータが存在しません。"
                )
        else:
            X_fit = X

        # 共通メタデータを保存（元のデータで）
        self._store_fit_metadata(X)
        
        # 標準化対象の列を決定
        self.target_columns_ = self._get_target_columns(X)
        
        # 標準化用のデータを準備
        X_fit_array = self._extract_target_data(X_fit)
        
        # StandardScalerでfit
        self.scaler_ = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        self.scaler_.fit(X_fit_array)
        
        # 統計量を属性として保存（アクセス用）
        if hasattr(self.scaler_, 'mean_'):
            self.mean_ = self.scaler_.mean_
        if hasattr(self.scaler_, 'scale_'):
            self.scale_ = self.scaler_.scale_
        if hasattr(self.scaler_, 'var_'):
            self.var_ = self.scaler_.var_

        

        # 重要: fit完了をマーク
        self._mark_as_fitted(
            scaler_=self.scaler_,
            target_columns_=self.target_columns_,
            mean_=getattr(self, 'mean_', None),
            scale_=getattr(self, 'scale_', None),
            var_=getattr(self, 'var_', None)
        )
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """標準化を実行（全期間のデータに適用）"""
        # シンプルなfit状態チェック
        self._check_is_fitted(['scaler_', 'target_columns_'])
        
        # 基本検証
        self._validate_input(X)
        
        # 出力データの準備
        X_transformed = self._prepare_output(X)
        
        # 標準化対象のデータを抽出
        X_target = self._extract_target_data(X_transformed)
        
        # 標準化を実行
        X_standardized = self.scaler_.transform(X_target)
        
        # 結果を元の形式に戻す
        X_transformed = self._restore_data_format(X_transformed, X_standardized)
        
        return X_transformed
    
    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """標準化を逆変換（標準化前の値に戻す）"""
        # fit状態チェック
        self._check_is_fitted(['scaler_', 'target_columns_'])
        
        # 基本検証
        self._validate_input(X)
        
        # 出力データの準備
        X_restored = self._prepare_output(X)
        
        # 逆変換対象のデータを抽出
        X_target = self._extract_target_data(X_restored)
        
        # 逆変換を実行
        X_inverse = self.scaler_.inverse_transform(X_target)
        
        # 結果を元の形式に戻す
        X_restored = self._restore_data_format(X_restored, X_inverse)
        
        return X_restored
    
    def _get_target_columns(self, X: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        """標準化対象の列を決定"""
        if isinstance(X, pd.DataFrame):
            if self.target_columns is not None:
                # 指定された列名をチェック
                missing_cols = [col for col in self.target_columns if col not in X.columns]
                if missing_cols:
                    raise ValueError(f"Specified target columns not found: {missing_cols}")
                return self.target_columns
            else:
                # 全ての数値列を対象（時間列は除外）
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                if self.time_column and self.time_column in numeric_cols:
                    numeric_cols.remove(self.time_column)
                return numeric_cols
        else:
            # numpy配列の場合
            feature_names = self._get_feature_names_from_input(X)
            if self.target_columns is not None:
                # 指定されたインデックスが範囲内かチェック
                n_features = len(feature_names)
                target_indices = []
                for col in self.target_columns:
                    if isinstance(col, str) and col.startswith('feature_'):
                        try:
                            idx = int(col.split('_')[1])
                            if 0 <= idx < n_features:
                                target_indices.append(col)
                        except (ValueError, IndexError):
                            pass
                    elif isinstance(col, int) and 0 <= col < n_features:
                        target_indices.append(f'feature_{col}')
                return target_indices
            else:
                return feature_names
    
    def _extract_target_data(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """標準化対象のデータを配列として抽出"""
        if isinstance(X, pd.DataFrame):
            # DataFrameの場合、対象列を選択して2次元配列として返す
            target_data = X[self.target_columns_]
            return target_data.values
        else:
            # numpy配列の場合、target_columns_は feature_0, feature_1 のような形式
            indices = []
            for col in self.target_columns_:
                if col.startswith('feature_'):
                    try:
                        idx = int(col.split('_')[1])
                        indices.append(idx)
                    except (ValueError, IndexError):
                        pass
            if indices:
                # 指定されたインデックスの列のみを抽出
                return X[:, indices]
            else:
                return X
    
    def _restore_data_format(self, X_original: Union[pd.DataFrame, np.ndarray], 
                           X_transformed: np.ndarray) -> Union[pd.DataFrame, np.ndarray]:
        """変換結果を元の形式に復元"""
        if isinstance(X_original, pd.DataFrame):
            # DataFrameの場合、対象列のみを置き換え
            X_result = X_original.copy() if self.copy else X_original
            # 変換された配列を対象列に代入（列の順序を保持）
            X_result[self.target_columns_] = X_transformed
            return X_result
        else:
            # numpy配列の場合
            X_result = X_original.copy() if self.copy else X_original
            # target_columns_からインデックスを取得
            indices = []
            for col in self.target_columns_:
                if col.startswith('feature_'):
                    try:
                        idx = int(col.split('_')[1])
                        indices.append(idx)
                    except (ValueError, IndexError):
                        pass
            
            if indices and len(indices) == X_transformed.shape[1]:
                # インデックス順序で列を復元
                X_result[:, indices] = X_transformed
            else:
                # 全列が対象の場合
                if X_transformed.shape[1] == X_result.shape[1]:
                    X_result[:] = X_transformed
            
            return X_result
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """変換後の特徴量名を取得（sklearn互換）"""
        if hasattr(self, 'target_columns_'):
            return self.target_columns_
        return super().get_feature_names_out(input_features)
    
    def get_statistics(self) -> dict:
        """fit期間の統計量を取得"""
        self._check_is_fitted(['scaler_'])
        
        stats = {
            'target_columns': self.target_columns_,
            'with_mean': self.with_mean,
            'with_std': self.with_std,
        }
        
        if hasattr(self, 'mean_'):
            stats['mean'] = self.mean_
        if hasattr(self, 'scale_'):
            stats['scale'] = self.scale_
        if hasattr(self, 'var_'):
            stats['var'] = self.var_
            
        return stats