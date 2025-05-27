"""
前処理エンジンの具体実装
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from data_processing.core.preprocessors.base import (
    Preprocessor, NoPreprocessor, StatisticalPreprocessor, 
    TimeSeriesPreprocessor, TrainTestSplitPreprocessor
)
from data_processing.core.contracts.input_contracts import get_input_contract
from data_processing.core.contracts.output_contracts import get_output_contract


class PCAMarketFactorRemovalPreprocessor(TrainTestSplitPreprocessor):
    """PCAによるマーケットファクター除去前処理"""
    
    def __init__(self, n_components: int, train_start: datetime, train_end: datetime, **kwargs):
        input_contract = get_input_contract('multi_return')
        output_contract = get_output_contract('pca', n_components=n_components, output_type='residuals')
        super().__init__(train_start=train_start, train_end=train_end, 
                        input_contract=input_contract, output_contract=output_contract, **kwargs)
        self.n_components = n_components
        self.pca_model = None
        self.fitted_data = None
    
    def _fit_on_train_data(self, train_data: pd.DataFrame, **kwargs) -> None:
        """訓練期間でPCAモデルを学習"""
        if train_data.index.nlevels != 2:
            raise ValueError('インデックスは2階層にしてください。')
        
        # データを準備
        target_column = kwargs.get('target_column', 'Target')
        df_for_pca = train_data[target_column].unstack(-1)
        df_for_pca = df_for_pca.dropna()
        
        if df_for_pca.empty:
            raise ValueError("訓練データが空です")
        
        # PCAモデルを学習
        self.pca_model = PCA(n_components=self.n_components)
        self.pca_model.fit(df_for_pca)
        self.fitted_data = df_for_pca
        
        # 学習統計を記録（StatisticalPreprocessorのメソッドを使用）
        self.fitted_statistics = {
            'training_data_shape': df_for_pca.shape,
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_.tolist(),
            'total_explained_variance': self.pca_model.explained_variance_ratio_.sum()
        }
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """PCA残差を算出"""
        if not self.is_fitted or self.pca_model is None:
            raise ValueError("PCAモデルが学習されていません")
        
        target_column = 'Target'  # デフォルト
        if target_column not in df.columns:
            target_column = df.columns[0]
        
        # データを準備
        df_for_pca = df[target_column].unstack(-1)
        
        # 欠損値を処理
        df_for_pca = df_for_pca.ffill().bfill()
        
        # PCA変換と逆変換
        extracted_array = self.pca_model.transform(df_for_pca)
        inverse_array = self.pca_model.inverse_transform(extracted_array)
        
        # 逆変換結果をDataFrameに
        inverse_df = pd.DataFrame(inverse_array, 
                                index=df_for_pca.index, 
                                columns=df_for_pca.columns)
        
        # 残差を計算
        residuals = (df_for_pca - inverse_df).stack()
        residuals_df = pd.DataFrame(residuals).reset_index()
        result = residuals_df.rename(columns={0: target_column}).set_index(['Date', 'Sector'])
        
        return result
    
    def get_principal_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """主成分のみを抽出（分析用）"""
        if not self.is_fitted or self.pca_model is None:
            raise ValueError("PCAモデルが学習されていません")
        
        target_column = 'Target'
        if target_column not in df.columns:
            target_column = df.columns[0]
        
        df_for_pca = df[target_column].unstack(-1)
        df_for_pca = df_for_pca.fillna(method='ffill').fillna(method='bfill')
        
        extracted_array = self.pca_model.transform(df_for_pca)
        result = pd.DataFrame(extracted_array, 
                            index=df_for_pca.index,
                            columns=[f'PC_{i:02d}' for i in range(self.n_components)])
        
        return result.sort_index()
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """各主成分の寄与率を取得"""
        if self.pca_model is None:
            raise ValueError("PCAモデルが学習されていません")
        return self.pca_model.explained_variance_ratio_
    
    def _get_model_state(self) -> Dict[str, Any]:
        return {
            'n_components': self.n_components,
            'pca_model': self.pca_model,
            'fitted_data_shape': self.fitted_data.shape if self.fitted_data is not None else None,
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_ if self.pca_model else None
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        self.n_components = state['n_components']
        self.pca_model = state['pca_model']


class PCADimensionReductionPreprocessor(StatisticalPreprocessor):
    """PCAによる次元削減前処理"""
    
    def __init__(self, n_components: Optional[int] = None, 
                 explained_variance_ratio: float = 0.95, **kwargs):
        input_contract = get_input_contract('multi_return')
        output_contract = get_output_contract('pca', 
                                            n_components=n_components or 10, 
                                            output_type='components')
        super().__init__(input_contract=input_contract, output_contract=output_contract, **kwargs)
        self.n_components = n_components
        self.explained_variance_ratio = explained_variance_ratio
        self.pca_model = None
        self.actual_n_components = None
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'PCADimensionReductionPreprocessor':
        """PCA次元削減モデルを学習"""
        self.validate_input(df)
        
        # 数値データのみを選択
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.dropna()
        
        if numeric_df.empty:
            raise ValueError("数値データが存在しません")
        
        # 成分数を決定
        if self.n_components is None:
            # 寄与率に基づいて成分数を自動決定
            temp_pca = PCA()
            temp_pca.fit(numeric_df)
            cumsum_ratio = np.cumsum(temp_pca.explained_variance_ratio_)
            self.actual_n_components = np.argmax(cumsum_ratio >= self.explained_variance_ratio) + 1
        else:
            self.actual_n_components = min(self.n_components, numeric_df.shape[1])
        
        # PCAモデルを学習
        self.pca_model = PCA(n_components=self.actual_n_components)
        self.pca_model.fit(numeric_df)
        
        self.is_fitted = True
        self._fit_metadata = {
            'fit_time': datetime.now(),
            'input_shape': numeric_df.shape,
            'n_components': self.actual_n_components,
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_.sum(),
            'individual_variance_ratios': self.pca_model.explained_variance_ratio_.tolist()
        }
        
        # 統計情報を記録
        self._record_fit_statistics(numeric_df)
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """PCA次元削減を適用"""
        if not self.is_fitted or self.pca_model is None:
            raise ValueError("PCAモデルが学習されていません")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        # 欠損値を処理
        numeric_df = numeric_df.fillna(method='ffill').fillna(method='bfill')
        
        # PCA変換
        transformed_array = self.pca_model.transform(numeric_df)
        
        # 結果をDataFrameに
        result = pd.DataFrame(
            transformed_array,
            index=numeric_df.index,
            columns=[f'PC_{i:02d}' for i in range(self.actual_n_components)]
        )
        
        return result
    
    def _get_model_state(self) -> Dict[str, Any]:
        return {
            'n_components': self.n_components,
            'explained_variance_ratio': self.explained_variance_ratio,
            'actual_n_components': self.actual_n_components,
            'pca_model': self.pca_model
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        self.n_components = state['n_components']
        self.explained_variance_ratio = state['explained_variance_ratio']
        self.actual_n_components = state['actual_n_components']
        self.pca_model = state['pca_model']


class StandardizationPreprocessor(StatisticalPreprocessor):
    """標準化前処理"""
    
    def __init__(self, method: str = 'standard', **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.scaler = None
        
        # スケーラーを選択
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"未対応の標準化手法です: {method}")
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'StandardizationPreprocessor':
        """標準化パラメータを学習"""
        self.validate_input(df)
        
        target_cols = self._get_target_columns(df)
        numeric_df = df[target_cols].select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("標準化対象の数値データが存在しません")
        
        # 欠損値を除いて学習
        clean_df = numeric_df.dropna()
        self.scaler.fit(clean_df)
        
        self.is_fitted = True
        self._fit_metadata = {
            'fit_time': datetime.now(),
            'method': self.method,
            'input_shape': clean_df.shape,
            'target_columns': target_cols
        }
        
        self._record_fit_statistics(clean_df)
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """標準化を適用"""
        if not self.is_fitted or self.scaler is None:
            raise ValueError("スケーラーが学習されていません")
        
        result = df.copy()
        target_cols = self._get_target_columns(df)
        numeric_cols = [col for col in target_cols if col in df.columns]
        
        if numeric_cols:
            # 欠損値を一時的に補完
            temp_df = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
            
            # 標準化を適用
            scaled_array = self.scaler.transform(temp_df)
            result[numeric_cols] = scaled_array
        
        return result
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """標準化を逆変換"""
        if not self.is_fitted or self.scaler is None:
            raise ValueError("スケーラーが学習されていません")
        
        result = df.copy()
        target_cols = self._get_target_columns(df)
        numeric_cols = [col for col in target_cols if col in df.columns]
        
        if numeric_cols:
            inversed_array = self.scaler.inverse_transform(df[numeric_cols])
            result[numeric_cols] = inversed_array
        
        return result
    
    def _get_model_state(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'scaler': self.scaler,
            'target_columns': self.target_columns
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        self.method = state['method']
        self.scaler = state['scaler']
        self.target_columns = state['target_columns']


class OutlierRemovalPreprocessor(StatisticalPreprocessor):
    """外れ値除去前処理"""
    
    def __init__(self, method: str = 'iqr', threshold: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold
        self.outlier_bounds = {}
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'OutlierRemovalPreprocessor':
        """外れ値の境界を学習"""
        self.validate_input(df)
        
        target_cols = self._get_target_columns(df)
        self.outlier_bounds = {}
        
        for col in target_cols:
            if col in df.columns:
                series = df[col].dropna()
                
                if self.method == 'iqr':
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.threshold * IQR
                    upper_bound = Q3 + self.threshold * IQR
                elif self.method == 'zscore':
                    mean = series.mean()
                    std = series.std()
                    lower_bound = mean - self.threshold * std
                    upper_bound = mean + self.threshold * std
                else:
                    raise ValueError(f"未対応の外れ値除去手法です: {self.method}")
                
                self.outlier_bounds[col] = {
                    'lower': lower_bound,
                    'upper': upper_bound
                }
        
        self.is_fitted = True
        self._fit_metadata = {
            'fit_time': datetime.now(),
            'method': self.method,
            'threshold': self.threshold,
            'outlier_bounds': self.outlier_bounds
        }
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """外れ値を除去またはクリッピング"""
        if not self.is_fitted:
            raise ValueError("外れ値境界が学習されていません")
        
        result = df.copy()
        
        for col, bounds in self.outlier_bounds.items():
            if col in result.columns:
                result[col] = result[col].clip(bounds['lower'], bounds['upper'])
        
        return result
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """外れ値を検出（除去はしない）"""
        if not self.is_fitted:
            raise ValueError("外れ値境界が学習されていません")
        
        outlier_flags = pd.DataFrame(index=df.index)
        
        for col, bounds in self.outlier_bounds.items():
            if col in df.columns:
                outlier_flags[f'{col}_outlier'] = (
                    (df[col] < bounds['lower']) | (df[col] > bounds['upper'])
                )
        
        return outlier_flags
    
    def _get_model_state(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'threshold': self.threshold,
            'outlier_bounds': self.outlier_bounds,
            'target_columns': self.target_columns
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        self.method = state['method']
        self.threshold = state['threshold']
        self.outlier_bounds = state['outlier_bounds']
        self.target_columns = state['target_columns']


class MissingValueImputationPreprocessor(TimeSeriesPreprocessor):
    """欠損値補完前処理"""
    
    def __init__(self, method: str = 'ffill', **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.fill_values = {}
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'MissingValueImputationPreprocessor':
        """補完値を学習"""
        self.validate_input(df)
        
        target_cols = self._get_target_columns(df)
        self.fill_values = {}
        
        if self.method in ['mean', 'median', 'mode']:
            for col in target_cols:
                if col in df.columns:
                    series = df[col].dropna()
                    if self.method == 'mean':
                        self.fill_values[col] = series.mean()
                    elif self.method == 'median':
                        self.fill_values[col] = series.median()
                    elif self.method == 'mode':
                        mode_val = series.mode()
                        self.fill_values[col] = mode_val.iloc[0] if not mode_val.empty else 0
        
        self.is_fitted = True
        self._fit_metadata = {
            'fit_time': datetime.now(),
            'method': self.method,
            'fill_values': self.fill_values
        }
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値を補完"""
        result = df.copy()
        
        if self.method == 'ffill':
            if self.group_column in df.index.names:
                result = result.groupby(level=self.group_column).fillna(method='ffill')
            else:
                result = result.fillna(method='ffill')
        elif self.method == 'bfill':
            if self.group_column in df.index.names:
                result = result.groupby(level=self.group_column).fillna(method='bfill')
            else:
                result = result.fillna(method='bfill')
        elif self.method in ['mean', 'median', 'mode']:
            for col, fill_val in self.fill_values.items():
                if col in result.columns:
                    result[col] = result[col].fillna(fill_val)
        elif self.method == 'interpolate':
            if self.group_column in df.index.names:
                result = result.groupby(level=self.group_column).apply(
                    lambda x: x.interpolate()
                )
            else:
                result = result.interpolate()
        
        return result
    
    def _get_model_state(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'fill_values': self.fill_values,
            'group_column': self.group_column,
            'target_columns': self.target_columns
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        self.method = state['method']
        self.fill_values = state['fill_values']
        self.group_column = state['group_column']
        self.target_columns = state['target_columns']


# ファクトリー関数
def get_preprocessor(preprocessor_type: str, **kwargs) -> Preprocessor:
    """前処理タイプに応じた前処理器を取得"""
    preprocessors = {
        'none': NoPreprocessor,
        'pca_market_factor_removal': PCAMarketFactorRemovalPreprocessor,
        'pca_dimension_reduction': PCADimensionReductionPreprocessor,
        'standardization': StandardizationPreprocessor,
        'outlier_removal': OutlierRemovalPreprocessor,
        'missing_value_imputation': MissingValueImputationPreprocessor
    }
    
    if preprocessor_type not in preprocessors:
        raise ValueError(f"未対応の前処理タイプです: {preprocessor_type}")
    
    preprocessor_class = preprocessors[preprocessor_type]
    return preprocessor_class(**kwargs)