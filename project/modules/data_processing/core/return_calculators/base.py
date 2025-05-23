"""
リターン算出エンジンの基底クラス
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime

from data_processing.core.contracts import (
    BaseInputContract, BaseOutputContract
)

class ReturnCalculator(ABC):
    """リターン算出の基底クラス"""
    
    def __init__(self, input_contract: Optional[BaseInputContract] = None,
                 output_contract: Optional[BaseOutputContract] = None):
        self.input_contract = input_contract
        self.output_contract = output_contract
        self._last_calculation_metadata = {}
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        リターンを算出する
        
        Args:
            df: 入力データフレーム
            **kwargs: 計算に必要な追加パラメータ
            
        Returns:
            リターンを含むデータフレーム
        """
        pass
    
    @property
    @abstractmethod
    def output_column_name(self) -> str:
        """出力列名"""
        pass
    
    @property
    @abstractmethod
    def required_input_columns(self) -> list[str]:
        """必要な入力列名"""
        pass
    
    def validate_input(self, df: pd.DataFrame) -> None:
        """入力データの検証"""
        # 必須カラムの存在確認
        missing_columns = set(self.required_input_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"必須カラムが不足しています: {missing_columns}")
        
        # 契約ベースの検証
        if self.input_contract:
            self.input_contract.validate(df)
    
    def validate_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """出力データの検証と整形"""
        if self.output_contract:
            self.output_contract.validate_output(df)
            return self.output_contract.format_output(df)
        return df
    
    def execute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        リターン算出の完全なパイプライン実行
        
        Args:
            df: 入力データフレーム
            **kwargs: 計算パラメータ
            
        Returns:
            検証済み・整形済みのリターンデータフレーム
        """
        # 入力検証
        self.validate_input(df)
        
        # 計算実行
        result = self.calculate(df, **kwargs)
        
        # メタデータを記録
        self._record_calculation_metadata(df, result, **kwargs)
        
        # 出力検証・整形
        return self.validate_output(result)
    
    def _record_calculation_metadata(self, input_df: pd.DataFrame, 
                                   output_df: pd.DataFrame, **kwargs) -> None:
        """計算のメタデータを記録"""
        self._last_calculation_metadata = {
            'input_shape': input_df.shape,
            'output_shape': output_df.shape,
            'calculation_time': datetime.now(),
            'parameters': kwargs,
            'non_null_ratio': output_df[self.output_column_name].notna().mean(),
            'value_range': {
                'min': output_df[self.output_column_name].min(),
                'max': output_df[self.output_column_name].max(),
                'mean': output_df[self.output_column_name].mean(),
                'std': output_df[self.output_column_name].std()
            }
        }
    
    def get_calculation_metadata(self) -> Dict[str, Any]:
        """最後の計算のメタデータを取得"""
        return self._last_calculation_metadata.copy()
    
    def _handle_missing_data(self, df: pd.DataFrame, method: str = 'drop') -> pd.DataFrame:
        """欠損データの処理"""
        if method == 'drop':
            return df.dropna(subset=self.required_input_columns)
        elif method == 'ffill':
            return df.fillna(method='ffill')
        elif method == 'bfill':
            return df.fillna(method='bfill')
        else:
            return df
    
    def _apply_outlier_detection(self, series: pd.Series, 
                               method: str = 'iqr', 
                               threshold: float = 3.0) -> pd.Series:
        """外れ値の検出と処理"""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return series.clip(lower_bound, upper_bound)
        elif method == 'zscore':
            mean = series.mean()
            std = series.std()
            return series.clip(mean - threshold * std, mean + threshold * std)
        else:
            return series


class TimeSeriesReturnCalculator(ReturnCalculator):
    """時系列データ用リターン算出の基底クラス"""
    
    def __init__(self, input_contract: Optional[BaseInputContract] = None,
                 output_contract: Optional[BaseOutputContract] = None,
                 handle_missing: str = 'drop',
                 detect_outliers: bool = False,
                 outlier_method: str = 'iqr'):
        super().__init__(input_contract, output_contract)
        self.handle_missing = handle_missing
        self.detect_outliers = detect_outliers
        self.outlier_method = outlier_method
    
    def execute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """時系列データ特有の前処理を含む実行"""
        # 入力検証
        self.validate_input(df)
        
        # 時系列データの前処理
        processed_df = self._preprocess_timeseries(df)
        
        # 計算実行
        result = self.calculate(processed_df, **kwargs)
        
        # 後処理
        result = self._postprocess_timeseries(result)
        
        # メタデータ記録
        self._record_calculation_metadata(df, result, **kwargs)
        
        # 出力検証・整形
        return self.validate_output(result)
    
    def _preprocess_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """時系列データの前処理"""
        processed_df = df.copy()
        
        # 日付でソート
        if 'Date' in processed_df.index.names:
            processed_df = processed_df.sort_index(level='Date')
        
        # 欠損データの処理
        if self.handle_missing != 'none':
            processed_df = self._handle_missing_data(processed_df, self.handle_missing)
        
        return processed_df
    
    def _postprocess_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """時系列データの後処理"""
        processed_df = df.copy()
        
        # 外れ値の処理
        if self.detect_outliers and self.output_column_name in processed_df.columns:
            if processed_df.index.nlevels > 1:
                # マルチインデックスの場合、グループごとに処理
                processed_df[self.output_column_name] = processed_df.groupby(
                    level=list(range(1, processed_df.index.nlevels))
                )[self.output_column_name].transform(
                    lambda x: self._apply_outlier_detection(x, self.outlier_method)
                )
            else:
                processed_df[self.output_column_name] = self._apply_outlier_detection(
                    processed_df[self.output_column_name], self.outlier_method
                )
        
        return processed_df


class GroupedReturnCalculator(TimeSeriesReturnCalculator):
    """グループ別リターン算出の基底クラス（セクター別等）"""
    
    def __init__(self, group_column: str = 'Sector', **kwargs):
        super().__init__(**kwargs)
        self.group_column = group_column
    
    def _apply_grouped_calculation(self, df: pd.DataFrame, 
                                 calc_func, **kwargs) -> pd.DataFrame:
        """グループ別に計算を適用"""
        if self.group_column in df.index.names:
            # マルチインデックスの場合
            return df.groupby(level=self.group_column).apply(
                lambda x: calc_func(x, **kwargs)
            ).reset_index(level=0, drop=True)
        elif self.group_column in df.columns:
            # 通常のカラムの場合
            return df.groupby(self.group_column).apply(
                lambda x: calc_func(x, **kwargs)
            ).reset_index(level=0, drop=True)
        else:
            # グループ列がない場合は全体に適用
            return calc_func(df, **kwargs)


class RollingReturnCalculator(GroupedReturnCalculator):
    """ローリング計算機能付きリターン算出基底クラス"""
    
    def __init__(self, window: int = 1, min_periods: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.min_periods = min_periods or window
    
    def _apply_rolling_calculation(self, series: pd.Series, 
                                 calc_func, **kwargs) -> pd.Series:
        """ローリング計算を適用"""
        return series.rolling(
            window=self.window, 
            min_periods=self.min_periods
        ).apply(calc_func, **kwargs)