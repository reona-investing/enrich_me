"""
統計的変換処理の実装
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from scipy import stats
import warnings

from .base import StatisticalTransformer, TimeSeriesTransformer
from ..contracts.input_contracts import get_input_contract
from ..contracts.output_contracts import get_output_contract


class StatisticalMomentsTransformer(StatisticalTransformer):
    """統計的モーメント（平均、分散、歪度、尖度）変換器"""
    
    def __init__(self, windows: List[int] = [21, 60],
                 moments: List[str] = ['mean', 'std', 'skew', 'kurt'],
                 base_column: str = '1d_return',
                 **kwargs):
        super().__init__(windows=windows, **kwargs)
        self.moments = moments
        self.base_column = base_column
    
    @property
    def output_column_names(self) -> List[str]:
        columns = []
        for window in self.windows:
            for moment in self.moments:
                columns.append(f'{window}d_{moment}')
        return columns
    
    @property
    def required_input_columns(self) -> List[str]:
        return [self.base_column]
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """統計的モーメントを算出"""
        if self.base_column not in df.columns:
            raise ValueError(f"必要なカラム '{self.base_column}' が存在しません")
        
        result = df.copy()
        
        for window in self.windows:
            series = df[self.base_column]
            
            if self.group_column in df.index.names:
                # セクター別に計算
                for moment in self.moments:
                    col_name = f'{window}d_{moment}'
                    if moment == 'mean':
                        result[col_name] = series.groupby(level=self.group_column).rolling(
                            window=window, min_periods=window//2
                        ).mean().reset_index(0, drop=True)
                    elif moment == 'std':
                        result[col_name] = series.groupby(level=self.group_column).rolling(
                            window=window, min_periods=window//2
                        ).std().reset_index(0, drop=True)
                    elif moment == 'skew':
                        result[col_name] = series.groupby(level=self.group_column).rolling(
                            window=window, min_periods=window//2
                        ).skew().reset_index(0, drop=True)
                    elif moment == 'kurt':
                        result[col_name] = series.groupby(level=self.group_column).rolling(
                            window=window, min_periods=window//2
                        ).kurt().reset_index(0, drop=True)
            else:
                # 全体で計算
                for moment in self.moments:
                    col_name = f'{window}d_{moment}'
                    if moment == 'mean':
                        result[col_name] = series.rolling(window=window, min_periods=window//2).mean()
                    elif moment == 'std':
                        result[col_name] = series.rolling(window=window, min_periods=window//2).std()
                    elif moment == 'skew':
                        result[col_name] = series.rolling(window=window, min_periods=window//2).skew()
                    elif moment == 'kurt':
                        result[col_name] = series.rolling(window=window, min_periods=window//2).kurt()
        
        return result


class CorrelationTransformer(TimeSeriesTransformer):
    """相関係数変換器"""
    
    def __init__(self, reference_columns: List[str],
                 target_columns: Optional[List[str]] = None,
                 windows: List[int] = [21, 60],
                 correlation_method: str = 'pearson',
                 **kwargs):
        super().__init__(**kwargs)
        self.reference_columns = reference_columns
        self.target_columns = target_columns
        self.windows = windows
        self.correlation_method = correlation_method
    
    @property
    def output_column_names(self) -> List[str]:
        columns = []
        ref_cols = self.reference_columns
        target_cols = self.target_columns or ['1d_return']
        
        for window in self.windows:
            for ref_col in ref_cols:
                for target_col in target_cols:
                    columns.append(f'{target_col}_vs_{ref_col}_{window}d_corr')
        return columns
    
    @property
    def required_input_columns(self) -> List[str]:
        required = self.reference_columns.copy()
        if self.target_columns:
            required.extend(self.target_columns)
        else:
            required.append('1d_return')
        return required
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """相関係数を算出"""
        result = df.copy()
        
        target_cols = self.target_columns or ['1d_return']
        
        for window in self.windows:
            for ref_col in self.reference_columns:
                if ref_col not in df.columns:
                    warnings.warn(f"参照列 '{ref_col}' が存在しません")
                    continue
                
                for target_col in target_cols:
                    if target_col not in df.columns:
                        warnings.warn(f"対象列 '{target_col}' が存在しません")
                        continue
                    
                    corr_col = f'{target_col}_vs_{ref_col}_{window}d_corr'
                    
                    if self.group_column in df.index.names:
                        # セクター別に計算
                        def rolling_corr(group):
                            return group[target_col].rolling(
                                window=window, min_periods=window//2
                            ).corr(group[ref_col])
                        
                        result[corr_col] = df.groupby(level=self.group_column).apply(
                            rolling_corr
                        ).reset_index(0, drop=True)
                    else:
                        # 全体で計算
                        result[corr_col] = df[target_col].rolling(
                            window=window, min_periods=window//2
                        ).corr(df[ref_col])
        
        return result


class QuantileTransformer(StatisticalTransformer):
    """分位数変換器"""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
                 windows: List[int] = [21, 60],
                 base_column: str = '1d_return',
                 **kwargs):
        super().__init__(windows=windows, **kwargs)
        self.quantiles = quantiles
        self.base_column = base_column
    
    @property
    def output_column_names(self) -> List[str]:
        columns = []
        for window in self.windows:
            for q in self.quantiles:
                q_str = f"{int(q*100):02d}"
                columns.append(f'{window}d_q{q_str}')
        return columns
    
    @property
    def required_input_columns(self) -> List[str]:
        return [self.base_column]
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """分位数を算出"""
        if self.base_column not in df.columns:
            raise ValueError(f"必要なカラム '{self.base_column}' が存在しません")
        
        result = df.copy()
        
        for window in self.windows:
            for q in self.quantiles:
                q_str = f"{int(q*100):02d}"
                col_name = f'{window}d_q{q_str}'
                
                if self.group_column in df.index.names:
                    # セクター別に計算
                    result[col_name] = df.groupby(level=self.group_column)[self.base_column].rolling(
                        window=window, min_periods=window//2
                    ).quantile(q).reset_index(0, drop=True)
                else:
                    # 全体で計算
                    result[col_name] = df[self.base_column].rolling(
                        window=window, min_periods=window//2
                    ).quantile(q)
        
        return result


class ZScoreTransformer(StatisticalTransformer):
    """Z-スコア変換器"""
    
    def __init__(self, windows: List[int] = [21, 60],
                 target_columns: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(windows=windows, **kwargs)
        self.target_columns = target_columns
    
    @property
    def output_column_names(self) -> List[str]:
        if self.target_columns:
            columns = []
            for window in self.windows:
                for col in self.target_columns:
                    columns.append(f'{col}_{window}d_zscore')
            return columns
        else:
            return []  # 実行時に動的決定
    
    @property
    def required_input_columns(self) -> List[str]:
        return self.target_columns or []
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Z-スコアを算出"""
        result = df.copy()
        
        # 対象カラムを決定
        if self.target_columns:
            target_cols = [col for col in self.target_columns if col in df.columns]
        else:
            # 数値カラムを自動選択
            target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for window in self.windows:
            for col in target_cols:
                zscore_col = f'{col}_{window}d_zscore'
                
                if self.group_column in df.index.names:
                    # セクター別に計算
                    def rolling_zscore(series):
                        rolling_mean = series.rolling(window=window, min_periods=window//2).mean()
                        rolling_std = series.rolling(window=window, min_periods=window//2).std()
                        return (series - rolling_mean) / rolling_std
                    
                    result[zscore_col] = df.groupby(level=self.group_column)[col].apply(
                        rolling_zscore
                    ).reset_index(0, drop=True)
                else:
                    # 全体で計算
                    rolling_mean = df[col].rolling(window=window, min_periods=window//2).mean()
                    rolling_std = df[col].rolling(window=window, min_periods=window//2).std()
                    result[zscore_col] = (df[col] - rolling_mean) / rolling_std
        
        return result


class PercentileRankTransformer(StatisticalTransformer):
    """パーセンタイルランク変換器"""
    
    def __init__(self, windows: List[int] = [21, 60],
                 target_columns: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(windows=windows, **kwargs)
        self.target_columns = target_columns
    
    @property
    def output_column_names(self) -> List[str]:
        if self.target_columns:
            columns = []
            for window in self.windows:
                for col in self.target_columns:
                    columns.append(f'{col}_{window}d_pctrank')
            return columns
        else:
            return []  # 実行時に動的決定
    
    @property
    def required_input_columns(self) -> List[str]:
        return self.target_columns or []
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """パーセンタイルランクを算出"""
        result = df.copy()
        
        # 対象カラムを決定
        if self.target_columns:
            target_cols = [col for col in self.target_columns if col in df.columns]
        else:
            # 数値カラムを自動選択
            target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for window in self.windows:
            for col in target_cols:
                pctrank_col = f'{col}_{window}d_pctrank'
                
                if self.group_column in df.index.names:
                    # セクター別に計算
                    def rolling_percentrank(series):
                        def pct_rank(x):
                            if len(x) < 2:
                                return np.nan
                            return (x.rank(method='average') - 1) / (len(x) - 1)
                        
                        return series.rolling(window=window, min_periods=window//2).apply(
                            lambda x: pct_rank(x).iloc[-1], raw=False
                        )
                    
                    result[pctrank_col] = df.groupby(level=self.group_column)[col].apply(
                        rolling_percentrank
                    ).reset_index(0, drop=True)
                else:
                    # 全体で計算
                    def pct_rank(x):
                        if len(x) < 2:
                            return np.nan
                        return (x.rank(method='average') - 1) / (len(x) - 1)
                    
                    result[pctrank_col] = df[col].rolling(
                        window=window, min_periods=window//2
                    ).apply(lambda x: pct_rank(x).iloc[-1], raw=False)
        
        return result


class OutlierDetectionTransformer(StatisticalTransformer):
    """外れ値検出変換器"""
    
    def __init__(self, windows: List[int] = [21, 60],
                 methods: List[str] = ['iqr', 'zscore'],
                 target_columns: Optional[List[str]] = None,
                 thresholds: Dict[str, float] = {'iqr': 1.5, 'zscore': 3.0},
                 **kwargs):
        super().__init__(windows=windows, **kwargs)
        self.methods = methods
        self.target_columns = target_columns
        self.thresholds = thresholds
    
    @property
    def output_column_names(self) -> List[str]:
        if self.target_columns:
            columns = []
            for window in self.windows:
                for col in self.target_columns:
                    for method in self.methods:
                        columns.append(f'{col}_{window}d_{method}_outlier')
            return columns
        else:
            return []  # 実行時に動的決定
    
    @property
    def required_input_columns(self) -> List[str]:
        return self.target_columns or []
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """外れ値フラグを算出"""
        result = df.copy()
        
        # 対象カラムを決定
        if self.target_columns:
            target_cols = [col for col in self.target_columns if col in df.columns]
        else:
            # 数値カラムを自動選択
            target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for window in self.windows:
            for col in target_cols:
                for method in self.methods:
                    outlier_col = f'{col}_{window}d_{method}_outlier'
                    threshold = self.thresholds.get(method, 3.0)
                    
                    if self.group_column in df.index.names:
                        # セクター別に計算
                        def detect_outliers(series):
                            if method == 'iqr':
                                rolling_q25 = series.rolling(window=window, min_periods=window//2).quantile(0.25)
                                rolling_q75 = series.rolling(window=window, min_periods=window//2).quantile(0.75)
                                rolling_iqr = rolling_q75 - rolling_q25
                                lower_bound = rolling_q25 - threshold * rolling_iqr
                                upper_bound = rolling_q75 + threshold * rolling_iqr
                                return (series < lower_bound) | (series > upper_bound)
                            elif method == 'zscore':
                                rolling_mean = series.rolling(window=window, min_periods=window//2).mean()
                                rolling_std = series.rolling(window=window, min_periods=window//2).std()
                                zscore = np.abs((series - rolling_mean) / rolling_std)
                                return zscore > threshold
                            else:
                                return pd.Series(False, index=series.index)
                        
                        result[outlier_col] = df.groupby(level=self.group_column)[col].apply(
                            detect_outliers
                        ).reset_index(0, drop=True)
                    else:
                        # 全体で計算
                        if method == 'iqr':
                            rolling_q25 = df[col].rolling(window=window, min_periods=window//2).quantile(0.25)
                            rolling_q75 = df[col].rolling(window=window, min_periods=window//2).quantile(0.75)
                            rolling_iqr = rolling_q75 - rolling_q25
                            lower_bound = rolling_q25 - threshold * rolling_iqr
                            upper_bound = rolling_q75 + threshold * rolling_iqr
                            result[outlier_col] = (df[col] < lower_bound) | (df[col] > upper_bound)
                        elif method == 'zscore':
                            rolling_mean = df[col].rolling(window=window, min_periods=window//2).mean()
                            rolling_std = df[col].rolling(window=window, min_periods=window//2).std()
                            zscore = np.abs((df[col] - rolling_mean) / rolling_std)
                            result[outlier_col] = zscore > threshold
        
        return result


class TrendAnalysisTransformer(StatisticalTransformer):
    """トレンド分析変換器"""
    
    def __init__(self, windows: List[int] = [21, 60],
                 trend_methods: List[str] = ['linear_slope', 'trend_strength'],
                 base_column: str = 'Close',
                 **kwargs):
        super().__init__(windows=windows, **kwargs)
        self.trend_methods = trend_methods
        self.base_column = base_column
    
    @property
    def output_column_names(self) -> List[str]:
        columns = []
        for window in self.windows:
            for method in self.trend_methods:
                columns.append(f'{window}d_{method}')
        return columns
    
    @property
    def required_input_columns(self) -> List[str]:
        return [self.base_column]
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """トレンド指標を算出"""
        if self.base_column not in df.columns:
            raise ValueError(f"必要なカラム '{self.base_column}' が存在しません")
        
        result = df.copy()
        
        for window in self.windows:
            for method in self.trend_methods:
                col_name = f'{window}d_{method}'
                
                if self.group_column in df.index.names:
                    # セクター別に計算
                    def calc_trend(series):
                        if method == 'linear_slope':
                            return series.rolling(window=window, min_periods=window//2).apply(
                                self._calculate_linear_slope, raw=False
                            )
                        elif method == 'trend_strength':
                            return series.rolling(window=window, min_periods=window//2).apply(
                                self._calculate_trend_strength, raw=False
                            )
                        else:
                            return pd.Series(np.nan, index=series.index)
                    
                    result[col_name] = df.groupby(level=self.group_column)[self.base_column].apply(
                        calc_trend
                    ).reset_index(0, drop=True)
                else:
                    # 全体で計算
                    if method == 'linear_slope':
                        result[col_name] = df[self.base_column].rolling(
                            window=window, min_periods=window//2
                        ).apply(self._calculate_linear_slope, raw=False)
                    elif method == 'trend_strength':
                        result[col_name] = df[self.base_column].rolling(
                            window=window, min_periods=window//2
                        ).apply(self._calculate_trend_strength, raw=False)
        
        return result
    
    def _calculate_linear_slope(self, series: pd.Series) -> float:
        """線形トレンドの傾きを計算"""
        if len(series) < 2:
            return np.nan
        
        x = np.arange(len(series))
        y = series.values
        
        # 有効な値のみを使用
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 2:
            return np.nan
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        try:
            slope, _, _, _, _ = stats.linregress(x_valid, y_valid)
            return slope
        except:
            return np.nan
    
    def _calculate_trend_strength(self, series: pd.Series) -> float:
        """トレンドの強さを計算（R²値）"""
        if len(series) < 2:
            return np.nan
        
        x = np.arange(len(series))
        y = series.values
        
        # 有効な値のみを使用
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 2:
            return np.nan
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        try:
            _, _, r_value, _, _ = stats.linregress(x_valid, y_valid)
            return r_value ** 2  # R²値
        except:
            return np.nan


class RollingBetaTransformer(StatisticalTransformer):
    """ローリングベータ変換器"""
    
    def __init__(self, benchmark_column: str,
                 target_columns: Optional[List[str]] = None,
                 windows: List[int] = [21, 60],
                 **kwargs):
        super().__init__(windows=windows, **kwargs)
        self.benchmark_column = benchmark_column
        self.target_columns = target_columns
    
    @property
    def output_column_names(self) -> List[str]:
        if self.target_columns:
            columns = []
            for window in self.windows:
                for col in self.target_columns:
                    columns.append(f'{col}_{window}d_beta')
            return columns
        else:
            return []  # 実行時に動的決定
    
    @property
    def required_input_columns(self) -> List[str]:
        required = [self.benchmark_column]
        if self.target_columns:
            required.extend(self.target_columns)
        return required
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """ローリングベータを算出"""
        if self.benchmark_column not in df.columns:
            raise ValueError(f"ベンチマークカラム '{self.benchmark_column}' が存在しません")
        
        result = df.copy()
        
        # 対象カラムを決定
        if self.target_columns:
            target_cols = [col for col in self.target_columns if col in df.columns]
        else:
            # 数値カラムを自動選択（ベンチマークカラムを除く）
            target_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                          if col != self.benchmark_column]
        
        for window in self.windows:
            for col in target_cols:
                beta_col = f'{col}_{window}d_beta'
                
                def calc_beta(target_series, benchmark_series):
                    if len(target_series) < 2 or len(benchmark_series) < 2:
                        return np.nan
                    
                    valid_mask = target_series.notna() & benchmark_series.notna()
                    if valid_mask.sum() < 2:
                        return np.nan
                    
                    target_valid = target_series[valid_mask]
                    benchmark_valid = benchmark_series[valid_mask]
                    
                    covariance = np.cov(target_valid, benchmark_valid)[0, 1]
                    benchmark_variance = np.var(benchmark_valid)
                    
                    if benchmark_variance == 0:
                        return np.nan
                    
                    return covariance / benchmark_variance
                
                if self.group_column in df.index.names:
                    # セクター別に計算
                    def rolling_beta(group):
                        betas = []
                        for i in range(len(group)):
                            start_idx = max(0, i - window + 1)
                            end_idx = i + 1
                            
                            target_window = group[col].iloc[start_idx:end_idx]
                            benchmark_window = group[self.benchmark_column].iloc[start_idx:end_idx]
                            
                            if len(target_window) >= window // 2:
                                beta = calc_beta(target_window, benchmark_window)
                            else:
                                beta = np.nan
                            
                            betas.append(beta)
                        
                        return pd.Series(betas, index=group.index)
                    
                    result[beta_col] = df.groupby(level=self.group_column).apply(
                        rolling_beta
                    ).reset_index(0, drop=True)
                else:
                    # 全体で計算
                    betas = []
                    for i in range(len(df)):
                        start_idx = max(0, i - window + 1)
                        end_idx = i + 1
                        
                        target_window = df[col].iloc[start_idx:end_idx]
                        benchmark_window = df[self.benchmark_column].iloc[start_idx:end_idx]
                        
                        if len(target_window) >= window // 2:
                            beta = calc_beta(target_window, benchmark_window)
                        else:
                            beta = np.nan
                        
                        betas.append(beta)
                    
                    result[beta_col] = betas
        
        return result


# ファクトリー関数
def get_statistical_transformer(transformer_type: str, **kwargs) -> StatisticalTransformer:
    """統計変換器タイプに応じた変換器を取得"""
    transformers = {
        'statistical_moments': StatisticalMomentsTransformer,
        'correlation': CorrelationTransformer,
        'quantile': QuantileTransformer,
        'zscore': ZScoreTransformer,
        'percentile_rank': PercentileRankTransformer,
        'outlier_detection': OutlierDetectionTransformer,
        'trend_analysis': TrendAnalysisTransformer,
        'rolling_beta': RollingBetaTransformer
    }
    
    if transformer_type not in transformers:
        raise ValueError(f"未対応の統計変換器タイプです: {transformer_type}")
    
    transformer_class = transformers[transformer_type]
    return transformer_class(**kwargs)


# 便利な組み合わせ関数
def create_statistical_analysis_transformers(
    base_column: str = '1d_return',
    windows: List[int] = [21, 60],
    include_moments: bool = True,
    include_quantiles: bool = True,
    include_zscore: bool = True,
    include_trend: bool = False
) -> List[StatisticalTransformer]:
    """統計分析用変換器のセットを作成"""
    
    transformers = []
    
    # 統計的モーメント
    if include_moments:
        transformers.append(StatisticalMomentsTransformer(
            windows=windows,
            base_column=base_column,
            moments=['mean', 'std', 'skew']
        ))
    
    # 分位数
    if include_quantiles:
        transformers.append(QuantileTransformer(
            windows=windows,
            base_column=base_column,
            quantiles=[0.1, 0.25, 0.75, 0.9]
        ))
    
    # Z-スコア
    if include_zscore:
        transformers.append(ZScoreTransformer(
            windows=windows,
            target_columns=[base_column]
        ))
    
    # トレンド分析
    if include_trend:
        transformers.append(TrendAnalysisTransformer(
            windows=windows,
            base_column='Close',  # 価格系列でトレンド分析
            trend_methods=['linear_slope', 'trend_strength']
        ))
    
    return transformers


def create_risk_analysis_transformers(
    base_column: str = '1d_return',
    benchmark_column: str = 'market_return',
    windows: List[int] = [21, 60]
) -> List[StatisticalTransformer]:
    """リスク分析用変換器のセットを作成"""
    
    transformers = [
        # ボラティリティ・リスク指標
        StatisticalMomentsTransformer(
            windows=windows,
            base_column=base_column,
            moments=['std', 'skew', 'kurt']
        ),
        
        # 外れ値検出
        OutlierDetectionTransformer(
            windows=windows,
            methods=['iqr', 'zscore'],
            target_columns=[base_column]
        ),
        
        # 分位数リスク
        QuantileTransformer(
            windows=windows,
            base_column=base_column,
            quantiles=[0.05, 0.1, 0.9, 0.95]  # VaR関連の分位数
        ),
        
        # ベータ計算
        RollingBetaTransformer(
            benchmark_column=benchmark_column,
            target_columns=[base_column],
            windows=windows
        )
    ]
    
    return transformers


def create_performance_analysis_transformers(
    base_column: str = '1d_return',
    benchmark_column: str = 'market_return',
    windows: List[int] = [21, 60, 252]
) -> List[StatisticalTransformer]:
    """パフォーマンス分析用変換器のセットを作成"""
    
    transformers = [
        # リターン統計
        StatisticalMomentsTransformer(
            windows=windows,
            base_column=base_column,
            moments=['mean', 'std']
        ),
        
        # 相関分析
        CorrelationTransformer(
            reference_columns=[benchmark_column],
            target_columns=[base_column],
            windows=windows
        ),
        
        # パーセンタイルランク
        PercentileRankTransformer(
            windows=windows,
            target_columns=[base_column]
        ),
        
        # ベータ分析
        RollingBetaTransformer(
            benchmark_column=benchmark_column,
            target_columns=[base_column],
            windows=windows
        )
    ]
    
    return transformers