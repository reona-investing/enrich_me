from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime

from data_processing.core.contracts import (
    BaseInputContract, BaseOutputContract
)


class Transformer(ABC):
    """変換処理の基底クラス"""
    
    def __init__(self, input_contract: Optional[BaseInputContract] = None,
                 output_contract: Optional[BaseOutputContract] = None):
        self.input_contract = input_contract
        self.output_contract = output_contract
        self._last_transformation_metadata = {}
    
    @abstractmethod
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        変換処理を実行する
        
        Args:
            df: 入力データフレーム
            **kwargs: 変換に必要な追加パラメータ
            
        Returns:
            変換済みデータフレーム
        """
        pass
    
    @property
    @abstractmethod
    def output_column_names(self) -> List[str]:
        """出力列名のリスト"""
        pass
    
    @property
    @abstractmethod
    def required_input_columns(self) -> List[str]:
        """必要な入力列名のリスト"""
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
        変換処理の完全なパイプライン実行
        
        Args:
            df: 入力データフレーム
            **kwargs: 変換パラメータ
            
        Returns:
            検証済み・整形済みの変換データフレーム
        """
        # 入力検証
        self.validate_input(df)
        
        # 変換実行
        result = self.transform(df, **kwargs)
        
        # メタデータを記録
        self._record_transformation_metadata(df, result, **kwargs)
        
        # 出力検証・整形
        return self.validate_output(result)
    
    def _record_transformation_metadata(self, input_df: pd.DataFrame, 
                                      output_df: pd.DataFrame, **kwargs) -> None:
        """変換のメタデータを記録"""
        self._last_transformation_metadata = {
            'input_shape': input_df.shape,
            'output_shape': output_df.shape,
            'transformation_time': datetime.now(),
            'parameters': kwargs,
            'added_columns': list(set(output_df.columns) - set(input_df.columns)),
            'modified_columns': list(set(output_df.columns) & set(input_df.columns))
        }
    
    def get_transformation_metadata(self) -> Dict[str, Any]:
        """最後の変換のメタデータを取得"""
        return self._last_transformation_metadata.copy()
    
    def _handle_missing_data(self, df: pd.DataFrame, method: str = 'skip') -> pd.DataFrame:
        """欠損データの処理"""
        if method == 'skip':
            return df
        elif method == 'drop':
            return df.dropna(subset=self.required_input_columns)
        elif method == 'ffill':
            return df.fillna(method='ffill')
        elif method == 'bfill':
            return df.fillna(method='bfill')
        else:
            return df


class FinancialTransformer(Transformer):
    """金融データ変換の基底クラス"""
    
    def __init__(self, group_column: str = 'Sector', **kwargs):
        super().__init__(**kwargs)
        self.group_column = group_column
    
    def _apply_grouped_transformation(self, df: pd.DataFrame, 
                                    transform_func, **kwargs) -> pd.DataFrame:
        """グループ別に変換を適用"""
        if self.group_column in df.index.names:
            # マルチインデックスの場合
            return df.groupby(level=self.group_column).apply(
                lambda x: transform_func(x, **kwargs)
            )
        elif self.group_column in df.columns:
            # 通常のカラムの場合
            return df.groupby(self.group_column).apply(
                lambda x: transform_func(x, **kwargs)
            ).reset_index(level=0, drop=True)
        else:
            # グループ列がない場合は全体に適用
            return transform_func(df, **kwargs)


class TimeSeriesTransformer(FinancialTransformer):
    """時系列データ変換の基底クラス"""
    
    def __init__(self, lookback_periods: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.lookback_periods = lookback_periods
    
    def _ensure_time_sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間軸でソートを確保"""
        if 'Date' in df.index.names:
            return df.sort_index(level='Date')
        elif df.index.dtype == 'datetime64[ns]':
            return df.sort_index()
        else:
            return df
    
    def _apply_rolling_transformation(self, series: pd.Series, 
                                    transform_func, window: int = None, 
                                    min_periods: int = None, **kwargs) -> pd.Series:
        """ローリング変換を適用"""
        window = window or self.lookback_periods
        min_periods = min_periods or window
        
        return series.rolling(
            window=window,
            min_periods=min_periods
        ).apply(transform_func, **kwargs)
    
    def _shift_for_lookahead_bias_prevention(self, df: pd.DataFrame, 
                                           columns: List[str],
                                           shift_periods: int = 1) -> pd.DataFrame:
        """ルックアヘッドバイアス防止のためのシフト処理"""
        result = df.copy()
        
        if self.group_column in df.index.names:
            # グループ別にシフト
            for col in columns:
                if col in result.columns:
                    result[col] = result.groupby(level=self.group_column)[col].shift(shift_periods)
        else:
            # 全体でシフト
            for col in columns:
                if col in result.columns:
                    result[col] = result[col].shift(shift_periods)
        
        return result


class StatisticalTransformer(TimeSeriesTransformer):
    """統計指標変換の基底クラス"""
    
    def __init__(self, windows: List[int] = [5, 21], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
    
    def _calculate_multiple_windows(self, series: pd.Series, 
                                  calc_func, **kwargs) -> pd.DataFrame:
        """複数ウィンドウで統計指標を計算"""
        results = pd.DataFrame(index=series.index)
        
        for window in self.windows:
            column_name = self._generate_column_name(calc_func.__name__, window, **kwargs)
            results[column_name] = self._apply_rolling_transformation(
                series, calc_func, window=window, **kwargs
            )
        
        return results
    
    def _generate_column_name(self, metric_name: str, window: int, **kwargs) -> str:
        """カラム名を生成"""
        base_name = kwargs.get('base_column', 'value')
        return f'{base_name}_{window}d_{metric_name}'


class RankingTransformer(FinancialTransformer):
    """ランキング変換の基底クラス"""
    
    def __init__(self, ranking_method: str = 'dense', ascending: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.ranking_method = ranking_method
        self.ascending = ascending
    
    def _apply_cross_sectional_ranking(self, df: pd.DataFrame, 
                                     columns: List[str]) -> pd.DataFrame:
        """横断面ランキングを適用"""
        result = df.copy()
        
        for col in columns:
            if col in df.columns:
                if 'Date' in df.index.names:
                    # 日付別にランキング
                    result[f'{col}_rank'] = df.groupby(level='Date')[col].rank(
                        method=self.ranking_method, ascending=self.ascending
                    )
                else:
                    # 全体でランキング
                    result[f'{col}_rank'] = df[col].rank(
                        method=self.ranking_method, ascending=self.ascending
                    )
        
        return result


class CompositeTransformer(Transformer):
    """複数の変換器を組み合わせる基底クラス"""
    
    def __init__(self, transformers: List[Transformer], **kwargs):
        super().__init__(**kwargs)
        self.transformers = transformers
        self.step_results = []
    
    @property
    def output_column_names(self) -> List[str]:
        """全変換器の出力列名を結合"""
        all_columns = []
        for transformer in self.transformers:
            all_columns.extend(transformer.output_column_names)
        return all_columns
    
    @property
    def required_input_columns(self) -> List[str]:
        """全変換器の必要入力列名を結合"""
        all_columns = set()
        for transformer in self.transformers:
            all_columns.update(transformer.required_input_columns)
        return list(all_columns)
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """各変換器を順次適用"""
        result = df.copy()
        self.step_results = []
        
        for i, transformer in enumerate(self.transformers):
            step_params = kwargs.get(f'step_{i}_params', {})
            step_result = transformer.execute(result, **step_params)
            
            # 結果をマージ（新しいカラムを追加）
            new_columns = set(step_result.columns) - set(result.columns)
            if new_columns:
                result = pd.concat([result, step_result[list(new_columns)]], axis=1)
            
            self.step_results.append(step_result.copy())
        
        return result
    
    def get_step_result(self, step_index: int) -> pd.DataFrame:
        """指定したステップの結果を取得"""
        if not self.step_results:
            raise ValueError("transform()が実行されていません")
        
        if step_index >= len(self.step_results):
            raise ValueError(f"ステップインデックスが範囲外です: {step_index}")
        
        return self.step_results[step_index]


class ConditionalTransformer(Transformer):
    """条件付き変換の基底クラス"""
    
    def __init__(self, condition_func, true_transformer: Transformer, 
                 false_transformer: Optional[Transformer] = None, **kwargs):
        super().__init__(**kwargs)
        self.condition_func = condition_func
        self.true_transformer = true_transformer
        self.false_transformer = false_transformer
        self.active_transformer = None
    
    @property
    def output_column_names(self) -> List[str]:
        """アクティブな変換器の出力列名"""
        if self.active_transformer:
            return self.active_transformer.output_column_names
        else:
            # 両方の可能性を返す
            true_cols = self.true_transformer.output_column_names
            false_cols = self.false_transformer.output_column_names if self.false_transformer else []
            return list(set(true_cols + false_cols))
    
    @property
    def required_input_columns(self) -> List[str]:
        """両方の変換器の必要入力列名を結合"""
        true_cols = set(self.true_transformer.required_input_columns)
        false_cols = set(self.false_transformer.required_input_columns) if self.false_transformer else set()
        return list(true_cols | false_cols)
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """条件に応じて適切な変換器を適用"""
        if self.condition_func(df):
            self.active_transformer = self.true_transformer
        else:
            self.active_transformer = self.false_transformer
            if self.active_transformer is None:
                # false_transformerが指定されていない場合は元のデータを返す
                return df
        
        return self.active_transformer.execute(df, **kwargs)


class CachingTransformer(Transformer):
    """キャッシュ機能付き変換器"""
    
    def __init__(self, base_transformer: Transformer, cache_size: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.base_transformer = base_transformer
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []
    
    @property
    def output_column_names(self) -> List[str]:
        return self.base_transformer.output_column_names
    
    @property
    def required_input_columns(self) -> List[str]:
        return self.base_transformer.required_input_columns
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """キャッシュを利用した変換"""
        # キャッシュキーを生成
        cache_key = self._generate_cache_key(df, **kwargs)
        
        # キャッシュヒットの確認
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        # 変換実行
        result = self.base_transformer.execute(df, **kwargs)
        
        # キャッシュに保存
        self._update_cache(cache_key, result)
        
        return result
    
    def _generate_cache_key(self, df: pd.DataFrame, **kwargs) -> str:
        """キャッシュキーを生成"""
        import hashlib
        
        # データフレームのハッシュを計算
        df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
        
        # パラメータのハッシュを計算
        params_str = str(sorted(kwargs.items()))
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        return f"{df_hash}_{params_hash}"
    
    def _update_cache(self, key: str, value: pd.DataFrame) -> None:
        """キャッシュを更新"""
        # キャッシュサイズ制限
        if len(self.cache) >= self.cache_size and key not in self.cache:
            # 最も古いエントリを削除
            oldest_key = self.cache_order.pop(0)
            del self.cache[oldest_key]
        
        # 新しいエントリを追加
        self.cache[key] = value.copy()
        if key not in self.cache_order:
            self.cache_order.append(key)
    
    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self.cache.clear()
        self.cache_order.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'cache_keys': list(self.cache.keys())
        }