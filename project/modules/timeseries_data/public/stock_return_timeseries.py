import pandas as pd
from typing import Union, Optional
from datetime import datetime
from timeseries_data.calculation_method import CalculationMethodBase
from timeseries_data.preprocessing import PreprocessingPipeline
from utils.timeseries import Duration


class StockReturnTimeseries:
    def __init__(self, original_timeseries: pd.DataFrame, 
                 date_column: str = 'Date', sector_column: str = 'Sector',
                 open_column: str = 'Open', close_column: str = 'Close',
                 target_column: str = 'Target'):
        self._original_timeseries = original_timeseries.copy()
        self._date_column = date_column
        self._sector_column = sector_column
        self._open_column = open_column
        self._close_column = close_column
        self._target_column = target_column
        self._raw_return = None
        self._processed_return = None

    def calculate(self, method: CalculationMethodBase, *args, **kwargs):
        """
        指定されたメソッドでリターンを計算します。
        
        Args:
            method: 計算メソッドのインスタンス
            *args, **kwargs: calculateメソッドに渡す追加引数
        """
        original_timeseries = self._original_timeseries.reset_index().copy()
        original_timeseries = original_timeseries[
            [self._date_column, self._sector_column, self._open_column, self._close_column]
            ].set_index([self._date_column, self._sector_column], drop=True)
        self._raw_return = \
            original_timeseries.groupby(self._sector_column, group_keys=False).apply(
                lambda group: method.calculate(group),
                ).reset_index(drop=False).set_index([self._date_column, self._sector_column])
        
        self._processed_return = self._raw_return.copy() # processed_returnは初期状態ではraw_returnと同じ
    
    def preprocess(self, pipeline: PreprocessingPipeline):
        """
        前処理パイプラインを実行します。
        
        Args:
            pipeline: PreprocessingPipelineのインスタンス
        """
        if self._processed_return is None:
            raise ValueError("calculate()メソッドを先に実行してください。")
        if not isinstance(pipeline, PreprocessingPipeline):
            raise ValueError("pipelineにはPreprocessingPipelineインスタンスを指定してください。")
        
        processed_return = self._processed_return.unstack(-1)

        pipeline.fit(processed_return) 
        processed_return = pipeline.transform(processed_return)
        processed_return.columns = pd.MultiIndex.from_tuples(processed_return.columns, names=['', self._sector_column])
        self._processed_return = processed_return.stack(future_stack=True)
    
    @property
    def raw_return(self) -> pd.DataFrame:
        """生のリターンデータを返します。"""
        if self._raw_return is None:
            raise ValueError("calculate()メソッドを先に実行してください。")
        return self._raw_return.copy()
    
    @property
    def processed_return(self) -> pd.DataFrame:
        """前処理済みのリターンデータを返します。"""
        if self._processed_return is None:
            raise ValueError("calculate()メソッドを先に実行してください。")
        return self._processed_return.copy()

    def evaluate(self):
        """評価メソッド（将来の実装用）"""
        pass