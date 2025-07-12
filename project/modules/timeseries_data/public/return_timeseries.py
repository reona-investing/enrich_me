import pandas as pd
from typing import Optional
from timeseries_data.calculation_method import CalculationMethodBase
from timeseries_data.preprocessing import PreprocessingPipeline


class ReturnTimeseries:
    """
    株価リターン（raw / processed）の計算と保持を行うユーティリティ。

    Parameters
    ----------
    original_timeseries : pd.DataFrame
        元となる時系列データ（行：日時 or (日時×セクター)）。
    calculated_column : str
        計算結果を格納する列名（例：'Return'）。
    name: str, default ''
        インスタンスの名前。空白の場合はcalculated_columnを流用。
    date_column : str, default 'Date'
        日時列の名称。
    sector_column : Optional[str], default 'Sector'
        セクター列の名称。None の場合は単一セクター扱い。
    open_column : str, default 'Open'
        始値列の名称。
    close_column : str, default 'Close'
        終値列の名称。
    """
    def __init__(self, original_timeseries: pd.DataFrame,
                 name: str = '',
                 date_column: str = 'Date', sector_column: Optional[str] = 'Sector',
                 open_column: str = 'Open', close_column: str = 'Close') -> None: 
        self._name = name
        self._original_timeseries = original_timeseries.copy()
        self._date_column = date_column
        self._sector_column = sector_column
        self._open_column = open_column
        self._close_column = close_column
        # 計算後にセットされる属性
        self._calculated_column: Optional[str] = None
        self._raw_return: pd.DataFrame = pd.DataFrame()
        self._processed_return: pd.DataFrame = pd.DataFrame()

    def calculate(self, method: CalculationMethodBase, *args, **kwargs):
        """
        指定されたメソッドでリターンを計算します。
        
        Args:
            method: 計算メソッドのインスタンス
            *args, **kwargs: calculateメソッドに渡す追加引数
        """
        # 必要な列をあらかじめ規定しておく
        index_cols = [self._date_column]
        if self._sector_column is not None:
            index_cols.append(self._sector_column)
        extract_cols = index_cols + [self._open_column, self._close_column]

        original_timeseries = self._original_timeseries.reset_index().copy()
        original_timeseries = original_timeseries[extract_cols].set_index(index_cols, drop=True)
        if self._sector_column is not None:
            self._raw_return = \
                original_timeseries.groupby(self._sector_column, group_keys=False).apply(
                    lambda group: method.calculate(group),
                    ).reset_index(drop=False).set_index(index_cols)
        else:
            self._raw_return = method.calculate(original_timeseries)

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
        
        if self._sector_column is None:
            processed_return = self._processed_return.copy()
        else:
            processed_return = self._processed_return.unstack(-1)
            if not isinstance(processed_return, pd.DataFrame):
                raise TypeError("raw_returnが予期しない形状で保存されています。calculate()を実行してください。")

        pipeline.fit(processed_return) 
        processed_return = pipeline.transform(processed_return)
        if self._sector_column is not None:
            processed_return.columns = pd.MultiIndex.from_tuples(processed_return.columns, names=['', self._sector_column])
            processed_return = processed_return.stack(future_stack=True)
            if not isinstance(processed_return, pd.DataFrame):
                raise TypeError("raw_returnが予期しない形状で保存されています。calculate()を実行してください。")
        self._processed_return = processed_return
    
    @property
    def name(self) -> str:
        """インスタンスの名称を兼ねます"""
        return self._name

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
            raise ValueError("calculate()およびprocessed()メソッドを先に実行してください。")
        return self._processed_return.copy()

    @property
    def statistic_data(self) -> pd.DataFrame:
        """raw_targetとprocessed_targetそれぞれの統計量を返します。"""
        if self._raw_return is None or self._processed_return is None:
            raise ValueError("calculate()およびprocessed()メソッドを先に実行してください。")
        raw_statistics = self._raw_return.describe()
        raw_statistics.columns = ['Raw']
        processed_statistics = self._processed_return.describe()
        processed_statistics.columns = ['Processed']
        return pd.concat([raw_statistics, processed_statistics], axis=1)