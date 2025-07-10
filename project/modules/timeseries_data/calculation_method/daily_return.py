from timeseries_data.calculation_method.base import CalculationMethodBase
import pandas as pd

class DailyReturn(CalculationMethodBase):
    def __init__(self, return_column: str = 'Target', close_column: str = 'Close'):
        '''
        Args:
            return_column (str): 目的変数をどのようなカラム名で返すか
            close_column (str): 終値のカラム名
        '''
        self._return_column = return_column
        self._close_column = close_column

    def calculate(self, return_timeseries: pd.DataFrame) -> pd.DataFrame:
        '''
        日次リターンを算出します。
        注意：あらかじめ日付昇順に並んでいる必要があります！

        Args:
            return_timeseries (pd.DataFrame): 価格データのDataFrame

        Returns:
            pd.DataFrame: 日次リターンが計算されたDataFrame (インデックスはreturn_timeseriesと一致)
        '''
        result_df = return_timeseries.copy()
        result_df[self._return_column] = return_timeseries[self._close_column] / return_timeseries[self._close_column].shift(1) - 1
        return result_df[[self._return_column]]