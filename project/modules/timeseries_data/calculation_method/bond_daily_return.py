from timeseries_data.calculation_method.base import CalculationMethodBase
import pandas as pd

class BondDailyReturn(CalculationMethodBase):
    """債券利回りの日次変化量を計算するクラス"""

    def __init__(self, return_column: str = 'Target', close_column: str = 'Close'):
        """
        Args:
            return_column (str): 返り値のカラム名
            close_column (str): 終値のカラム名
        """
        self._return_column = return_column
        self._close_column = close_column

    def calculate(self, return_timeseries: pd.DataFrame) -> pd.DataFrame:
        """当日終値と前日終値の差分でリターンを計算します。"""
        result_df = return_timeseries.copy()
        result_df[self._return_column] = (
            return_timeseries[self._close_column] - return_timeseries[self._close_column].shift(1)
        )
        return result_df[[self._return_column]]
