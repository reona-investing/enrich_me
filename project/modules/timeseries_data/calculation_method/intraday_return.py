from timeseries_data.calculation_method.base import CalculationMethodBase
import pandas as pd

class IntradayReturn(CalculationMethodBase):
    def __init__(self):
        pass

    def calculate(self, return_timeseries: pd.DataFrame, 
                  return_column: str = 'Target', open_column: str = 'Open', close_column: str = 'Close', *args, **kwargs) -> pd.DataFrame:
        '''
        日内リターンを算出します。
        注意：あらかじめ日付昇順に並んでいる必要があります！

        Args:
            return_timeseries (pd.DataFrame): 価格データのDataFrame
            open_column (str): 始値のカラム名
            close_column (str): 終値のカラム名

        Returns:
            pd.DataFrame: 日内リターンが計算されたDataFrame
        '''
        result_df = return_timeseries.copy()
        result_df[return_column] = return_timeseries[close_column] / return_timeseries[open_column] - 1
        return result_df