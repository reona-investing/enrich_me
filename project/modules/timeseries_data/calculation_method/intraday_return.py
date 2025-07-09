from timeseries_data.calculation_method.base import CalculationMethodBase
import pandas as pd

class IntradayReturn(CalculationMethodBase):
    def __init__(self):
        pass

    def calculate(self, return_timeseries: pd.DataFrame, open_column: str = 'Open', close_column: str = 'Close'):
        '''
        日内リターンを算出します。
        注意：あらかじめ日付昇順に並んでいる必要があります！

        Args:

        Returns:
        
        '''
        return return_timeseries[close_column] / return_timeseries[open_column] - 1