import pandas as pd

from utils.timeseries import Duration
from timeseries_data.public import StockReturnTimeseries
from timeseries_data.calculation_method import IntradayReturn
from timeseries_data.preprocessing import PreprocessingPipeline, PCAHandler

class Pc1RemovedIntradayReturn:

    def __init__(self, original_timeseries: pd.DataFrame,
                 date_column: str = 'Date', sector_column: str = 'Sector',
                 open_column: str = 'Open', close_column: str = 'Close',
                 target_column: str = 'Target'):
        self._return_timeseries = StockReturnTimeseries(original_timeseries=original_timeseries,
                                                       date_column=date_column, sector_column=sector_column,
                                                       open_column=open_column, close_column=close_column,
                                                       target_column=target_column)
        self._is_calculated = False

    def calculate(self, fit_duration: Duration):
        self._return_timeseries.calculate(IntradayReturn())
        ppp = PreprocessingPipeline([
            ('remove_pc1', PCAHandler(n_components=1, mode='residuals', fit_duration=fit_duration))
        ])
        self._return_timeseries.preprocess(pipeline=ppp)
        self._is_calculated = True
    
    @property
    def processed_return(self):
        if not self._is_calculated:
            raise ValueError("calculate()メソッドを先に実行してください。")
        return self._return_timeseries.processed_return
    
    @property
    def raw_return(self):
        if not self._is_calculated:
            raise ValueError("calculate()メソッドを先に実行してください。")
        return self._return_timeseries.raw_return