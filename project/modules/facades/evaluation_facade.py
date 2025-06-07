from models.evaluation import LSDataHandler, MetricsCalculator, Visualizer
from datetime import datetime
import pandas as pd

class EvaluationFacade:
    def __init__(self, 
                 pred_result_df: pd.DataFrame, 
                 raw_target_df: pd.DataFrame,
                 start_day: datetime,
                 end_day: datetime,
                 trade_sector_num: int = 3,
                 bin_num: int = None,
                 top_slope: float = 1.0,
                 ):
        self.ls_data_handler = LSDataHandler(pred_result_df, raw_target_df, start_day = start_day, end_day = end_day)
        self.met_calc = MetricsCalculator(self.ls_data_handler, trade_sector_num, bin_num, top_slope)
        self.visualizer = Visualizer(self.met_calc)
    def display(self):
        self.visualizer.display_result()