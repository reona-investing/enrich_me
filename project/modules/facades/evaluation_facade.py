from models.evaluation import LSDataHandler, MetricsCalculator, Visualizer
from models import MLDataset
from datetime import datetime

class EvaluationFacade:
    def __init__(self, 
                 ml_dataset: MLDataset, 
                 start_day: datetime,
                 end_day: datetime,
                 trade_sector_num: int = 3,
                 bin_num: int = None,
                 top_slope: float = 1.0,
                 ):
        self.ls_data_handler = LSDataHandler(ml_dataset, start_day = datetime(2022, 1, 1), end_day = datetime.today())
        self.met_calc = MetricsCalculator(self.ls_data_handler, bin_num=5)
        self.visualizer = Visualizer(self.met_calc)
        self.visualizer.display_result()