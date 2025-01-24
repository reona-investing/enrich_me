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
        materials = ml_dataset.get_materials_for_evaluation()
        self.ls_data_handler = LSDataHandler(materials.pred_result_df, materials.raw_target_df, start_day = start_day, end_day = end_day)
        self.met_calc = MetricsCalculator(self.ls_data_handler, trade_sector_num, bin_num, top_slope)
        self.visualizer = Visualizer(self.met_calc)
    def display(self):
        self.visualizer.display_result()