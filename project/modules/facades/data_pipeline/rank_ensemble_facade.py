from machine_learning.ensembles import EnsembleMethodFactory
#from machine_learning.ml_dataset.core import MLDataset
from machine_learning.ml_dataset import MLDataset
from typing import List, Literal

class RankEnsembleFacade:
    def __init__(self, mode: Literal["train_and_predict", "predict_only", "load_only", "none"], ensembled_dataset_path: str, datasets: List[MLDataset], ensemble_rates: List[float]):
        self.mode = mode
        self.ensemble_dataset_path = ensembled_dataset_path
        self.ensemble_method = EnsembleMethodFactory().create_method(method_name='by_rank')
        self.main_ml_dataset = datasets[0]
        if len(datasets) > 0:
            self.inputs = [(dataset.pred_result_df, ensemble_rate) for dataset, ensemble_rate in zip(datasets, ensemble_rates) if dataset]

    def execute(self) -> MLDataset | None:
        if self.mode == "train_and_predict" or self.mode == "predict_only":
            pred_result_df = self.main_ml_dataset.pred_result_df.groupby('Date')[['Target']].rank(ascending=False)
            pred_result_df['Pred'] = self.ensemble_method.ensemble(self.inputs)
            self.main_ml_dataset.output_collection.pred_result_df = pred_result_df
            self.main_ml_dataset.dataset_path = self.ensemble_dataset_path
            self.main_ml_dataset.save()
            return self.main_ml_dataset
        else:
            return None