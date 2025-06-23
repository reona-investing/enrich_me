from __future__ import annotations

from typing import Literal

from models.machine_learning.loaders.loader import DatasetLoader
from models.machine_learning.models.lasso_model import LassoModel
from models.machine_learning.ml_dataset.ml_datasets import MLDatasets


class LassoLearningFacade:
    """LASSOモデルの予測処理だけを担当するシンプルなファサード"""

    def __init__(self, mode: Literal["predict_only", "none"], dataset_path: str) -> None:
        self.mode = mode
        self.dataset_path = dataset_path

    def execute(self) -> MLDatasets:
        loader = DatasetLoader(self.dataset_path)
        ml_datasets = loader.load_datasets()
        if self.mode == "predict_only":
            model = LassoModel()
            for _, single_ml in ml_datasets.items():
                pred_df = model.predict(
                    single_ml.train_test_materials.target_test_df,
                    single_ml.train_test_materials.features_test_df,
                    single_ml.ml_object_materials.model,
                    single_ml.ml_object_materials.scaler,
                )
                single_ml.archive_pred_result(pred_df)
                single_ml.save()
                ml_datasets.replace_model(single_ml_dataset=single_ml)
        return ml_datasets
