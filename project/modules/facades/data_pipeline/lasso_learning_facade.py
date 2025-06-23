from __future__ import annotations

from typing import Literal

from models.machine_learning.loaders.loader import DatasetLoader
from models.machine_learning.models.lasso_model import LassoModel
from models.machine_learning.ml_dataset.ml_datasets import MLDatasets


class LassoLearningFacade:
    """LASSOモデルの学習・予測を担当するシンプルなファサード"""

    def __init__(
        self,
        mode: Literal["train_and_predict", "predict_only", "load_only", "none"],
        dataset_path: str,
    ) -> None:
        self.mode = mode
        self.dataset_path = dataset_path

    def execute(self) -> MLDatasets | None:
        if self.mode == "none":
            return None

        loader = DatasetLoader(self.dataset_path)
        ml_datasets = loader.load_datasets()

        if self.mode == "train_and_predict":
            self._train(ml_datasets)
            self._predict(ml_datasets)
        elif self.mode == "predict_only":
            self._predict(ml_datasets)
        elif self.mode == "load_only":
            pass
        else:
            raise NotImplementedError

        return ml_datasets

    def _train(self, ml_datasets: MLDatasets) -> None:
        model = LassoModel()
        for _, single_ml in ml_datasets.items():
            trainer_outputs = model.train(
                single_ml.train_test_materials.target_train_df,
                single_ml.train_test_materials.features_train_df,
            )
            single_ml.archive_ml_objects(trainer_outputs.model, trainer_outputs.scaler)
            single_ml.save()
            ml_datasets.replace_model(single_ml_dataset=single_ml)

    def _predict(self, ml_datasets: MLDatasets) -> None:
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
