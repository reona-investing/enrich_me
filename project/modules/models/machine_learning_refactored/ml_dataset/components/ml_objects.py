from typing import Optional
from models.machine_learning_refactored.ml_dataset.components.base_data_component import BaseDataComponent
from models.machine_learning_refactored.outputs import TrainerOutputs

class MLObjects(BaseDataComponent):
    """機械学習オブジェクトの管理"""
    
    instance_vars = {
        'model': '.pkl',
        'scaler': '.pkl',
    }

    def archive_ml_objects(self, model: object, scaler: Optional[object] = None):
        """機械学習のモデルとスケーラーを格納"""
        self._model = model
        self._scaler = scaler

    def getter(self) -> TrainerOutputs:
        """データクラスとして返却"""
        return TrainerOutputs(model=self._model, scaler=self._scaler)